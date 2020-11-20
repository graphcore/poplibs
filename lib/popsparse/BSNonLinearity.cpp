// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "BSNonLinearity.hpp"
#include "BSOps.hpp"
#include "BSUtils.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/ElementWiseUtil.hpp"
#include "popops/Fill.hpp"
#include "popops/Reduce.hpp"
#include "popops/Zero.hpp"
#include "popsparse/experimental/BlockSparse.hpp"
#include <popops/Pad.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;
using namespace popsparse::experimental;

namespace logging = poplibs_support::logging;

#define MIN_HALF_VALUE -65504.0
#define MIN_FLOAT_VALUE -3.4028235e+38

namespace poputil {
template <>
poplar::ProfileValue
toProfileValue(const popsparse::experimental::SubBlockMask &t) {
  switch (t) {
  case popsparse::experimental::SubBlockMask::None:
    return poplar::ProfileValue("None");
  case popsparse::experimental::SubBlockMask::ZeroUpperTriangle:
    return poplar::ProfileValue("ZeroUpperTriangle");
  case popsparse::experimental::SubBlockMask::ZeroLowerTriangle:
    return poplar::ProfileValue("ZeroLowerTriangle");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
}
} // namespace poputil

namespace popsparse {
namespace experimental {

struct SparseToPaddedSlices {
  std::vector<Tensor> slicesAsRectCopyVec;
  std::vector<std::vector<unsigned>> origRowMappingVec;
  Tensor sparseTensorCopyBack;
};

static SparseToPaddedSlices sliceAndPad(Graph &graph, Tensor sparseTensor,
                                        unsigned blockRow, unsigned blockCol,
                                        unsigned blockRows, unsigned blockCols,
                                        const unsigned char *sparsity,
                                        float padValue, Sequence &prog,
                                        const DebugNameAndId &dnai) {
  const unsigned rows = blockRow * blockRows;
  const auto dataType = sparseTensor.elementType();

  std::vector<std::size_t> slicesLen;
  std::vector<Tensor> slices;
  std::size_t maxSliceLen = 0;
  for (unsigned r = 0; r < rows; ++r) {
    Tensor curSlice = slice(sparseTensor, r, 0, blockRow, blockCol, blockRows,
                            blockCols, false, sparsity);
    assert(curSlice.rank() == 1);
    slices.push_back(curSlice);
    slicesLen.push_back(curSlice.dim(0));
    maxSliceLen = std::max(maxSliceLen, curSlice.dim(0));
  }

  logging::popsparse::debug("Maximum slice length = {}", maxSliceLen);

  SparseToPaddedSlices sparseToPaddedSlices;
  const unsigned SLICES_DECREASE_STEP = 2;

  // This is a simple memory scheme to use just one block column as a seed
  // constant and broadcasting it to full pad length. It decreases performance a
  // bit compared to using the full pad length. Increasing seed constant length
  // seems does not bring too much performance improvement.
  Tensor padSpace0 =
      graph.addConstant(dataType, {blockCol}, padValue, {dnai, "/pad"});
  mapTensorLinearly(graph, padSpace0);
  std::size_t totalPadLength = 0;
  for (std::size_t i = 0; i < slicesLen.size(); ++i) {
    long padLength =
        static_cast<long>(maxSliceLen) - static_cast<long>(slicesLen[i]);
    totalPadLength += static_cast<std::size_t>(padLength);
  }
  logging::popsparse::debug("Total pad length = {}", totalPadLength);
  assert(totalPadLength % blockCol == 0);
  Tensor padSpace = padSpace0.broadcast(totalPadLength / blockCol, 0);

  std::size_t padStart = 0;
  std::vector<Tensor> slicesAsRectFlatVec;
  std::vector<Tensor> slicesAsRectCopyFlatVec;
  std::vector<Tensor> slicesAsRectCopyAll(rows);
  for (unsigned curSliceLenUpper = maxSliceLen, nStep = 0; curSliceLenUpper > 0;
       ++nStep) {
    unsigned curSliceLenLower = curSliceLenUpper / SLICES_DECREASE_STEP;
    std::vector<unsigned> origRowMapping;
    std::vector<Tensor> slicesPadded;
    for (unsigned r = 0; r < rows; ++r) {
      if (slicesLen[r] <= curSliceLenUpper && slicesLen[r] > curSliceLenLower) {
        long padLength = static_cast<long>(curSliceLenUpper) -
                         static_cast<long>(slicesLen[r]);
        if (padLength > 0) {
          std::size_t padEnd = padStart + static_cast<std::size_t>(padLength);
          slicesPadded.push_back(
              concat(slices[r], padSpace.slice(padStart, padEnd)));
          padStart = padEnd;
        } else {
          slicesPadded.push_back(slices[r]);
        }
        origRowMapping.push_back(r);
      }
    }
    for (std::size_t i = 0; i < slicesPadded.size(); ++i) {
      slicesPadded[i] = slicesPadded[i].expand({0});
    }
    if (!slicesPadded.empty()) {
      Tensor slicesAsRect = concat(slicesPadded);
      assert(slicesAsRect.shape() ==
             std::vector<std::size_t>({slicesPadded.size(), curSliceLenUpper}));
      Tensor slicesAsRectCopy = graph.addVariable(
          dataType, slicesAsRect.shape(), {dnai, "slice_rect_copy"});
      mapTensorLinearly(graph, slicesAsRectCopy);

      slicesAsRectFlatVec.push_back(slicesAsRect.flatten());
      slicesAsRectCopyFlatVec.push_back(slicesAsRectCopy.flatten());

      for (unsigned i = 0; i < slicesAsRectCopy.dim(0); ++i) {
        unsigned r = origRowMapping[i];
        slicesAsRectCopyAll[r] = slicesAsRectCopy[i];
      }

      sparseToPaddedSlices.slicesAsRectCopyVec.push_back(slicesAsRectCopy);
      sparseToPaddedSlices.origRowMappingVec.push_back(origRowMapping);

      logging::popsparse::debug("Padding step {}: lengths = {}-{}", nStep,
                                curSliceLenLower, curSliceLenUpper);
    }
    curSliceLenUpper = curSliceLenLower;
  }

  Tensor slicesAsRectFlat = concat(slicesAsRectFlatVec);
  Tensor slicesAsRectCopyFlat = concat(slicesAsRectCopyFlatVec);
  prog.add(Copy(slicesAsRectFlat, slicesAsRectCopyFlat));

  // The following represents 2-dimensional jagged array of
  // [blockRows [packed blockCols [rows in block]]]
  // slices in the original block-sparse order
  std::vector<std::vector<std::vector<Tensor>>> slicesCopyBack(blockRows);
  for (unsigned r = 0; r < rows; ++r) {
    if (slicesLen[r] > 0) {
      Tensor curSlice = slicesAsRectCopyAll[r].slice(0, slicesLen[r]);
      assert(curSlice.rank() == 1);
      assert(curSlice.dim(0) % blockCol == 0);
      unsigned br = r / blockRow;
      unsigned blockColsInSlice = curSlice.dim(0) / blockCol;
      for (unsigned bc = 0, c = 0; bc < blockColsInSlice; ++bc, c += blockCol) {
        if (slicesCopyBack[br].size() <= bc) {
          slicesCopyBack[br].push_back({});
          assert(slicesCopyBack[br].size() == bc + 1);
        }
        slicesCopyBack[br][bc].push_back(curSlice.slice(c, c + blockCol));
      }
    }
  }
  std::vector<Tensor> slicesCopyBackFlat;
  for (unsigned i = 0; i < slicesCopyBack.size(); ++i) {
    for (unsigned j = 0; j < slicesCopyBack[i].size(); ++j) {
      for (unsigned k = 0; k < slicesCopyBack[i][j].size(); ++k) {
        slicesCopyBackFlat.push_back(slicesCopyBack[i][j][k]);
      }
    }
  }

  Tensor sparseTensorCopyBack = concat(slicesCopyBackFlat);
  assert(sparseTensorCopyBack.numElements() == sparseTensor.numElements());
  sparseToPaddedSlices.sparseTensorCopyBack =
      sparseTensorCopyBack.reshape(sparseTensor.shape());

  return sparseToPaddedSlices;
}

Tensor bsSoftmaxInternal(Graph &graph, Tensor sparseTensor, bool inPlace,
                         unsigned blockRow, unsigned blockCol,
                         unsigned blockRows, unsigned blockCols,
                         const unsigned char *sparsity,
                         SubBlockMask subBlockMaskType, unsigned numGroups,
                         Sequence &prog, const DebugNameAndId &dnai) {
  const std::string layer = "BSSoftMax";
  logging::popsparse::info(
      "blocksparse softmax: sparseTensor={}, name={}, inPlace={}",
      sparseTensor.shape(), dnai.getPathName() + "/" + layer, inPlace);

  const auto dataType = sparseTensor.elementType();
  if (dataType != FLOAT && dataType != HALF) {
    throw poplibs_error("Only FLOAT and HALF types are supported");
  }
  if (sparseTensor.rank() != 2) {
    throw poplibs_error("input tensor to blocksparse softmax must have "
                        "2 dimensions");
  }
  if (blockRow == 0 || blockCol == 0 || blockRows == 0 || blockCols == 0) {
    throw poplibs_error("Block dimension cannot be zero");
  }

  const unsigned numBlocks = blockRows * blockCols;
  const unsigned blockArea = blockRow * blockCol;
  const unsigned rows = blockRow * blockRows;
  unsigned nz = 0;
  for (unsigned i = 0; i < numBlocks; ++i) {
    if (sparsity[i]) {
      ++nz;
    }
  }
  if (sparseTensor.dim(0) != nz || sparseTensor.dim(1) != blockArea) {
    throw poplibs_error("Sparse tensor must "
                        "have shape [non-zero blocks x block area]");
  }

  Tensor source = sparseTensor;
  Tensor target = sparseTensor;
  if (!inPlace) {
    target = graph.addVariable(dataType, sparseTensor.shape(),
                               {dnai, layer + "/source_copy"});
    mapTensorLinearly(graph, target);
  }

  const float minValue = dataType == FLOAT ? MIN_FLOAT_VALUE : MIN_HALF_VALUE;

  // 1. Apply sub-block mask if any
  std::vector<bool> emptyRowsMask(rows, false);
  std::vector<bool> ignoreEmptyRowsMask;
  if (subBlockMaskType != SubBlockMask::None) {
    std::vector<Tensor> maskAsZeroBlocks;
    std::vector<Tensor> maskAsMinValueBlocks;
    std::vector<unsigned> diagBlockIdxs;
    bsCreateMaskTensor(graph, blockRow, blockCol, blockRows, blockCols,
                       sparsity, subBlockMaskType, numGroups, 0.0f, 1.0f,
                       dataType, maskAsZeroBlocks, diagBlockIdxs, emptyRowsMask,
                       {dnai, layer});
    assert(maskAsZeroBlocks.size() == diagBlockIdxs.size());
    std::vector<unsigned> ignoreDiagBlockIdxs;
    bsCreateMaskTensor(graph, blockRow, blockCol, blockRows, blockCols,
                       sparsity, subBlockMaskType, numGroups, minValue, 0.0f,
                       dataType, maskAsMinValueBlocks, ignoreDiagBlockIdxs,
                       ignoreEmptyRowsMask, {dnai, layer});
    if (diagBlockIdxs.size() > 0) {
      if (!inPlace) {
        prog.add(Copy(source, target));
        source = target;
      }
      std::vector<Tensor> diagBlocksArr;
      for (std::size_t idx = 0; idx < diagBlockIdxs.size(); ++idx) {
        diagBlocksArr.push_back(source[diagBlockIdxs[idx]].expand({0}));
      }
      Tensor diagSoftmaxTensor = concat(diagBlocksArr);
      Tensor maskAsZeroTensor = concat(maskAsZeroBlocks);
      // unchanged_element = unchanged_element * 1.0f = unchanged_element
      // changed_element = changed_element * 0.0f = 0.0
      popops::mulInPlace(graph, diagSoftmaxTensor, maskAsZeroTensor, prog,
                         {dnai, layer + "/subBlockZeroMasked"});
      Tensor maskAsMinValueTensor = concat(maskAsMinValueBlocks);
      // unchanged_element = unchanged_element + 0.0f = unchanged_element
      // changed_element = 0.0f + minValue = minValue
      popops::addInPlace(graph, diagSoftmaxTensor, maskAsMinValueTensor, prog,
                         {dnai, layer + "/subBlockMinValueMasked"});
    }
  }

  // Fully empty rows will contain minValue.
  // All we need to do on them is exp().
  // They will contain 0 after it and that is what we want.
  SparseToPaddedSlices sparseToPaddedSlices =
      sliceAndPad(graph, source, blockRow, blockCol, blockRows, blockCols,
                  sparsity, minValue, prog, {dnai, layer});

  std::vector<Tensor> slicesAsRectFlatVec;
  std::vector<Tensor> slicesAsRectFilteredVec;
  std::vector<Tensor> slicesAsRectFilteredFlatVec;
  std::vector<Tensor> maxRowValuesBcastFlatVec;
  std::vector<Tensor> sumByRowVec;
  std::vector<Tensor> sumByRowFlatVec;
  std::vector<Tensor> oneOverSumFlatVec;

  assert(sparseToPaddedSlices.origRowMappingVec.size() ==
         sparseToPaddedSlices.slicesAsRectCopyVec.size());

  std::vector<ComputeSet> css;
  for (unsigned i = 0; i < sparseToPaddedSlices.slicesAsRectCopyVec.size();
       ++i) {
    const auto &slicesAsRect = sparseToPaddedSlices.slicesAsRectCopyVec[i];
    const auto &origRowMapping = sparseToPaddedSlices.origRowMappingVec[i];
    // Filter out empty rows, but keep them in unfiltered tensor.
    // We will not calculate softmax() on it,
    // but call exp() to assign them zeros.
    Tensor slicesAsRectFiltered = slicesAsRect;
    std::vector<Tensor> slicesVec;
    for (std::size_t j = 0; j < slicesAsRect.dim(0); ++j) {
      unsigned r = origRowMapping[j];
      if (!emptyRowsMask[r]) {
        slicesVec.push_back(slicesAsRect[j].expand({0}));
      }
    }
    if (slicesVec.size() != slicesAsRectFiltered.dim(0)) {
      slicesAsRectFiltered = concat(slicesVec);
    }

    // 2. Compute maximum element for each row
    Tensor maxRowValues =
        popops::reduce(graph, slicesAsRectFiltered, {1}, popops::Operation::MAX,
                       css, {dnai, layer});
    assert(maxRowValues.shape() ==
           std::vector<std::size_t>({slicesAsRectFiltered.shape()[0]}));
    Tensor maxRowValuesBcast =
        maxRowValues.expand({1}).broadcast(slicesAsRectFiltered.shape()[1], 1);
    assert(maxRowValuesBcast.shape() ==
           std::vector<std::size_t>({slicesAsRectFiltered.shape()}));

    slicesAsRectFlatVec.push_back(slicesAsRect.flatten());
    slicesAsRectFilteredVec.push_back(slicesAsRectFiltered);
    slicesAsRectFilteredFlatVec.push_back(slicesAsRectFiltered.flatten());
    maxRowValuesBcastFlatVec.push_back(maxRowValuesBcast.flatten());
  }
  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }

  Tensor slicesAsRectFilteredFlat = concat(slicesAsRectFilteredFlatVec);
  Tensor maxRowValuesBcastFlat = concat(maxRowValuesBcastFlatVec);
  Tensor slicesAsRectFlat = concat(slicesAsRectFlatVec);

  // 3. Subtract maximum element from each row
  popops::subInPlace(graph, slicesAsRectFilteredFlat, maxRowValuesBcastFlat,
                     prog, {dnai, layer});

  // 4. exp
  // No filtering !
  popops::expInPlace(graph, slicesAsRectFlat, prog, {dnai, layer});

  // 5. Compute the sum of each row
  css.clear();
  for (const auto &slicesAsRectFiltered : slicesAsRectFilteredVec) {
    Tensor sumByRow =
        popops::reduce(graph, slicesAsRectFiltered, FLOAT, {1},
                       popops::Operation::ADD, css, {dnai, layer});
    assert(sumByRow.shape() ==
           std::vector<std::size_t>({slicesAsRectFiltered.shape()[0]}));

    sumByRowVec.push_back(sumByRow);
    sumByRowFlatVec.push_back(sumByRow.flatten());
  }
  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }

  // 6. Compute 1 / (sum for each row)
  Tensor sumByRowFlat = concat(sumByRowFlatVec);
  popops::invInPlace(graph, sumByRowFlat, prog, {dnai, layer});

  for (unsigned i = 0; i < slicesAsRectFilteredVec.size(); ++i) {
    const auto &slicesAsRectFiltered = slicesAsRectFilteredVec[i];
    const auto &sumByRow = sumByRowVec[i];
    Tensor sum = (dataType == HALF)
                     ? popops::cast(graph, sumByRow, HALF, prog, {dnai, layer})
                     : sumByRow;

    Tensor oneOverSum =
        sum.expand({1}).broadcast(slicesAsRectFiltered.shape()[1], 1);
    assert(oneOverSum.shape() == slicesAsRectFiltered.shape());

    oneOverSumFlatVec.push_back(oneOverSum.flatten());
  }

  // 7. Compute division by sum for each row
  Tensor oneOverSumFlat = concat(oneOverSumFlatVec);
  popops::mulInPlace(graph, slicesAsRectFilteredFlat, oneOverSumFlat, prog,
                     {dnai, layer});

  prog.add(Copy(sparseToPaddedSlices.sparseTensorCopyBack, target));
  return target;
}

Tensor bsSoftmaxGradInternal(Graph &graph, Tensor sparseOut,
                             Tensor sparseOutGrad, unsigned blockRow,
                             unsigned blockCol, unsigned blockRows,
                             unsigned blockCols, const unsigned char *sparsity,
                             poplar::program::Sequence &prog,
                             const DebugNameAndId &dnai) {

  const std::string layer = "BSSoftMaxGrad";

  logging::popsparse::info(
      "blocksparse softmaxGrad: sparseOut={}, sparseOutGrad={}, name={}",
      sparseOut.shape(), sparseOutGrad.shape(),
      dnai.getPathName() + "/" + layer);

  const auto outType = sparseOut.elementType();
  const auto outGradType = sparseOutGrad.elementType();
  if ((outType != FLOAT && outType != HALF) ||
      (outGradType != FLOAT && outGradType != HALF)) {
    throw poplibs_error("Only FLOAT and HALF types are supported");
  }
  if (outType != outGradType) {
    throw poplibs_error(
        "Activation and gradient tensors must be of the same type");
  }
  if (sparseOut.rank() != 2) {
    throw poplibs_error("input tensors to blocksparse softmaxGrad must have "
                        "2 dimensions");
  }
  if (sparseOut.shape() != sparseOutGrad.shape()) {
    throw poplibs_error("sparseOut and sparseOutGrad tensors must have "
                        "equal shapes");
  }
  if (blockRow == 0 || blockCol == 0 || blockRows == 0 || blockCols == 0) {
    throw poplibs_error("Block dimension cannot be zero");
  }
  unsigned numBlocks = blockRows * blockCols;
  unsigned blockArea = blockRow * blockCol;
  unsigned rows = blockRow * blockRows;
  unsigned nz = 0;
  for (unsigned i = 0; i < numBlocks; ++i) {
    if (sparsity[i]) {
      ++nz;
    }
  }
  if (sparseOut.dim(0) != nz || sparseOut.dim(1) != blockArea) {
    throw poplibs_error("sparseOut tensor must "
                        "have shape [non-zero blocks x block area]");
  }

  // 1. Multiply output by out gradient
  Tensor sparseOutMulOutGrad =
      popops::mul(graph, sparseOut, sparseOutGrad, prog, {dnai, layer});

  SparseToPaddedSlices sparseToPaddedSlices =
      sliceAndPad(graph, sparseOutMulOutGrad, blockRow, blockCol, blockRows,
                  blockCols, sparsity, 0.0f, prog, {dnai, layer});

  std::vector<Tensor> sumOutMulOutGradVec;
  std::vector<std::pair<int, int>> slicesIdxsByRow(rows,
                                                   std::pair<int, int>(-1, -1));

  // 2. Compute sum for each row of the (output x out gradient) product
  for (unsigned i = 0; i < sparseToPaddedSlices.slicesAsRectCopyVec.size();
       ++i) {
    const auto &slicedOutMulOutGrad =
        sparseToPaddedSlices.slicesAsRectCopyVec[i];
    const auto &origRowMapping = sparseToPaddedSlices.origRowMappingVec[i];

    assert(slicedOutMulOutGrad.rank() == 2);
#ifndef NDEBUG
    std::size_t numSumRows = slicedOutMulOutGrad.shape()[0];
#endif
    Tensor sumOutMulOutGrad =
        popops::reduce(graph, slicedOutMulOutGrad, outType, {1},
                       popops::Operation::ADD, prog, {dnai, layer});

    assert(sumOutMulOutGrad.shape() == std::vector<std::size_t>({numSumRows}));
    sumOutMulOutGradVec.push_back(sumOutMulOutGrad);

    for (unsigned j = 0; j < origRowMapping.size(); ++j) {
      unsigned r = origRowMapping[j];
      slicesIdxsByRow[r] = std::pair<int, int>(i, j);
    }
  }

  // 3. Broadcast sum by every non-empty row block
  std::vector<Tensor> sumBcastBySparseRowBlocks;
  for (unsigned br = 0, idxDense = 0; br < blockRows; ++br) {
    for (unsigned bc = 0; bc < blockCols; ++bc, ++idxDense) {
      if (sparsity[idxDense]) {
        std::vector<Tensor> sumBlockElems;
        unsigned r = br * blockRow;
        for (unsigned rb = 0; rb < blockRow; ++rb, ++r) {
          int idxChunk = slicesIdxsByRow[r].first;
          int idxInChunk = slicesIdxsByRow[r].second;
          assert(idxChunk >= 0 && idxInChunk >= 0);
          const auto &sumOutMulOutGrad = sumOutMulOutGradVec[idxChunk];

          Tensor sumRowBlock = sumOutMulOutGrad[idxInChunk].expand({0});
          for (unsigned cb = 0; cb < blockCol; ++cb) {
            sumBlockElems.push_back(sumRowBlock);
          }
        }
        Tensor sumBlock = concat(sumBlockElems).expand({0});
        sumBcastBySparseRowBlocks.push_back(sumBlock);
      }
    }
  };
  Tensor sumBcastBySparseRow = concat(sumBcastBySparseRowBlocks);
  assert(sumBcastBySparseRow.shape() == sparseOut.shape());

  // 4. Multiply out by sum by row
  Tensor sparseOutMulSumOutGrad =
      popops::mul(graph, sparseOut, sumBcastBySparseRow, prog, {dnai, layer});

  // 5. Substruct (out x sum by row) product from (output x out gradient)
  // product
  popops::subInPlace(graph, sparseOutMulOutGrad, sparseOutMulSumOutGrad, prog,
                     {dnai, layer});
  return sparseOutMulOutGrad;
}

Tensor bsSoftmax(Graph &graph, Tensor sparseTensor,
                 const std::array<int, 2> &dim,
                 const std::array<int, 2> &blockSize,
                 const std::vector<unsigned char> &sparsity,
                 SubBlockMask subBlockMaskType, unsigned numGroups,
                 program::Sequence &prog,
                 const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(sparseTensor, dim, blockSize, sparsity, subBlockMaskType));

  for (int iDim = 0; iDim < 2; ++iDim) {
    if (dim[iDim] % blockSize[iDim] != 0) {
      throw poputil::poplibs_error(
          "Input error: input dimension " + std::to_string(iDim) + ": " +
          std::to_string(dim[iDim]) +
          " is not divisible by block size dimension " + std::to_string(iDim) +
          ": " + std::to_string(blockSize[iDim]));
    }
  }

  auto output = bsSoftmaxInternal(
      graph, sparseTensor, false, static_cast<unsigned>(blockSize[0]),
      static_cast<unsigned>(blockSize[1]),
      static_cast<unsigned>(dim[0] / blockSize[0] * numGroups),
      static_cast<unsigned>(dim[1] / blockSize[1]), sparsity.data(),
      subBlockMaskType, numGroups, prog, {di});
  di.addOutput(output);
  return output;
}

void bsSoftmaxInPlace(Graph &graph, Tensor sparseTensor,
                      const std::array<int, 2> &dim,
                      const std::array<int, 2> &blockSize,
                      const std::vector<unsigned char> &sparsity,
                      SubBlockMask subBlockMaskType, unsigned numGroups,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext) {

  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(sparseTensor, dim, blockSize, sparsity, subBlockMaskType));

  for (int iDim = 0; iDim < 2; ++iDim) {
    if (dim[iDim] % blockSize[iDim] != 0) {
      throw poputil::poplibs_error(
          "Input error: input dimension " + std::to_string(iDim) + ": " +
          std::to_string(dim[iDim]) +
          " is not divisible by block size dimension " + std::to_string(iDim) +
          ": " + std::to_string(blockSize[iDim]));
    }
  }

  bsSoftmaxInternal(graph, sparseTensor, true,
                    static_cast<unsigned>(blockSize[0]),
                    static_cast<unsigned>(blockSize[1]),
                    static_cast<unsigned>(dim[0] / blockSize[0] * numGroups),
                    static_cast<unsigned>(dim[1] / blockSize[1]),
                    sparsity.data(), subBlockMaskType, numGroups, prog, {di});
}

Tensor bsSoftmaxGrad(Graph &graph, Tensor sparseOut, Tensor sparseOutGrad,
                     const std::array<int, 2> &dim,
                     const std::array<int, 2> &blockSize,
                     const std::vector<unsigned char> &sparsity,
                     poplar::program::Sequence &prog,
                     const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(sparseOut, sparseOutGrad, dim, blockSize, sparsity));

  for (int iDim = 0; iDim < 2; ++iDim) {
    if (dim[iDim] % blockSize[iDim] != 0) {
      throw poputil::poplibs_error(
          "Input error: input dimension " + std::to_string(iDim) + ": " +
          std::to_string(dim[iDim]) +
          " is not divisible by block size dimension " + std::to_string(iDim) +
          ": " + std::to_string(blockSize[iDim]));
    }
  }

  auto output = bsSoftmaxGradInternal(
      graph, sparseOut, sparseOutGrad, static_cast<unsigned>(blockSize[0]),
      static_cast<unsigned>(blockSize[1]),
      static_cast<unsigned>(dim[0] / blockSize[0]),
      static_cast<unsigned>(dim[1] / blockSize[1]), sparsity.data(), prog,
      {di});
  di.addOutput(output);
  return output;
}

} // namespace experimental
} // namespace popsparse
