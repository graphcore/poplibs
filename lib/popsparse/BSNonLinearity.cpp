// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "BSNonLinearity.hpp"
#include "BSOps.hpp"
#include "BSUtils.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/ElementWiseUtil.hpp"
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

static Tensor sliceAndPad(Graph &graph, Tensor sparseTensor, unsigned blockRow,
                          unsigned blockCol, unsigned blockRows,
                          unsigned blockCols, const unsigned char *sparsity,
                          float padValue,
                          const std::vector<bool> &emptyRowsMaskIn,
                          std::vector<bool> &emptyRowsMaskOut, Sequence &prog,
                          const DebugNameAndId &dnai) {
  const unsigned rows = blockRow * blockRows;
  const auto dataType = sparseTensor.elementType();
  emptyRowsMaskOut.resize(rows, false);

  std::vector<std::size_t> slicesLen;
  std::vector<Tensor> slicesPadded;
  std::size_t maxSliceLen = 0;
  for (unsigned r = 0; r < rows; ++r) {
    if (!emptyRowsMaskIn[r]) {
      Tensor curSlice = slice(sparseTensor, r, 0, blockRow, blockCol, blockRows,
                              blockCols, false, sparsity);
      assert(curSlice.rank() == 1);
      if (curSlice.dim(0) > 0) {
        slicesPadded.push_back(curSlice);
        slicesLen.push_back(curSlice.dim(0));
        maxSliceLen = std::max(maxSliceLen, curSlice.dim(0));
      } else {
        emptyRowsMaskOut[r] = true;
      }
    }
  }

  logging::popsparse::debug("Maximum slice length = {}", maxSliceLen);

  // TODO: To effectively handle sparsity masks
  // with large number of non-zeros in some rows -
  // implement reduce in steps
  std::size_t totalPadLength = 0;
  for (std::size_t i = 0; i < slicesPadded.size(); ++i) {
    long padLength =
        static_cast<long>(maxSliceLen) - static_cast<long>(slicesLen[i]);
    totalPadLength += static_cast<std::size_t>(padLength);
  }
  logging::popsparse::debug("Total pad length = {}", totalPadLength);
  if (totalPadLength > 0) {
    // TODO:
    // After moving to SDK 1.2
    // of all dependent projects, use new fill() operation
    // to eliminate extra add()
    Tensor padSpace =
        graph.addVariable(dataType, {totalPadLength}, {dnai, "zero_pad"});
    mapTensorLinearly(graph, padSpace);
    popops::zero(graph, padSpace, prog, {dnai, "zero"});

    if (padValue != 0.0f) {
      Tensor padValueConst = graph.addConstant(dataType, {totalPadLength},
                                               padValue, {dnai, "zero_pad"});
      mapTensorLinearly(graph, padValueConst);
      popops::addInPlace(graph, padSpace, padValueConst, prog,
                         {dnai, "value_pad"});
    }

    std::size_t padStart = 0;
    for (std::size_t i = 0; i < slicesPadded.size(); ++i) {
      long padLength =
          static_cast<long>(maxSliceLen) - static_cast<long>(slicesLen[i]);
      if (padLength > 0) {
        std::size_t padEnd = padStart + static_cast<std::size_t>(padLength);
        slicesPadded[i] =
            concat(slicesPadded[i], padSpace.slice(padStart, padEnd));
        padStart = padEnd;
      }
    }
  }
  for (std::size_t i = 0; i < slicesPadded.size(); ++i) {
    slicesPadded[i] = slicesPadded[i].expand({0});
  }
  Tensor slicesAsRect = concat(slicesPadded);
  assert(slicesAsRect.shape() ==
         std::vector<std::size_t>({slicesPadded.size(), maxSliceLen}));
  return slicesAsRect;
}

Tensor bsSoftmaxInternal(Graph &graph, Tensor sparseTensor, bool inPlace,
                         unsigned blockRow, unsigned blockCol,
                         unsigned blockRows, unsigned blockCols,
                         const unsigned char *sparsity,
                         SubBlockMask subBlockMaskType, Sequence &prog,
                         const DebugNameAndId &dnai) {
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
  bool needsCopy = !inPlace;

  if (needsCopy) {
    source = graph.addVariable(dataType, sparseTensor.shape(),
                               {dnai, layer + "/source_copy"});
    mapTensorLinearly(graph, source);
    prog.add(Copy(sparseTensor, source, false, {dnai}));
  }

  const float minValue = dataType == FLOAT ? MIN_FLOAT_VALUE : MIN_HALF_VALUE;

  // 1. Apply sub-block mask if any
  std::vector<bool> noEmptyRowsMask(rows, false);
  std::vector<bool> emptyRowsMask = noEmptyRowsMask;
  std::vector<bool> ignoreEmptyRowsMask;
  bool areEmptyRows = false;
  if (subBlockMaskType != SubBlockMask::None) {
    std::vector<Tensor> maskAsZeroBlocks;
    std::vector<Tensor> maskAsMinValueBlocks;
    std::vector<unsigned> diagBlockIdxs;
    bsCreateMaskTensor(graph, blockRow, blockCol, blockRows, blockCols,
                       sparsity, subBlockMaskType, 0.0f, 1.0f, dataType,
                       maskAsZeroBlocks, diagBlockIdxs, emptyRowsMask,
                       {dnai, layer});
    assert(maskAsZeroBlocks.size() == diagBlockIdxs.size());
    std::vector<unsigned> ignorediagBlockIdxs;
    bsCreateMaskTensor(graph, blockRow, blockCol, blockRows, blockCols,
                       sparsity, subBlockMaskType, minValue, 0.0f, dataType,
                       maskAsMinValueBlocks, ignorediagBlockIdxs,
                       ignoreEmptyRowsMask, {dnai, layer});
    if (diagBlockIdxs.size() > 0) {
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
    for (bool emptyRow : emptyRowsMask) {
      if (emptyRow) {
        areEmptyRows = true;
        break;
      }
    }
  }

  // Fully empty rows will contain minValue.
  // All we need to do on them is exp().
  // They will contain 0 after it and that is what we want.
  Tensor slicesAsRect = sliceAndPad(
      graph, source, blockRow, blockCol, blockRows, blockCols, sparsity,
      minValue, noEmptyRowsMask, ignoreEmptyRowsMask, prog, {dnai, layer});
  Tensor slicesAsRectFiltered = slicesAsRect;
  if (areEmptyRows) {
    std::vector<Tensor> slicesVec;
    for (std::size_t i = 0; i < slicesAsRect.dim(0); ++i) {
      if (!emptyRowsMask[i]) {
        slicesVec.push_back(slicesAsRect[i].expand({0}));
      }
    }
    slicesAsRectFiltered = concat(slicesVec);
  }

  // 2. Subtract maximum element from each row
  Tensor maxRowValues =
      popops::reduce(graph, slicesAsRectFiltered, {1}, popops::Operation::MAX,
                     prog, {dnai, layer});
  assert(maxRowValues.shape() ==
         std::vector<std::size_t>({slicesAsRectFiltered.shape()[0]}));
  Tensor maxRowValuesBcast =
      maxRowValues.expand({1}).broadcast(slicesAsRectFiltered.shape()[1], 1);
  assert(maxRowValuesBcast.shape() ==
         std::vector<std::size_t>({slicesAsRectFiltered.shape()}));
  popops::subInPlace(graph, slicesAsRectFiltered, maxRowValuesBcast, prog,
                     {dnai, layer});

  // 3. exp
  // No filtering !
  popops::expInPlace(graph, slicesAsRect, prog, {dnai, layer});

  // 4. Divide by sum for each row
  Tensor sumByRow = popops::reduce(graph, slicesAsRectFiltered, FLOAT, {1},
                                   popops::Operation::ADD, prog, {dnai, layer});
  assert(sumByRow.shape() ==
         std::vector<std::size_t>({slicesAsRectFiltered.shape()[0]}));

  popops::invInPlace(graph, sumByRow, prog, {dnai, layer});
  Tensor sum = (dataType == HALF)
                   ? popops::cast(graph, sumByRow, HALF, prog, {dnai, layer})
                   : sumByRow;

  Tensor oneOverSum =
      sum.expand({1}).broadcast(slicesAsRectFiltered.shape()[1], 1);
  assert(oneOverSum.shape() == slicesAsRectFiltered.shape());

  popops::mulInPlace(graph, slicesAsRectFiltered, oneOverSum, prog,
                     {dnai, layer});

  return source;
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

  Tensor sparseOutMulOutGrad =
      popops::mul(graph, sparseOut, sparseOutGrad, prog, {dnai, layer});

  std::vector<bool> emptyRowsMaskIn(rows, false);
  std::vector<bool> emptyRowsMaskOut;
  Tensor slicedOutMulOutGrad = sliceAndPad(
      graph, sparseOutMulOutGrad, blockRow, blockCol, blockRows, blockCols,
      sparsity, 0.0f, emptyRowsMaskIn, emptyRowsMaskOut, prog, {dnai, layer});
  assert(slicedOutMulOutGrad.rank() == 2);
#ifndef NDEBUG
  std::size_t numSumRows = slicedOutMulOutGrad.shape()[0];
#endif
  Tensor sumOutMulOutGrad =
      popops::reduce(graph, slicedOutMulOutGrad, outType, {1},
                     popops::Operation::ADD, prog, {dnai, layer});

  assert(sumOutMulOutGrad.shape() == std::vector<std::size_t>({numSumRows}));

  std::vector<int> filteredRowsIdxs(rows, -1);
  int rf = -1;
  for (unsigned r = 0; r < rows; ++r) {
    if (!emptyRowsMaskOut[r]) {
      ++rf;
    }
    filteredRowsIdxs[r] = rf;
  }
  assert(filteredRowsIdxs[rows - 1] == static_cast<int>(numSumRows) - 1);

  std::vector<Tensor> sumBcastBySparseRowBlocks;
  for (unsigned br = 0, idxDense = 0; br < blockRows; ++br) {
    for (unsigned bc = 0; bc < blockCols; ++bc, ++idxDense) {
      if (sparsity[idxDense]) {
        std::vector<Tensor> sumBlockElems;
        unsigned r = br * blockRow;
        for (unsigned rb = 0; rb < blockRow; ++rb, ++r) {
          int rf = filteredRowsIdxs[r];
          assert(rf >= 0 && rf < static_cast<int>(numSumRows));
          Tensor sumBlock = sumOutMulOutGrad[rf].expand({0});
          for (unsigned cb = 0; cb < blockCol; ++cb) {
            sumBlockElems.push_back(sumBlock);
          }
        }
        Tensor sumBlock = concat(sumBlockElems).expand({0});
        sumBcastBySparseRowBlocks.push_back(sumBlock);
      }
    }
  };
  Tensor sumBcastBySparseRow = concat(sumBcastBySparseRowBlocks);
  assert(sumBcastBySparseRow.shape() == sparseOut.shape());

  Tensor sparseOutMulSumOutGrad =
      popops::mul(graph, sparseOut, sumBcastBySparseRow, prog, {dnai, layer});

  Tensor sparseInGrad = sparseOutMulOutGrad;
  popops::subInPlace(graph, sparseInGrad, sparseOutMulSumOutGrad, prog,
                     {dnai, layer});
  return sparseInGrad;
}

Tensor bsSoftmax(Graph &graph, Tensor sparseTensor,
                 const std::array<int, 2> &dim,
                 const std::array<int, 2> &blockSize,
                 const std::vector<unsigned char> &sparsity,
                 SubBlockMask subBlockMaskType, program::Sequence &prog,
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
      static_cast<unsigned>(dim[0] / blockSize[0]),
      static_cast<unsigned>(dim[1] / blockSize[1]), sparsity.data(),
      subBlockMaskType, prog, {di});
  di.addOutput(output);
  return output;
}

void bsSoftmaxInPlace(Graph &graph, Tensor sparseTensor,
                      const std::array<int, 2> &dim,
                      const std::array<int, 2> &blockSize,
                      const std::vector<unsigned char> &sparsity,
                      SubBlockMask subBlockMaskType,
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
                    static_cast<unsigned>(dim[0] / blockSize[0]),
                    static_cast<unsigned>(dim[1] / blockSize[1]),
                    sparsity.data(), subBlockMaskType, prog, {di});
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
