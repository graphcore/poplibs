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

namespace popsparse {
namespace experimental {

static Tensor sliceAndPad(Graph &graph, poplar::Tensor sparseTensor,
                          unsigned blockRow, unsigned blockCol,
                          unsigned blockRows, unsigned blockCols,
                          const unsigned char *sparsity,
                          std::vector<bool> &emptyRowsMask, Sequence &prog,
                          const std::string &debugStr) {
  const unsigned rows = blockRow * blockRows;
  const auto dataType = sparseTensor.elementType();

  std::vector<std::size_t> slicesLen;
  std::vector<Tensor> slicesPadded;
  std::size_t maxSliceLen = 0;
  for (unsigned r = 0; r < rows; ++r) {
    if (!emptyRowsMask[r]) {
      Tensor curSlice = slice(sparseTensor, r, 0, blockRow, blockCol, blockRows,
                              blockCols, false, sparsity);
      assert(curSlice.rank() == 1);
      if (curSlice.dim(0) > 0) {
        slicesPadded.push_back(curSlice);
        slicesLen.push_back(curSlice.dim(0));
        maxSliceLen = std::max(maxSliceLen, curSlice.dim(0));
      } else {
        emptyRowsMask[r] = true;
      }
    }
  }

  logging::debug("Maximum slice length = {}", maxSliceLen);

  // TODO: To effectively handle sparsity masks
  // with large number of non-zeros in some rows -
  // implement reduce in steps
  std::size_t totalPadLength = 0;
  for (std::size_t i = 0; i < slicesPadded.size(); ++i) {
    long padLength =
        static_cast<long>(maxSliceLen) - static_cast<long>(slicesLen[i]);
    totalPadLength += static_cast<std::size_t>(padLength);
  }
  logging::debug("Total pad length = {}", totalPadLength);
  if (totalPadLength > 0) {
    Tensor padSpace =
        graph.addVariable(dataType, {totalPadLength}, debugStr + "/zero_pad");
    mapTensorLinearly(graph, padSpace);
    popops::zero(graph, padSpace, prog, debugStr + "/zero");

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

Tensor bsSoftmaxInternal(Graph &graph, poplar::Tensor sparseTensor,
                         bool inPlace, unsigned blockRow, unsigned blockCol,
                         unsigned blockRows, unsigned blockCols,
                         const unsigned char *sparsity,
                         SubBlockMask subBlockMaskType, Sequence &prog,
                         const std::string &debugStr) {
  const auto debugStr1 = debugStr + "/BSSoftMax";
  logging::info("blocksparse softmax: sparseTensor={}, name={}, inPlace={}",
                sparseTensor.shape(), debugStr1, inPlace);

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

  Tensor sourceTensor = sparseTensor;
  bool needsCopy = !inPlace;

  Tensor softmaxTensor = sparseTensor;
  if (needsCopy) {
    softmaxTensor = popops::exp(graph, sourceTensor, prog, debugStr1);
  } else {
    popops::expInPlace(graph, softmaxTensor, prog, debugStr1);
  }
  assert(dataType == softmaxTensor.elementType());

  std::vector<bool> emptyRowsMask(rows, false);
  if (subBlockMaskType != SubBlockMask::None) {
    std::vector<Tensor> maskBlocks;
    std::vector<unsigned> diagBlockIdxs;
    bsCreateMaskTensor(graph, blockRow, blockCol, blockRows, blockCols,
                       sparsity, subBlockMaskType, 0.0f, 1.0f, dataType,
                       maskBlocks, diagBlockIdxs, emptyRowsMask, debugStr1);
    assert(maskBlocks.size() == diagBlockIdxs.size());
    if (diagBlockIdxs.size() > 0) {
      std::vector<Tensor> diagBlocksArr;
      for (std::size_t idx = 0; idx < diagBlockIdxs.size(); ++idx) {
        diagBlocksArr.push_back(softmaxTensor[diagBlockIdxs[idx]].expand({0}));
      }
      Tensor diagSoftmaxTensor = concat(diagBlocksArr);
      Tensor maskTensor = concat(maskBlocks);
      popops::mulInPlace(graph, diagSoftmaxTensor, maskTensor, prog,
                         debugStr1 + "/subBlockMasked");
    }
  }

  Tensor slicesAsRect =
      sliceAndPad(graph, softmaxTensor, blockRow, blockCol, blockRows,
                  blockCols, sparsity, emptyRowsMask, prog, debugStr1);

  Tensor sumF = popops::reduce(graph, slicesAsRect, poplar::FLOAT, {1},
                               popops::Operation::ADD, prog, debugStr1);
  assert(sumF.shape() == std::vector<std::size_t>({slicesAsRect.shape()[0]}));

  popops::invInPlace(graph, sumF, prog, debugStr1);
  Tensor sum = (dataType == poplar::HALF)
                   ? popops::cast(graph, sumF, poplar::HALF, prog, debugStr1)
                   : sumF;

  Tensor oneOverSum = sum.expand({1}).broadcast(slicesAsRect.shape()[1], 1);
  assert(oneOverSum.shape() == slicesAsRect.shape());

  popops::mulInPlace(graph, slicesAsRect, oneOverSum, prog, debugStr1);

  return softmaxTensor;
}

poplar::Tensor bsSoftmaxGradInternal(
    poplar::Graph &graph, poplar::Tensor sparseOut,
    poplar::Tensor sparseOutGrad, unsigned blockRow, unsigned blockCol,
    unsigned blockRows, unsigned blockCols, const unsigned char *sparsity,
    poplar::program::Sequence &prog, const std::string &debugStr) {

  const auto debugStr1 = debugStr + "/BSSoftMaxGrad";

  logging::info(
      "blocksparse softmaxGrad: sparseOut={}, sparseOutGrad={}, name={}",
      sparseOut.shape(), sparseOutGrad.shape(), debugStr1);

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
      popops::mul(graph, sparseOut, sparseOutGrad, prog, debugStr1);

  std::vector<bool> emptyRowsMask(rows, false);
  Tensor slicedOutMulOutGrad =
      sliceAndPad(graph, sparseOutMulOutGrad, blockRow, blockCol, blockRows,
                  blockCols, sparsity, emptyRowsMask, prog, debugStr1);
  assert(slicedOutMulOutGrad.rank() == 2);
#ifndef NDEBUG
  std::size_t numSumRows = slicedOutMulOutGrad.shape()[0];
#endif
  Tensor sumOutMulOutGrad =
      popops::reduce(graph, slicedOutMulOutGrad, outType, {1},
                     popops::Operation::ADD, prog, debugStr1);

  assert(sumOutMulOutGrad.shape() == std::vector<std::size_t>({numSumRows}));

  std::vector<int> filteredRowsIdxs(rows, -1);
  int rf = -1;
  for (unsigned r = 0; r < rows; ++r) {
    if (!emptyRowsMask[r]) {
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
      popops::mul(graph, sparseOut, sumBcastBySparseRow, prog, debugStr1);

  Tensor sparseInGrad = sparseOutMulOutGrad;
  popops::subInPlace(graph, sparseInGrad, sparseOutMulSumOutGrad, prog,
                     debugStr1);
  return sparseInGrad;
}

poplar::Tensor bsSoftmax(poplar::Graph &graph, poplar::Tensor sparseTensor,
                         const std::array<int, 2> &dim,
                         const std::array<int, 2> &blockSize,
                         const std::vector<unsigned char> &sparsity,
                         SubBlockMask subBlockMaskType,
                         poplar::program::Sequence &prog,
                         const std::string &debugStr) {

  for (int iDim = 0; iDim < 2; ++iDim) {
    if (dim[iDim] % blockSize[iDim] != 0) {
      throw poputil::poplibs_error(
          "Input error: input dimension " + std::to_string(iDim) + ": " +
          std::to_string(dim[iDim]) +
          " is not divisible by block size dimension " + std::to_string(iDim) +
          ": " + std::to_string(blockSize[iDim]));
    }
  }

  return bsSoftmaxInternal(graph, sparseTensor, false,
                           static_cast<unsigned>(blockSize[0]),
                           static_cast<unsigned>(blockSize[1]),
                           static_cast<unsigned>(dim[0] / blockSize[0]),
                           static_cast<unsigned>(dim[1] / blockSize[1]),
                           sparsity.data(), subBlockMaskType, prog, debugStr);
}

void bsSoftmaxInPlace(poplar::Graph &graph, poplar::Tensor sparseTensor,
                      const std::array<int, 2> &dim,
                      const std::array<int, 2> &blockSize,
                      const std::vector<unsigned char> &sparsity,
                      SubBlockMask subBlockMaskType,
                      poplar::program::Sequence &prog,
                      const std::string &debugStr) {

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
                    sparsity.data(), subBlockMaskType, prog, debugStr);
}

poplar::Tensor bsSoftmaxGrad(poplar::Graph &graph, poplar::Tensor sparseOut,
                             poplar::Tensor sparseOutGrad,
                             const std::array<int, 2> &dim,
                             const std::array<int, 2> &blockSize,
                             const std::vector<unsigned char> &sparsity,
                             poplar::program::Sequence &prog,
                             const std::string &debugStr) {
  for (int iDim = 0; iDim < 2; ++iDim) {
    if (dim[iDim] % blockSize[iDim] != 0) {
      throw poputil::poplibs_error(
          "Input error: input dimension " + std::to_string(iDim) + ": " +
          std::to_string(dim[iDim]) +
          " is not divisible by block size dimension " + std::to_string(iDim) +
          ": " + std::to_string(blockSize[iDim]));
    }
  }

  return bsSoftmaxGradInternal(graph, sparseOut, sparseOutGrad,
                               static_cast<unsigned>(blockSize[0]),
                               static_cast<unsigned>(blockSize[1]),
                               static_cast<unsigned>(dim[0] / blockSize[0]),
                               static_cast<unsigned>(dim[1] / blockSize[1]),
                               sparsity.data(), prog, debugStr);
}

} // namespace experimental
} // namespace popsparse