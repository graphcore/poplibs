// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popsparse/experimental/BlockSparseMatMul.hpp"
#include "BSMatrix.hpp"
#include "BSOps.hpp"
#include "HyperGraphBlockZoltan.hpp"
#include <iostream>
#include <memory>
#include <poplibs_support/logging.hpp>
#include <poputil/OptionParsing.hpp>
#include <poputil/exceptions.hpp>

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

using namespace poplibs;

class BSMatMulImpl {

public:
  BSMatMulImpl(const std::array<int, 3> &dim,
               const std::array<int, 3> &blockSize,
               const std::vector<unsigned char> &rhsSparsity,
               bool rhsNeedTransposeIn, poplar::Type inDataTypeIn,
               poplar::Type outDataTypeIn, poplar::Type partialDataTypeIn)
      : inDataType(inDataTypeIn), outDataType(outDataTypeIn),
        partialDataType(partialDataTypeIn), isLhsSparse(false),
        isRhsSparse(true), isResSparse(false),
        rhsNeedTranspose(rhsNeedTransposeIn), subBlockMask(SubBlockMask::None) {

    for (int iDim = 0; iDim < 3; ++iDim) {
      if (dim[iDim] % blockSize[iDim] != 0) {
        throw poputil::poplibs_error(
            "Input error: input dimension " + std::to_string(iDim) + ": " +
            std::to_string(dim[iDim]) +
            " is not divisible by block size dimension " +
            std::to_string(iDim) + ": " + std::to_string(blockSize[iDim]));
      }
    }
    int numBlocks = dim[1] / blockSize[1] * dim[2] / blockSize[2];
    if (static_cast<int>(rhsSparsity.size()) != numBlocks) {
      throw poputil::poplibs_error("Input error: the sparsity mask size: " +
                                   std::to_string(rhsSparsity.size()) +
                                   " does not match total number of blocks: " +
                                   std::to_string(numBlocks));
    }
    lhsMatrix.reset(new BlockDenseMatrix(dim[0], dim[1], blockSize[0],
                                         blockSize[1], false));
    if (!rhsNeedTranspose) {
      rhsMatrix.reset(new BlockSparseMatrix(dim[1], dim[2], blockSize[1],
                                            blockSize[2], rhsNeedTranspose,
                                            rhsSparsity.data()));
    } else {
      rhsMatrix.reset(new BlockSparseMatrix(dim[2], dim[1], blockSize[2],
                                            blockSize[1], rhsNeedTranspose,
                                            rhsSparsity.data()));
    }
  }

  BSMatMulImpl(const std::array<int, 3> &dim,
               const std::array<int, 3> &blockSize,
               const std::vector<unsigned char> &lhsSparsity,
               bool lhsNeedTranspose,
               const std::vector<unsigned char> &rhsSparsity,
               bool rhsNeedTranspose, poplar::Type inDataTypeIn,
               poplar::Type outDataTypeIn, poplar::Type partialDataTypeIn)
      : inDataType(inDataTypeIn), outDataType(outDataTypeIn),
        partialDataType(partialDataTypeIn), isLhsSparse(true),
        isRhsSparse(true), isResSparse(false), rhsNeedTranspose(false),
        subBlockMask(SubBlockMask::None) {
    throw poputil::poplibs_error("Sparse x sparse case is not supported.");
  }

  BSMatMulImpl(const std::array<int, 3> &dim,
               const std::array<int, 3> &blockSize,
               const std::vector<unsigned char> &resSparsityIn,
               poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
               poplar::Type partialDataTypeIn,
               SubBlockMask subBlockMaskIn = SubBlockMask::None,
               unsigned numGroupsIn = 1)
      : resSparsity(resSparsityIn), inDataType(inDataTypeIn),
        outDataType(outDataTypeIn), partialDataType(partialDataTypeIn),
        isLhsSparse(false), isRhsSparse(false), isResSparse(true),
        rhsNeedTranspose(false), subBlockMask(subBlockMaskIn) {
    for (int iDim = 0; iDim < 3; ++iDim) {
      if (dim[iDim] % blockSize[iDim] != 0) {
        throw poputil::poplibs_error(
            "Input error: input dimension " + std::to_string(iDim) + ": " +
            std::to_string(dim[iDim]) +
            " is not divisible by block size dimension " +
            std::to_string(iDim) + ": " + std::to_string(blockSize[iDim]));
      }
    }
    int numBlocks = dim[0] / blockSize[0] * dim[2] / blockSize[2];
    if (static_cast<int>(resSparsityIn.size()) != numBlocks) {
      throw poputil::poplibs_error("Input error: the sparsity mask size: " +
                                   std::to_string(resSparsityIn.size()) +
                                   " does not match total number of blocks: " +
                                   std::to_string(numBlocks));
    }
    lhsMatrix.reset(new BlockDenseMatrix(dim[0], dim[1], blockSize[0],
                                         blockSize[1], false));
    rhsMatrix.reset(new BlockDenseMatrix(dim[1], dim[2], blockSize[1],
                                         blockSize[2], false));
    resSparsity = resSparsityIn;
  }

  std::unique_ptr<BlockMatrix> lhsMatrix;
  std::unique_ptr<BlockMatrix> rhsMatrix;
  std::vector<unsigned char> resSparsity;
  const poplar::Type inDataType;
  const poplar::Type outDataType;
  const poplar::Type partialDataType;
  const bool isLhsSparse;
  const bool isRhsSparse;
  const bool isResSparse;
  const bool rhsNeedTranspose;
  const SubBlockMask subBlockMask;
};

BSMatMulParams::BSMatMulParams(const std::array<int, 3> &dim,
                               const std::array<int, 3> &blockSize,
                               const std::vector<unsigned char> &rhsSparsity,
                               bool rhsNeedTranspose, poplar::Type inDataType,
                               poplar::Type outDataType,
                               poplar::Type partialDataType)
    : impl(new BSMatMulImpl(dim, blockSize, rhsSparsity, rhsNeedTranspose,
                            inDataType, outDataType, partialDataType)) {}

BSMatMulParams::BSMatMulParams(const std::array<int, 3> &dim,
                               const std::array<int, 3> &blockSize,
                               const std::vector<unsigned char> &lhsSparsity,
                               bool lhsNeedTranspose,
                               const std::vector<unsigned char> &rhsSparsity,
                               bool rhsNeedTranspose, poplar::Type inDataType,
                               poplar::Type outDataType,
                               poplar::Type partialDataType)
    : impl(new BSMatMulImpl(dim, blockSize, lhsSparsity, lhsNeedTranspose,
                            rhsSparsity, rhsNeedTranspose, inDataType,
                            outDataType, partialDataType)) {}

BSMatMulParams::BSMatMulParams(const std::array<int, 3> &dim,
                               const std::array<int, 3> &blockSize,
                               const std::vector<unsigned char> &resSparsity,
                               poplar::Type inDataType,
                               poplar::Type outDataType,
                               poplar::Type partialDataType,
                               SubBlockMask subBlockMask)
    : impl(new BSMatMulImpl(dim, blockSize, resSparsity, inDataType,
                            outDataType, partialDataType, subBlockMask)) {}

BSMatMulParams::BSMatMulParams(BSMatMulParams &&other) = default;

BSMatMulParams::~BSMatMulParams() = default;

poplar::Tensor createBSMatMulInputLHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name) {
  BSMatMulImpl *impl = bsMatMul.impl.get();

  BlockMatrix *lhsMatrix = impl->lhsMatrix.get();

  return lhsMatrix->createTensor(graph, impl->inDataType, name);
}

poplar::Tensor createBSMatMulInputRHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name) {
  BSMatMulImpl *impl = bsMatMul.impl.get();

  BlockMatrix *rhsMatrix = impl->rhsMatrix.get();

  return rhsMatrix->createTensor(graph, impl->inDataType, name);
}

static void parseOptions(const poplar::OptionFlags &options,
                         double &memoryCycleRatio) {
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec bsSpec{
      {"memory-cycle-ratio", OptionHandler::createWithDouble(memoryCycleRatio)},
  };
  for (const auto &entry : options) {
    bsSpec.parse(entry.first, entry.second);
  }
}

poplar::Tensor bsMatMul(poplar::Graph &graph, const BSMatMulParams &bsMatMul,
                        poplar::program::Sequence &prog,
                        const poplar::Tensor &lhsMatrix,
                        const poplar::Tensor &rhsMatrix,
                        const poplar::OptionFlags &optionFlags,
                        const std::string &debugPrefix) {
  BSMatMulImpl &bsMatMulImpl = *(bsMatMul.impl.get());

  double memoryCycleRatio =
      0.2; // Default value which works well. Found by in tests.
  parseOptions(optionFlags, memoryCycleRatio);

  auto &lhs = *bsMatMulImpl.lhsMatrix;
  auto &rhs = *bsMatMulImpl.rhsMatrix;

  lhs.setBlockTensor(lhsMatrix);
  rhs.setBlockTensor(rhsMatrix);

  std::unique_ptr<HyperGraph> hg;
  unsigned numTiles = graph.getTarget().getTilesPerIPU();
  hg = std::make_unique<HyperGraphBlockZoltan>(
      lhs, rhs, bsMatMulImpl.inDataType, bsMatMulImpl.outDataType,
      bsMatMulImpl.partialDataType, numTiles,
      static_cast<float>(memoryCycleRatio));

  if (!bsMatMulImpl.isResSparse) {
    hg->createGraphMatMul(graph, debugPrefix);
  } else {
    hg->createGraphMatMulSparsifyResult(graph, bsMatMulImpl.resSparsity.data(),
                                        debugPrefix);
  }

  hg->createProgramMatMul(graph, bsMatMulImpl.subBlockMask, prog, debugPrefix);

  return hg->getResultTensor();
}

} // namespace experimental
} // namespace popsparse
