// Copyright (c) 2020 Graphcore Ltd, All rights reserved.

#include "popsparse/experimental/BlockSparseMatMul.hpp"
#include "BSMatrix.hpp"
#include "HyperGraph.hpp"
#include <iostream>
#include <memory>
#include <poplibs_support/OptionParsing.hpp>

#include <poputil/exceptions.hpp>

namespace popsparse {
namespace experimental {

using namespace poplibs;

class BSMatMulImpl {

public:
  BSMatMulImpl(const std::array<int, 3> &dim,
               const std::array<int, 3> &blockSize,
               const std::vector<unsigned char> &rhsSparsity,
               bool rhsNeedTranspose, poplar::Type inDataTypeIn,
               poplar::Type outDataTypeIn, poplar::Type partialDataTypeIn,
               float memoryCycleRatioIn)
      : inDataType(inDataTypeIn), outDataType(outDataTypeIn),
        partialDataType(partialDataTypeIn),
        memoryCycleRatio(memoryCycleRatioIn), isLhsSparse(false),
        isRhsSparse(true), isResSparse(false) {

    assert(static_cast<int>(rhsSparsity.size()) ==
           dim[1] / blockSize[1] * dim[2] / blockSize[2]);
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
               poplar::Type outDataTypeIn, poplar::Type partialDataTypeIn,
               float memoryCycleRatioIn)
      : inDataType(inDataTypeIn), outDataType(outDataTypeIn),
        partialDataType(partialDataTypeIn),
        memoryCycleRatio(memoryCycleRatioIn), isLhsSparse(true),
        isRhsSparse(true), isResSparse(false) {
    assert(0);
  }

  BSMatMulImpl(const std::array<int, 3> &dim,
               const std::array<int, 3> &blockSize,
               const std::vector<unsigned char> &resSparsityIn,
               poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
               poplar::Type partialDataTypeIn, float memoryCycleRatioIn)
      : inDataType(inDataTypeIn), outDataType(outDataTypeIn),
        partialDataType(partialDataTypeIn),
        memoryCycleRatio(memoryCycleRatioIn), isLhsSparse(false),
        isRhsSparse(false), isResSparse(true) {
    assert(static_cast<int>(resSparsityIn.size()) ==
           dim[0] / blockSize[0] * dim[2] / blockSize[2]);
    lhsMatrix.reset(new BlockDenseMatrix(dim[0], dim[1], blockSize[0],
                                         blockSize[1], false));
    rhsMatrix.reset(new BlockDenseMatrix(dim[1], dim[2], blockSize[1],
                                         blockSize[2], false));
    resSparsity = resSparsityIn;
  }

  std::unique_ptr<BlockMatrix> lhsMatrix;
  std::unique_ptr<BlockMatrix> rhsMatrix;
  std::vector<unsigned char> resSparsity;
  poplar::Type inDataType;
  poplar::Type outDataType;
  poplar::Type partialDataType;
  float memoryCycleRatio;
  bool isLhsSparse;
  bool isRhsSparse;
  bool isResSparse;
};

BSMatMulParams::BSMatMulParams(const std::array<int, 3> &dim,
                               const std::array<int, 3> &blockSize,
                               const std::vector<unsigned char> &rhsSparsity,
                               bool rhsNeedTranspose, poplar::Type inDataType,
                               poplar::Type outDataType,
                               poplar::Type partialDataType)
    : impl(new BSMatMulImpl(dim, blockSize, rhsSparsity, rhsNeedTranspose,
                            inDataType, outDataType, partialDataType, 0.0)) {}

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
                            outDataType, partialDataType, 0.0)) {}

BSMatMulParams::BSMatMulParams(const std::array<int, 3> &dim,
                               const std::array<int, 3> &blockSize,
                               const std::vector<unsigned char> &resSparsity,
                               poplar::Type inDataType,
                               poplar::Type outDataType,
                               poplar::Type partialDataType)
    : impl(new BSMatMulImpl(dim, blockSize, resSparsity, inDataType,
                            outDataType, partialDataType, 0.0)) {}

BSMatMulParams::BSMatMulParams(BSMatMulParams &&other) = default;

BSMatMulParams::~BSMatMulParams() = default;

poplar::Tensor createBSMatMulInputLHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name) {
  BSMatMulImpl *impl = bsMatMul.impl.get();

  BlockMatrix *lhsMatrix = impl->lhsMatrix.get();

  return lhsMatrix->createTensor(graph, impl->inDataType, name);
}

/**
 * Create a tensor that is used as the right operand of block sparse matrix
 * multiplication.
 *
 * \param bsMatMul        The object for block sparse information, includes the
 *                        sparsity mask, the matrix size, the block size,
 *                        and the data type
 *
 * \param graph           The Poplar graph.
 *
 * \param name            The debug name of the required matrix.
 *
 * \returns               The return tensor is an array of non zero
 *                        blocks for block sparse matrix
 */
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
  double memoryCycleRatio;
  parseOptions(optionFlags, memoryCycleRatio);

  BSMatMulImpl *impl = (BSMatMulImpl *)(bsMatMul.impl.get());
  impl->memoryCycleRatio = (float)memoryCycleRatio;

  impl->lhsMatrix->setBlockTensor(lhsMatrix);
  impl->rhsMatrix->setBlockTensor(rhsMatrix);

  HyperGraph hg(*impl->lhsMatrix, *impl->rhsMatrix, impl->inDataType,
                impl->outDataType, impl->partialDataType,
                graph.getTarget().getTilesPerIPU());

  if (!impl->isResSparse) {
    hg.createGraphMatMul(impl->memoryCycleRatio, graph, debugPrefix);
  } else {
    hg.createGraphMatMulSparsifyResult(
        impl->resSparsity.data(), impl->memoryCycleRatio, graph, debugPrefix);
  }

  std::vector<int> tileAssignment;
  hg.partitionGraph(tileAssignment);
  hg.createProgramMatMul(tileAssignment, graph, prog, debugPrefix);

  if (hg.matC->isDense()) {
    return ((BlockDenseMatrix *)hg.matC.get())->denseMatrix;
  } else {
    std::vector<poplar::Tensor> blocks;
    for (auto &b : hg.matC->getBlockTensor()) {
      blocks.push_back(b.expand({0}));
    }

    return concat(blocks);
  }
}

} // namespace experimental
} // namespace popsparse
