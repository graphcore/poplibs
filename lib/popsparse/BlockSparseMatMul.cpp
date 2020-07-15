// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include "popsparse/experimental/BlockSparseMatMul.hpp"
#include "BSMatrix.hpp"
#include "BSOps.hpp"
#include "HyperGraphBlockNaive.hpp"
#include "HyperGraphBlockZoltan.hpp"
#include "HyperGraphStrip.hpp"
#include "HyperGraphStripV0.hpp"
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
               poplar::Type outDataTypeIn, poplar::Type partialDataTypeIn,
               unsigned numGroupsIn = 1)
      : inDataType(inDataTypeIn), outDataType(outDataTypeIn),
        partialDataType(partialDataTypeIn), isLhsSparse(false),
        isRhsSparse(true), isResSparse(false),
        rhsNeedTranspose(rhsNeedTransposeIn), subBlockMask(SubBlockMask::None),
        numGroups(numGroupsIn) {

    if (numGroupsIn == 0) {
      throw poputil::poplibs_error("Input error: zero number of groups.");
    }
    for (int iDim = 0; iDim < 3; ++iDim) {
      if (dim[iDim] % blockSize[iDim] != 0) {
        throw poputil::poplibs_error(
            "Input error: input dimension " + std::to_string(iDim) + ": " +
            std::to_string(dim[iDim]) +
            " is not divisible by block size dimension " +
            std::to_string(iDim) + ": " + std::to_string(blockSize[iDim]));
      }
    }
    std::size_t sparsitySize = rhsSparsity.size();
    if (sparsitySize % numGroups != 0) {
      throw poputil::poplibs_error(
          "Input error: sparsity mask size " + std::to_string(sparsitySize) +
          " is not divisible by number of groups " + std::to_string(numGroups));
    }
    std::size_t sparsitySizePerGroup = sparsitySize / numGroups;
    int numBlocks = dim[1] / blockSize[1] * dim[2] / blockSize[2];
    if (static_cast<int>(sparsitySizePerGroup) != numBlocks) {
      throw poputil::poplibs_error(
          "Input error: sparsity mask size" +
          std::string((numGroups > 1 ? " per group:" : ":")) +
          std::to_string(sparsitySizePerGroup) +
          " does not match total number of blocks: " +
          std::to_string(numBlocks));
    }
    const unsigned char *sparsityBuf = rhsSparsity.data();
    for (unsigned idxGroup = 0; idxGroup < numGroups; ++idxGroup) {
      lhsMatrices.emplace_back(new BlockDenseMatrix(
          dim[0], dim[1], blockSize[0], blockSize[1], false));
      if (!rhsNeedTransposeIn) {
        rhsMatrices.emplace_back(
            new BlockSparseMatrix(dim[1], dim[2], blockSize[1], blockSize[2],
                                  rhsNeedTransposeIn, sparsityBuf));
      } else {
        rhsMatrices.emplace_back(
            new BlockSparseMatrix(dim[2], dim[1], blockSize[2], blockSize[1],
                                  rhsNeedTransposeIn, sparsityBuf));
      }
      sparsityBuf += sparsitySizePerGroup;
    }

    logging::info(
        "bsMatMul dsd: {} x {} x {}, block: {} x {} x {} {} {} group(s)",
        dim[0], dim[1], dim[2], blockSize[0], blockSize[1], blockSize[2],
        (rhsNeedTransposeIn ? "rhs transposed" : ""), numGroups);
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
        rhsNeedTranspose(false), subBlockMask(subBlockMaskIn),
        numGroups(numGroupsIn) {
    if (numGroupsIn == 0) {
      throw poputil::poplibs_error("Input error: zero number of groups.");
    }
    for (int iDim = 0; iDim < 3; ++iDim) {
      if (dim[iDim] % blockSize[iDim] != 0) {
        throw poputil::poplibs_error(
            "Input error: input dimension " + std::to_string(iDim) + ": " +
            std::to_string(dim[iDim]) +
            " is not divisible by block size dimension " +
            std::to_string(iDim) + ": " + std::to_string(blockSize[iDim]));
      }
    }
    std::size_t sparsitySize = resSparsityIn.size();
    if (sparsitySize % numGroups != 0) {
      throw poputil::poplibs_error(
          "Input error: sparsity mask size " + std::to_string(sparsitySize) +
          " is not divisible by number of groups " + std::to_string(numGroups));
    }
    std::size_t sparsitySizePerGroup = sparsitySize / numGroups;
    int numBlocks = dim[0] / blockSize[0] * dim[2] / blockSize[2];
    if (static_cast<int>(sparsitySizePerGroup) != numBlocks) {
      throw poputil::poplibs_error(
          "Input error: sparsity mask size " +
          std::string((numGroups > 1 ? "per group:" : ":")) +
          std::to_string(sparsitySizePerGroup) +
          " does not match total number of blocks: " +
          std::to_string(numBlocks));
    }
    for (unsigned idxGroup = 0; idxGroup < numGroups; ++idxGroup) {
      lhsMatrices.emplace_back(new BlockDenseMatrix(
          dim[0], dim[1], blockSize[0], blockSize[1], false));
      rhsMatrices.emplace_back(new BlockDenseMatrix(
          dim[1], dim[2], blockSize[1], blockSize[2], false));
    }

    logging::info("bsMatMul dds: {} x {} x {}, block: {} x {} x {} {} group(s)",
                  dim[0], dim[1], dim[2], blockSize[0], blockSize[1],
                  blockSize[2], numGroups);
  }

  std::vector<std::unique_ptr<BlockMatrix>> lhsMatrices;
  std::vector<std::unique_ptr<BlockMatrix>> rhsMatrices;
  std::vector<unsigned char> resSparsity;
  const poplar::Type inDataType;
  const poplar::Type outDataType;
  const poplar::Type partialDataType;
  const bool isLhsSparse;
  const bool isRhsSparse;
  const bool isResSparse;
  const bool rhsNeedTranspose;
  const SubBlockMask subBlockMask;
  const unsigned numGroups;
};

BSMatMulParams::BSMatMulParams(const std::array<int, 3> &dim,
                               const std::array<int, 3> &blockSize,
                               const std::vector<unsigned char> &rhsSparsity,
                               bool rhsNeedTranspose, poplar::Type inDataType,
                               poplar::Type outDataType,
                               poplar::Type partialDataType,
                               unsigned numGroupsIn)
    : impl(new BSMatMulImpl(dim, blockSize, rhsSparsity, rhsNeedTranspose,
                            inDataType, outDataType, partialDataType,
                            numGroupsIn)) {}

BSMatMulParams::BSMatMulParams(const std::array<int, 3> &dim,
                               const std::array<int, 3> &blockSize,
                               const std::vector<unsigned char> &resSparsity,
                               poplar::Type inDataType,
                               poplar::Type outDataType,
                               poplar::Type partialDataType,
                               SubBlockMask subBlockMask, unsigned numGroupsIn)
    : impl(new BSMatMulImpl(dim, blockSize, resSparsity, inDataType,
                            outDataType, partialDataType, subBlockMask,
                            numGroupsIn)) {}

BSMatMulParams::BSMatMulParams(BSMatMulParams &&other) = default;

BSMatMulParams::~BSMatMulParams() = default;

poplar::Tensor createBSMatMulInputLHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name) {
  BSMatMulImpl *impl = bsMatMul.impl.get();

  assert(impl->lhsMatrices.size() == impl->numGroups);
  std::vector<poplar::Tensor> ts;
  for (unsigned idxGroup = 0; idxGroup < impl->numGroups; ++idxGroup) {
    poplar::Tensor t = impl->lhsMatrices[idxGroup]->createTensor(
        graph, impl->inDataType, name);
    ts.push_back(t);
  }
  return poplar::concat(ts);
}

poplar::Tensor createBSMatMulInputRHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name) {
  BSMatMulImpl *impl = bsMatMul.impl.get();

  assert(impl->rhsMatrices.size() == impl->numGroups);
  std::vector<poplar::Tensor> ts;
  for (unsigned idxGroup = 0; idxGroup < impl->numGroups; ++idxGroup) {
    poplar::Tensor t = impl->rhsMatrices[idxGroup]->createTensor(
        graph, impl->inDataType, name);
    ts.push_back(t);
  }
  return poplar::concat(ts);
}

static void parseOptions(const poplar::OptionFlags &options,
                         double &memoryCycleRatio, int &nPass,
                         std::string &partitionMethod) {
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec bsSpec{
      {"memoryCycleRatio", OptionHandler::createWithDouble(memoryCycleRatio)},
      {"numberOfPass", OptionHandler::createWithInteger(nPass)},
      {"partitionMethod", OptionHandler::createWithString(partitionMethod)},
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

  logging::info("blocksparse matmul: number of groups = {}",
                bsMatMulImpl.numGroups);

  if (lhsMatrix.rank() != 2 || rhsMatrix.rank() != 2) {
    throw poputil::poplibs_error("The rank of both input matrices must be 2");
  }

  double memoryCycleRatio = 0.2;
  int nPass = 1;
  std::string partitionMethod = std::string("strip");
  parseOptions(optionFlags, memoryCycleRatio, nPass, partitionMethod);

  if (nPass > 1 && bsMatMulImpl.numGroups > 1) {
    throw poputil::poplibs_error(
        "For group operation number-of-pass option must be 1");
  }

  enum class PartitionMethod {
    STRIPV0,
    STRIP,
    BLOCK,
    BLOCK_NAIVE,
  };
  PartitionMethod pm = PartitionMethod::STRIP;
  logging::info("Partition method used: {}", partitionMethod.c_str());
  if (partitionMethod.compare("block") == 0) {
    pm = PartitionMethod::BLOCK;
  } else if (partitionMethod.compare("block-naive") == 0) {
    pm = PartitionMethod::BLOCK_NAIVE;
  } else if (partitionMethod.compare("stripv0") == 0) {
    pm = PartitionMethod::STRIPV0;
  } else if (partitionMethod.compare("strip") == 0) {
    pm = PartitionMethod::STRIP;
  } else {
    logging::warn(
        "Unknown partition method {}. Default method strip will be used.",
        partitionMethod.c_str());
  }

  logging::info("matrix dimension [{}, {}, {}, {}] rhs need transpose {} "
                "inner group size = {}",
                (int)bsMatMulImpl.lhsMatrices.size(),
                bsMatMulImpl.lhsMatrices[0]->getRowCount(),
                bsMatMulImpl.lhsMatrices[0]->getColCount(),
                bsMatMulImpl.rhsMatrices[0]->getColCount(),
                bsMatMulImpl.rhsNeedTranspose, bsMatMulImpl.numGroups);

  if (bsMatMulImpl.isResSparse) {
    logging::info("result sparse");
  }

  logging::info("block dimention [{}, {}, {}]",
                bsMatMulImpl.lhsMatrices[0]->getBlockRow(),
                bsMatMulImpl.lhsMatrices[0]->getBlockCol(),
                bsMatMulImpl.rhsMatrices[0]->getBlockCol());

  float sparsity = 0.0;
  if (bsMatMulImpl.isResSparse) {
    int zeros = 0;
    for (unsigned i = 0; i < bsMatMulImpl.resSparsity.size(); i++) {
      if (bsMatMulImpl.resSparsity[i] == 0)
        zeros++;
    }
    sparsity = (float)zeros / (float)bsMatMulImpl.resSparsity.size();
    logging::info("non zero blocks {}, sparsity {}",
                  (int)(bsMatMulImpl.resSparsity.size() - zeros), sparsity);
  } else {
    int nonZeros = bsMatMulImpl.rhsMatrices[0]->getNonZeroBlockCount();
    int blocks = bsMatMulImpl.rhsMatrices[0]->getBlockRowCount() *
                 bsMatMulImpl.rhsMatrices[0]->getBlockColCount();
    logging::info("non zero blocks {}, sparsity {}\n", nonZeros,
                  1.0 - (float)nonZeros / (float)blocks);
  }

  std::function<std::unique_ptr<HyperGraph>(const BlockMatrix &,
                                            const BlockMatrix &, unsigned int)>
      createHyperGraph = [&](const BlockMatrix &lhs, const BlockMatrix &rhs,
                             unsigned int numTiles) {
        std::unique_ptr<HyperGraph> hg;
        switch (pm) {
        case (PartitionMethod::BLOCK):
          hg = std::make_unique<HyperGraphBlockZoltan>(
              lhs, rhs, bsMatMulImpl.inDataType, bsMatMulImpl.outDataType,
              bsMatMulImpl.partialDataType, numTiles,
              static_cast<float>(memoryCycleRatio));
          break;
        case (PartitionMethod::BLOCK_NAIVE):
          hg = std::make_unique<HyperGraphBlockNaive>(
              lhs, rhs, bsMatMulImpl.inDataType, bsMatMulImpl.outDataType,
              bsMatMulImpl.partialDataType, numTiles,
              static_cast<float>(memoryCycleRatio));
          break;
        case (PartitionMethod::STRIPV0):
          hg = std::make_unique<HyperGraphStripV0>(
              lhs, rhs, bsMatMulImpl.inDataType, bsMatMulImpl.outDataType,
              bsMatMulImpl.partialDataType, numTiles,
              static_cast<float>(memoryCycleRatio));
          break;
        default:
          hg = std::make_unique<HyperGraphStrip>(
              lhs, rhs, bsMatMulImpl.inDataType, bsMatMulImpl.outDataType,
              bsMatMulImpl.partialDataType, numTiles, nPass);
          break;
        };
        return hg;
      };

  if (bsMatMulImpl.numGroups == 1) {
    auto &lhs = *bsMatMulImpl.lhsMatrices[0];
    auto &rhs = *bsMatMulImpl.rhsMatrices[0];

    lhs.setBlockTensor(lhsMatrix);
    rhs.setBlockTensor(rhsMatrix);

    unsigned numTiles = graph.getTarget().getTilesPerIPU();
    std::unique_ptr<HyperGraph> hg = createHyperGraph(lhs, rhs, numTiles);

    if (!bsMatMulImpl.isResSparse) {
      hg->createGraphMatMul(graph, debugPrefix);
    } else {
      hg->createGraphMatMulSparsifyResult(
          graph, bsMatMulImpl.resSparsity.data(), debugPrefix);
    }

    hg->createProgramMatMul(graph, bsMatMulImpl.subBlockMask, prog,
                            debugPrefix);

    return hg->getResultTensor();
  } else {
    std::unique_ptr<poplar::ComputeSet> transposeCS;
    if (!bsMatMulImpl.rhsNeedTranspose) {
      transposeCS.reset(new poplar::ComputeSet(
          graph.addComputeSet(debugPrefix + "/transposeCS")));
      prog.add(poplar::program::Execute(*transposeCS.get()));
    }
    poplar::ComputeSet mulCS = graph.addComputeSet(debugPrefix + "/mulCS");
    poplar::ComputeSet reduceCS =
        graph.addComputeSet(debugPrefix + "/reduceCS");

    unsigned numTilesTotal = graph.getTarget().getTilesPerIPU();
    unsigned numTilesPerGroupLow = numTilesTotal / bsMatMulImpl.numGroups;
    unsigned numTilesPerGroupLeftover = numTilesTotal % bsMatMulImpl.numGroups;

    int resBlockRow, resBlockCol;
    int resBlockRowCount, resBlockColCount;

    std::vector<poplar::Tensor> resTensors;
    unsigned idxLowerTile = 0;
    const unsigned char *sparsityBuf = bsMatMulImpl.resSparsity.data();
    std::size_t sparsitySizePerGroup =
        bsMatMulImpl.resSparsity.size() / bsMatMulImpl.numGroups;
    std::size_t lhsBegin = 0;
    std::size_t rhsBegin = 0;
    std::size_t lhsEnd, rhsEnd;
    for (unsigned idxGroup = 0; idxGroup < bsMatMulImpl.numGroups; ++idxGroup) {
      unsigned numTiles = (idxGroup < numTilesPerGroupLeftover)
                              ? numTilesPerGroupLow + 1
                              : numTilesPerGroupLow;
      unsigned idxUpperTile = idxLowerTile + numTiles;
      poplar::Graph subGraph =
          graph.createVirtualGraph(idxLowerTile, idxUpperTile);

      auto &lhs = *bsMatMulImpl.lhsMatrices[idxGroup];
      auto &rhs = *bsMatMulImpl.rhsMatrices[idxGroup];
      std::array<int, 2> lhsDims = lhs.getDimensions();
      std::array<int, 2> rhsDims = rhs.getDimensions();
      lhsEnd = lhsBegin + lhsDims[0];
      rhsEnd = rhsBegin + rhsDims[0];
      if (lhsEnd > lhsMatrix.dim(0)) {
        throw poputil::poplibs_error("0 dimension of LHS matrix is too small");
      }
      if (rhsEnd > rhsMatrix.dim(0)) {
        throw poputil::poplibs_error("0 dimension of RHS matrix is too small");
      }

      lhs.setBlockTensor(lhsMatrix.slice(lhsBegin, lhsEnd, 0));
      rhs.setBlockTensor(rhsMatrix.slice(rhsBegin, rhsEnd, 0));

      std::unique_ptr<HyperGraph> hg = createHyperGraph(lhs, rhs, numTiles);

      if (!bsMatMulImpl.isResSparse) {
        hg->createGraphMatMul(subGraph, debugPrefix);
      } else {
        hg->createGraphMatMulSparsifyResult(subGraph, sparsityBuf, debugPrefix);
      }

      hg->createProgramMatMul(subGraph, transposeCS.get(), mulCS, reduceCS,
                              prog, debugPrefix);

      resTensors.push_back(hg->getResultTensor());
      hg->getResultBlockSize(resBlockRow, resBlockCol);
      hg->getResultBlockCount(resBlockRowCount, resBlockColCount);

      sparsityBuf += sparsitySizePerGroup;
      idxLowerTile = idxUpperTile;
      lhsBegin = lhsEnd;
      rhsBegin = rhsEnd;
    }
    if (lhsEnd != lhsMatrix.dim(0)) {
      throw poputil::poplibs_error("0 dimension of LHS matrix is too big");
    }
    if (rhsEnd != rhsMatrix.dim(0)) {
      throw poputil::poplibs_error("0 dimension of RHS matrix is too big");
    }
    prog.add(poplar::program::Execute(mulCS));
    prog.add(poplar::program::Execute(reduceCS));

    poplar::Tensor resTensor = poplar::concat(resTensors);
    if (bsMatMulImpl.subBlockMask != SubBlockMask::None) {
      applySubBlockMask(graph, resTensor, bsMatMulImpl.subBlockMask,
                        resBlockRow, resBlockCol, resBlockRowCount,
                        resBlockColCount, bsMatMulImpl.resSparsity.data(),
                        bsMatMulImpl.numGroups, prog, debugPrefix);
    }

    return resTensor;
  }
}

} // namespace experimental
} // namespace popsparse
