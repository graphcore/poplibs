// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include "popsparse/experimental/BlockSparseMatMul.hpp"
#include "BSMatrix.hpp"
#include "BSOps.hpp"
#include "HyperGraphBlockGroup2.hpp"
#include "HyperGraphBlockNaive.hpp"
#include "HyperGraphBlockZoltan.hpp"
#include "HyperGraphStrip.hpp"
#include "HyperGraphStripV0.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include <iostream>
#include <memory>
#include <poplibs_support/logging.hpp>
#include <popops/Rearrange.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/OptionParsing.hpp>
#include <poputil/exceptions.hpp>

namespace logging = poplibs_support::logging;

namespace poputil {
template <>
poplar::ProfileValue
toProfileValue(const popsparse::experimental::BSMatMulParams &t) {
  poplar::ProfileValue::Map v;
  return poplar::ProfileValue("<popsparse::experimental::BSMatMulParams>");
}
} // namespace poputil

namespace popsparse {
namespace experimental {

using namespace poplibs;

class BSMatMulImpl {

  static void
  checkCommonParameters(const std::array<int, 3> &dim,
                        const std::array<int, 3> &blockSize,
                        const std::vector<unsigned char> &sparsityMask,
                        const std::array<int, 2> &sparseDims,
                        const poplar::Type &inType, const poplar::Type &outType,
                        const poplar::Type &partialType, unsigned numGroups) {
    if (inType != poplar::HALF && inType != poplar::FLOAT) {
      throw poputil::poplibs_error(
          "Input error: input data type must be half or float but got " +
          inType.toString());
    }
    if (outType != poplar::HALF && outType != poplar::FLOAT) {
      throw poputil::poplibs_error(
          "Input error: output data type must be half or float but got " +
          outType.toString());
    }
    if (partialType != poplar::HALF && partialType != poplar::FLOAT) {
      throw poputil::poplibs_error(
          "Input error: partial data type must be half or float but got " +
          partialType.toString());
    }
    if (inType == poplar::FLOAT && partialType == poplar::HALF) {
      throw poputil::poplibs_error(
          "Input error: partial data type's precision must be equal to or "
          "greater than input data type's precision. Input type was " +
          inType.toString() + " and partial data type was " +
          partialType.toString() + ".");
    }
    if (numGroups == 0) {
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
    const std::size_t sparsitySize = sparsityMask.size();
    if (sparsitySize % numGroups != 0) {
      throw poputil::poplibs_error(
          "Input error: sparsity mask size " + std::to_string(sparsitySize) +
          " is not divisible by number of groups " + std::to_string(numGroups));
    }
    const std::size_t sparsitySizePerGroup = sparsitySize / numGroups;
    const std::size_t numBlocks =
        std::accumulate(sparseDims.begin(), sparseDims.end(), std::size_t(1),
                        [&](std::size_t t, int iDim) {
                          return t * static_cast<std::size_t>(dim[iDim]) /
                                 static_cast<std::size_t>(blockSize[iDim]);
                        });
    if (numBlocks != sparsitySizePerGroup) {
      throw poputil::poplibs_error(
          "Input error: sparsity mask size " +
          std::string((numGroups > 1 ? "per group:" : ":")) +
          std::to_string(sparsitySizePerGroup) +
          " does not match total number of blocks: " +
          std::to_string(numBlocks));
    }
  }

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
    checkCommonParameters(dim, blockSize, rhsSparsity, {1, 2}, inDataType,
                          outDataType, partialDataType, numGroups);
    const unsigned char *sparsityBuf = rhsSparsity.data();
    const std::size_t sparsitySizePerGroup = rhsSparsity.size() / numGroups;
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

    logging::popsparse::info(
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
    checkCommonParameters(dim, blockSize, resSparsityIn, {0, 2}, inDataType,
                          outDataType, partialDataType, numGroupsIn);
    for (unsigned idxGroup = 0; idxGroup < numGroups; ++idxGroup) {
      lhsMatrices.emplace_back(new BlockDenseMatrix(
          dim[0], dim[1], blockSize[0], blockSize[1], false));
      rhsMatrices.emplace_back(new BlockDenseMatrix(
          dim[1], dim[2], blockSize[1], blockSize[2], false));
    }

    logging::popsparse::info(
        "bsMatMul dds: {} x {} x {}, block: {} x {} x {} {} group(s)", dim[0],
        dim[1], dim[2], blockSize[0], blockSize[1], blockSize[2], numGroups);
  }

  void createPartitionPlan(poplar::Graph &graph,
                           const poplar::OptionFlags &options,
                           const poplar::DebugNameAndId &dnai);

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
  std::vector<std::unique_ptr<HyperGraph>> hyperGraphs;
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

void BSMatMulImpl::createPartitionPlan(poplar::Graph &graph,
                                       const poplar::OptionFlags &options,
                                       const poplar::DebugNameAndId &dnai) {
  double memoryCycleRatio = 0.2;
  int nPass = 1;
  std::string partitionMethod = std::string("strip");
  parseOptions(options, memoryCycleRatio, nPass, partitionMethod);

  if (nPass > 1 && numGroups > 1) {
    throw poputil::poplibs_error(
        "For group operation number-of-pass option must be 1");
  }

  enum class PartitionMethod {
    STRIPV0,
    STRIP,
    BLOCK,
    BLOCK_NAIVE,
    BLOCK_GROUP2,
  };

  PartitionMethod pm = PartitionMethod::STRIP;
  logging::popsparse::info("Partition method used: {}",
                           partitionMethod.c_str());
  if (partitionMethod.compare("block") == 0) {
    pm = PartitionMethod::BLOCK;
  } else if (partitionMethod.compare("block-naive") == 0) {
    pm = PartitionMethod::BLOCK_NAIVE;
  } else if (partitionMethod.compare("block-group2") == 0) {
    pm = PartitionMethod::BLOCK_GROUP2;
  } else if (partitionMethod.compare("stripv0") == 0) {
    pm = PartitionMethod::STRIPV0;
  } else if (partitionMethod.compare("strip") == 0) {
    pm = PartitionMethod::STRIP;
  } else {
    logging::popsparse::warn(
        "Unknown partition method {}. Default method strip will be used.",
        partitionMethod.c_str());
  }

  logging::popsparse::info(
      "matrix dimension [{}, {}, {}, {}] rhs need transpose {} "
      "inner group size = {}",
      (int)lhsMatrices.size(), lhsMatrices[0]->getRowCount(),
      lhsMatrices[0]->getColCount(), rhsMatrices[0]->getColCount(),
      rhsNeedTranspose, numGroups);

  if (isResSparse) {
    logging::popsparse::info("result sparse");
  }

  logging::popsparse::info(
      "block dimention [{}, {}, {}]", lhsMatrices[0]->getBlockRow(),
      lhsMatrices[0]->getBlockCol(), rhsMatrices[0]->getBlockCol());

  if (isResSparse) {
    int zeros = 0;
    for (unsigned i = 0; i < resSparsity.size(); i++) {
      if (resSparsity[i] == 0)
        zeros++;
    }
    float sparsity = (float)zeros / (float)resSparsity.size();
    logging::popsparse::info("non zero blocks {}, sparsity {}",
                             (int)(resSparsity.size() - zeros), sparsity);
  } else {
    int nonZeros = rhsMatrices[0]->getNonZeroBlockCount();
    int blocks =
        rhsMatrices[0]->getBlockRowCount() * rhsMatrices[0]->getBlockColCount();

    logging::popsparse::info("non zero blocks {}, sparsity {}\n", nonZeros,
                             1.0 - (float)nonZeros / (float)blocks);
  }

  std::function<HyperGraph *(BlockMatrix &, BlockMatrix &, unsigned int)>
      createHyperGraph = [&](BlockMatrix &lhs, BlockMatrix &rhs,
                             unsigned int numTiles) {
        HyperGraph *hg;
        switch (pm) {
        case (PartitionMethod::BLOCK):
          hg = new HyperGraphBlockZoltan(lhs, rhs, inDataType, outDataType,
                                         partialDataType, numTiles,
                                         static_cast<float>(memoryCycleRatio));
          break;
        case (PartitionMethod::BLOCK_NAIVE):
          hg = new HyperGraphBlockNaive(lhs, rhs, inDataType, outDataType,
                                        partialDataType, numTiles);
          break;
        case (PartitionMethod::STRIPV0):
          hg = new HyperGraphStripV0(lhs, rhs, inDataType, outDataType,
                                     partialDataType, numTiles, nPass);
          break;
        case (PartitionMethod::BLOCK_GROUP2):
          hg = new HyperGraphBlockGroup2(lhs, rhs, inDataType, outDataType,
                                         partialDataType, numTiles);
          break;
        default:
          hg = new HyperGraphStrip(lhs, rhs, inDataType, outDataType,
                                   partialDataType, numTiles, nPass);
          break;
        };
        return hg;
      };

  const std::string layer = "bsMatMul";
  if (numGroups == 1) {
    hyperGraphs.resize(1);

    auto &lhs = *lhsMatrices[0];
    auto &rhs = *rhsMatrices[0];

    unsigned numTiles = graph.getTarget().getTilesPerIPU();
    logging::popsparse::debug(
        "{}: {} x {} x {}, block: {} x {} x {}. Tiles: {}",
        isResSparse ? "dds" : "dsd", lhs.getRowCount(), lhs.getColCount(),
        rhs.getColCount(), lhs.getBlockRow(), lhs.getBlockCol(),
        rhs.getBlockCol(), numTiles);
    HyperGraph *hg = createHyperGraph(lhs, rhs, numTiles);

    if (!isResSparse) {
      hg->createGraphMatMul(graph, {dnai, layer});
    } else {
      hg->createGraphMatMulSparsifyResult(graph, resSparsity.data(),
                                          {dnai, layer});
    }

    hyperGraphs[0].reset(hg);
  } else {
    hyperGraphs.resize(numGroups);

    unsigned numTilesTotal = graph.getTarget().getTilesPerIPU();
    unsigned numTilesPerGroupLow = numTilesTotal / numGroups;
    unsigned numTilesPerGroupLeftover = numTilesTotal % numGroups;

    unsigned idxLowerTile = 0;
    const unsigned char *sparsityBuf = resSparsity.data();
    std::size_t sparsitySizePerGroup = resSparsity.size() / numGroups;
    for (unsigned idxGroup = 0; idxGroup < numGroups; ++idxGroup) {
      unsigned numTiles = (idxGroup < numTilesPerGroupLeftover)
                              ? numTilesPerGroupLow + 1
                              : numTilesPerGroupLow;
      unsigned idxUpperTile = idxLowerTile + numTiles;
      poplar::Graph subGraph =
          graph.createVirtualGraph(idxLowerTile, idxUpperTile);

      auto &lhs = *lhsMatrices[idxGroup];
      auto &rhs = *rhsMatrices[idxGroup];
      HyperGraph *hg = createHyperGraph(lhs, rhs, numTiles);

      if (!isResSparse) {
        hg->createGraphMatMul(subGraph, {dnai, layer});
      } else {
        hg->createGraphMatMulSparsifyResult(subGraph, sparsityBuf,
                                            {dnai, layer});
      }

      sparsityBuf += sparsitySizePerGroup;
      idxLowerTile = idxUpperTile;

      hyperGraphs[idxGroup].reset(hg);
    }
  }
}

poplar::Tensor createBSMatMulInputLHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const poplar::DebugContext &debugContext,
                                      const poplar::OptionFlags &options) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(bsMatMul, options));

  BSMatMulImpl *impl = bsMatMul.impl.get();
  if (impl->hyperGraphs.empty()) {
    impl->createPartitionPlan(graph, options, {di});
  }
  unsigned numGroups = impl->numGroups;
  assert(impl->lhsMatrices.size() == impl->numGroups);

  std::vector<poplar::Tensor> ts;
  if (numGroups == 1) {
    poplar::Tensor t =
        impl->lhsMatrices[0]->createTensor(graph, impl->inDataType, {di});
    ts.push_back(t);

    impl->hyperGraphs[0]->setTileMappingLHS(graph, t);
  } else {
    unsigned numTilesTotal = graph.getTarget().getTilesPerIPU();
    unsigned numTilesPerGroupLow = numTilesTotal / numGroups;
    unsigned numTilesPerGroupLeftover = numTilesTotal % numGroups;

    unsigned idxLowerTile = 0;
    for (unsigned idxGroup = 0; idxGroup < numGroups; ++idxGroup) {
      unsigned numTiles = (idxGroup < numTilesPerGroupLeftover)
                              ? numTilesPerGroupLow + 1
                              : numTilesPerGroupLow;
      unsigned idxUpperTile = idxLowerTile + numTiles;
      poplar::Graph subGraph =
          graph.createVirtualGraph(idxLowerTile, idxUpperTile);

      poplar::Tensor t = impl->lhsMatrices[idxGroup]->createTensor(
          subGraph, impl->inDataType, {di});
      ts.push_back(t);

      idxLowerTile = idxUpperTile;

      impl->hyperGraphs[idxGroup]->setTileMappingLHS(subGraph, t);
    }
  }

  return poplar::concat(ts);
}

poplar::Tensor createBSMatMulInputRHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const poplar::DebugContext &debugContext,
                                      const poplar::OptionFlags &options) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(bsMatMul, options));

  BSMatMulImpl *impl = bsMatMul.impl.get();
  if (impl->hyperGraphs.empty()) {
    impl->createPartitionPlan(graph, options, {di});
  }
  unsigned numGroups = impl->numGroups;
  assert(impl->rhsMatrices.size() == impl->numGroups);

  std::vector<poplar::Tensor> ts;
  if (numGroups == 1) {
    poplar::Tensor t =
        impl->rhsMatrices[0]->createTensor(graph, impl->inDataType, {di});
    ts.push_back(t);

    impl->hyperGraphs[0]->setTileMappingRHS(graph, t);
  } else {
    unsigned numTilesTotal = graph.getTarget().getTilesPerIPU();
    unsigned numTilesPerGroupLow = numTilesTotal / numGroups;
    unsigned numTilesPerGroupLeftover = numTilesTotal % numGroups;

    unsigned idxLowerTile = 0;
    for (unsigned idxGroup = 0; idxGroup < numGroups; ++idxGroup) {
      unsigned numTiles = (idxGroup < numTilesPerGroupLeftover)
                              ? numTilesPerGroupLow + 1
                              : numTilesPerGroupLow;
      unsigned idxUpperTile = idxLowerTile + numTiles;
      poplar::Graph subGraph =
          graph.createVirtualGraph(idxLowerTile, idxUpperTile);

      poplar::Tensor t = impl->rhsMatrices[idxGroup]->createTensor(
          subGraph, impl->inDataType, {di});
      ts.push_back(t);

      idxLowerTile = idxUpperTile;

      impl->hyperGraphs[idxGroup]->setTileMappingRHS(subGraph, t);
    }
  }

  return poplar::concat(ts);
}

poplar::Tensor bsMatMul(poplar::Graph &graph, const BSMatMulParams &bsMatMul,
                        poplar::program::Sequence &prog,
                        const poplar::Tensor &lhsMatrixIn,
                        const poplar::Tensor &rhsMatrixIn,
                        const poplar::OptionFlags &options,
                        const poplar::DebugContext &debugContext) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(bsMatMul, lhsMatrixIn, rhsMatrixIn, options));

  if (lhsMatrixIn.rank() != 2 || rhsMatrixIn.rank() != 2) {
    throw poputil::poplibs_error("The rank of both input matrices must be 2");
  }

  BSMatMulImpl &bsMatMulImpl = *(bsMatMul.impl.get());
  if (bsMatMulImpl.hyperGraphs.empty()) {
    bsMatMulImpl.createPartitionPlan(graph, options, {di});
  }

  // If the input tensors are re-grouped, it makes huge performance difference
  // in HyperGraph::preprocessBlocks to copy the blocks to make it contiguous.
  int groupSize = bsMatMulImpl.inDataType == poplar::HALF ? 16 : 8;

  poplar::Tensor lhsMatrix = popops::rearrange::regroupIfBeneficial(
      graph, lhsMatrixIn, groupSize, prog, {di, "bs-matmul-regroup-lhs"});

  poplar::Tensor rhsMatrix = popops::rearrange::regroupIfBeneficial(
      graph, rhsMatrixIn, groupSize, prog, {di, "bs-matmul-regroup-rhs"});

  if (bsMatMulImpl.numGroups == 1) {
    auto &lhs = *bsMatMulImpl.lhsMatrices[0];
    auto &rhs = *bsMatMulImpl.rhsMatrices[0];

    lhs.setBlockTensor(lhsMatrix);
    rhs.setBlockTensor(rhsMatrix);

    HyperGraph *hg = bsMatMulImpl.hyperGraphs[0].get();

    hg->createProgramMatMul(graph, bsMatMulImpl.subBlockMask, prog, {di});

    return hg->getResultTensor();
  } else {
    std::unique_ptr<poplar::ComputeSet> transposeCS;
    if (!bsMatMulImpl.rhsNeedTranspose) {
      transposeCS.reset(
          new poplar::ComputeSet(graph.addComputeSet({di, "transposeCS"})));
      prog.add(poplar::program::Execute(*transposeCS.get(), {di}));
    }
    poplar::ComputeSet mulCS = graph.addComputeSet({di, "mulCS"});
    poplar::ComputeSet reduceCS = graph.addComputeSet({di, "reduceCS"});

    unsigned numTilesTotal = graph.getTarget().getTilesPerIPU();
    unsigned numTilesPerGroupLow = numTilesTotal / bsMatMulImpl.numGroups;
    unsigned numTilesPerGroupLeftover = numTilesTotal % bsMatMulImpl.numGroups;

    int resBlockRow, resBlockCol;
    int resBlockRowCount, resBlockColCount;

    std::vector<poplar::Tensor> resTensors;
    unsigned idxLowerTile = 0;
    std::size_t lhsBegin = 0;
    std::size_t rhsBegin = 0;
    std::size_t lhsEnd, rhsEnd;
    for (unsigned idxGroup = 0; idxGroup < bsMatMulImpl.numGroups; ++idxGroup) {
      unsigned numTilesPerGroup = (idxGroup < numTilesPerGroupLeftover)
                                      ? numTilesPerGroupLow + 1
                                      : numTilesPerGroupLow;
      unsigned idxUpperTile = idxLowerTile + numTilesPerGroup;
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

      logging::popsparse::debug(
          "dds: {} x {} x {}, block: {} x {} x {}, group: {}, num tiles: {}",
          lhs.getRowCount(), lhs.getColCount(), rhs.getColCount(),
          lhs.getBlockRow(), lhs.getBlockCol(), rhs.getBlockCol(), idxGroup,
          numTilesTotal);

      HyperGraph *hg = bsMatMulImpl.hyperGraphs[idxGroup].get();

      hg->createProgramMatMul(subGraph, transposeCS.get(), mulCS, reduceCS,
                              prog, {di});

      resTensors.push_back(hg->getResultTensor());
      hg->getResultBlockSize(resBlockRow, resBlockCol);
      hg->getResultBlockCount(resBlockRowCount, resBlockColCount);

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
    prog.add(poplar::program::Execute(mulCS, {di}));
    prog.add(poplar::program::Execute(reduceCS, {di}));

    poplar::Tensor resTensor = poplar::concat(resTensors);
    if (bsMatMulImpl.subBlockMask != SubBlockMask::None) {
      applySubBlockMask(graph, resTensor, bsMatMulImpl.subBlockMask,
                        resBlockRow, resBlockCol, resBlockRowCount,
                        resBlockColCount, bsMatMulImpl.resSparsity.data(),
                        bsMatMulImpl.numGroups, prog, {di});
    }

    return resTensor;
  }
}

} // namespace experimental
} // namespace popsparse
