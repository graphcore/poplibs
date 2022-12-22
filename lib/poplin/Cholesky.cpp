// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "poplin/Cholesky.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplin/MatMul.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Encoding.hpp"
#include "popops/Expr.hpp"
#include "popops/Fill.hpp"
#include "popops/Pad.hpp"
#include "popops/Reduce.hpp"
#include "popops/Zero.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/GraphFunction.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <math.h>
#include <sstream>

namespace pe = popops::expr;

namespace poplin {
namespace {

using VoidFunctionPtr = std::unique_ptr<poputil::graphfn::VoidFunction>;

struct CholeskyOptions {
  std::size_t blockSize;
  poplar::OptionFlags matmulOptions;

  CholeskyOptions(const poplar::OptionFlags &opts, std::size_t maxBlockSize)
      : blockSize(8) {
    for (auto &opt : opts) {
      if (opt.first == "blockSize") {
        blockSize = poplibs::parse::asInteger<std::size_t>(opt.second);
      } else {
        matmulOptions.set(opt.first, opt.second);
      }
    }
    if (blockSize > maxBlockSize)
      blockSize = maxBlockSize;
  }
};

struct CholeskyParams {
  std::size_t blockSize;
  poplar::OptionFlags options;
  PlanningCache *cache;

  CholeskyParams(const CholeskyOptions &options, PlanningCache *cache)
      : blockSize(options.blockSize), options(options.matmulOptions),
        cache(cache) {}
};

void validateInput(std::vector<uint64_t> shape) {
  auto rank = shape.size();
  if (rank < 2) {
    throw poputil::poplibs_error("tensor A must have rank of 2 or higher.");
  }

  auto m = shape[rank - 2];
  auto n = shape[rank - 1];
  if (n != m) {
    throw poputil::poplibs_error(
        "2 minor dimension of tensor A must have the same size.");
  }
}

void validateInput(poplar::Graph &graph, const poplar::Tensor &a) {
  validateInput(a.shape());
}

void zeroTensor(poplar::Graph &graph, const poplar::Tensor &t,
                poplar::program::Sequence &prog) {
  auto zero = graph.addConstant(t.elementType(), {}, 0);
  graph.setTileMapping(zero, 0);
  poputil::broadcastToMatch(zero, t.shape());
  prog.add(poplar::program::Copy(zero, t));
}

std::size_t bestTile(poplar::Graph &graph, const poplar::Tensor &tensor) {
  auto mappings = graph.getTileMapping(tensor);
  std::size_t best = mappings.size();
  std::size_t bestElements = 0;
  for (std::size_t t = 0; t < mappings.size(); ++t) {
    auto &mapping = mappings[t];
    auto numElements =
        std::accumulate(mapping.begin(), mapping.end(), std::size_t(0),
                        [](std::size_t total, const poplar::Interval &i) {
                          return total + i.size();
                        });
    if (numElements > bestElements) {
      best = t;
      bestElements = numElements;
    }
  }
  if (best == mappings.size())
    throw poputil::poplibs_error("Invalid tensor mapping");
  return best;
}

void invertAndTranspose(poplar::Graph &graph, const poplar::Tensor &dst,
                        const poplar::Tensor &a,
                        poplar::program::Sequence &prog,
                        const poplar::DebugNameAndId &dnai,
                        CholeskyParams &params) {
  std::size_t n = a.dim(1);

  if (n == 0)
    throw poputil::poplibs_error("received an empty matrix.");

  auto g = a.dim(0);

  const auto numWorkers = graph.getTarget().getNumWorkerContexts();
  const auto numTiles = graph.getTarget().getNumTiles();

  const std::string name = "triangularInverse_n=" + std::to_string(n);
  auto cs = graph.addComputeSet({dnai, name});
  for (std::size_t i = 0; i < g; ++i) {
    auto aSlice = a.slice({i, 0, 0}, {i + 1, n, n}).reshape({n * n});
    for (size_t tile = 0, col = 0; col < n; col += numWorkers, ++tile) {
      auto end_col = col + numWorkers > n ? n : col + numWorkers;
      auto oSlice = dst.slice({i, col, 0}, {i + 1, end_col, n})
                        .reshape({(end_col - col) * n});
      auto v = graph.addVertex(
          cs,
          poputil::templateVertex("poplin::TriangularInverseWithTranspose",
                                  a.elementType()),
          {{"in", aSlice}, {"out", oSlice}});
      graph.setInitialValue(v["dim"], n);
      graph.setInitialValue(v["colBegin"], col);
      graph.setInitialValue(v["colCnt"], end_col - col);

      graph.setTileMapping(v, (tile + bestTile(graph, oSlice)) % numTiles);
    }
  }
  prog.add(poplar::program::Execute(cs));
}

poplar::Tensor invertAndTranspose(poplar::Graph &graph, const poplar::Tensor &a,
                                  poplar::program::Sequence &prog,
                                  const poplar::DebugNameAndId &dnai,
                                  CholeskyParams &params) {
  auto inv = graph.clone(a);
  invertAndTranspose(graph, inv, a, prog, dnai, params);
  return inv;
}

void factorise(poplar::Graph &graph, const poplar::Tensor &a,
               poplar::program::Sequence &prog,
               const poplar::DebugNameAndId &dnai, CholeskyParams &params) {
  std::size_t g = a.dim(0), n = a.dim(1);

  if (n == 0)
    return;

  auto cs = graph.addComputeSet({dnai, "cholesky"});
  for (std::size_t i = 0; i < g; ++i) {
    auto aSlice = a.slice({i, 0, 0}, {i + 1, n, n});
    // Only lower triangular Cholesky decomposition could be optimised with
    // vectorised dot-product.
    auto v = graph.addVertex(
        cs, poputil::templateVertex("poplin::Cholesky", a.elementType(), true),
        {{"in", aSlice.reshape({n * n})}});
    graph.setInitialValue(v["dim"], n);

    graph.setTileMapping(v, bestTile(graph, aSlice));
  }
  prog.add(poplar::program::Execute(cs));
}

std::pair<poplar::Tensor, poplar::Tensor> rearrangeTensorsForGroupedMatMul(
    poplar::Graph &graph, poplar::Tensor &lhs, poplar::Tensor &rhs,
    poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai,
    CholeskyParams &params) {
  auto rearrangedLHS = createMatMulGroupedInputLHS(
      graph, lhs.elementType(), rhs.elementType(), lhs.shape(), rhs.shape(),
      {dnai, "createLHS"}, params.options, params.cache);
  prog.add(poplar::program::Copy(lhs, rearrangedLHS));

  auto rearrangedRHS = createMatMulGroupedInputRHS(
      graph, lhs.elementType(), rhs.elementType(), lhs.shape(), rhs.shape(),
      {dnai, "createRHS"}, params.options, params.cache);
  prog.add(poplar::program::Copy(rhs, rearrangedRHS));

  return {rearrangedLHS, rearrangedRHS};
}

poplar::Tensor groupedMatMulWithRearrange(poplar::Graph &graph,
                                          poplar::Tensor &lhs,
                                          poplar::Tensor &rhs,
                                          poplar::program::Sequence &prog,
                                          const poplar::DebugNameAndId &dnai,
                                          CholeskyParams &params) {
  auto [rearrangedLHS, rearrangedRHS] = rearrangeTensorsForGroupedMatMul(
      graph, lhs, rhs, prog, {dnai, "createGroupedMatmulInputs"}, params);

  return poplin::matMulGrouped(graph, rearrangedLHS, rearrangedRHS, prog,
                               lhs.elementType(), {dnai, "matmul"},
                               params.options, params.cache);
}

void groupedMatMulAccWithRearrange(poplar::Graph &graph, poplar::Tensor &acc,
                                   float scale, poplar::Tensor &lhs,
                                   poplar::Tensor &rhs,
                                   poplar::program::Sequence &prog,
                                   const poplar::DebugNameAndId &dnai,
                                   CholeskyParams &params) {
  auto [rearrangedLHS, rearrangedRHS] = rearrangeTensorsForGroupedMatMul(
      graph, lhs, rhs, prog, {dnai, "createGroupedMatmulInputs"}, params);

  poplin::matMulGroupedAcc(graph, acc, scale, rearrangedLHS, rearrangedRHS,
                           prog, {dnai, "matmul"}, params.options,
                           params.cache);
}

void factoriseBlocked(poplar::Graph &graph, const poplar::Tensor &A,
                      poplar::program::Sequence &prog,
                      const poplar::DebugNameAndId &dnai,
                      CholeskyParams &params) {
  std::size_t nBatches = A.dim(0);
  std::size_t n = A.dim(1);

  if (n == 0)
    return;

  if (n <= params.blockSize) {
    factorise(graph, A, prog, {dnai, "factorise"}, params);
    return;
  }

  auto A11 = A.slice({0, 0, 0}, {nBatches, params.blockSize, params.blockSize});
  factorise(graph, A11, prog, {dnai, "factorise A11"}, params);

  auto inv_t_A11 =
      invertAndTranspose(graph, A11, prog, {dnai, "inverse(A11)"}, params);
  auto A22 = A.slice({0, params.blockSize, params.blockSize}, {nBatches, n, n});

  auto A21 = A.slice({0, params.blockSize, 0}, {nBatches, n, params.blockSize});

  auto A21xA11IT = groupedMatMulWithRearrange(graph, A21, inv_t_A11, prog,
                                              {dnai, "A21*A11IT"}, params);

  prog.add(poplar::program::Copy(A21xA11IT, A21));

  auto A21T = poplin::transposeGroupedMatrix(A21);
  groupedMatMulAccWithRearrange(graph, A22, -1, A21, A21T, prog,
                                {dnai, "A21*A21T"}, params);

  factoriseBlocked(graph, A22, prog, {dnai, "factoriseBlockedRecurse"}, params);
}

void maskOutput(poplar::Graph &graph, const poplar::Tensor &A,
                poplar::program::Sequence &prog,
                const poplar::DebugNameAndId &dnai, CholeskyParams &params) {
  std::size_t nBatches = A.dim(0);
  std::size_t n = A.dim(1);

  for (std::size_t i = 0; i < n; i++) {
    zeroTensor(graph, A.slice({0, i, i + 1}, {nBatches, i + 1, n}), prog);
  }
}

std::vector<std::size_t> groupedShape(const std::vector<std::size_t> &shape) {
  const auto rank = shape.size();
  if (rank < 2)
    throw poputil::poplibs_error(
        "groupedShape: rank of tensor must be greater or equal to 2.");

  std::size_t batches = 1;
  if (rank >= 3) {
    for (std::size_t i = 0; i < rank - 2; ++i) {
      batches *= shape[i];
    }
  }
  return {batches, shape[rank - 2], shape[rank - 1]};
}

void computePrePlanParametersNonBlocked(
    const poplar::Type &type, std::size_t g, std::size_t n,
    const CholeskyOptions &options, std::set<poplin::MatMulParams> &paramSet) {
  // No matmuls at the moment
}

void computePrePlanParametersBlocked(const poplar::Type &type, std::size_t g,
                                     std::size_t n,
                                     const CholeskyOptions &options,
                                     std::set<poplin::MatMulParams> &paramSet) {
  auto b = options.blockSize;
  if (n <= b) {
    computePrePlanParametersNonBlocked(type, g, n, options, paramSet);
    return;
  }

  // factorise A11
  computePrePlanParametersNonBlocked(type, g, n, options, paramSet);

  // A21 * inv(transpose(A11))
  paramSet.insert({type, type, {g, n - b, b}, {g, b, b}});

  // A21 * transpose(A21)
  paramSet.insert({type, type, {g, n - b, b}, {g, b, n - b}});

  // A22
  computePrePlanParametersBlocked(type, g, n - b, options, paramSet);
}

} // anonymous namespace

std::vector<std::pair<MatMulParams, poplar::OptionFlags>>
getCholeskyMatMulPrePlanParameters(const poplar::Type &type,
                                   const std::vector<std::size_t> &shape,
                                   bool lower, poplar::OptionFlags options) {
  POPLIN_TRACEPOINT();

  validateInput(shape);
  auto gshape = groupedShape(shape);

  auto g = gshape[0];
  auto n = gshape[1];

  std::set<poplin::MatMulParams> paramSet;
  CholeskyOptions choleskyOptions(options, n);
  computePrePlanParametersBlocked(type, g, n, choleskyOptions, paramSet);

  std::vector<std::pair<MatMulParams, poplar::OptionFlags>> matmulParams;

  for (auto &param : paramSet) {
    matmulParams.emplace_back(param, choleskyOptions.matmulOptions);
  }

  return matmulParams;
}

unsigned setTileMappingLowerTriangular(poplar::Graph &graph,
                                       const poplar::Tensor &a,
                                       unsigned blockSize, unsigned beginTileId,
                                       unsigned ipuTiles) {
  const unsigned nofBlocks = a.dim(1) / blockSize;
  for (unsigned row = 0; row < nofBlocks; ++row) {
    for (unsigned col = 0; col < row + 1; ++col) {
      const auto blockSlice =
          a.slice({0, row * blockSize, col * blockSize},
                  {a.dim(0), (row + 1) * blockSize, (col + 1) * blockSize});
      graph.setTileMapping(blockSlice, beginTileId);
      beginTileId = (beginTileId + 1) % ipuTiles;
    }
  }
  return beginTileId;
}

unsigned setTileMappingUpperTriangular(poplar::Graph &graph,
                                       const poplar::Tensor &a,
                                       unsigned blockSize,
                                       unsigned blockSizeMultiplier,
                                       unsigned beginTileId,
                                       unsigned tileIdJump, unsigned ipuTiles) {
  for (unsigned col = 0; col < a.dim(2) / blockSize; ++col) {
    const unsigned maxRow =
        col - (col % blockSizeMultiplier) + blockSizeMultiplier;
    for (unsigned row = 0; row < maxRow && row < a.dim(1) / blockSize; ++row) {
      const auto blockSlice =
          a.slice({0, row * blockSize, col * blockSize},
                  {a.dim(0), (row + 1) * blockSize, (col + 1) * blockSize});
      graph.setTileMapping(blockSlice, beginTileId);
      beginTileId = (beginTileId + tileIdJump) % ipuTiles;
    }
  }
  return beginTileId;
}

unsigned setTileMappingRectangle(poplar::Graph &graph, const poplar::Tensor &a,
                                 unsigned blockSize, unsigned beginTileId,
                                 unsigned ipuTiles) {
  const unsigned nofRowBlocks = a.dim(1) / blockSize;
  const unsigned nofColBlocks = a.dim(2) / blockSize;
  for (unsigned row = 0; row < nofRowBlocks; ++row) {
    for (unsigned col = 0; col < nofColBlocks; ++col) {
      const auto blockSlice =
          a.slice({0, row * blockSize, col * blockSize},
                  {a.dim(0), (row + 1) * blockSize, (col + 1) * blockSize});
      graph.setTileMapping(blockSlice, beginTileId);
      beginTileId = (beginTileId + 1) % ipuTiles;
    }
  }
  return beginTileId;
}

void setTileMapping(poplar::Graph &graph, const poplar::Tensor &a,
                    unsigned blockSize, unsigned beginTileId) {
  const unsigned n = a.dim(1);

  const auto numTiles = graph.getTarget().getNumTiles();

  if (n < blockSize * 2)
    setTileMappingRectangle(graph, a, blockSize, beginTileId, numTiles);

  const bool isOddAlligned = (n / blockSize) & 0x1;
  const unsigned blockMultiplier = n % blockSize || isOddAlligned ? 1 : 2;
  unsigned nDiv2RoundedToBlockSize =
      static_cast<unsigned>(floor(n / 2 + blockSize - 1) / blockSize) *
      blockSize;

  const auto a12 = a.slice({0, 0, nDiv2RoundedToBlockSize},
                           {a.dim(0), nDiv2RoundedToBlockSize, n});
  const auto a22 = a.slice(
      {0, nDiv2RoundedToBlockSize, nDiv2RoundedToBlockSize}, {a.dim(0), n, n});

  setTileMappingLowerTriangular(graph, a, blockSize * blockMultiplier,
                                ((beginTileId - 2) % numTiles), numTiles);
  setTileMappingRectangle(graph, a12, blockSize * blockMultiplier, beginTileId,
                          numTiles);
  setTileMappingUpperTriangular(graph, a22, blockSize, blockMultiplier,
                                beginTileId, 1, numTiles);
  if (nDiv2RoundedToBlockSize > blockSize * blockMultiplier) {
    // set mapping for unused part of A11 (upper triangle, over diagonal)
    const auto a11_ut =
        a.slice({0, 0, blockSize * blockMultiplier},
                {a.dim(0), nDiv2RoundedToBlockSize, nDiv2RoundedToBlockSize});
    setTileMappingUpperTriangular(graph, a11_ut, blockSize, blockMultiplier,
                                  numTiles - 1, -1, numTiles);
  }
}

poplar::Tensor
createCholeskyInput(poplar::Graph &graph, const poplar::Type &type,
                    const std::vector<std::size_t> &shape, bool lower,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options, PlanningCache *cache) {
  POPLIN_TRACEPOINT();

  poputil::PoplibsOpDebugInfo di(debugContext);
  auto gShape = groupedShape(shape);

  auto g = gShape[0];
  auto m = gShape[1];
  auto n = gShape[2];

  if (m != n)
    throw poputil::poplibs_error(
        "createCholeskyInput: last two dimensions must be of the same size.");

  CholeskyOptions choleskyOptions(options, n);
  auto blockSize = choleskyOptions.blockSize;

  const auto nb = gccs::ceildiv(n, blockSize);

  const std::vector<std::size_t> blockShape = {g, nb, nb, blockSize, blockSize};
  auto tensor = graph.addVariable(type, blockShape, debugContext);

  tensor = tensor.dimShuffle({0, 1, 3, 2, 4});
  tensor = tensor.reshape({g, nb * blockSize, nb * blockSize});

  const auto startTileId =
      std::min<unsigned>(graph.getTarget().getNumTiles() - 1, 2);
  setTileMapping(graph, tensor, blockSize, startTileId);

  tensor = tensor.slice({0, 0, 0}, {g, n, n});

  if (!lower)
    tensor = tensor.dimShuffle({0, 2, 1});

  return tensor.reshape(shape);
}

void choleskyInPlace(poplar::Graph &graph, const poplar::Tensor &a, bool lower,
                     poplar::program::Sequence &prog,
                     const poplar::DebugContext &debugContext,
                     poplar::OptionFlags options, PlanningCache *cache) {
  POPLIN_TRACEPOINT();

  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(a));

  validateInput(graph, a);

  auto rank = a.rank();
  poplar::Tensor As;
  if (rank > 2)
    As = a.flatten(0, rank - 2);
  else
    As = a.reshape({1, a.dim(rank - 2), a.dim(rank - 1)});

  CholeskyOptions choleskyOptions(options, As.dim(1));

  PlanningCache localCache;
  CholeskyParams params(choleskyOptions, cache ? cache : &localCache);

  const poplar::Tensor &AsEff = lower ? As : poplin::transposeGroupedMatrix(As);
  factoriseBlocked(graph, AsEff, prog, {di, "factoriseBlockedDivideBy2Top"},
                   params);

  maskOutput(graph, AsEff, prog, {di, "maskOutput"}, params);
}

poplar::Tensor cholesky(poplar::Graph &graph, const poplar::Tensor &a,
                        bool lower, poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext,
                        poplar::OptionFlags options, PlanningCache *cache) {
  POPLIN_TRACEPOINT();

  auto a2 = createCholeskyInput(graph, a.elementType(), a.shape(), lower,
                                debugContext, options, cache);
  prog.add(poplar::program::Copy(a, a2));

  choleskyInPlace(graph, a2, lower, prog, debugContext, options, cache);

  return a2;
}

} // namespace poplin
