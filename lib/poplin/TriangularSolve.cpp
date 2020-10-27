// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplin/TriangularSolve.hpp"
#include "poplin/MatMul.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Encoding.hpp"
#include "popops/Expr.hpp"
#include "popops/Pad.hpp"
#include "popops/Zero.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/GraphFunction.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include <sstream>

namespace pe = popops::expr;

namespace poplin {
namespace {

struct SolveParams {
  std::size_t k;
  bool lower;
  bool unitDiagonal;
  std::size_t blockSize;
  poplar::OptionFlags options;
  matmul::PlanningCache *cache;
  poplar::Tensor x;
  mutable std::unique_ptr<poputil::graphfn::VoidFunction> solver;

  SolveParams(std::size_t k, bool lower, bool unitDiagonal,
              std::size_t blockSize, poplar::OptionFlags options,
              matmul::PlanningCache *cache)
      : k(k), lower(lower), unitDiagonal(unitDiagonal), blockSize(blockSize),
        options(options), cache(cache) {}
};

template <typename S, typename T>
S &operator<<(S &stream, const std::vector<T> &value) {
  stream << "{";
  for (std::size_t i = 0; i < value.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << value[i];
  }
  stream << "}";
  return stream;
}

poplar::Tensor triangle(poplar::Graph &graph, const poplar::Tensor &a,
                        const SolveParams &params,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "") {
  if (a.rank() != 3) {
    throw poputil::poplibs_error("triangle: tensor must have rank of 3");
  }
  auto n = a.dim(2);
  auto m = a.dim(1);
  if (n != m) {
    throw poputil::poplibs_error(
        "triangularMask: tensor must have shape of [..., N, N].");
  }

  auto iotaOut = graph.addVariable(poplar::UNSIGNED_INT, {n},
                                   "iotaOut" + std::to_string(n));
  poputil::mapTensorLinearly(graph, iotaOut);
  popops::iota(graph, iotaOut, 0u, prog,
               debugPrefix + "/triangleIota" + std::to_string(n));

  auto totalBatches = a.dim(0);
  auto indices0 =
      iotaOut.reshape({1, 1, n}).broadcast(n, 1).broadcast(totalBatches, 0);
  auto indices1 =
      iotaOut.reshape({1, n, 1}).broadcast(n, 2).broadcast(totalBatches, 0);

  pe::Any triangularMask =
      pe::Select(pe::_1, pe::Const(0.0f),
                 params.lower ? pe::Any(pe::Lte(pe::_2, pe::_3))
                              : pe::Any(pe::Gte(pe::_2, pe::_3)));
  if (params.unitDiagonal) {
    triangularMask =
        pe::Select(pe::Const(1.0f), triangularMask, pe::Equal(pe::_2, pe::_3));
  }

  return popops::map(graph, triangularMask, {a, indices0, indices1}, prog,
                     debugPrefix + "/mapTriangularMask");
}

poplar::Tensor diag(const poplar::Tensor &a) {
  auto ndims = a.rank();
  if (ndims < 2) {
    throw poputil::poplibs_error(
        "diag: tensor must have rank greater or equal to 2.");
  }

  auto n = a.dim(ndims - 1);
  auto m = a.dim(ndims - 2);

  return a.reshapePartial(ndims - 2, ndims, {n * m})
      .subSample(n + 1, ndims - 2); // reshape reduced ndims by 1
}

void solve(poplar::Graph &graph, const poplar::Tensor &a,
           const poplar::Tensor &b, std::size_t bLeftPos, std::size_t bTopPos,
           const SolveParams &params, poplar::program::Sequence &prog,
           const std::string &debugPrefix) {
  if (a.rank() != 3) {
    throw poputil::poplibs_error("solve: invalid rank of tensor A");
  }
  if (b.rank() != 3) {
    throw poputil::poplibs_error("solve: invalid rank of tensor B");
  }

  auto an = a.dim(2);
  auto bn = b.dim(2);
  auto totalBatches = a.dim(0);
  if (totalBatches != b.dim(0)) {
    throw poputil::poplibs_error("solve: batch dimensions must match");
  }

  if (an > params.blockSize) {
    auto an2 = an >> 1;
    auto bn2 = bn >> 1;
    auto bMiddlePos = bLeftPos + bn2;

    auto a11 = a.slice({0, 0, 0}, {totalBatches, an2, an2});
    auto a12 = a.slice({0, 0, an2}, {totalBatches, an2, an});
    auto a21 = a.slice({0, an2, 0}, {totalBatches, an, an2});
    auto a22 = a.slice({0, an2, an2}, {totalBatches, an, an});

    auto b11 = b.slice({0, 0, 0}, {totalBatches, an2, bn2});
    auto b12 = b.slice({0, 0, bn2}, {totalBatches, an2, bn});
    auto b21 = b.slice({0, an2, 0}, {totalBatches, an, bn2});
    auto b22 = b.slice({0, an2, bn2}, {totalBatches, an, bn});

    if (params.lower) {
      // Lower solver:
      // A11       X11 X12   B11 B12
      // A21 A22   X21 X22   B21 B22

      // Recursively splitting it:
      // A11 * X11 = B11
      // A11 * X12 = B12
      // A21 * X11 + A22 * X21 = B21
      // A21 * X12 + A22 * X22 = B22

      solve(graph, a11, b11, bLeftPos, bTopPos, params, prog,
            debugPrefix + "/X11");

      if (bMiddlePos < params.k) {
        solve(graph, a11, b12, bMiddlePos, bTopPos, params, prog,
              debugPrefix + "/X12");
      }

      {
        auto x11 =
            params.x.slice({0, bTopPos, bLeftPos},
                           {totalBatches, bTopPos + an2, bLeftPos + an2});
        auto a21x11 = matMulGrouped(graph, a21, x11, prog, a.elementType(),
                                    debugPrefix + "/A21*X11", params.options,
                                    params.cache);
        b21 = popops::sub(graph, b21, a21x11, prog, "/B21-A21*X11");
      }
      solve(graph, a22, b21, bLeftPos, bTopPos + an2, params, prog,
            debugPrefix + "/X21");

      if (bMiddlePos < params.k) {
        auto x12 = params.x.slice({0, bTopPos, bLeftPos + an2},
                                  {totalBatches, bTopPos + an2, bLeftPos + an});
        auto a21x12 = matMulGrouped(graph, a21, x12, prog, a.elementType(),
                                    debugPrefix + "/A21*X12", params.options,
                                    params.cache);
        b22 = popops::sub(graph, b22, a21x12, prog, "/B22-A21*X12");
        solve(graph, a22, b22, bMiddlePos, bTopPos + an2, params, prog,
              debugPrefix + "/X22");
      }
    } else {
      // Upper solver:
      // A11 A12   X11 X12  B11 B12
      //     A22   X21 X22  B21 B22

      // Recursively splitting it:
      // A22 * X21 = B21
      // A22 * X22 = B22
      // A11 * X11 + A12 * X21 = B11
      // A11 * X12 + A12 * X22 = B12

      solve(graph, a22, b21, bLeftPos, bTopPos + an2, params, prog,
            debugPrefix + "/X21");
      if (bMiddlePos < params.k) {
        solve(graph, a22, b22, bMiddlePos, bTopPos + an2, params, prog,
              debugPrefix + "/X22");
      }

      {
        auto x21 = params.x.slice({0, bTopPos + an2, bLeftPos},
                                  {totalBatches, bTopPos + an, bLeftPos + an2});
        auto a12x21 = matMulGrouped(graph, a12, x21, prog, a.elementType(),
                                    debugPrefix + "/A12*X21", params.options,
                                    params.cache);
        b11 = popops::sub(graph, b11, a12x21, prog, "B11-A12*X21");
      }
      solve(graph, a11, b11, bLeftPos, bTopPos, params, prog,
            debugPrefix + "/X11");

      if (bMiddlePos < params.k) {
        auto x22 = params.x.slice({0, bTopPos + an2, bLeftPos + an2},
                                  {totalBatches, bTopPos + an, bLeftPos + an});
        auto a12x22 = matMulGrouped(graph, a12, x22, prog, a.elementType(),
                                    debugPrefix + "/A12*X22", params.options,
                                    params.cache);
        b12 = popops::sub(graph, b12, a12x22, prog, "/B12-A12*X22");
        solve(graph, a11, b12, bLeftPos + bn2, bTopPos, params, prog,
              debugPrefix + "/X12");
      }
    }
  } else {
    // direct solver: back/forward substitution
    if (!params.solver) {
      params.solver.reset(new poputil::graphfn::VoidFunction(
          graph,
          {poputil::graphfn::input(a), poputil::graphfn::input(b),
           poputil::graphfn::inout(b)},
          [&graph, an, bn, totalBatches, &params,
           &debugPrefix](std::vector<poplar::Tensor> &args,
                         poplar::program::Sequence &prog) {
            auto a = args.at(0), b = args.at(1), x = args.at(2);
            auto b0 =
                (params.lower ? b.slice({0, 0, 0}, {totalBatches, 1, bn})
                              : b.slice({0, an - 1, 0}, {totalBatches, an, bn}))
                    .reshape({totalBatches, bn});
            auto x0 = (params.lower
                           ? x.slice({0, 0, 0}, {totalBatches, 1, bn})
                           : x.slice({0, an - 1, 0}, {totalBatches, an, bn}));

            prog.add(poplar::program::Copy(b0, x0));

            for (std::size_t idx = 1; idx < an; ++idx) {
              auto row = params.lower ? a.slice({0, idx, 0},
                                                {totalBatches, idx + 1, idx})
                                      : a.slice({0, an - 1 - idx, an - idx},
                                                {totalBatches, an - idx, an});
              auto bValue =
                  params.lower
                      ? b.slice({0, idx, 0}, {totalBatches, idx + 1, bn})
                      : b.slice({0, an - 1 - idx, 0},
                                {totalBatches, an - idx, bn});

              auto xValue =
                  params.lower
                      ? x.slice({0, 0, 0}, {totalBatches, idx, bn})
                      : x.slice({0, an - idx, 0}, {totalBatches, an, bn});

              auto dot = poplin::matMulGrouped(
                  graph, row, xValue, prog, a.elementType(),
                  debugPrefix + "/substituteX" + std::to_string(idx),
                  params.options, params.cache);

              dot = popops::sub(graph, bValue, dot, prog,
                                debugPrefix + "/substituteB" +
                                    std::to_string(idx));

              prog.add(poplar::program::Copy(
                  dot, params.lower
                           ? x.slice({0, idx, 0}, {totalBatches, idx + 1, bn})
                           : x.slice({0, an - 1 - idx, 0},
                                     {totalBatches, an - idx, bn})));
            }
          },
          false));
    }

    auto dst = params.x.slice({0, bTopPos, bLeftPos},
                              {totalBatches, bTopPos + an, bLeftPos + bn});
    std::vector<poplar::Tensor> args{a, b, dst};
    (*params.solver)(args, prog);
  }
}

void validateInput(poplar::Graph &graph, const poplar::Tensor &a) {
  auto ndims = a.rank();
  if (ndims < 2) {
    throw poputil::poplibs_error("tensor A must have rank of 2 or higher.");
  }

  auto n = a.dim(ndims - 1);
  auto m = a.dim(ndims - 2);
  if (n != m) {
    throw poputil::poplibs_error(
        "2 minor dimension of tensor A must have the same size.");
  }
}

} // anonymous namespace

poplar::Tensor triangularMask(poplar::Graph &graph, const poplar::Tensor &a,
                              bool lower, bool unitDiagonal,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  validateInput(graph, a);

  SolveParams params(0, lower, unitDiagonal, 0, {}, nullptr);

  auto batchShape = a.shape();
  batchShape.resize(batchShape.size() - 2);
  auto batchA = a.rank() >= 3 ? a.flatten(0, batchShape.size()) : a.expand({0});

  return triangle(graph, batchA, params, prog, debugPrefix + "/triangularMask")
      .reshape(a.shape());
}

poplar::Tensor triangularSolve(
    poplar::Graph &graph, const poplar::Tensor &a, const poplar::Tensor &b,
    bool leftSide, bool lower, bool unitDiagonal, std::size_t blockSize,
    poplar::program::Sequence &prog, const poplar::DebugContext &debugContext,
    poplar::OptionFlags options, matmul::PlanningCache *cache) {
  const auto debugPrefix = debugContext.getPathName();
  validateInput(graph, a);
  if (blockSize == 0) {
    throw poputil::poplibs_error("blockSize must be greater than zero");
  }

  auto aRank = a.rank();
  auto an = a.dim(aRank - 1);

  auto bRank = b.rank();
  if (bRank < 2) {
    throw poputil::poplibs_error(
        "rank of tensor B must be greater or equal to 2.");
  }

  if (aRank != bRank) {
    throw poputil::poplibs_error("ranks of tensors A and B must match");
  }

  auto bn = b.dim(bRank - 1);
  if (leftSide) {
    auto bm = b.dim(bRank - 2);
    if (bm != an)
      throw poputil::poplibs_error(
          "mismatched shape for triangular solver. For A [..., N, N], B must "
          "have shape of [..., N, K]");
  } else {
    if (bn != an)
      throw poputil::poplibs_error(
          "mismatched shape for triangular solver. For A [..., N, N], B must "
          "have shape of [..., K, N]");
    bn = b.dim(bRank - 2);
  }

  // everything except two minor dims must match
  auto batchShape = a.shape();
  batchShape.resize(batchShape.size() - 2);
  {
    auto batchShapeB = b.shape();
    batchShapeB.resize(batchShapeB.size() - 2);
    if (batchShape != batchShapeB) {
      throw poputil::poplibs_error(
          "major (batch) dimensions A and B must match");
    }
  }

  // if we have any batch dimensions, flatten them into single one
  // if we have tensors of rank 2, expand with singlular batch dimension

  auto batchA = aRank >= 3 ? a.flatten(0, batchShape.size()) : a.expand({0});
  auto batchB = aRank >= 3 ? b.flatten(0, batchShape.size()) : b.expand({0});

  // even though cache == null, matmul could benefit of planning cache,
  // provide local ephemeral cache for the solver only.
  matmul::PlanningCache localCache;
  SolveParams params(bn, lower, unitDiagonal, blockSize, options,
                     cache ? cache : &localCache);

  bool needPadding = an > blockSize || bn > blockSize;
  std::size_t paddedSize = blockSize;
  while (paddedSize < an || paddedSize < bn) {
    paddedSize <<= 1;
  }

  // how many columns/rows required for A along two minor dimensions
  long toPad = paddedSize - an;
  // how many columns/rows required for B along dimension not bound to A
  long toPadB = paddedSize - bn;

  batchA =
      triangle(graph, batchA, params, prog, debugPrefix + "/triangularMask");
  // Ax = B is effectively xAT = BT
  if (!leftSide) {
    batchA = batchA.dimShuffle({0, 2, 1});
    batchB = batchB.dimShuffle({0, 2, 1});
    params.lower = !lower;
  }

  auto totalBatches = batchA.dim(0);

  // do not scale A to have unit diagonal in case of unitDiagonal explicitly
  // passed
  if (!unitDiagonal) {
    auto diagVector = diag(batchA);
    auto diagMatrix = diagVector.expand({2});
    poputil::broadcastToMatch(diagMatrix, batchA.shape());
    batchA =
        popops::div(graph, batchA, diagMatrix, prog, debugPrefix + "/scaleA");
    auto diagVectorB =
        diagVector.reshape({totalBatches, an, 1}).broadcast(bn, 2);
    poputil::broadcastToMatch(diagVectorB, batchB.shape());
    batchB =
        popops::div(graph, batchB, diagVectorB, prog, debugPrefix + "/scaleB");
  }

  if (needPadding) {
    // create two writeable padding regions and fill them with 1 on main
    // diagonal between [N and paddedSize]

    // align paddings as
    //  A  A  A  PR PR
    //  A  A  A  PR PR
    //  A  A  A  PR PR
    //  PB PB PB PB PB
    //  PB PB PB PB PB

    auto paddingRight = graph.addVariable(
        a.elementType(), {totalBatches, an, (std::size_t)toPad});
    poputil::mapTensorLinearly(graph, paddingRight);
    popops::zero(graph, paddingRight, prog, debugPrefix + "/rightZeroPad");

    auto paddingBottom = graph.addVariable(
        a.elementType(), {totalBatches, (std::size_t)toPad, paddedSize});
    poputil::mapTensorLinearly(graph, paddingBottom);
    popops::zero(graph, paddingBottom, prog, debugPrefix + "/bottomZeroPad");

    batchA = poplar::concat(batchA, paddingRight, 2);
    batchA = poplar::concat(batchA, paddingBottom, 1);

    auto tOne =
        graph.addConstant(a.elementType(), {1}, 1.0f, debugPrefix + "/const:1");
    graph.setTileMapping(tOne, 0);

    auto paddedDiagA = diag(batchA).slice({0, an}, {totalBatches, paddedSize});
    prog.add(poplar::program::Copy(
        tOne.reshape({1, 1, 1}).broadcast(toPad, 1).broadcast(totalBatches, 0),
        paddedDiagA));

    batchB = popops::pad(graph, batchB, {0, 0, 0}, {0, toPad, toPadB});
  }

  auto x = graph.addVariable(a.elementType(), batchB.shape());
  poputil::mapTensorLinearly(graph, x);
  popops::zero(graph, x, prog);
  params.x = x;

  solve(graph, batchA, batchB, 0, 0, params, prog, debugPrefix + "/solve");

  if (needPadding) {
    // remove padding
    x = popops::pad(graph, x, {0, 0, 0}, {0, -toPad, -toPadB});
  }

  if (!leftSide) {
    // Ax = B is effectively xAT = BT, transpose back
    x = x.dimShuffle({0, 2, 1});
  }

  // restore batch dimensions
  return x.reshape(b.shape());
}

} // namespace poplin
