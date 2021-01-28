// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplin/TriangularSolve.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplin/MatMul.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Expr.hpp"
#include "popops/Pad.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/DebugInfo.hpp"
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
                        const poplar::DebugNameAndId &dnai) {
  if (a.rank() != 3) {
    throw poputil::poplibs_error("triangle: tensor must have rank of 3");
  }
  auto n = a.dim(2);
  if (n != a.dim(1)) {
    throw poputil::poplibs_error(
        "triangularMask: tensor must have shape of [..., N, N].");
  }

  auto totalBatches = a.dim(0);
  auto out = graph.clone(a.elementType(), a, {dnai, "cloneTriangularInput"});
  auto zero =
      graph.addConstant(a.elementType(), {1}, 0).broadcast(totalBatches, 0);
  graph.setTileMapping(zero, 0);
  auto one =
      graph.addConstant(a.elementType(), {1}, 1).broadcast(totalBatches, 0);
  graph.setTileMapping(one, 0);

  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::size_t> begin, end;
    if (params.lower) {
      begin = {0, i, i + 1};
      end = {totalBatches, i + 1, n};
    } else {
      begin = {0, i, 0};
      end = {totalBatches, i + 1, i};
    }
    poplar::Tensor dstZero = out.slice(begin, end);
    poplar::Tensor srcZero =
        zero.reshape({totalBatches, 1, 1}).broadcast(end[2] - begin[2], 2);
    prog.add(poplar::program::Copy(srcZero, dstZero));

    if (params.unitDiagonal) {
      prog.add(poplar::program::Copy(
          one, out.slice({0, i, i}, {totalBatches, i + 1, i + 1})
                   .reshape({totalBatches})));
    }

    if (params.lower) {
      begin = {0, i, 0};
      end = {totalBatches, i + 1, params.unitDiagonal ? i : i + 1};
    } else {
      begin = {0, i, params.unitDiagonal ? i + 1 : i};
      end = {totalBatches, i + 1, n};
    }

    prog.add(poplar::program::Copy(a.slice(begin, end), out.slice(begin, end)));
  }

  return out;
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
           const poplar::DebugNameAndId &dnai) {
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
    auto an2 = (an + 1) >> 1;
    auto bn2 = (bn + 1) >> 1;
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

      solve(graph, a11, b11, bLeftPos, bTopPos, params, prog, {dnai, "X11"});

      if (bMiddlePos < params.k) {
        solve(graph, a11, b12, bMiddlePos, bTopPos, params, prog,
              {dnai, "X12"});
      }

      {
        auto x11 =
            params.x.slice({0, bTopPos, bLeftPos},
                           {totalBatches, bTopPos + an2, bLeftPos + bn2});
        auto a21x11 =
            matMulGrouped(graph, a21, x11, prog, a.elementType(),
                          {dnai, "A21*X11"}, params.options, params.cache);
        b21 = popops::sub(graph, b21, a21x11, prog, {dnai, "B21-A21*X11"});
      }
      solve(graph, a22, b21, bLeftPos, bTopPos + an2, params, prog,
            {dnai, "X21"});

      if (bMiddlePos < params.k) {
        auto x12 = params.x.slice({0, bTopPos, bLeftPos + bn2},
                                  {totalBatches, bTopPos + an2, bLeftPos + bn});
        auto a21x12 =
            matMulGrouped(graph, a21, x12, prog, a.elementType(),
                          {dnai, "A21*X12"}, params.options, params.cache);
        b22 = popops::sub(graph, b22, a21x12, prog, {dnai, "B22-A21*X12"});
        solve(graph, a22, b22, bMiddlePos, bTopPos + an2, params, prog,
              {dnai, "X22"});
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
            {dnai, "X21"});
      if (bMiddlePos < params.k) {
        solve(graph, a22, b22, bMiddlePos, bTopPos + an2, params, prog,
              {dnai, "X22"});
      }

      {
        auto x21 = params.x.slice({0, bTopPos + an2, bLeftPos},
                                  {totalBatches, bTopPos + an, bLeftPos + bn2});
        auto a12x21 =
            matMulGrouped(graph, a12, x21, prog, a.elementType(),
                          {dnai, "A12*X21"}, params.options, params.cache);
        b11 = popops::sub(graph, b11, a12x21, prog, {dnai, "B11-A12*X21"});
      }
      solve(graph, a11, b11, bLeftPos, bTopPos, params, prog, {dnai, "X11"});

      if (bMiddlePos < params.k) {
        auto x22 = params.x.slice({0, bTopPos + an2, bLeftPos + bn2},
                                  {totalBatches, bTopPos + an, bLeftPos + bn});
        auto a12x22 =
            matMulGrouped(graph, a12, x22, prog, a.elementType(),
                          {dnai, "A12*X22"}, params.options, params.cache);
        b12 = popops::sub(graph, b12, a12x22, prog, {dnai, "B12-A12*X22"});
        solve(graph, a11, b12, bLeftPos + bn2, bTopPos, params, prog,
              {dnai, "X12"});
      }
    }
  } else {
    // direct solver: back/forward substitution
    auto solverSize = params.k > params.blockSize ? params.blockSize : params.k;
    if (!params.solver) {
      // Create nicely layed out tensors for function input, so clone won't
      // expand padding.
      poplar::Tensor inputA =
          graph.addVariable(a.elementType(), a.shape(), "inputA");
      poputil::mapTensorLinearly(graph, inputA);
      poplar::Tensor inputB = graph.addVariable(
          b.elementType(), {b.dim(0), b.dim(1), solverSize}, "inputB");
      poputil::mapTensorLinearly(graph, inputB);
      params.solver.reset(new poputil::graphfn::VoidFunction(
          graph,
          {poputil::graphfn::input(inputA), poputil::graphfn::input(inputB),
           poputil::graphfn::inout(inputB)}, // X has the same shape as B
          [&graph, an, bn, totalBatches, &params,
           &dnai](std::vector<poplar::Tensor> &args,
                  poplar::program::Sequence &prog) {
            auto a = args.at(0), b = args.at(1), x = args.at(2);
            auto b0 =
                (params.lower ? b.slice({0, 0, 0}, {totalBatches, 1, bn})
                              : b.slice({0, an - 1, 0}, {totalBatches, an, bn}))
                    .reshape({totalBatches, bn});
            auto x0 = (params.lower
                           ? x.slice({0, 0, 0}, {totalBatches, 1, bn})
                           : x.slice({0, an - 1, 0}, {totalBatches, an, bn}));

            prog.add(poplar::program::Copy(b0, x0, false, {dnai}));

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
                  {dnai, "substituteX" + std::to_string(idx)}, params.options,
                  params.cache);

              dot = popops::sub(graph, bValue, dot, prog,
                                {dnai, "substituteB" + std::to_string(idx)});

              prog.add(poplar::program::Copy(
                  dot,
                  params.lower
                      ? x.slice({0, idx, 0}, {totalBatches, idx + 1, bn})
                      : x.slice({0, an - 1 - idx, 0},
                                {totalBatches, an - idx, bn}),
                  false, {dnai}));
            }
          },
          false));
    }

    auto dst = params.x.slice({0, bTopPos, bLeftPos},
                              {totalBatches, bTopPos + an, bLeftPos + bn});

    std::ptrdiff_t toPad = solverSize - bn;
    std::vector<poplar::Tensor> args{a, b, dst};
    if (toPad != 0) {
      args[1] = popops::pad(graph, args[1], {0, 0, 0}, {0, 0, toPad});
      poplar::Tensor solverPadding = graph.addVariable(
          dst.elementType(), {totalBatches, solverSize, std::size_t(toPad)});
      poputil::mapTensorLinearly(graph, solverPadding);
      args[2] = poplar::concat(args[2], solverPadding, 2);
    }
    (*params.solver)(args, prog);
  }
}

void validateInput(const std::vector<std::size_t> &shape) {
  const auto rank = shape.size();
  if (rank < 2) {
    throw poputil::poplibs_error("tensor A must have rank of 2 or higher.");
  }

  const auto n = shape[rank - 1];
  const auto m = shape[rank - 2];
  if (n != m) {
    throw poputil::poplibs_error(
        "2 minor dimension of tensor A must have the same size.");
  }
}

std::tuple<std::size_t, std::size_t>
validateInputs(const std::vector<std::size_t> &aShape,
               const std::vector<std::size_t> &bShape, bool leftSide) {
  validateInput(aShape);

  const auto aRank = aShape.size();
  const auto an = aShape[aRank - 1];

  const auto bRank = bShape.size();
  if (bRank < 2) {
    throw poputil::poplibs_error(
        "rank of tensor B must be greater or equal to 2.");
  }

  if (aRank != bRank) {
    throw poputil::poplibs_error("ranks of tensors A and B must match");
  }

  auto bn = bShape[bRank - 1];
  if (leftSide) {
    const auto bm = bShape[bRank - 2];
    if (bm != an)
      throw poputil::poplibs_error(
          "mismatched shape for triangular solver. For A [..., N, N], B must "
          "have shape of [..., N, K]");
  } else {
    if (bn != an)
      throw poputil::poplibs_error(
          "mismatched shape for triangular solver. For A [..., N, N], B must "
          "have shape of [..., K, N]");
    bn = bShape[bRank - 2];
  }

  // everything except two minor dims must match
  auto batchShape = aShape;
  batchShape.resize(batchShape.size() - 2);
  {
    auto batchShapeB = bShape;
    batchShapeB.resize(batchShapeB.size() - 2);
    if (batchShape != batchShapeB) {
      throw poputil::poplibs_error(
          "major (batch) dimensions A and B must match");
    }
  }

  return std::make_tuple(an, bn);
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

poplar::Tensor createInput(poplar::Graph &graph, poplar::Type type,
                           const std::vector<std::size_t> &shape, bool leftSide,
                           std::size_t blockSize,
                           const poplar::DebugContext &debugContext) {
  const auto g = shape[0];
  const auto n = shape[leftSide ? 1 : 2];
  const auto m = shape[leftSide ? 2 : 1];
  if (n <= blockSize && m <= blockSize) {
    blockSize = 1; // No blocks
  }

  const auto nb = poplibs_support::ceildiv(n, blockSize);
  const auto mb = poplibs_support::ceildiv(m, blockSize);

  // Allocating "blockwise layout", instead of
  //   B1 B1 B2 B2
  //   B1 B1 B2 B2
  //   B3 B3 B4 B4
  //   B3 B3 B4 B4
  // Layout block elements as:
  //   B1 B1 B1 B1
  //   B2 B2 B2 B2
  //   B3 B3 B3 B3
  //   B4 B4 B4 B4
  // It will keep block elements closer to each other,
  // so they can share the same tile

  const std::vector<std::size_t> blockShape = {g, nb, mb, blockSize, blockSize};
  auto tensor = graph.addVariable(type, blockShape, debugContext);

  poputil::mapTensorLinearly(graph, tensor);

  tensor = tensor.dimShuffle({0, 1, 3, 2, 4});
  tensor = tensor.reshape({g, nb * blockSize, mb * blockSize});
  tensor = tensor.slice({0, 0, 0}, {g, n, m});

  if (!leftSide)
    tensor = tensor.dimShuffle({0, 2, 1});

  return tensor;
}

poplar::Tensor createTriangularSolveOutput(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &xShape, bool leftSide,
    std::size_t blockSize, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, matmul::PlanningCache *cache) {
  auto xGrouped = groupedShape(xShape);

  auto tensor = createInput(graph, inputType, xGrouped, false, blockSize,
                            {debugContext, "X"});

  return tensor.reshape(xShape);
}

} // anonymous namespace

poplar::Tensor createTriangularSolveInputLHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide,
    std::size_t blockSize, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, matmul::PlanningCache *cache) {
  validateInputs(aShape, bShape, leftSide);
  auto aGrouped = groupedShape(aShape);

  auto tensor = createInput(graph, inputType, aGrouped, leftSide, blockSize,
                            {debugContext, "A"});

  return tensor.reshape(aShape);
}

poplar::Tensor createTriangularSolveInputRHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide,
    std::size_t blockSize, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, matmul::PlanningCache *cache) {
  validateInputs(aShape, bShape, leftSide);
  auto bGrouped = groupedShape(bShape);

  if (leftSide) {
    std::swap(bGrouped[1], bGrouped[2]);
  }

  auto tensor = graph.addVariable(outputType, bGrouped, debugContext);
  poputil::mapTensorLinearly(graph, tensor);

  if (leftSide) {
    tensor = tensor.dimShuffle({0, 2, 1});
  }

  return tensor.reshape(bShape);
}

poplar::Tensor triangularMask(poplar::Graph &graph, const poplar::Tensor &a,
                              bool lower, bool unitDiagonal,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(a, lower, unitDiagonal));

  validateInput(a.shape());

  SolveParams params(0, lower, unitDiagonal, 0, {}, nullptr);

  auto batchShape = a.shape();
  batchShape.resize(batchShape.size() - 2);
  auto batchA = a.rank() >= 3 ? a.flatten(0, batchShape.size()) : a.expand({0});

  auto output = triangle(graph, batchA, params, prog, {di, "triangularMask"})
                    .reshape(a.shape());
  di.addOutput(output);
  return output;
}

poplar::Tensor triangularSolve(
    poplar::Graph &graph, const poplar::Tensor &a, const poplar::Tensor &b,
    bool leftSide, bool lower, bool unitDiagonal, std::size_t blockSize,
    poplar::program::Sequence &prog, const poplar::DebugContext &debugContext,
    poplar::OptionFlags options, matmul::PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(a, b, leftSide, lower, unitDiagonal, blockSize, options, cache));

  if (blockSize == 0) {
    throw poputil::poplibs_error("blockSize must be greater than zero");
  }

  const auto [an, bn] = validateInputs(a.shape(), b.shape(), leftSide);
  const auto aRank = a.rank();

  // everything except two minor dims must match
  auto batchShape = a.shape();
  batchShape.resize(batchShape.size() - 2);

  // if we have any batch dimensions, flatten them into single one
  // if we have tensors of rank 2, expand with singular batch dimension

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

  batchA = triangle(graph, batchA, params, prog, {di, "triangularMask"});
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
    auto diagVectorSrc = diag(batchA);
    auto diagVector = graph.clone(diagVectorSrc, {di, "diagClone"});
    prog.add(poplar::program::Copy(diagVectorSrc, diagVector));
    auto diagMatrix = diagVector.expand({2});
    poputil::broadcastToMatch(diagMatrix, batchA.shape());
    // batchA returned from triangular() is a clone.
    popops::divInPlace(graph, batchA, diagMatrix, prog, {di, "scaleA"});
    auto diagVectorB =
        diagVector.reshape({totalBatches, an, 1}).broadcast(bn, 2);
    poputil::broadcastToMatch(diagVectorB, batchB.shape());
    batchB = popops::div(graph, batchB, diagVectorB, prog, {di, "scaleB"});
  }

  auto zero = graph.addConstant(a.elementType(), {1}, 0)
                  .broadcast(totalBatches, 0)
                  .reshape({totalBatches, 1, 1});
  graph.setTileMapping(zero, 0);

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
        a.elementType(), {totalBatches, an, (std::size_t)toPad}, {di});
    poputil::mapTensorLinearly(graph, paddingRight);
    prog.add(poplar::program::Copy(zero.broadcast(an, 1).broadcast(toPad, 2),
                                   paddingRight, false, {di, "rightZeroPad"}));

    auto paddingBottom = graph.addVariable(
        a.elementType(), {totalBatches, (std::size_t)toPad, paddedSize}, {di});
    poputil::mapTensorLinearly(graph, paddingBottom);
    prog.add(
        poplar::program::Copy(zero.broadcast(toPad, 1).broadcast(paddedSize, 2),
                              paddingBottom, false, {di, "bottomZeroPad"}));

    batchA = poplar::concat(batchA, paddingRight, 2);
    batchA = poplar::concat(batchA, paddingBottom, 1);

    auto tOne = graph.addConstant(a.elementType(), {1}, 1.0f, {di, "const:1"});
    graph.setTileMapping(tOne, 0);

    auto paddedDiagA = diag(batchA).slice({0, an}, {totalBatches, paddedSize});
    prog.add(poplar::program::Copy(
        tOne.reshape({1, 1, 1}).broadcast(toPad, 1).broadcast(totalBatches, 0),
        paddedDiagA, false, {di}));

    batchB = popops::pad(graph, batchB, {0, 0, 0}, {0, toPad, 0});
  }

  poplar::Tensor x = createTriangularSolveOutput(
      graph, a.elementType(), b.elementType(), batchA.shape(), batchB.shape(),
      leftSide, blockSize, debugContext, options, cache);

  params.x = x;

  solve(graph, batchA, batchB, 0, 0, params, prog, {di, "solve"});

  if (needPadding) {
    // remove padding
    x = popops::pad(graph, x, {0, 0, 0}, {0, -toPad, 0});
  }

  if (!leftSide) {
    // Ax = B is effectively xAT = BT, transpose back
    x = x.dimShuffle({0, 2, 1});
  }

  // restore batch dimensions
  auto output = x.reshape(b.shape());
  di.addOutput(output);
  return output;
}

std::vector<std::pair<MatMulParams, poplar::OptionFlags>>
getTriangularSolveMatMulPrePlanParameters(
    const poplar::Type &inputType, const poplar::Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide, bool lower,
    std::size_t blockSize, poplar::OptionFlags opts) {
  std::vector<std::pair<poplin::MatMulParams, poplar::OptionFlags>> matmuls;

  const auto [an, bn] = validateInputs(aShape, bShape, leftSide);
  bool blocked = an > blockSize || bn > blockSize;

  const auto aGrouped = groupedShape(aShape);

  const auto g = aGrouped[0];

  // Find padded size of A.
  std::size_t paddedSize = blockSize;
  while (paddedSize < an || paddedSize < bn) {
    paddedSize <<= 1;
  }

  std::set<poplin::MatMulParams> paramSet;
  // Preplan half-size matmuls until size is greater than blockSize.
  while (paddedSize > blockSize) {
    paddedSize >>= 1;
    MatMulParams params{inputType,
                        inputType, // A*X
                        {g, paddedSize, paddedSize},
                        {g, paddedSize, paddedSize}};
    paramSet.insert(params);
    if (paddedSize <= blockSize && bn % blockSize) { // leaf unaligned block
      MatMulParams params{inputType,
                          inputType, // A*X
                          {g, paddedSize, paddedSize},
                          {g, paddedSize, bn % blockSize}};
      paramSet.insert(params);
    }
  }

  // Substitution dot products
  std::size_t dots = blocked ? blockSize : an;
  for (std::size_t k = 1; k < dots; ++k) {
    MatMulParams params{inputType,
                        inputType, // A*X
                        {g, 1, k},
                        {g, k, bn > blockSize ? blockSize : bn}};
    paramSet.insert(params);
  }

  for (auto &params : paramSet) {
    matmuls.emplace_back(params, opts);
  }

  return matmuls;
}

} // namespace poplin
