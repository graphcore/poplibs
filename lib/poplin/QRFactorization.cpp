// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poplin/experimental/QRFactorization.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poputil/GraphFunction.hpp"
#include "poputil/OptionParsing.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/Loop.hpp>
#include <popops/Operation.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>

#include <vector>

#define DIV_UP(a, b) (a + b - 1) / b

using namespace poplar;
using namespace program;

/*
 * QR factorization is implemented using Householder reflections.
 * The matrix on the input is transposed, so we operate on rows.
 *
 *  Algorithm:
 *  [m, n] = size(A)
 *  for k = 1:n
 *      rowToProcess = A(k, k:m)
 *      diagonalValue = A(k, k)
 *
 *      ip = InnerProduct(rowToProcess)
 *
 *      # The first element is calculated according to the following formula.
 *      rowToProcess = -rowToProcess
 *      rowToProcess(1) = diagonalValue - sign(diagonalValue) * sqrt(ip)
 *
 *      # We can skip the calculation of the next inner product by updating
 *      # the previously calculated.
 *      ip = ip + ip + 2 * sqrt(ip) * abs(rowToProcess(1));
 *      v = rowToProcess / sqrt(ip)
 *
 *      # Update matrix A and Q
 *      for j = 1:n
 *          dot = dotProduct(v, A(j, k:m))
 *          A(j, k:m) -= v(j) * dot
 *      for j = 1:n
 *          dot = dotProduct(v, Q(j, k:m))
 *          Q(j, k:m) -= v(j) * dot
 */

namespace poplin {

namespace experimental {

namespace {
struct QRParams {
  const size_t m; // rows
  const size_t n; // columns
  size_t vSize;   // size of the vector v
  const unsigned tiles;
  const unsigned workers;
};

void validateInputShape(const std::vector<std::size_t> &shape) {
  const auto rank = shape.size();
  if (rank != 2) {
    throw poputil::poplibs_error("tensor A must have a rank of 2.");
  }
  if (shape[0] < shape[1]) {
    throw poputil::poplibs_error("number of matrix A rows must be greater or "
                                 "equal to the number of columns.");
  }
}

Tensor createInput(Graph &graph, const Type &type,
                   const std::vector<std::size_t> &shape,
                   const DebugContext &debugContext) {
  const auto m = shape[0];
  const auto n = shape[1];
  // Create a matrix transposed in memory.
  auto tensor = graph.addVariable(type, {n, m}, debugContext);

  const auto numWork = n + m; // A columns + Q columns
  const auto numTiles = graph.getTarget().getNumTiles();

  const auto rowsPerTile = numWork / numTiles;
  const auto remainder = numWork % numTiles;
  std::size_t offset = 0;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto startIdx = offset;
    offset += rowsPerTile + (tile < remainder);
    const auto endIdx = std::min(offset, numWork);

    if (startIdx >= numWork)
      break;

    // Distribute the blocks of the rows A on the following tiles.
    if (endIdx <= n) {
      graph.setTileMapping(tensor.slice(startIdx, endIdx, 0), tile);
    } else if (startIdx < n) {
      graph.setTileMapping(tensor.slice(startIdx, n, 0), tile);
    }
  }
  return tensor;
}

Tensor createOutput(Graph &graph, const Type &type, const std::size_t dimension,
                    const std::size_t AColumns,
                    const DebugContext &debugContext) {
  auto tensor = graph.addVariable(type, {dimension, dimension}, debugContext);

  const auto numWork = AColumns + dimension; // A columns + Q columns
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto rowsPerTile = numWork / numTiles;
  const auto remainder = numWork % numTiles;

  std::size_t offset = 0;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto startIdx = offset;
    offset += rowsPerTile + (tile < remainder);
    const auto endIdx = std::min(offset, numWork);

    if (startIdx >= numWork)
      break;

    // Distribute the blocks of the rows Q on the following tiles, taking into
    // account the memory for the matrix A.
    if (startIdx >= AColumns) {
      graph.setTileMapping(
          tensor.slice(startIdx - AColumns, endIdx - AColumns, 0), tile);
    } else if (endIdx > AColumns) {
      graph.setTileMapping(tensor.slice(0, endIdx - AColumns, 0), tile);
    }
  }
  return tensor;
}

void validateParams(const std::vector<std::size_t> &shapeA,
                    const std::vector<std::size_t> &shapeQ, const Type &AType,
                    const Type &QType) {
  const auto rankA = shapeA.size();
  const auto rankQ = shapeQ.size();
  if (rankA != 2 || rankQ != 2) {
    throw poputil::poplibs_error("tensors A and Q must have a rank of 2.");
  }
  if (shapeA[0] < shapeA[1]) {
    throw poputil::poplibs_error("number of matrix A rows must be greater or "
                                 "equal to the number of columns.");
  }
  if (shapeA[0] != shapeQ[0]) {
    throw poputil::poplibs_error(
        "number of rows in matrices A and Q must be equal.");
  }
  if (shapeQ[0] != shapeQ[1]) {
    throw poputil::poplibs_error("matrix Q must be square.");
  }
  if (AType != FLOAT || QType != FLOAT) {
    throw poputil::poplibs_error("only float is supported for this operation.");
  }
}

std::size_t getRowsPerIteration(const OptionFlags &options) {
  constexpr std::size_t defaultValue = 32;
  if (options.find("rowsPerIteration") != options.end())
    return poplibs::parse::asInteger<std::size_t>(
        options.at("rowsPerIteration"));
  return defaultValue;
}

void generateRowSteps(std::vector<std::size_t> &rowSteps, const std::size_t n,
                      const std::size_t rowsPerIteration) {
  for (int i = n; i > 0; i -= rowsPerIteration)
    rowSteps.push_back(i);
  rowSteps.push_back(0);
}

void initAuxVectors(Graph &graph, const Type &type, Tensor &diagonalValueVector,
                    Tensor &rowToProcess, Tensor &v, Tensor &diagonalValue,
                    Tensor &dotProduct, const size_t m, const unsigned workers,
                    const DebugContext &debugContext) {
  diagonalValueVector =
      graph.addVariable(type, {m}, {debugContext, "diagonalValueVector"});
  rowToProcess = graph.addVariable(type, {m}, {debugContext, "rowToProcess"});
  v = graph.addVariable(type, {m}, {debugContext, "v"});
  diagonalValue = graph.addVariable(type, {}, {debugContext, "diagonalValue"});
  dotProduct = graph.addVariable(type, {}, {debugContext, "dotProduct"});

  poputil::mapTensorLinearly(graph, diagonalValueVector, workers, workers);
  poputil::mapTensorLinearly(graph, v, workers, workers);
  poputil::mapTensorLinearly(graph, rowToProcess, workers, workers);
  poputil::mapTensorLinearly(graph, diagonalValue);
  poputil::mapTensorLinearly(graph, dotProduct);
}

// Dynamic slice custom function. This will also extract the value on the
// diagonal needed for further calculations.
void sliceTensor(Graph &graph, Sequence &main, Tensor &A, Tensor &copiedRow,
                 Tensor &diagonalValueVector, Tensor &padding,
                 const QRParams &params) {
  // Distribute 6 per tile to utilize all workers.
  const auto vectorSize = params.workers;
  const std::size_t numWork = params.n + params.m;
  const std::size_t vectors = DIV_UP(numWork, vectorSize);
  const std::size_t blockSize = vectors / params.tiles;
  const std::size_t remainder = vectors % params.tiles;

  std::size_t offset = 0;
  auto rowCopySet = graph.addComputeSet("rowCopySet");
  for (unsigned tile = 0; tile != params.tiles; ++tile) {
    const std::size_t startIdx = offset;
    offset += (blockSize + (tile < remainder)) * vectorSize;
    const std::size_t endIdx = std::min(offset, params.vSize);

    if (startIdx >= params.vSize)
      break;

    VertexRef vtx =
        graph.addVertex(rowCopySet, "poplin::experimental::RowCopy");
    graph.connect(vtx["copiedRow"], copiedRow.slice(startIdx, endIdx));
    graph.connect(vtx["diagonalValueVector"],
                  diagonalValueVector.slice(startIdx, endIdx));
    graph.connect(vtx["A"], A.slice(startIdx, endIdx, 0));
    graph.connect(vtx["padding"], padding);
    graph.connect(vtx["offset"], startIdx);

    graph.setTileMapping(vtx, tile);
  }

  main.add(Execute(rowCopySet));
}

// The function to squared each value in the vector and zero the padding.
void partialSquareElements(Graph &graph, Sequence &main, Tensor &rowToProcess,
                           Tensor &padding, const QRParams &params) {
  // Distribute 6 per tile to utilize all workers.
  const auto vSize = params.vSize;
  const auto vectorSize = params.workers;
  const std::size_t vectors = DIV_UP(params.vSize, vectorSize);
  const std::size_t blockSize = vectors / params.tiles;
  const std::size_t remainder = vectors % params.tiles;

  std::size_t offset = 0;
  auto partialSquareElementsSet =
      graph.addComputeSet("partialSquareElementsSet");
  for (unsigned tile = 0; tile != params.tiles; ++tile) {
    const std::size_t startIdx = offset;
    offset += (blockSize + (tile < remainder)) * vectorSize;
    const std::size_t endIdx = std::min(offset, vSize);

    if (startIdx >= vSize)
      break;

    VertexRef vtx =
        graph.addVertex(partialSquareElementsSet,
                        "poplin::experimental::PartialSquareElements");
    graph.connect(vtx["rowToProcess"], rowToProcess.slice(startIdx, endIdx));
    graph.connect(vtx["offset"], startIdx);
    graph.connect(vtx["padding"], padding);

    graph.setTileMapping(vtx, tile);
  }
  main.add(Execute(partialSquareElementsSet));
}

// Function to reduce diagonal value and dot product components.
void reduceStage(Graph &graph, Sequence &main, Tensor &rowToProcess,
                 Tensor &diagonalValue, Tensor &diagonalValueVector,
                 Tensor &dotProduct, const size_t vSize) {
  auto reduceInput = concat(rowToProcess.reshape({1, vSize}),
                            diagonalValueVector.reshape({1, vSize}), 0);
  auto reduceOutput =
      concat(dotProduct.reshape({1, 1}), diagonalValue.reshape({1, 1}), 0);
  reduceWithOutput(graph, reduceInput, reduceOutput, {1},
                   popops::Operation::ADD, main);
}

// A function that computes the vector v.
void householder(Graph &graph, Sequence &main, Tensor &v, Tensor &rowToProcess,
                 Tensor &diagonalValue, Tensor &diagonalValueVector,
                 Tensor &padding, Tensor &dotProduct, const QRParams &params) {
  const auto vSize = params.vSize;

  // Save row in tensor v.
  main.add(Copy(rowToProcess, v));
  // Squared each value in the vector and zero the padding.
  partialSquareElements(graph, main, rowToProcess, padding, params);
  // Get value from diagonal (computed during dynamic slice) and calculate dot
  // product from the same vector based on the squared values.
  reduceStage(graph, main, rowToProcess, diagonalValue, diagonalValueVector,
              dotProduct, vSize);

  // Calculate the vector v from the formula.
  // Distribute 6 per tile to utilize all workers.
  const auto vectorSize = params.workers;
  const std::size_t vectors = DIV_UP(params.vSize, vectorSize);
  const std::size_t blockSize = vectors / params.tiles;
  const std::size_t remainder = vectors % params.tiles;
  std::size_t offset = 0;

  auto householderSet = graph.addComputeSet("HouseholderSet");
  for (unsigned tile = 0; tile != params.tiles; ++tile) {
    const std::size_t startIdx = offset;
    offset += (blockSize + (tile < remainder)) * vectorSize;
    const std::size_t endIdx = std::min(offset, vSize);

    if (startIdx >= vSize)
      break;

    VertexRef vtx =
        graph.addVertex(householderSet, "poplin::experimental::Householder");
    graph.connect(vtx["diagonalValue"], diagonalValue);
    graph.connect(vtx["v"], v.slice(startIdx, endIdx));
    graph.connect(vtx["dotProduct"], dotProduct);
    graph.connect(vtx["offset"], startIdx);
    graph.connect(vtx["padding"], padding);

    graph.setTileMapping(vtx, tile);
  }

  main.add(Execute(householderSet));
}

// Function to update matrices A and Q with vector v. Rows of both matrices are
// split into tiles.
void update(Graph &graph, Sequence &main, Tensor &A, Tensor &Q, Tensor &padding,
            Tensor &v, const std::size_t row, const QRParams &params) {
  const auto m = params.m;
  const auto n = params.n;
  // We update one column less in each iteration.
  Tensor ASlice = A.slice(row, m, 1);
  Tensor QSlice = Q.slice(row, m, 1);

  const std::size_t numWork = n + m;
  const std::size_t blockSize = numWork / params.tiles;
  const std::size_t remainder = numWork % params.tiles;

  std::size_t offset = 0;
  auto dotProductsSet = graph.addComputeSet("dotProductsSet");
  for (unsigned tile = 0; tile != params.tiles; ++tile) {
    const std::size_t startIdx = offset;
    offset += blockSize + (tile < remainder);
    const std::size_t endIdx = std::min(offset, numWork);

    if (startIdx >= numWork)
      break;

    VertexRef vtx =
        graph.addVertex(dotProductsSet, "poplin::experimental::Update");

    // Distribute row blocks on successive tiles from A to Q.
    if (endIdx <= n) {
      graph.connect(vtx["AQRows"], ASlice.slice(startIdx, endIdx, 0));
    } else if (startIdx >= n) {
      graph.connect(vtx["AQRows"], QSlice.slice(startIdx - n, endIdx - n, 0));
    } else {
      Tensor concatedSlices =
          concat(ASlice.slice(startIdx, n), QSlice.slice(0, endIdx - n));
      graph.connect(vtx["AQRows"], concatedSlices);
    }

    graph.connect(vtx["v"], v);
    graph.connect(vtx["padding"], padding);
    graph.setTileMapping(vtx, tile);
  }

  main.add(Execute(dotProductsSet));
}

// A function for single iteration of factorization.
void QRFactorizationIteration(Graph &graph, Sequence &main, Tensor &A,
                              Tensor &Q, Tensor &padding, Tensor &rowToProcess,
                              Tensor &diagonalValue,
                              Tensor &diagonalValueVector, Tensor &v,
                              Tensor &dotProduct, const std::size_t row,
                              const std::size_t toProcess,
                              const QRParams &params) {
  // The vectors are one element shorter in each iteration.
  // In a poplar loop, upper values are filled with zeros.
  const auto vSize = params.vSize;
  diagonalValueVector = diagonalValueVector.slice(0, vSize);
  rowToProcess = rowToProcess.slice(0, vSize);
  v = v.slice(0, vSize);

  Tensor ASlice =
      A.slice(row, row + toProcess, 0).slice(row, params.m, 1).transpose();
  // Get row from A matrix dynamically.
  sliceTensor(graph, main, ASlice, rowToProcess, diagonalValueVector, padding,
              params);
  // Calculate the value of the vector v.
  householder(graph, main, v, rowToProcess, diagonalValue, diagonalValueVector,
              padding, dotProduct, params);
  // Update the values in Q and A with the vector v.
  update(graph, main, A, Q, padding, v, row, params);
}

} // namespace

std::array<poplar::Tensor, 2>
createQRFactorizationMatrices(poplar::Graph &graph, const poplar::Type &type,
                              const std::size_t m, const std::size_t n,
                              const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  std::vector<std::size_t> shape{m, n};
  validateInputShape(shape);

  auto tensorA = createInput(graph, type, shape, {debugContext, "A"});
  auto tensorQ = createOutput(graph, type, m, n, {debugContext, "Q"});
  return {tensorA.transpose(), tensorQ};
}

void QRFactorization(Graph &graph, Tensor &A, Tensor &Q, Sequence &prog,
                     const DebugContext &debugContext,
                     const OptionFlags &options) {
  validateParams(A.shape(), Q.shape(), A.elementType(), Q.elementType());

  const auto shape = A.shape();
  const auto m = shape[0];
  const auto n = shape[1];
  const auto &target = graph.getTarget();
  const auto tiles = target.getNumTiles();
  const auto workers = target.getNumWorkerContexts();
  QRParams params{m, n, m, tiles, workers};

  // Generate row steps to be processed in each iteration.
  std::vector<std::size_t> rowSteps;
  generateRowSteps(rowSteps, n, getRowsPerIteration(options));
  const auto numPrograms = rowSteps.size();

  // Auxiliary tensors
  Tensor diagonalValueVector, rowToProcess, v, diagonalValue, dotProduct;
  const auto type = A.elementType();
  initAuxVectors(graph, type, diagonalValueVector, rowToProcess, v,
                 diagonalValue, dotProduct, m, workers, debugContext);

  // Transpose matrix to operate on rows instead of columns.
  A = A.transpose();

  // Hybrid poplar/C++ loop.
  Sequence mainLoop;
  for (std::size_t p = 0; p < numPrograms - 1; p++) {
    const int row = n - rowSteps[p];
    const int toProcess = rowSteps[p] - rowSteps[p + 1];
    params.vSize = m - row;

    mainLoop.add(popops::countedLoop(graph, toProcess, [&](Tensor padding) {
      Sequence it;
      QRFactorizationIteration(graph, it, A, Q, padding, rowToProcess,
                               diagonalValue, diagonalValueVector, v,
                               dotProduct, row, toProcess, params);
      return it;
    }));
  }

  A = A.transpose();

  prog.add(mainLoop);
}

} // namespace experimental

} // namespace poplin
