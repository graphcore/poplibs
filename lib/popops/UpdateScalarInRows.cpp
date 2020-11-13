// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "SparseUtils.hpp"

#include <popops/UpdateScalarInRows.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplibs_support/Algorithms.hpp>
#include <popops/codelets.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include <algorithm>
#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;

namespace {

constexpr int BATCH_DIM = 0;
constexpr int CLASSES_DIM = 1;

std::string getCodeletName(Type type) {
  return "popops::UpdateIntervalDEC<" + type.toString() + ">";
}

std::string get2DCodeletName(Type type) {
  return "popops::UpdateIntervalsDEC<" + type.toString() + ">";
}

std::string getColumnsCodeletName(Type type) {
  return "popops::UpdateColumnsDEC<" + type.toString() + ">";
}

void create1DVertex(Graph &graph, ComputeSet &computeSet, Type type,
                    const Interval &interval, const Tensor &flatParams,
                    const Tensor &indices, unsigned int width, int tile) {
  if (interval.size() == 0)
    return;

  std::vector<Interval> bounds;
  std::size_t startRow = getBounds(interval, width, bounds);

  std::vector<int> rowsStart;
  int sum = 0;
  for (const auto &bound : bounds) {
    rowsStart.push_back(sum);
    sum += bound.size();
  }

  Interval rows(startRow, startRow + bounds.size());
  const Tensor &vertexParams = flatParams.slice(interval);
  const Tensor &vertexIndices = indices.slice(rows);

  graph.setTileMapping(vertexIndices, tile);

  VertexRef v =
      graph.addVertex(computeSet, getCodeletName(type),
                      {{"params", vertexParams}, {"indices", vertexIndices}});
  graph.setInitialValue(v["rowsStart"], rowsStart);
  graph.setInitialValue(v["paramsWidth"], width);
  graph.setInitialValue(v["firstStartCol"], bounds.front().begin());
  graph.setInitialValue(v["lastEndCol"], bounds.back().end());
  graph.setInitialValue(v["rowCount"], vertexIndices.numElements());
  graph.setTileMapping(v, tile);
}

void create2DVertex(Graph &graph, ComputeSet &computeSet, Type type,
                    const std::vector<Interval> &intervals,
                    const Tensor &flatParams, const Tensor &indices, int width,
                    int tile) {
  std::vector<Tensor> vertexParams;
  std::vector<Tensor> vertexIndices;
  std::vector<unsigned> rowsStart;
  std::vector<unsigned> firstStartCol;
  std::vector<unsigned> lastEndCol;
  std::vector<unsigned> rowCounts;

  for (const Interval &interval : intervals) {
    if (interval.size() == 0) {
      continue;
    }

    vertexParams.push_back(flatParams.slice(interval));
    std::vector<Interval> bounds;
    std::size_t startRow = getBounds(interval, width, bounds);

    int sum = 0;
    for (const auto &bound : bounds) {
      rowsStart.push_back(sum);
      sum += bound.size();
    }

    Interval rows(startRow, startRow + bounds.size());
    const Tensor &intervalIndices = indices.slice(rows);
    graph.setTileMapping(intervalIndices, tile);
    vertexIndices.push_back(intervalIndices);

    firstStartCol.push_back(bounds.front().begin());
    lastEndCol.push_back(bounds.back().end());
    rowCounts.push_back(bounds.size());
  }

  if (vertexParams.empty()) {
    return;
  }

  VertexRef v =
      graph.addVertex(computeSet, get2DCodeletName(type),
                      {{"params", vertexParams}, {"indices", vertexIndices}});
  graph.setInitialValue(v["rowsStart"], rowsStart);
  graph.setInitialValue(v["paramsWidth"], width);
  graph.setInitialValue(v["firstStartCol"], firstStartCol);
  graph.setInitialValue(v["lastEndCol"], lastEndCol);
  graph.setInitialValue(v["rowCounts"], rowCounts);
  graph.setTileMapping(v, tile);
}

// Create a vertex specialized to work with intervals laid-out in column
// format. This is typical for tensor used also by matrix multiplication.
void createColumnsVertex(Graph &graph, ComputeSet &computeSet, Type type,
                         const Regions &regions, const Tensor &flatParams,
                         const Tensor &indices, int width, int tile) {
  if (regions.empty()) {
    return;
  }

  std::vector<Tensor> vertexParams;
  std::vector<Tensor> vertexIndices;
  std::vector<unsigned> columnWidths;
  std::vector<unsigned> regionHeights;
  std::vector<unsigned> regionWidths;
  std::vector<unsigned> firstColumns;

  vertexParams.reserve(regions.size());
  vertexIndices.reserve(regions.size());
  regionHeights.reserve(regions.size());
  regionWidths.reserve(regions.size());
  firstColumns.reserve(regions.size());

  for (const Region &region : regions) {
    vertexParams.push_back(concat(flatParams.slices(region)));

    int regionHeight = 0, regionWidth = 0;
    std::tie(regionHeight, regionWidth) =
        getRegionBounds(region, width, columnWidths);

    std::size_t firstRow = getSingleRowIntervalRowIndex(region.front(), width);
    Tensor indicesSlice = indices.slice({firstRow, firstRow + regionHeight});
    graph.setTileMapping(indicesSlice, tile);
    vertexIndices.push_back(std::move(indicesSlice));

    regionHeights.push_back(regionHeight);
    regionWidths.push_back(regionWidth);
    int firstColumn =
        getSingleRowIntervalColumnIndices(region.front(), width).begin();
    firstColumns.push_back(firstColumn);
  }

  VertexRef v =
      graph.addVertex(computeSet, getColumnsCodeletName(type),
                      {{"params", vertexParams}, {"indices", vertexIndices}});
  graph.setInitialValue(v["columnWidths"], columnWidths);
  graph.setInitialValue(v["regionHeights"], regionHeights);
  graph.setInitialValue(v["regionWidths"], regionWidths);
  graph.setInitialValue(v["firstColumns"], firstColumns);
  graph.setInitialValue(v["paramsWidth"], width);
  graph.setTileMapping(v, tile);
}

} // namespace
/*
 * TODO: T12980 This function should receive in input a poplar::Expr
 * that describes the computation to perform when updating the matrix.
 * The current implementation subtracts 1.f from the original scalar.
 * This is because this is the operation is the building block of the tensorflow
 * operation operation: tf.nn.sparse_softmax_cross_entropy_with_logits
 * https://www.tensorflow.org/api_docs/python/tf/nn/
 * sparse_softmax_cross_entropy_with_logits
 */
void popops::updateScalarInRows(Graph &graph, const Tensor &params,
                                const Tensor &indices, Sequence &program,
                                const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, indices));

  // Check preconditions.
  expect(indices.rank() == 1, "indices must have rank 1");
  expect(indices.elementType() == UNSIGNED_INT,
         "indices must have type UNSIGNED_INT");
  expect(params.rank() == 2, "params must have rank 2");
  Type elementType = params.elementType();
  expect(elementType == FLOAT || elementType == HALF,
         "params must have type FLOAT or HALF");
  std::size_t width = params.dim(CLASSES_DIM);
  std::size_t height = params.dim(BATCH_DIM);
  expect(height == indices.dim(0),
         "length of indices must match height of params");

  auto mapping = graph.getTileMapping(params);

  const Tensor flatParams = params.flatten();
  ComputeSet computeSet = graph.addComputeSet({di, "UpdateScalarInRows"});
  const auto target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(elementType);

  // For each tile.
  for (std::size_t tile = 0; tile < mapping.size(); ++tile) {
    const std::vector<Interval> &tileIntervals = mapping[tile];
    // If the tile has no intervals mapped, we have nothing to do.
    if (tileIntervals.empty()) {
      continue;
    }

    Regions tileRegions =
        graph.getSortedContiguousRegions(params, tileIntervals);

    // If all the intervals span a single row, use a version that uses
    // bookmarking metadata. This is meant primarily for layouts which have
    // some kind of 2D structure on-tile (e.g. matmul layouts).
    if (checkRegionShapes(tileRegions, width)) {
      std::vector<int> regionsPerVertex =
          balancedPartition(tileRegions.size(), target.getNumWorkerContexts());

      // TODO: T12971 Add a 1D codelet to handle the case where we have a single
      // region.
      int counter = 0;
      for (unsigned i = 0; i < regionsPerVertex.size(); ++i) {
        int regionsPerThisVertex = regionsPerVertex[i];

        if (regionsPerThisVertex == 0) {
          continue;
        }

        Regions vertexRegions(tileRegions.begin() + counter,
                              tileRegions.begin() + counter +
                                  regionsPerThisVertex);

        createColumnsVertex(graph, computeSet, elementType, vertexRegions,
                            flatParams, indices, width, tile);
        counter += regionsPerThisVertex;
      }
      continue;
    }

    const auto &regionsPerVertex =
        splitRegionsBetweenWorkers(target, tileRegions, 2 * vectorWidth);
    for (const auto &thisWorkerRegions : regionsPerVertex) {
      std::vector<Interval> flatThisWorkerRegions;
      for (unsigned i = 0; i < thisWorkerRegions.size(); i++) {
        for (unsigned j = 0; j < thisWorkerRegions[i].size(); j++) {
          flatThisWorkerRegions.push_back(thisWorkerRegions[i][j]);
        }
      }
      if (flatThisWorkerRegions.empty()) {
        continue;
      }
      if (flatThisWorkerRegions.size() == 1) {
        create1DVertex(graph, computeSet, elementType,
                       flatThisWorkerRegions.front(), flatParams, indices,
                       width, tile);
      } else {
        create2DVertex(graph, computeSet, elementType, flatThisWorkerRegions,
                       flatParams, indices, width, tile);
      }
    }
  }

  program.add(Execute(computeSet, {di}));
}
