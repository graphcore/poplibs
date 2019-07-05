#include "SparseUtils.hpp"

#include <popops/SelectScalarFromRows.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;

namespace {

constexpr int BATCH_DIM = 0;
constexpr int CLASSES_DIM = 1;

std::string getCodeletName(Type type) {
  return "popops::SelectFromInterval<" + type.toString() + ">";
}

std::string get2DCodeletName(Type type) {
  return "popops::SelectFromIntervals<" + type.toString() + ">";
}

std::string getColumnsCodeletName(Type type) {
  return "popops::SelectFromRowsInColumns<" + type.toString() + ">";
}

void create1DVertex(Graph &graph, ComputeSet &computeSet, Type type,
                    const Interval &interval, const Tensor &flatParams,
                    const Tensor &indices,
                    std::vector<std::vector<Tensor>> &partials,
                    unsigned int width, int tile) {
  if (interval.size() == 0) {
    return;
  }

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
  Tensor output = graph.addVariable(flatParams.elementType(), {bounds.size()});
  graph.setTileMapping(output, tile);
  for (unsigned r = 0; r < bounds.size(); ++r) {
    partials[r + startRow].push_back(output.slice(r, r + 1));
  }

  graph.setTileMapping(vertexIndices, tile);

  VertexRef v = graph.addVertex(computeSet, getCodeletName(type),
                                {{"params", vertexParams},
                                 {"indices", vertexIndices},
                                 {"output", output}});
  graph.setInitialValue(v["rowsStart"], rowsStart);
  graph.setInitialValue(v["paramsWidth"], width);
  graph.setInitialValue(v["firstStartCol"], bounds.front().begin());
  graph.setInitialValue(v["lastEndCol"], bounds.back().end());
  graph.setInitialValue(v["rowCount"], vertexIndices.numElements());
  graph.setTileMapping(v, tile);
}

void create2DVertex(Graph &graph, ComputeSet &computeSet, Type type,
                    const std::vector<Interval> &intervals,
                    const Tensor &flatParams, const Tensor &indices,
                    std::vector<std::vector<Tensor>> &partials, int width,
                    int tile) {
  std::vector<Tensor> vertexParams;
  std::vector<Tensor> vertexIndices;
  std::vector<Tensor> output;
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

    Tensor currentOutput =
        graph.addVariable(flatParams.elementType(), {bounds.size()});
    graph.setTileMapping(currentOutput, tile);
    for (unsigned r = 0; r < bounds.size(); ++r) {
      partials[r + startRow].push_back(currentOutput.slice(r, r + 1));
    }

    firstStartCol.push_back(bounds.front().begin());
    lastEndCol.push_back(bounds.back().end());
    rowCounts.push_back(bounds.size());
    output.push_back(currentOutput);
  }

  if (vertexParams.empty()) {
    return;
  }

  VertexRef v = graph.addVertex(computeSet, get2DCodeletName(type),
                                {{"params", vertexParams},
                                 {"indices", vertexIndices},
                                 {"output", output}});
  graph.setInitialValue(v["rowsStart"], rowsStart);
  graph.setInitialValue(v["paramsWidth"], width);
  graph.setInitialValue(v["firstStartCol"], firstStartCol);
  graph.setInitialValue(v["lastEndCol"], lastEndCol);
  graph.setInitialValue(v["rowCounts"], rowCounts);
  graph.setTileMapping(v, tile);
}

// Create a vertex specialized to work with intervals layed out in column
// format. This is typical for tensor used also by matrix multiplication.
void createColumnsVertex(Graph &graph, ComputeSet &computeSet, Type type,
                         const Regions &regions, const Tensor &flatParams,
                         const Tensor &indices,
                         std::vector<std::vector<Tensor>> &partials, int width,
                         int tile) {
  if (regions.empty()) {
    return;
  }

  std::vector<Tensor> vertexParams;
  std::vector<Tensor> vertexIndices;
  std::vector<Tensor> output;
  std::vector<unsigned> columnWidths;
  std::vector<unsigned> regionHeights;
  std::vector<unsigned> regionWidths;
  std::vector<unsigned> firstColumns;

  for (const Region &region : regions) {
    vertexParams.push_back(concat(flatParams.slices(region)));

    unsigned regionHeight = 0, regionWidth = 0;
    std::tie(regionHeight, regionWidth) =
        getRegionBounds(region, width, columnWidths);

    std::size_t firstRow = getSingleRowIntervalRowIndex(region.front(), width);
    Tensor indicesSlice = indices.slice({firstRow, firstRow + regionHeight});
    graph.setTileMapping(indicesSlice, tile);
    vertexIndices.push_back(std::move(indicesSlice));

    regionHeights.push_back(regionHeight);
    regionWidths.push_back(regionWidth);
    int firstColumn = getSingleRowIntervalColumnIndices(region.front(),
                                                        width).begin();
    firstColumns.push_back(firstColumn);

    Tensor currentOutput = graph.addVariable(
        flatParams.elementType(), {static_cast<std::size_t>(regionHeight)});
    graph.setTileMapping(currentOutput, tile);
    for (unsigned r = 0; r < regionHeight; ++r) {
      partials[r + firstRow].push_back(currentOutput.slice(r, r + 1));
    }
    output.push_back(currentOutput);
  }
  VertexRef v = graph.addVertex(computeSet, getColumnsCodeletName(type),
                                {{"params", vertexParams},
                                 {"indices", vertexIndices},
                                 {"output", output}});
  graph.setInitialValue(v["columnWidths"], columnWidths);
  graph.setInitialValue(v["regionHeights"], regionHeights);
  graph.setInitialValue(v["regionWidths"], regionWidths);
  graph.setInitialValue(v["firstColumns"], firstColumns);
  graph.setInitialValue(v["paramsWidth"], width);
  graph.setTileMapping(v, tile);
}

} // namespace

Tensor popops::selectScalarFromRows(Graph &graph, const Tensor &params,
                                    const Tensor &indices, Sequence &program,
                                    const std::string &debugPrefix) {
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
  ComputeSet computeSet =
      graph.addComputeSet(debugPrefix + "/SelectScalarFromRows");
  const auto target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(elementType);

  // Data structure to store the partial result from the selection from the
  // params matrix. For each row in the params matrix we are going to have a
  // list of partial results. It is possible that different rows have a
  // different number of partial results. For each row only one partial should
  // be != 0.0
  std::vector<std::vector<Tensor>> partials(height);

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
    if (checkRegionShapes(tileRegions, width)){
      std::vector<int> regionsPerVertex = balancedPartition(
          tileRegions.size(), target.getNumWorkerContexts());

      // TODO: almagni
      // Add a 1D codelet to handle the case where we have a single region.
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
                            flatParams, indices, partials, width, tile);
        counter += regionsPerThisVertex;
      }

      continue;
    }
    const auto &regionsPerVertex = splitRegionsBetweenWorkers(
        target, tileRegions, 2 * vectorWidth);
    for (const auto &thisWorkerRegions : regionsPerVertex) {
      std::vector<Interval> flatThisWorkerRegions;
      for(unsigned i = 0; i < thisWorkerRegions.size(); i++) {
        for(unsigned j = 0; j< thisWorkerRegions[i].size(); j++) {
          flatThisWorkerRegions.push_back(thisWorkerRegions[i][j]);
        }
      }
      if (flatThisWorkerRegions.empty()) {
        continue;
      }
      if (flatThisWorkerRegions.size() == 1) {
        create1DVertex(graph, computeSet, elementType,
                       flatThisWorkerRegions.front(),
                       flatParams, indices, partials, width, tile);
      } else {
        create2DVertex(graph, computeSet, elementType, flatThisWorkerRegions,
                       flatParams, indices, partials, width, tile);
      }
    }
  }

  program.add(Execute(computeSet));

  // Now that we have the set of partial results we reduce over the 'columns' so
  // to get a 1D tensor.
  // If all the vectors in tensors have the same size we can perform a single
  // 2D->1D reduction.
  bool allSameWidth = std::all_of(partials.begin(), partials.end(),
                                  [&](const std::vector<Tensor> &is) {
                                    return is.size() == partials.front().size();
                                  });
  if (allSameWidth) {
    std::vector<Tensor> rows(height);
    for (std::size_t row = 0; row < partials.size(); ++row) {
      rows[row] = concat(partials[row]).expand({1});
    }
    Tensor toReduce = concat(rows, 1);

    Tensor output = reduce(graph, toReduce, elementType, {0}, Operation::ADD,
                           program, debugPrefix);
    return output;
  } else {
    // If the vectors in partials don't have the same size we perform
    // multiple reductions, one per row of the param input.
    // The version of reduction that is used here allows to run multiple compute
    // sets in parallel.
    std::vector<ComputeSet> css;
    std::vector<Tensor> rowOutputs;
    for (std::size_t row = 0; row < partials.size(); ++row) {
      Tensor toReduce = concat(partials[row]).expand({1});
      rowOutputs.push_back(reduce(graph, toReduce, elementType, {0},
                                  Operation::ADD, css, debugPrefix));
    }
    for (const auto &cs : css) {
      program.add(Execute(cs));
    }

    return concat(rowOutputs);
  }
}
