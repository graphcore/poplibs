#include "MatMulPlan.hpp"

#include "poplar/Graph.hpp"
#include "PerformanceEstimation.hpp"
#include <limits>

using namespace poplin;

static unsigned
estimateCost(const poplar::DeviceInfo &deviceInfo, bool isFloat,
             unsigned numRows, unsigned numCols, unsigned tilesPerRow,
             unsigned tilesPerColumn) {
  auto numTiles = tilesPerRow * tilesPerColumn;
  auto numVertices = numRows * tilesPerRow;
  auto numWorkerContexts = deviceInfo.numWorkerContexts;
  auto vertexElements = (numCols + tilesPerRow - 1) / tilesPerRow;
  auto partialSumsPerTile = (numRows + tilesPerColumn - 1) / tilesPerColumn;
  auto vertexRuntime =
      getMatMul1PartialCycleEstimate(isFloat, vertexElements,
                                     deviceInfo.dataPathWidth);
  auto verticesPerWorker = (numVertices + numTiles * numWorkerContexts - 1) /
                           (numTiles * numWorkerContexts);
  auto computeCycles = vertexRuntime * verticesPerWorker * numWorkerContexts;
  auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;
  auto inputBytes = vertexElements * (isFloat ? 4 : 2);
  auto partialSumBytes = partialSumsPerTile * 4;
  auto exchangeCycles =
      (inputBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (partialSumBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle;
  return computeCycles + exchangeCycles;
}

static unsigned
estimateTransposedCost(const poplar::DeviceInfo &deviceInfo, bool isFloat,
                       unsigned numRows, unsigned numCols,
                       unsigned tilesPerRow, unsigned tilesPerColumn) {
  auto vectorWidth = isFloat ? deviceInfo.getFloatVectorWidth() :
                               deviceInfo.getHalfVectorWidth();
  auto numWorkerContexts = deviceInfo.numWorkerContexts;
  auto vertexElements = (numRows + tilesPerColumn - 1) / tilesPerColumn;
  auto partialSumsPerTile = (numCols + tilesPerRow - 1) / tilesPerRow;
  auto verticesPerTile = (partialSumsPerTile + vectorWidth - 1) / vectorWidth;
  auto vertexRuntime = getMatMul2CycleEstimate(vertexElements);
  auto verticesPerWorker = (verticesPerTile + numWorkerContexts - 1) /
                           numWorkerContexts;
  auto computeCycles = vertexRuntime * verticesPerWorker * numWorkerContexts;
  auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;
  auto inputBytes = vertexElements * (isFloat ? 4 : 2);
  auto partialSumBytes = partialSumsPerTile * 4;
  auto exchangeCycles =
      (inputBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (partialSumBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle;
  return computeCycles + exchangeCycles;
}

Plan
poplin::getPlan(const poplar::Graph &graph,
                std::string dType_,
                std::vector<std::size_t> aShape_,
                std::vector<std::size_t> bShape_,
                MatMulOptions options_) {
  PlanningCache::Params params(std::move(dType_), std::move(aShape_),
                               std::move(bShape_), std::move(options_));
  if (params.options.cache) {
    auto &plans = params.options.cache->plans;
    auto match = plans.find(params);
    if (match != plans.end())
      return *match->second;
  }
  // In theory a 2D tiling of the matrix across IPUs could decrease the
  // amount of communication. Unfortunately it introduces a new causal layer.
  // It turns out that, at least up to 16 IPUs, it is better to always keep
  // all row elements on the same IPU to avoid the need for an extra sync.

  const auto numCols = params.aShape[1];
  const auto numRows = params.aShape[0];
  bool usedInTranspose = !params.options.leftHandArgUsedInTranspose;
  bool isFloat = params.dType == "float";
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
  const auto maxRowsPerIPU = (numRows + numIPUs - 1) / numIPUs;

  unsigned lowestCost = std::numeric_limits<unsigned>::max();
  unsigned bestTilesPerColumn, bestTilesPerRow;
  for (unsigned tilesPerRow = 1; tilesPerRow <= tilesPerIPU; ++tilesPerRow) {
    unsigned tilesPerColumn = tilesPerIPU / tilesPerRow;
    auto cost = estimateCost(deviceInfo, isFloat, maxRowsPerIPU,
                                numCols, tilesPerRow, tilesPerColumn);
    if (usedInTranspose) {
      cost += estimateTransposedCost(deviceInfo, isFloat, maxRowsPerIPU,
                              numCols, tilesPerRow, tilesPerColumn);
    }
    if (cost < lowestCost) {
      lowestCost = cost;
      bestTilesPerColumn = tilesPerColumn;
      bestTilesPerRow = tilesPerRow;
    }
  }
  if (params.options.cache) {
    auto &plans = params.options.cache->plans;
    auto plan =
      std::unique_ptr<Plan>(
        new Plan(Plan::Partition(bestTilesPerColumn, bestTilesPerRow))
       );
    auto res = plans.emplace(std::make_pair(params, std::move(plan)));
    return *res.first->second;
  }
  auto plan = Plan(Plan::Partition(bestTilesPerColumn, bestTilesPerRow));
  return plan;
}

poplin::PlanningCache::PlanningCache() {}
poplin::PlanningCache::~PlanningCache() {}
