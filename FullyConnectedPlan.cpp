#include "FullyConnectedPlan.hpp"

#include "DeviceInfo.hpp"
#include "PerformanceEstimation.hpp"
#include <limits>

using namespace fc;

static unsigned
estimatePartitionCost(const DeviceInfo &deviceInfo, bool isFloat,
                      unsigned numRows, unsigned numCols, unsigned tilesPerRow,
                      unsigned tilesPerColumn) {
  auto numTiles = tilesPerRow * tilesPerColumn;
  auto numVertices = numRows * tilesPerRow;
  auto numWorkerContexts = deviceInfo.getNumWorkerContexts();
  auto vertexElements = (numCols + tilesPerRow - 1) / tilesPerRow;
  auto partialSumsPerTile = (numRows + tilesPerColumn - 1) / tilesPerColumn;
  auto vertexRuntime =
      getFullyConnectedPartialCycleEstimate(isFloat, vertexElements,
                                            deviceInfo.dataPathWidth);
  auto verticesPerWorker = (numVertices + numTiles * numWorkerContexts - 1) /
                           (numTiles * numWorkerContexts);
  auto computeCycles = vertexRuntime * verticesPerWorker * numWorkerContexts;
  auto exchangeBytesPerCycle = deviceInfo.getIPUExchangeBandwidth();
  auto inputBytes = vertexElements * (isFloat ? 4 : 2);
  auto partialSumBytes = partialSumsPerTile * 4;
  auto exchangeCycles =
      (inputBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (partialSumBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle;
  return computeCycles + exchangeCycles;
}

Plan
fc::createPlan(const DeviceInfo &deviceInfo,
               const std::string &dType, unsigned numRows,
               unsigned numCols) {
  // In theory a 2D tiling of the matrix across IPUs could decrease the
  // amount of communication. Unfortunately it introduces a new causal layer.
  // It turns out that, at least up to 16 IPUs, it is better to always keep
  // all row elements on the same IPU to avoid the need for an extra sync.
  bool isFloat = dType == "float";
  const auto numIPUs = deviceInfo.getNumIPUs();
  unsigned rowsPerIPU = (numRows + numIPUs - 1) / numIPUs;
  const auto tilesPerIPU = deviceInfo.getTilesPerIPU();
  unsigned lowestCost = std::numeric_limits<unsigned>::max();
  unsigned bestTilesPerColumn, bestTilesPerRow;
  for (unsigned tilesPerRow = 1; tilesPerRow <= tilesPerIPU; ++tilesPerRow) {
    unsigned tilesPerColumn = tilesPerIPU / tilesPerRow;
    const auto cost = estimatePartitionCost(deviceInfo, isFloat, rowsPerIPU,
                                            numCols, tilesPerRow,
                                            tilesPerColumn);
    if (cost < lowestCost) {
      lowestCost = cost;
      bestTilesPerColumn = tilesPerColumn;
      bestTilesPerRow = tilesPerRow;
    }
  }
  return Plan(Partition(bestTilesPerColumn, bestTilesPerRow));
}
