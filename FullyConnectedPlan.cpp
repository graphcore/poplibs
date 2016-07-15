#include "FullyConnectedPlan.hpp"

#include "DeviceInfo.hpp"
#include "PerformanceEstimation.hpp"
#include <limits>

using namespace fc;

static unsigned
estimateFwdCost(const DeviceInfo &deviceInfo, bool isFloat,
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

static unsigned
estimateBwdCost(const DeviceInfo &deviceInfo, bool isFloat,
                unsigned numRows, unsigned numCols, unsigned tilesPerRow,
                unsigned tilesPerColumn) {
  auto vectorWidth = isFloat ? deviceInfo.getFloatVectorWidth() :
                               deviceInfo.getHalfVectorWidth();
  auto numWorkerContexts = deviceInfo.getNumWorkerContexts();
  auto vertexElements = (numRows + tilesPerColumn - 1) / tilesPerColumn;
  auto partialSumsPerTile = (numCols + tilesPerRow - 1) / tilesPerRow;
  auto verticesPerTile = (partialSumsPerTile + vectorWidth - 1) / vectorWidth;
  auto vertexRuntime =
      getFullyConnectedBwdCycleEstimate(vertexElements);
  auto verticesPerWorker = (verticesPerTile + numWorkerContexts - 1) /
                           numWorkerContexts;
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
               const std::string &dType,
               unsigned numCols, std::vector<unsigned> outputMapping,
               bool forwardOnly) {
  // In theory a 2D tiling of the matrix across IPUs could decrease the
  // amount of communication. Unfortunately it introduces a new causal layer.
  // It turns out that, at least up to 16 IPUs, it is better to always keep
  // all row elements on the same IPU to avoid the need for an extra sync.
  unsigned maxRowsPerIPU = 0;
  const auto numIPUs = deviceInfo.getNumIPUs();
  const auto tilesPerIPU = deviceInfo.getTilesPerIPU();
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    const auto ipuBeginRow = outputMapping[ipu * tilesPerIPU];
    const auto ipuEndRow = outputMapping[(ipu + 1) * tilesPerIPU];
    const auto rows = ipuEndRow - ipuBeginRow;
    maxRowsPerIPU = std::max(maxRowsPerIPU, rows);
  }

  bool isFloat = dType == "float";
  unsigned lowestCost = std::numeric_limits<unsigned>::max();
  unsigned bestTilesPerColumn, bestTilesPerRow;
  for (unsigned tilesPerRow = 1; tilesPerRow <= tilesPerIPU; ++tilesPerRow) {
    unsigned tilesPerColumn = tilesPerIPU / tilesPerRow;
    auto cost = estimateFwdCost(deviceInfo, isFloat, maxRowsPerIPU,
                                numCols, tilesPerRow, tilesPerColumn);
    if (!forwardOnly) {
      cost += estimateBwdCost(deviceInfo, isFloat, maxRowsPerIPU,
                              numCols, tilesPerRow, tilesPerColumn);
    }
    if (cost < lowestCost) {
      lowestCost = cost;
      bestTilesPerColumn = tilesPerColumn;
      bestTilesPerRow = tilesPerRow;
    }
  }
  return Plan(Partition(bestTilesPerColumn, bestTilesPerRow),
              std::move(outputMapping));
}
