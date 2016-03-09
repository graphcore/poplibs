#include <cmath>
#include <limits>
#include <iostream>

int main()
{
  int numTiles = 1152;
  // Receptive field size.
  int f = 3;
  // Stride.
  int s = 1;
  // Batch size.
  int b = 1;

  int inputDepth = 256;
  int outputWidth = 13;
  int outputHeight = 13;
  int outputDepth = 256;
  int numPartialSums;

  int lowestCost = std::numeric_limits<int>::max();
  int bestTilesPerX, bestTilesPerY, bestTilesPerZ, bestNumPartialSums;

  for (int tilesPerX = 1; tilesPerX < std::sqrt(numTiles); ++tilesPerX) {
    int tilesPerY = tilesPerX;
    for (int tilesPerZ = 1;
         tilesPerZ < numTiles / (tilesPerX * tilesPerY);
         ++tilesPerZ) {
      int numPartialSums = numTiles / (tilesPerX * tilesPerY * tilesPerZ);
      if (numPartialSums == 0)
        continue;
      if (numPartialSums > inputDepth)
        continue;
      const auto usedTiles = tilesPerX * tilesPerY * numPartialSums;
      const auto x = (outputWidth - 1) / tilesPerX + 1;
      const auto y = (outputHeight - 1) / tilesPerY + 1;
      const auto z = (outputDepth - 1) / tilesPerZ + 1;
      const auto d = (inputDepth - 1) / numPartialSums + 1;
      const auto cost = (x * s + f - 1) * (y * s + f - 1) * d * b + f * f * d * z + x * y * z * d * b;
      if (cost < lowestCost) {
        std::cout << "cost=" << cost << "\n";
        lowestCost = cost;
        bestTilesPerX = tilesPerX;
        bestTilesPerY = tilesPerY;
        bestTilesPerZ = tilesPerZ;
        bestNumPartialSums = numPartialSums;
      }
    }
  }
  std::cout << "bestTilesPerX=" << bestTilesPerX << "\n";
  std::cout << "bestTilesPerY=" << bestTilesPerY << "\n";
  std::cout << "bestTilesPerZ=" << bestTilesPerZ << "\n";
  std::cout << "numPartialSums=" << bestNumPartialSums << "\n";
  return 0;
}
