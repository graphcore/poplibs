#include "popnn/Convolution.hpp"
#include <limits>
#include <cassert>
#include "ConvUtil.hpp"
#include "popnn/ActivationMapping.hpp"
#include "VertexTemplates.hpp"
#include "gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "popnn/exceptions.hpp"

#define DEBUG_PRINT 0

using namespace poplar;
using namespace poplar::program;

namespace conv {

struct WgdFwdTrfPartition {
    /**
     * Maximum Number of tiles allocated to kernel transforms
     */
    unsigned maxTilesForKernel;

    /**
     * Maximum number of kernel transform blocks allocated to a tile
     */
    unsigned maxBlocksForKernel;

    /**
     * Maximum number of tiles allocated to data transforms
     */
    unsigned maxTilesForData;

    /**
     * Maximum number of data transform blocks allocated to a tile
     */
    unsigned maxBlocksForData;
};

/**
 * Specifies type of optimisation for forward transforms
 */
enum class TransformMapOption {SEPARATE, JOINT};

/**
 * @brief Mapping of tiles for Data and Kernel transforms to be done concurrently
 *
 * @param kernelUnitCost
 *        cost to compute a kernel transform unit
 *
 * @param dataUnitCost
 *        cost to compute a data transform unit
 *
 * @param kernelUnits
 *        number of kernel transform units
 *
 * @param unitsData
 *        number of data transform units
 *
 * @partition
 *        computed partition
 */
static void wgdTrfTileMapping(
    const unsigned kernelUnitCost,
    const unsigned dataUnitCost,
    const unsigned kernelUnits,
    const unsigned dataUnits,
    const DeviceInfo &deviceInfo,
    const TransformMapOption mapOption,
    WgdFwdTrfPartition &partition) {
  /* This is a simplification to ensure that the default case of rounding
   * both up fits
   */
  unsigned workers = deviceInfo.numWorkerContexts;

  /* assumption here that the two different transform types are not assigned to the same tile.
   * Units are grouped as blocks.
   */
  unsigned kernelBlocks = (kernelUnits + workers - 1)/workers;
  unsigned dataBlocks = (dataUnits + workers - 1)/workers;

  if (mapOption == TransformMapOption::JOINT) {
    const double   maxTiles = deviceInfo.getNumTiles() - 2.0;

    /* tiles assigned to each transform assuming an equal average cost
     * per tile is attributed to each tile. Note that this will result in allocations
     * with fractional cost per tile and is the average we wish to achieve.
     */
    double dataTiles = maxTiles/
                       (1.0 + static_cast<double>(kernelBlocks)/dataBlocks * (static_cast<double>(kernelUnitCost)/dataUnitCost));
    double kernelTiles = maxTiles - dataTiles;

    const double   maxDataBlocks = std::ceil(dataBlocks/dataTiles);
    const double   maxKernelBlocks = std::ceil(kernelBlocks/kernelTiles);

    unsigned bestDataTiles = std::ceil(dataTiles);
    unsigned bestKernelTiles = std::ceil(kernelTiles);

    /* Use unit cost in the computation of block cost as both kernel and data are multiplies
     * by the same factor
     */
    double bestCost = std::max(maxDataBlocks * dataUnitCost, maxKernelBlocks * kernelUnitCost);

    //std::cout << "Candidate 0:" << bestKernelTiles << " tilesData: " << bestDataTiles << " cost: " << bestCost << std::endl;

    /* At least 1 block must be allocated */
    double minDataBlocks = std::max(maxDataBlocks - 1, 1.0);
    double minKernelBlocks = std::max(maxKernelBlocks - 1, 1.0);

    /* Try first candidate: (minDataBlocks) */
    double tilesThisCand = std::ceil(dataBlocks/minDataBlocks);

    if (tilesThisCand < maxTiles) {
      const double tilesData = tilesThisCand;
      const double tilesKernel = maxTiles - tilesData;
      const double cost = std::max(minDataBlocks * dataUnitCost, std::ceil(kernelBlocks/tilesKernel) * kernelUnitCost);

      //std::cout << "Candidate 1:" << tilesKernel << " tilesData: " << tilesData << " cost: " << cost << std::endl;

      /* select candidate if cost is lower */
      if (cost < bestCost) {
        bestCost = cost;
        bestDataTiles = tilesData;
        bestKernelTiles = tilesKernel;

        //std::cout << "Selected candidate 1" << std::endl;
      }
    }

    /* Try second candidate: (minKernelBlocks) */
    tilesThisCand = ceil(kernelBlocks/minKernelBlocks);
    if (tilesThisCand < maxTiles) {
      const double tilesKernel = tilesThisCand;
      const double tilesData = maxTiles - tilesKernel;
      const double cost = std::max(minKernelBlocks * kernelUnitCost, ceil(dataBlocks/tilesData) * dataUnitCost);

      //std::cout << "Candidate 2:" << tilesKernel << " tilesData: " << tilesData << " cost: " << cost << std::endl;

      /* select candidate if cost is lower */
      if (cost < bestCost) {
        bestCost = cost;
        bestDataTiles = tilesData;
        bestKernelTiles = tilesKernel;
        //std::cout << "Selected candidate 2" << std::endl;
      }
    }

    /* Split data and kernel transform across tiles:
     * Not all tiles may have the same number of units
     */
    partition.maxBlocksForData = (dataBlocks + bestDataTiles - 1)/bestDataTiles;
    partition.maxTilesForData = bestDataTiles;

    partition.maxBlocksForKernel = (kernelBlocks + bestKernelTiles - 1)/bestKernelTiles;
    partition.maxTilesForKernel = bestKernelTiles;

    assert( (partition.maxTilesForData + partition.maxTilesForKernel) <= deviceInfo.getNumTiles());
    assert( (partition.maxBlocksForData * partition.maxTilesForData + partition.maxBlocksForKernel * partition.maxTilesForKernel) * deviceInfo.numWorkerContexts >= (kernelUnits + dataUnits));
  } else {
    const unsigned maxTiles = deviceInfo.getNumTiles();
    partition.maxBlocksForData = (dataBlocks + maxTiles - 1)/maxTiles;
    partition.maxBlocksForKernel = (kernelBlocks + maxTiles -1)/maxTiles;
  }
}

/**
 * @brief Number of output patches of size patchSize given
 *        an input length and kernel size
 */
static unsigned getNumPatches(unsigned dim, unsigned kernelSize, unsigned patchSize) {
  assert(patchSize >= kernelSize);
  auto overlap = patchSize - kernelSize + 1;
  return (dim + patchSize - 1)/overlap - 1;
}


/**
 *  @brief Maximum number of valid outputs per patch
 */
static unsigned getNumOutputsPerPatch(unsigned kernelSize, unsigned patchSize) {
  return patchSize - kernelSize + 1;
}

struct WgdMapElem{
  unsigned xin;
  unsigned yin;
  unsigned xout;
  unsigned yout;
  unsigned inGroup;
  unsigned prepadX;
  unsigned prepadY;
  unsigned postpadX;
  unsigned postpadY;
};

struct DataPatchGen {
private:
  unsigned numInGroups;
  unsigned maxX;
  unsigned maxY;
  unsigned kernelX;
  unsigned kernelY;
  unsigned patchX;
  unsigned patchY;

  /* running counts for mapping patches */
  int xin;
  int yin;
  unsigned xout = 0;
  unsigned yout = 0;
  unsigned inGroup = 0;

public:
  DataPatchGen(unsigned x, unsigned y, unsigned inGroups, unsigned patchX, unsigned patchY, unsigned kernelX, unsigned kernelY) : maxX(x), maxY(y), numInGroups(inGroups), patchX(patchX), patchY(patchY), kernelX(kernelX), kernelY(kernelY) {
    assert(kernelX == 3);
    assert(kernelY == 3);
    assert(patchX == 4);
    assert(patchY == 4);
    xin = -static_cast<int>(kernelX - 1)/2;
    yin = -static_cast<int>(kernelY - 1)/2;
  }

  void alloc(WgdMapElem& el) {
    assert(inGroup < numInGroups);
    el.xin = (static_cast<int>(xin) < 0) ? 0 : xin;
    el.prepadX = (static_cast<int>(xin) < 0)? -xin : 0;
    el.postpadX = (xin > static_cast<int>(maxX - patchX)) ? patchX - (maxX - xin) : 0;
    el.yin = (static_cast<int>(yin) < 0) ? 0 : yin;
    el.prepadY = (static_cast<int>(yin) < 0) ? -yin : 0;
    el.postpadY = (yin > static_cast<int>(maxY - patchY)) ? patchY - (maxY - yin) : 0;
    el.inGroup = inGroup;
    el.xout = xout;
    el.yout = yout;

    ++xout;
    if ((xin += static_cast<int>(patchX - kernelX + 1)) > (maxX - 1 -(kernelX - 1)/2)) {
      xin = -static_cast<int>(kernelX - 1)/2;
      xout = 0;
      ++yout;
      if ((yin += static_cast<int>(patchY - kernelY + 1)) > (maxY -1 - (kernelY - 1)/2)) {
        yin = -static_cast<int>(kernelY - 1)/2;
        yout = 0;
        ++inGroup;
      }
    }
  }
};


/**
 * Set-up vertices and their tile mapping for data transforms in a compute set
 */
static void computeDataTransform(Graph &graph,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned xDim, unsigned yDim,
            unsigned patchSizeX, unsigned patchSizeY,
            const std::string dType,
            Tensor in, Tensor dataTf,
            unsigned numDataUnits, const WgdFwdTrfPartition &partition,
            ComputeSet &cs, ComputeSet &zeroCS, unsigned &tile) {

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const unsigned numInpChanGroups = in.dim(0);
  const unsigned numInpChansInGroup = in.dim(3);

  const auto nMaxWorkers = deviceInfo.numWorkerContexts;
  const auto maxTiles = deviceInfo.getNumTiles();

  assert(partition.maxBlocksForData);

  /* Map data transform units to workers on tiles. Each worker is assigned the Maximum
   * number of units until all units are consumed
   */
  DataPatchGen dTfPatch(xDim, yDim, numInpChanGroups, patchSizeX, patchSizeY, kernelSize, kernelSize);
  do {
    unsigned vertex = 0;
    bool zeroTensorCreated = false;
    Tensor zeroVec;

    /* allocated data units to workers */
    for (; vertex < nMaxWorkers&&numDataUnits; vertex++) {
      const auto numUnitsThisVertex = (numDataUnits > partition.maxBlocksForData) ? partition.maxBlocksForData : numDataUnits;

#if DEBUG_PRINT==1
      std::cout << "tile: " << tile << " vertex: " << vertex << "  numUnitsThisVertex: " << numUnitsThisVertex << " numDataUnits :" << numDataUnits << std::endl;
#endif

      /* allocate units to this vertex */
      auto v = graph.addVertex(cs,
                               templateVertex("Wgd3x3DataTransform2x2", dType));

      /* Each unit requires patchSizeX x patchSize vectors */
      graph.setFieldSize(v["dIn"], numUnitsThisVertex * patchSizeX * patchSizeY);
      graph.setFieldSize(v["dTf"], numUnitsThisVertex * patchSizeX * patchSizeY);
      const unsigned numOutChansPerGroup = in.dim(3);
      graph.setInitialValue(v["depth"], numInpChanGroups);
      graph.setInitialValue(v["nPatches"], numUnitsThisVertex);


      graph.setTileMapping(v, tile);

      for (auto unit = 0; unit < numUnitsThisVertex; ++unit) {
        WgdMapElem el;
        dTfPatch.alloc(el);

        auto yin_off = 0;
        for (auto y = 0; y < patchSizeY; ++y) {
          auto addZeroVecY = y < el.prepadY || y >= patchSizeY - el.postpadY;

          for (unsigned x = 0, xin_off=0; x < patchSizeX; ++x  ) {
            auto idx = unit * patchSizeX * patchSizeY + y * patchSizeX + x;

            /* zeros vector if element contains padding */
            auto addZeroVecX = x < el.prepadX || x >= patchSizeX - el.postpadX;

            if ((addZeroVecX || addZeroVecY)  && !zeroTensorCreated)
            {
              zeroVec = graph.addTensor(dType, {numInpChansInGroup}, "zero");
              zeroTensorCreated = true;

              auto v = graph.addVertex(zeroCS, templateVertex("Zero", dType));
              graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
              graph.connect(v["out"], zeroVec);
              graph.setTileMapping(v, tile);
              graph.setTileMapping(zeroVec, tile);
            }

            graph.connect((addZeroVecX || addZeroVecY)?zeroVec:in[el.inGroup][el.yin+yin_off][el.xin+xin_off], v["dIn"][idx]);

            if (!addZeroVecX)
              ++xin_off;

            Tensor dataTfPart = dataTf[el.inGroup][el.yout][el.xout][y][x];
            graph.connect(v["dTf"][idx], dataTfPart);
            graph.setTileMapping(dataTfPart, tile);
          }
          if (!addZeroVecY)
            ++yin_off;
        }
      }
      numDataUnits -= numUnitsThisVertex;
    }

    if (vertex)
      ++tile;

  } while (numDataUnits);
}


/**
 * Set-up vertices and their tile mapping for kernel transforms in a compute set
 */
static void computeKernelTransform(Graph &graph,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned xDim, unsigned yDim,
            unsigned patchSizeX, unsigned patchSizeY,
            const std::string dType,
            Tensor weights, Tensor kernelTf,
            unsigned numKernelUnits, const WgdFwdTrfPartition &partition,
            ComputeSet &cs, unsigned &tile) {

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto nMaxWorkers = deviceInfo.numWorkerContexts;

  const unsigned numOutPartialChanGroups = weights.dim(0);
  const unsigned numInpChanGroups = weights.dim(1);
  const unsigned numOutPartialChansInGroup = weights.dim(4);
  const unsigned numInpChansInGroup = weights.dim(5);

  unsigned inpChanGroup = 0;
  unsigned outChanGroup = 0;
  unsigned outPartialChan = 0;

  do {
    unsigned vertex = 0;

    /* allocate data units to workers */
    for (; vertex < nMaxWorkers && numKernelUnits; ++vertex) {
      const unsigned numUnitsThisVertex = (numKernelUnits > partition.maxBlocksForKernel) ? partition.maxBlocksForKernel : numKernelUnits;

#if DEBUG_PRINT==1
      std::cout << "tile: " << tile << " vertex: " << vertex << "  numUnitsThisVertex: " << numUnitsThisVertex << " numKernelUnits :" << numKernelUnits << std::endl;
#endif

      /* allocate units to this worker */
      auto v = graph.addVertex(cs,
                               templateVertex("Wgd3x3KernelTransform2x2", dType));
      graph.setFieldSize(v["wIn"], numUnitsThisVertex * kernelSize * kernelSize);
      graph.setFieldSize(v["wTf"], numUnitsThisVertex * patchSizeX * patchSizeY);
      graph.setInitialValue(v["depth"], numInpChansInGroup);
      graph.setInitialValue(v["nGroups"], numUnitsThisVertex);


      graph.setTileMapping(v, tile);

      for (auto unit = 0; unit < numUnitsThisVertex; ++unit) {
        for (auto x = 0; x < kernelSize; ++x) {
          for (auto y = 0; y < kernelSize; ++y) {
            graph.connect(weights[outChanGroup][inpChanGroup][y][x][outPartialChan].flatten(), v["wIn"][unit * kernelSize * kernelSize + y * kernelSize + x]);
          }
        }

        for (auto x = 0; x < patchSizeX; ++x) {
          for (auto y = 0; y < patchSizeY; ++y) {
            Tensor wTfPart = kernelTf[outChanGroup][inpChanGroup][y][x][outPartialChan].flatten();
            graph.connect(v["wTf"][unit * patchSizeX * patchSizeY + y * patchSizeX + x], wTfPart);
            graph.setTileMapping(wTfPart, tile);
          }
        }
        if (++outPartialChan == numOutPartialChansInGroup) {
          outPartialChan = 0;
          if (++inpChanGroup == numInpChanGroups) {
            inpChanGroup = 0;
            ++outChanGroup;
            assert(outChanGroup <= numOutPartialChanGroups);
          }
        }
      }
      numKernelUnits -= numUnitsThisVertex;
    }

    if (vertex)
      ++tile;
  } while (numKernelUnits);
}

/**
 * @brief  construct program for joint computation of data and kernel transforms
 */
static Program computeFwdTransforms(Graph &graph,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned xDim, unsigned yDim,
            unsigned patchSizeX, unsigned patchSizeY,
            const std::string dType,
            const std::string layerName,
            Tensor in, Tensor weights, Tensor dataTf, Tensor kernelTf) {

  const TransformMapOption fwdMapOption = TransformMapOption::SEPARATE;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();

  const unsigned numOutPartialChanGroups = weights.dim(0);
  const unsigned numInpChanGroups = weights.dim(1);
  const unsigned numOutPartialChansInGroup = weights.dim(4);
  const unsigned numInpChansInGroup = weights.dim(5);

  unsigned numKernelUnits = numOutPartialChanGroups * numInpChanGroups * numOutPartialChansInGroup;
  const unsigned kernelUnitCost = numInpChansInGroup * 35/4 + 2;

  const auto oneDTransformX = getNumPatches(xDim, kernelSize, patchSizeX);
  const auto oneDTransformY = getNumPatches(yDim, kernelSize, patchSizeY);

  unsigned numDataUnits = numInpChanGroups * oneDTransformX * oneDTransformY;
  const unsigned dataUnitCost = numInpChansInGroup * 56/4 + 15;

  WgdFwdTrfPartition partition;
  wgdTrfTileMapping(kernelUnitCost, dataUnitCost, numKernelUnits, numDataUnits, deviceInfo, fwdMapOption, partition);

  unsigned tile = 0;

  ComputeSet zeroCS = graph.createComputeSet(layerName + ".WgdZeros");
  ComputeSet jointCS = graph.createComputeSet(layerName + ((fwdMapOption == TransformMapOption::JOINT) ? ".kernelAndDataTrf" : ".dataTrf"));

  computeDataTransform(graph, kernelSize, stride, padding,
                       xDim, yDim,
                       patchSizeX, patchSizeY,
                       dType,
                       in, dataTf,
                       numDataUnits, partition, jointCS, zeroCS, tile);

  if (fwdMapOption == TransformMapOption::JOINT)
  {
    computeKernelTransform(graph,
                           kernelSize, stride, padding,
                           xDim, yDim,
                           patchSizeX, patchSizeY,
                           dType,
                           weights, kernelTf,
                           numKernelUnits, partition,
                           jointCS, tile);
    return Sequence(Execute(zeroCS), Execute(jointCS));
  } else {
    ComputeSet kernelCS = graph.createComputeSet(layerName + ".kernelTrf");
    tile = 0;
    computeKernelTransform(graph,
                           kernelSize, stride, padding,
                           xDim, yDim,
                           patchSizeX, patchSizeY,
                           dType,
                           weights, kernelTf,
                           numKernelUnits, partition,
                           kernelCS, tile);
    return Sequence(Execute(zeroCS), Execute(jointCS), Execute(kernelCS));
  }
}


/**
 * @brief  compute product of kernel and data transform and accumulate over partial input dimension
 */
static Program accumulate(Graph &graph,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned xDim, unsigned yDim,
            unsigned patchSizeX, unsigned patchSizeY,
            const std::string dType,
            const std::string layerName,
            Tensor in, Tensor weights, Tensor dataTf, Tensor kernelTf, Tensor accumTf)
{
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();

  const auto nMaxWorkers = deviceInfo.sharedConvWeights?1:deviceInfo.numWorkerContexts;
  const char *baseClass = deviceInfo.sharedConvWeights ? "poplar::SupervisorVertex" : "poplar::Vertex";
  const auto maxTiles = deviceInfo.getNumTiles();

  const auto oneDTransformX = getNumPatches(xDim, kernelSize, patchSizeX);
  const auto oneDTransformY = getNumPatches(yDim, kernelSize, patchSizeY);

  const unsigned numInpChanGroups = kernelTf.dim(1);
  const unsigned numOutPartialChanGroups = kernelTf.dim(0);
  const unsigned numOutPartialChansInGroup = kernelTf.dim(4);
  const unsigned numInpChansInGroup = kernelTf.dim(5);

  auto totalAccUnits = numOutPartialChanGroups * numInpChanGroups * patchSizeX * patchSizeY;
  const auto accUnitsPerVertex = (totalAccUnits + nMaxWorkers * maxTiles - 1)/(nMaxWorkers * maxTiles);

  ComputeSet cs = graph.createComputeSet(layerName +".accumulate");

  unsigned tile = 0;

  unsigned patchElemX = 0;
  unsigned patchElemY = 0;
  unsigned inpChanGroup = 0;
  unsigned outChanGroup = 0;
  do {
    unsigned vertex = 0;
    for (; vertex < nMaxWorkers && totalAccUnits; ++vertex) {
      const unsigned numUnitsThisVertex = totalAccUnits >= accUnitsPerVertex ? accUnitsPerVertex : totalAccUnits;

#if DEBUG_PRINT == 1
      std::cout << "tile: " << tile << " vertex: " << vertex << "  numUnitsThisVertex: " << numUnitsThisVertex << " totalAccUnits :" << totalAccUnits << std::endl;
#endif

      auto v = graph.addVertex(cs,
                               templateVertex("WgdPartials", baseClass, dType));

      graph.setInitialValue(v["inpChanDepth"], numInpChansInGroup);
      graph.setInitialValue(v["outChanDepth"], numOutPartialChansInGroup);
      graph.setInitialValue(v["nGroups"], numUnitsThisVertex);
      graph.setInitialValue(v["elemsPerFeature"], oneDTransformX * oneDTransformY);
      graph.setInitialValue(v["numWorkers"], deviceInfo.numWorkerContexts);
      graph.setFieldSize(v["wTf"], numUnitsThisVertex);
      graph.setFieldSize(v["dTf"], numUnitsThisVertex * oneDTransformX * oneDTransformY);
      graph.setFieldSize(v["partials"], numUnitsThisVertex * oneDTransformX * oneDTransformY);

      for (auto unit = 0; unit < numUnitsThisVertex; ++unit) {
        for (auto x = 0; x < oneDTransformX; ++x) {
          for (auto y = 0; y < oneDTransformY; ++y) {
            const auto idx = unit * oneDTransformX * oneDTransformY + y * oneDTransformX + x;
            graph.connect(dataTf[inpChanGroup][y][x][patchElemY][patchElemX], v["dTf"][idx]);
            auto aPart = accumTf[outChanGroup][inpChanGroup][y][x][patchElemY][patchElemX];
            graph.connect(v["partials"][idx], aPart);
            graph.setTileMapping(aPart, tile);
          }
        }

#if DEBUG_PRINT == 1
        std::cout << "outChanGroup: " << outChanGroup << " inChanGroup: " << inpChanGroup << "  patchElemX: " << patchElemX << " patchElemY :" << patchElemY << std::endl;
#endif

        graph.connect(kernelTf[outChanGroup][inpChanGroup][patchElemY][patchElemX].flatten(), v["wTf"][unit]);
        if (++patchElemX == patchSizeX) {
          patchElemX = 0;
          if (++patchElemY == patchSizeY) {
            patchElemY = 0;
            if (++inpChanGroup == numInpChanGroups) {
              inpChanGroup = 0;
              ++outChanGroup;
            }
          }
        }
      }
      graph.setTileMapping(v, tile);
      totalAccUnits -= numUnitsThisVertex;
    }
    if (vertex)
      ++tile;
  } while(totalAccUnits);
  return Execute(cs);
}

/**
 * @brief reduce accumulated transform product. Each element of patchSizeX x patchSizeY
 *        is assigned to a different vertex
 */
static Program reduce(Graph &graph,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned xDim, unsigned yDim,
            unsigned patchSizeX, unsigned patchSizeY,
            const std::string dType,
            const std::string layerName,
            Tensor weights, Tensor accumTf, Tensor invTf) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto nMaxWorkers = deviceInfo.numWorkerContexts;
  const auto maxTiles = deviceInfo.getNumTiles();

  const auto oneDTransformX = getNumPatches(xDim, kernelSize, patchSizeX);
  const auto oneDTransformY = getNumPatches(yDim, kernelSize, patchSizeY);

  const unsigned numInpChanGroups = weights.dim(1);
  const unsigned numOutPartialChanGroups = weights.dim(0);
  const unsigned numPartialChansInGroup = weights.dim(4);

  ComputeSet cs = graph.createComputeSet(layerName +".reduce");

  auto totalReduceUnits = numOutPartialChanGroups * oneDTransformY * oneDTransformX * patchSizeY * patchSizeX;
  const auto redUnitsPerVertex = (totalReduceUnits + maxTiles * nMaxWorkers - 1)/(nMaxWorkers * maxTiles);
  unsigned tile = 0;
  unsigned outChanGroup = 0;
  unsigned xPatch = 0;
  unsigned yPatch = 0;
  unsigned xElem = 0;
  unsigned yElem = 0;

  do {
    unsigned vertex = 0;
    for (; vertex < nMaxWorkers && totalReduceUnits; ++vertex){
      const unsigned numUnitsThisVertex = totalReduceUnits>=redUnitsPerVertex?redUnitsPerVertex:totalReduceUnits;

#if DEBUG_PRINT == 1
      std::cout << "tile: " << tile << " vertex: " << vertex << "  numUnitsThisVertex: " << numUnitsThisVertex << " totalReduceUnits :" << totalReduceUnits << std::endl;
#endif

      auto v = graph.addVertex(cs,
                               templateVertex("WgdReduce", dType));
      graph.setFieldSize(v["partials"], numUnitsThisVertex * numInpChanGroups);
      graph.setFieldSize(v["outPartial"], numUnitsThisVertex);
      graph.setInitialValue(v["inpLength"], numPartialChansInGroup);
      graph.setInitialValue(v["partialSumLength"], numInpChanGroups);
      graph.setInitialValue(v["nGroups"], numUnitsThisVertex);

      graph.setTileMapping(v, tile);

      /* set up tensors */
      for (auto unit = 0; unit < numUnitsThisVertex; ++unit) {
        for (auto part = 0; part < numInpChanGroups; ++part) {
          graph.connect(accumTf[outChanGroup][part][yPatch][xPatch][yElem][xElem], v["partials"][unit * numInpChanGroups + part]);
        }
        Tensor iPart = invTf[outChanGroup][yPatch][xPatch][yElem][xElem];
        graph.connect(v["outPartial"][unit], iPart);
        graph.setTileMapping(iPart, tile);

        if (++xPatch == oneDTransformX){
          xPatch = 0;
          if (++yPatch == oneDTransformY) {
            yPatch = 0;
            if (++yElem == patchSizeY) {
              yElem = 0;
              if (++xElem == patchSizeX) {
                xElem = 0;
                ++outChanGroup;
                assert(outChanGroup <= numOutPartialChanGroups);
              }
            }
          }
        }
      }
      totalReduceUnits -= numUnitsThisVertex;
    }
    if (vertex)
      ++tile;
  } while(totalReduceUnits);
  return Execute(cs);
}


/**
 * @brief compute inverse winograd transform
 */
static Program inverseTransform(Graph &graph,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned xDim, unsigned yDim,
            unsigned patchSizeX, unsigned patchSizeY,
            const std::string dType,
            const std::string layerName,
            Tensor invTfIn, Tensor invTfOut) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto nMaxWorkers = deviceInfo.numWorkerContexts;
  const auto maxTiles = deviceInfo.getNumTiles();

  const auto oneDTransformX = getNumPatches(xDim, kernelSize, patchSizeX);
  const auto oneDTransformY = getNumPatches(yDim, kernelSize, patchSizeY);
  const auto outputsPerPatchX = getNumOutputsPerPatch(kernelSize, patchSizeX);
  const auto outputsPerPatchY = getNumOutputsPerPatch(kernelSize, patchSizeX);

  const unsigned partialChansPerGroup = invTfIn.dim(5);
  const unsigned numOutChanPartialGroups = invTfIn.dim(0);

  ComputeSet cs = graph.createComputeSet(layerName + ".inverseTf");

  auto totalInvTfUnits = numOutChanPartialGroups * oneDTransformY * oneDTransformX;
  const auto invUnitsPerWorker = (totalInvTfUnits + maxTiles * nMaxWorkers - 1)/(nMaxWorkers * maxTiles);

  unsigned tile = 0;
  unsigned outChanGroup = 0;
  unsigned patchX = 0;
  unsigned patchY = 0;

  do {
    unsigned vertex = 0;
    for (; vertex < nMaxWorkers && totalInvTfUnits; ++vertex) {
      const unsigned numUnitsThisVertex = totalInvTfUnits >= invUnitsPerWorker ? invUnitsPerWorker : totalInvTfUnits;

#if DEBUG_PRINT == 1
      std::cout << "tile: " << tile << " vertex: " << vertex << "  numUnitsThisVertex: " << numUnitsThisVertex << " totalInvTfUnits :" << totalInvTfUnits << std::endl;
#endif

      auto v = graph.addVertex(cs,
                               templateVertex("Wgd3x3InverseTransform2x2", dType));
      graph.setFieldSize(v["dTf"], numUnitsThisVertex * patchSizeX * patchSizeY);
      graph.setFieldSize(v["dOut"], numUnitsThisVertex * outputsPerPatchX * outputsPerPatchY);
      graph.setInitialValue(v["depthDim"], partialChansPerGroup);
      graph.setInitialValue(v["nGroups"], numUnitsThisVertex);

      graph.setTileMapping(v, tile);

      for (auto unit = 0; unit < numUnitsThisVertex; ++unit) {

        /* connect input */
        for (auto x = 0; x < patchSizeX; ++x) {
          for (auto y = 0; y < patchSizeY; ++y) {
            graph.connect(invTfIn[outChanGroup][patchY][patchX][y][x], v["dTf"][unit * patchSizeX * patchSizeY + patchSizeX * y + x]);
          }
        }

        /* connect output */
        for (auto x = 0; x < outputsPerPatchX; ++x) {
          for (auto y = 0; y < outputsPerPatchY; ++y) {
            Tensor oPart = invTfOut[outChanGroup][patchY][patchX][y][x];
            graph.connect(v["dOut"][unit * outputsPerPatchX * outputsPerPatchY + outputsPerPatchX * y + x], oPart);
            graph.setTileMapping(oPart, tile);
          }
        }

        if (++patchX == oneDTransformX) {
          patchX = 0;
          if (++patchY == oneDTransformY) {
            patchY = 0;
            ++outChanGroup;
          }
        }
      }
      totalInvTfUnits -= numUnitsThisVertex;
    }
    if (vertex)
      ++tile;
  } while(totalInvTfUnits);
  return Execute(cs);
}


static Program complete(Graph &graph,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned xDim, unsigned yDim,
            unsigned patchSizeX, unsigned patchSizeY,
            NonLinearityType nonLinearityType,
            std::string dType,
            const std::string layerName,
            Tensor biases, Tensor activations,
            ResidualMethod resMethod, Tensor resIn, Tensor invTrfOut) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto nMaxWorkers = deviceInfo.numWorkerContexts;
  const auto maxTiles = deviceInfo.getNumTiles();

  const auto oneDTransformX = getNumPatches(xDim, kernelSize, patchSizeX);
  const auto oneDTransformY = getNumPatches(yDim, kernelSize, patchSizeY);

  const auto outputsPerPatchX = getNumOutputsPerPatch(kernelSize, patchSizeX);
  const auto outputsPerPatchY = getNumOutputsPerPatch(kernelSize, patchSizeY);

  assert(xDim == activations.dim(2));
  assert(yDim == activations.dim(1));

  /* work with partialGroups per channel as some of them are to be discarded */
  const auto padX = outputsPerPatchX * oneDTransformX - activations.dim(1);
  const auto padY = outputsPerPatchX * oneDTransformY - activations.dim(2);

  auto cs = graph.createComputeSet(layerName + ".complete");

  const unsigned numChansPerPartialGroup = invTrfOut.dim(5);
  const unsigned numPartialGroupsPerOutChanGroup = activations.dim(3)/numChansPerPartialGroup;
  const unsigned numPartialChanGroups = invTrfOut.dim(0);

  assert(activations.dim(3) % invTrfOut.dim(5) == 0);
  assert(numPartialGroupsPerOutChanGroup);

#if DEBUG_PRINT == 1
  std::cout << "numChansPerPartialGroup :" << numChansPerPartialGroup << " numPartialGroupsPerOutChanGroup " << numPartialGroupsPerOutChanGroup << " activations.dim(3): " << activations.dim(3) << std::endl;
  std::cout << "activations dimensions " << activations.dim(0) << "  " << activations.dim(1) << " " << activations.dim(2) << " " << activations.dim(3) << std::endl;
#endif

  /* these excludes elements arising from post padding */
  auto totalUnits = numPartialChanGroups * oneDTransformX * oneDTransformY;

  auto numUnitsPerWorker = (totalUnits + maxTiles * nMaxWorkers - 1)/(nMaxWorkers * maxTiles);
  unsigned tile = 0;

  unsigned outPartialGroup = 0;
  unsigned patchX = 0;
  unsigned patchY = 0;
  do {
    unsigned vertex = 0;
    for (; vertex < nMaxWorkers && totalUnits; ++vertex) {
      const unsigned numUnitsThisVertex = totalUnits >= numUnitsPerWorker ? numUnitsPerWorker : totalUnits;

      auto v = graph.addVertex(cs, templateVertex("WgdConvComplete", dType));

      unsigned patch = 0;

      for (auto unit = 0; unit < numUnitsThisVertex; ++unit) {
        for (auto y = 0; y < outputsPerPatchY; ++y) {
          for (auto x = 0; x < outputsPerPatchX; ++x) {
            if (!( ((patchX == oneDTransformX - 1) && (x >= outputsPerPatchX - padX)) ||
                 ((patchY == oneDTransformY - 1) && (y >= outputsPerPatchY - padY)) )) {
#if DEBUG_PRINT == 1
              std::cout << " outPartialGroup: " << outPartialGroup << " patchX: " << patchX << " patchY: " << patchY << " x: " << x << " y: " << y << std::endl;
#endif
              graph.connect(invTrfOut[outPartialGroup][patchY][patchX][y][x], v["dIn"][patch]);

              const auto outX = patchX * outputsPerPatchX + x;
              const auto outY = patchY * outputsPerPatchY + y;
              const auto outCGroup = outPartialGroup / numPartialGroupsPerOutChanGroup;
              const auto outPGroup = outPartialGroup % numPartialGroupsPerOutChanGroup;
              graph.connect(v["act"][patch], activations[outCGroup][outY][outX].slice(outPGroup * numChansPerPartialGroup, (outPGroup + 1) * numChansPerPartialGroup));
              graph.connect(biases.slice(outPartialGroup * numChansPerPartialGroup, (outPartialGroup + 1) * numChansPerPartialGroup), v["bias"][patch]);
              ++patch;
            }
          }
        }
#if DEBUG_PRINT == 1
      std::cout << "tile: " << tile << " vertex: " << vertex << "  numUnitsThisVertex: " << numUnitsThisVertex << " totalUnits :" << totalUnits << std::endl;
#endif

        if (++patchX == oneDTransformX) {
          patchX = 0;
          if (++patchY == oneDTransformY) {
            patchY = 0;
            ++outPartialGroup;
          }
        }
      }
      graph.setFieldSize(v["dIn"], patch);
      graph.setFieldSize(v["act"], patch);
      graph.setFieldSize(v["bias"], patch);
      graph.setInitialValue(v["vecLen"], numChansPerPartialGroup);
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      graph.setInitialValue(v["nGroups"], patch);
      graph.setTileMapping(v, tile);
      totalUnits -= numUnitsThisVertex;
    }
    if (vertex)
      ++tile;
  } while (totalUnits);
  return Execute(cs);
}

extern Program winogradConvolution(Graph &graph,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned xDim, unsigned yDim,
            unsigned outNumChans, unsigned patchSizeX, unsigned patchSizeY,
            NonLinearityType nonLinearityType,
            std::string dType,
            Tensor in, Tensor weights, Tensor biases, Tensor activations,
            ResidualMethod resMethod, Tensor resIn) {

#if DEBUG_PRINT==1
  std::cout << "xDim: " << xDim << std::endl;
  std::cout << "yDim: " << yDim << std::endl;
  std::cout << "in.dim(0) :" << in.dim(0) << std::endl;
  std::cout << "in.dim(1) :" << in.dim(1) << std::endl;
  std::cout << "in.dim(2) :" << in.dim(2) << std::endl;
  std::cout << "in.dim(3) :" << in.dim(3) << std::endl;

  std::cout << "weights.dim(0) :" << weights.dim(0) << std::endl;
  std::cout << "weights.dim(1) :" << weights.dim(1) << std::endl;
  std::cout << "weights.dim(2) :" << weights.dim(2) << std::endl;
  std::cout << "weights.dim(3) :" << weights.dim(3) << std::endl;
  std::cout << "weights.dim(4) :" << weights.dim(4) << std::endl;
  std::cout << "weights.dim(5) :" << weights.dim(5) << std::endl;
#endif

  /* assumption that number of input channels per group must be same for input activations and weights */
  assert(in.dim(0) == weights.dim(1));
  assert(in.dim(3) == weights.dim(5));


  /* Number of patchSizeX x patchSizeY dimension patches per feature */
  const auto oneDTransformX = getNumPatches(xDim, kernelSize, patchSizeX);
  const auto oneDTransformY = getNumPatches(yDim, kernelSize, patchSizeY);

  auto prog = Sequence();

  const auto layerName = "Wgd Conv" + std::to_string(kernelSize) + "x" + std::to_string(kernelSize) + ".fwd";


  /* create tensor for Data transform */
  Tensor dataTf = graph.addTensor(dType,
                                  { in.dim(0),
                                    oneDTransformY,
                                    oneDTransformX,
                                    patchSizeY, patchSizeX,
                                    in.dim(3)},
                                  "WgdDataTransform");

  Tensor kernelTf = graph.addTensor(dType,
                                    { weights.dim(0),
                                      weights.dim(1),
                                      patchSizeY, patchSizeX,
                                      weights.dim(4),
                                      weights.dim(5)},
                                    "WgdKernelTransform");


  prog.add(computeFwdTransforms(graph, kernelSize, stride, padding,
                       xDim, yDim, patchSizeX, patchSizeY, dType, layerName,
                       in, weights, dataTf, kernelTf));


  const unsigned numInChanGroups = weights.dim(1);
  const unsigned numOutChanPartialGroups = weights.dim(0);
  const unsigned partialChansPerGroup = weights.dim(4);

  /* accumulate across tiles */
  Tensor accumTf = graph.addTensor(dType,
                                   {numOutChanPartialGroups,
                                    numInChanGroups,
                                    oneDTransformY,
                                    oneDTransformX,
                                    patchSizeY, patchSizeX,
                                    partialChansPerGroup},
                                  "WgdAccumulate");

  prog.add(accumulate(graph, kernelSize, stride, padding,
                       xDim, yDim, patchSizeX, patchSizeY, dType, layerName,
                       in , weights, dataTf, kernelTf, accumTf));

  Tensor invTfIn = graph.addTensor(dType,
                                   {numOutChanPartialGroups,
                                    oneDTransformY,
                                    oneDTransformX,
                                    patchSizeY, patchSizeX,
                                    partialChansPerGroup},
                                   "WgdInvTrfIn");

  prog.add(reduce(graph, kernelSize, stride, padding,
                       xDim, yDim, patchSizeX, patchSizeY, dType, layerName,
                       weights, accumTf, invTfIn));


  const auto outputsPerPatchX = getNumOutputsPerPatch(kernelSize, patchSizeX);
  const auto outputsPerPatchY = getNumOutputsPerPatch(kernelSize, patchSizeY);

  Tensor invTfOut = graph.addTensor(dType,
                                   {numOutChanPartialGroups,
                                    oneDTransformY,
                                    oneDTransformX,
                                    outputsPerPatchY, outputsPerPatchX,
                                    partialChansPerGroup},
                                   "WgdInvTrfOut");



  prog.add(inverseTransform(graph, kernelSize, stride, padding,
                       xDim, yDim, patchSizeX, patchSizeY, dType, layerName,
                       invTfIn, invTfOut));

  prog.add(complete(graph, kernelSize, stride, padding,
                    xDim, yDim, patchSizeX, patchSizeY, nonLinearityType, dType, layerName, biases, activations,
                    resMethod, resIn, invTfOut));

  return prog;
}

} // namespace conv
