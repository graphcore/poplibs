#include "popnn/Convolution.hpp"
#include "ConvUtil.hpp"
#include "popnn/ActivationMapping.hpp"
#include "VertexTemplates.hpp"
#include "gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "popnn/exceptions.hpp"
#include <cassert>
#include <iostream>
#include <array>
#include <numeric>
#include <utility>


#define DEBUG_PRINT 0


using namespace poplar;
using namespace poplar::program;

namespace conv {

/**
 * compute stages for a full convolutional layer
 */
enum WgdStage {
    DATA_TRANSFORM,
    KERNEL_TRANSFORM,
    ACCUM,
    REDUCTION,
    INVERSE_TRANSFORM,
    COMPLETE,
    NUM_STAGES
};

/**
 * Class containing information to compute tile partition for the
 * Winograd transform. The naming convention for input and output
 * channels is the following
 *  All channels used variable name "z"" followed by "i" or "o"
 *  which indicates input or output. The following character could
 *  be wither "c" or "g" to indicate channels per grouping and
 *  the number of groups respectively.
 *
 * eg: zi is total number of input channels
 *     zoc is the total number of output channels in a group
 */

class WgdTilePartition {
  unsigned getOverlapX() const { return patchSizeX - kernelX + 1; }
  unsigned getOverlapY() const { return patchSizeY - kernelY + 1; }

public:
  using Cost = uint64_t;

  WgdTilePartition(unsigned padX, unsigned padY,
                  unsigned dimInX, unsigned dimInY,
                  unsigned patchSizeX, unsigned patchSizeY,
                  unsigned kernelX, unsigned kernelY,
                  unsigned zi, unsigned zo, std::string dType) :

                  padX(padX), padY(padY), dimInX(dimInX), dimInY(dimInY),
                  patchSizeX(patchSizeX), patchSizeY(patchSizeY),
                  kernelX(kernelX), kernelY(kernelY),
                  zi(zi), zo(zo), dType(dType) {
  }
  static constexpr unsigned dUnitSize = 4;
  static constexpr unsigned kUnitSize = 4;
  static constexpr unsigned iUnitSize = 4;

  const unsigned padX, padY;
  const unsigned dimInX, dimInY;
  const unsigned patchSizeX, patchSizeY;
  const unsigned kernelX, kernelY;
  const unsigned zi, zo;
  const std::string dType;


  /* Tiles over which all patches are distributed */
  unsigned tilesForPatches;

  /* Tiles for all Output Channel Groups */
  unsigned tilesForZog;

  /* Tiles for all Input Channel Groups */
  unsigned tilesForZig;

  /* Maximum number of patches assigned to each tile */
  unsigned patchesPerTile;

  /* Maximum number of output channel groups per tile */
  unsigned zogPerTile;

  /* Maximum number of input channel groups per tile */
  unsigned zigPerTile;

  /* true if kernel transform is computed on tile on which it is used
   * (i.e. same kernel transform may be done on multiple tiles)
   */
  bool replicateKTf;

  /* true if data transform is computed on tile on which it is used
   * (i.e. same data transform may be done on multiple tiles)
   */
  bool replicateDTf;

  /* number of input channel groups
   */
  unsigned zig;

  /* number of channels in an input channel group
   */
  unsigned zic;

  /* number of output channel groups
   */
  unsigned zog;

  /* number of channels in an output channel group
   */
  unsigned zoc;

  /* Number in channels in output channel groups
   */
  unsigned zocOut;

  unsigned outTotalPatches;

  /* Number of patches over which output activtions are kept.
   * These patches are over the whole output space and hence
   * differs from what is defined as "patchesPerTile"
   */
  unsigned outPatchesPerTile;

  /* Maintain compute and exchange cost for logging
   */
  std::array<Cost, NUM_STAGES> cc;
  std::array<Cost, NUM_STAGES> ec;

  uint64_t tilePartition(unsigned inpZic,
                     unsigned weightsZoc,
                     unsigned outZoc,
                     NonLinearityType nonLinearityType,
                     const DeviceInfo &deviceInfo);

  std::pair<unsigned, unsigned> getPaddingX(unsigned patchX) const {
    unsigned prepad = 0;
    unsigned postpad = 0;
    unsigned overlapX = getOverlapX();

    if (patchX * overlapX < padX) {
      prepad = std::min(padX - patchX * overlapX, patchSizeX);
    }

    if (patchX * overlapX + patchSizeX > dimInX + padX) {
      postpad = std::min(patchX * overlapX + patchSizeX - (dimInX + padX),
                         patchSizeX);
    }
    return std::make_pair(prepad, postpad);
  }

  std::pair<unsigned, unsigned> getPaddingY(unsigned patchY) const {
    unsigned prepad = 0;
    unsigned postpad = 0;
    unsigned overlapY = getOverlapY();

    if (patchY * overlapY < padY) {
      prepad = std::min(padY - patchY * overlapY, patchSizeY);
    }

    if (patchY * overlapY + patchSizeY > dimInY + padY) {
      postpad = std::min(patchY * overlapY + patchSizeY - (dimInY + padY),
                         patchSizeY);
    }
    return std::make_pair(prepad, postpad);
  }

  unsigned getInpPosX(unsigned patchX) const {
    unsigned inpPos = 0;
    unsigned overlapX = getOverlapX();
    if (patchX * overlapX > padX)
      inpPos = patchX * overlapX - padX;
    if (patchX * overlapX > padX + dimInX)
      inpPos = dimInX;
    return inpPos;
  }

  unsigned getInpPosY(unsigned patchY) const {
    unsigned inpPos = 0;
    unsigned overlapY = getOverlapY();
    if (patchY * overlapY > padY)
      inpPos = patchY * overlapY - padY;
    if (patchY * overlapY > padY + dimInY)
      inpPos = dimInY-1;
    return inpPos;
  }

  std::pair<unsigned, unsigned> getPatchIdx(unsigned patch) const {
    return std::make_pair(patch % getNumPatchesX(), patch / getNumPatchesX());
  }

  std::pair<unsigned, unsigned> getOutIdxForPatch(unsigned patch) const {
    unsigned patchX, patchY;
    std::tie(patchX, patchY) = getPatchIdx(patch);
    return std::make_pair(patchX * getOverlapX(), patchY * getOverlapY());
  }

  unsigned getOutputSizeX() const {
    return dimInX + 2 * padX - (kernelX - 1);
  }

  unsigned getOutputSizeY() const {
    return dimInY + 2 * padY - (kernelY - 1);
  }

  unsigned getNumPatchesX() const {
    return (getOutputSizeX() + getOverlapX() - 1)/getOverlapX();;
  }

  unsigned getNumPatchesY() const {
    return (getOutputSizeY() + getOverlapY() - 1)/getOverlapY();;
  }

  unsigned getNumPatches() const {
      return getNumPatchesX() * getNumPatchesY();
  }

  unsigned getNumOutputsPerPatchX() const {
    return getOverlapX();
  }

  unsigned getNumOutputsPerPatchY() const {
    return getOverlapY();
  }

  unsigned getTotalOutputPatches() const {
    return getNumPatches() * zo/zocOut;
  }

};

uint64_t WgdTilePartition::tilePartition(unsigned inpZic,
                                     unsigned weightsZoc,
                                     unsigned outZoc,
                                     NonLinearityType nonLinearityType,
                                     const DeviceInfo &deviceInfo) {

  const unsigned numTiles = deviceInfo.getNumTiles();
  const unsigned numWorkers = deviceInfo.numWorkerContexts;
  const auto numPatches = getNumPatches();
  const auto isFloat = dType == "float";
  const unsigned exchEfficiency = 100;

  /* for now use number of channel groups to be what is input.
   * this may change later
   */
  zic = inpZic;
  zoc = weightsZoc;
  zocOut = outZoc;

  assert(zi % zic == 0);
  assert(zo % zoc == 0);

  zog = zo/zoc;
  zig = zi/zic;

  std::array<unsigned, NUM_STAGES> enableCost{};
  //enableCost.fill(1);
  std::get<DATA_TRANSFORM>(enableCost) = 1;
  std::get<KERNEL_TRANSFORM>(enableCost) = 1;
  std::get<ACCUM>(enableCost) = 1;
  std::get<REDUCTION>(enableCost) = 1;
  std::get<INVERSE_TRANSFORM>(enableCost) = 1;
  std::get<COMPLETE>(enableCost) = 1;

  outPatchesPerTile = (getNumPatches() * zo/outZoc + numTiles - 1)
                            / numTiles;

  /* exchange transfer wordlength */
  const unsigned eWl = deviceInfo.exchangeBytesPerCycle;

  /* this may not be portable: use boost?? */
  Cost bestCost = std::numeric_limits<uint64_t>::max();

  for (unsigned tilesForPatches = 1;
                tilesForPatches <= std::min(numTiles, numPatches);
                ++tilesForPatches) {

    /* assign all remaining tiles to input channel groups */
    const unsigned numTilesZig = numTiles/tilesForPatches;
    const unsigned patchesPerTile = (numPatches + tilesForPatches - 1)
                                    / tilesForPatches;

    for (unsigned tilesForZig = 1;
                  tilesForZig <= std::min(numTilesZig, zig);
                  ++tilesForZig) {
      const unsigned zigPerTile = (zig + tilesForZig - 1)/tilesForZig;

      /* maximum tiles to which all output channel groups may be allocate to */
      const unsigned numTilesZog = numTiles/(tilesForPatches * tilesForZig);

      for (unsigned tilesForZog = 1;
                    tilesForZog <= std::min(numTilesZog, zog);
                    ++tilesForZog) {
        const unsigned zogPerTile = (zog + tilesForZog - 1)/tilesForZog;

        std::array<Cost, NUM_STAGES> cc{}, ec{};

        #if DEBUG_PRINT >= 3
        std::cout << "tilesForPatches :" << tilesForPatches;
        std::cout << "  numTilesZig :" << numTilesZig;
        std::cout << "  patchesPerTile :" << patchesPerTile;
        std::cout << "  tilesForZig :" << tilesForZig;
        std::cout << "  zigPerTile :" << zigPerTile;
        std::cout << "  numTilesZog :" << numTilesZog;
        std::cout << "  tilesForZog :" << tilesForZog;
        std::cout << "  zogPerTile :" << zogPerTile << std::endl;
        #endif

        /* the kernel transform could be computed in different ways:
         * 1) Copy untransformed data into tiles that need the kernels and
         *    transform them on those tiles.
         * 2) Use all available tiles to minimise computation cost and copy
         *    transform to the tiles that need the transformed kernels
         *
         * we compute both costs and select the one which is cheaper
         */
        const unsigned kWl   = isFloat ? 4 : 2;
        const unsigned kTfWl = isFloat ? 4 : 2;
        const unsigned numKTfUnits1 = zigPerTile * zogPerTile * zic * zoc;
        unsigned numKTfBlocks1 = (numKTfUnits1 + numWorkers - 1)/numWorkers;

        Cost ecKTf1 = (numKTfUnits1 * kernelX * kernelY * kWl) / eWl;
        Cost ccKTf1 = getWgdKernelTransformCycles(
                                                  numKTfBlocks1,
                                                  dType == "float")
                      * numWorkers;

        const unsigned numKTfUnits2 = (zi * zo/kUnitSize
                                       + numTiles - 1)/numTiles;
        const unsigned numKTfBlocks2 = (numKTfUnits2 + numWorkers - 1)
                                       / numWorkers;
        Cost ecKTf2 = (numKTfUnits1 * patchSizeX * patchSizeY * kTfWl
                      + kUnitSize * numKTfUnits2 * kernelX * kernelY * kWl)/eWl;
        Cost ccKTf2 = getWgdKernelTransformCycles(numKTfBlocks2 * kUnitSize,
                                                isFloat) * numWorkers;

        bool replicateKTf = true;
        std::get<KERNEL_TRANSFORM>(cc) = ccKTf1;
        std::get<KERNEL_TRANSFORM>(ec) = ecKTf1;

        if (ecKTf2 * 100/exchEfficiency + ccKTf2 < 
                                 ecKTf1 * 100/exchEfficiency + ccKTf1) {
          replicateKTf = false;
          std::get<KERNEL_TRANSFORM>(cc) = ccKTf2;
          std::get<KERNEL_TRANSFORM>(ec) = ecKTf2;
        }

        #if DEBUG_PRINT >= 3
        std::cout << "Kernel cost: ec1: " << ecKTf1;
        std::cout << " ec2: " << ecKTf2 << " cc1: " << ccKTf1;
        std::cout << " cc2: " << ccKTf2 << " repl: ";
        std::cout << replicateKTf << std::endl;
        #endif

        /* the data transform could be computed in different ways:
         * 1) Copy untransformed data into tiles that need the transform and
         *    transform them on those tiles.
         * 2) Use all available tiles to minimise computation cost and copy
         *    transforms to the tiles that need them
         *
         * we compute both costs and select the one which is cheaper
         */

        const unsigned dTfWl = isFloat ? 4 : 2;
        const unsigned dWl = isFloat ? 4 : 2;
        const unsigned numDTfUnits1 = patchesPerTile * zigPerTile * zic;
        const unsigned numDTfBlocks1 = (numDTfUnits1 + numWorkers - 1)
                                       / numWorkers;
        const unsigned numDTfUnits2 = (numPatches * zi/dUnitSize
                                       + numTiles - 1)/numTiles;
        const unsigned numDTfBlocks2 = (numDTfUnits2 + numWorkers -1)
                                        / numWorkers;

        Cost ecDTf1 = (numDTfUnits1 * patchSizeX * patchSizeY * dWl) / eWl;
        Cost ccDTf1 = getWgdDataTransformCycles(numDTfBlocks1, isFloat)
                      * numWorkers;
        Cost ecDTf2 = (numDTfUnits1 * patchSizeX * patchSizeY * dTfWl
                      + dUnitSize * numDTfUnits2 * patchSizeX * patchSizeY
                        * dWl) / eWl;
        Cost ccDTf2 = getWgdDataTransformCycles(numDTfBlocks2 * dUnitSize,
                                              isFloat) * numWorkers;
        bool replicateDTf = true;

        std::get<DATA_TRANSFORM>(cc) = ccDTf1;
        std::get<DATA_TRANSFORM>(ec) = ecDTf1;

        if (ecDTf2 * 100/exchEfficiency + ccDTf2 < 
                        ecDTf1 * 100/exchEfficiency + ccDTf1) {
          replicateDTf = false;

          std::get<DATA_TRANSFORM>(cc) = ccDTf2;
          std::get<DATA_TRANSFORM>(ec) = ecDTf2;
        }

        #if DEBUG_PRINT >= 3
        std::cout << "Data cost: ec1: " << ecDTf1;
        std::cout << " ec2: " << ecDTf2 << " cc1: " << ccDTf1;
        std::cout << " cc2: " << ccDTf2;
        std::cout << " repl: " << replicateDTf << std::endl;
        #endif

        /* compute accumulate cost: all the data is local to a tile and hence
         * needn't be exchanged
         */
        const unsigned accWl = isFloat ? 4 : 2;
        Cost ecAcc = 0;
        const auto weightsPerConvUnit =
                                deviceInfo.getWeightsPerConvUnit(isFloat);
        const auto numConvUnits = isFloat ?
                                deviceInfo.fp32AccumConvUnitsPerTile :
                                deviceInfo.fp16AccumConvUnitsPerTile;

        Cost ccAcc;
        if (deviceInfo.sharedConvWeights) {
          unsigned numBlocks = patchSizeX * patchSizeY * zogPerTile
                               * zigPerTile;
          ccAcc = getWgdAccumCycles(true, numBlocks, patchesPerTile, zic,
                                    zoc, numWorkers, numConvUnits,
                                    weightsPerConvUnit, isFloat);

        } else {
          unsigned numBlocks = (patchSizeX * patchSizeY * zogPerTile *
                                zigPerTile + numWorkers - 1)/numWorkers;

          ccAcc = getWgdAccumCycles(false, numBlocks, patchesPerTile, zic,
                                    zoc, numWorkers, numConvUnits,
                                    weightsPerConvUnit, isFloat) * numWorkers;
        }
        /* add cost of zeroing of partials */
        if (zigPerTile > 1) {
          auto numZeroUnits = (patchSizeX * patchSizeY * zoc
                              * patchesPerTile * zigPerTile + numWorkers - 1)
                              / numWorkers;
          ccAcc += numZeroUnits * accWl/4 * numWorkers;
        }

        std::get<ACCUM>(cc) = ccAcc;
        std::get<ACCUM>(ec) = ecAcc;

        #if DEBUG_PRINT >= 3
        std::cout << "Accum cost: ec: " << ecAcc;
        std::cout << " cc: " << ccAcc << std::endl;
        #endif


        /* compute exchange cost for reduction. All the input channels
         * must be brought onto a tile. The exchange cost is the max between
         * the amount a tile has to receive and the amount each tile has to
         * send
         */
        //const unsigned redWl = dType == "float" ? 4 : 2;
        const auto sendCost = zogPerTile * patchesPerTile * zoc
                              * patchSizeX * patchSizeY * accWl/eWl;

        const auto recvCost = outPatchesPerTile * outZoc * patchSizeX
                              * patchSizeY
                              * ((zig + zigPerTile - 1)/zigPerTile)
                              * accWl/eWl;

        Cost ecRed =  std::max(sendCost, recvCost);

        unsigned numRedBlocks = (outPatchesPerTile * zocOut
                                * patchSizeX * patchSizeY
                                + numWorkers - 1)/numWorkers;
        const auto numPartials = (zig + zigPerTile - 1)/zigPerTile;

        Cost ccRed = getWgdReduceCycles(numRedBlocks, numPartials, isFloat)
                     * numWorkers;

        std::get<REDUCTION>(cc) = ccRed;
        std::get<REDUCTION>(ec) = ecRed;

        #if DEBUG_PRINT >= 3
        std::cout << "Red cost: ec: " << ecRed;
        std::cout << " cc: " << ccRed << std::endl;
        #endif

        /* Inverse kernel transform doesn't require exchange */
        //const unsigned iTfWl = isFloat ? 4 : 2;
        Cost ecITf = 0;
        const unsigned numITfUnits = outPatchesPerTile * outZoc/iUnitSize;
        const unsigned numItfBlocks = (numITfUnits + numWorkers -1)/numWorkers;
        Cost ccITf = getWgdInvTransformCycles(
                                              iUnitSize * numItfBlocks,
                                              isFloat)
                     * numWorkers;

        std::get<INVERSE_TRANSFORM>(cc) = ccITf;
        std::get<INVERSE_TRANSFORM>(ec) = ecITf;

        #if DEBUG_PRINT >= 3
        std::cout << "Inverse cost: ec: " << ecITf;
        std::cout << " cc: " << ccITf << "\n";
        #endif


        /* cost to complete */
        const unsigned cWl = isFloat ? 4 : 2;
        Cost ecComp = (outPatchesPerTile * zocOut
                       * getOverlapX() * getOverlapY() * cWl) / eWl;

        std::get<COMPLETE>(ec) = ecComp;

        Cost ccComp = outPatchesPerTile
                      * getWgdCompleteCycles(
                              zocOut * getOverlapX() * getOverlapY(),
                              nonLinearityType,
                              isFloat);
        std::get<COMPLETE>(cc) = ccComp;

        #if DEBUG_PRINT >= 3
        std::cout << "Complete cost: ec: " << ecComp << "\n";
        #endif

        Cost totalECost = std::inner_product(ec.begin(), ec.end(),
                                            enableCost.begin(), 0) 
                          * 100/exchEfficiency;
        Cost totalCCost = std::inner_product(cc.begin(), cc.end(),
                                             enableCost.begin(), 0);
        Cost totalCost  = totalECost + totalCCost;

        #if DEBUG_PRINT >= 3
        std::cout << "Total cost: ec: " << totalECost;
        std::cout << "  cc: " << totalCCost << "\n\n\n";
        #endif


        if (totalCost < bestCost) {
            bestCost = totalCost;
            WgdTilePartition::patchesPerTile = patchesPerTile;
            WgdTilePartition::tilesForPatches = tilesForPatches;
            WgdTilePartition::tilesForZig = tilesForZig;
            WgdTilePartition::tilesForZog = tilesForZog;
            WgdTilePartition::zigPerTile = zigPerTile;
            WgdTilePartition::zogPerTile = zogPerTile;

            WgdTilePartition::replicateDTf = replicateDTf;
            WgdTilePartition::replicateKTf = replicateKTf;

            WgdTilePartition::cc = cc;
            WgdTilePartition::ec = ec;

        }
      }
    }
  }

  #if DEBUG_PRINT >= 1
  std::cout << "patchesPerTile :" << WgdTilePartition::patchesPerTile << "\n";
  std::cout << "tilesForPatches :" << WgdTilePartition::tilesForPatches << "\n";
  std::cout << "tilesForZig :" << WgdTilePartition::tilesForZig << "\n";
  std::cout << "tilesForZog :" << WgdTilePartition::tilesForZog << "\n";
  std::cout << "zigPerTile :" << WgdTilePartition::zigPerTile << "\n";
  std::cout << "zogPerTile :" << WgdTilePartition::zogPerTile << "\n";
  std::cout << "replicateDTf :" << WgdTilePartition::replicateDTf << "\n";
  std::cout << "replicateKTf :" << WgdTilePartition::replicateKTf << "\n";
  std::cout << "outPatchesPerTile :" << outPatchesPerTile << "\n\n";
  std::cout << "Exchange cost :\n";
  std::cout << " DATA_TRANSFORM " << WgdTilePartition::ec[DATA_TRANSFORM]<<"\n";
  std::cout << " KER_TRANSFORM "<< WgdTilePartition::ec[KERNEL_TRANSFORM]<<"\n";
  std::cout << " ACCUM "<< WgdTilePartition::ec[ACCUM]<<"\n";
  std::cout << " REDUCE "<< WgdTilePartition::ec[REDUCTION]<<"\n";
  std::cout << " INVERSE "<< WgdTilePartition::ec[INVERSE_TRANSFORM]<<"\n";
  std::cout << " COMPLETE "<< WgdTilePartition::ec[COMPLETE]<<"\n\n";

  std::cout << "Compute cost :\n";
  std::cout << " DATA_TRANSFORM " << WgdTilePartition::cc[DATA_TRANSFORM]<<"\n";
  std::cout << " KER_TRANSFORM "<< WgdTilePartition::cc[KERNEL_TRANSFORM]<<"\n";
  std::cout << " ACCUM "<< WgdTilePartition::cc[ACCUM]<<"\n";
  std::cout << " REDUCE "<< WgdTilePartition::cc[REDUCTION]<<"\n";
  std::cout << " INVERSE "<< WgdTilePartition::cc[INVERSE_TRANSFORM]<<"\n";
  std::cout << " COMPLETE "<< WgdTilePartition::cc[COMPLETE]<<"\n\n";
  std::cout << "Total cost : " << bestCost << "\n\n";
  #endif

  return bestCost;
}


static Program kernelTransform(Graph &graph,
                        const WgdTilePartition &tp,
                        const std::string layerName,
                        Tensor weights,
                        std::vector<Tensor> &kTfMapping) {

  ComputeSet cs = graph.createComputeSet(layerName + ".kernelTrf");
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const unsigned numWorkers = deviceInfo.numWorkerContexts;

  unsigned numZig = tp.zig;
  for (unsigned zigTile = 0; zigTile < tp.tilesForZig; ++zigTile) {

    unsigned numZog = tp.zog;
    const unsigned zigThisTile = std::min(numZig, tp.zigPerTile);

    for (unsigned zogTile = 0; zogTile < tp.tilesForZog; ++zogTile) {

      unsigned numPatches = tp.getNumPatches();
      const unsigned zogThisTile = std::min(numZog, tp.zogPerTile);

      for (unsigned pTile = 0; pTile < tp.tilesForPatches; ++pTile) {

        const auto tile = zigTile * tp.tilesForZog * tp.tilesForPatches
                          + zogTile * tp.tilesForPatches
                          + pTile;

        /* number assigned this tile */
        const auto patchesThisTile = std::min(numPatches, tp.patchesPerTile);

        if (!patchesThisTile)
          continue;

        /* start indices */
        const auto zigS = zigTile * tp.zigPerTile;
        const auto zogS = zogTile * tp.zogPerTile;

        auto numUnits = (tp.zic * tp.zoc + WgdTilePartition::kUnitSize - 1)
                        / WgdTilePartition::kUnitSize;

        Tensor kTf = graph.addTensor(tp.dType,
                                     {
                                      zogThisTile,
                                      zigThisTile,
                                      tp.patchSizeY,
                                      tp.patchSizeX,
                                      tp.zoc,
                                      tp.zic
                                     },
                                     "kernelTf"+std::to_string(tile));

        graph.setTileMapping(kTf, tile);
        assert(tile < kTfMapping.size());
        kTfMapping[tile] = kTf;

        /* divide Units over vertices */
        auto numUnitsPerVertex = (numUnits + numWorkers - 1) / numWorkers;

        for (unsigned vertex = 0; vertex < numWorkers && numUnits; ++vertex) {
          auto unitsThisVertex = std::min(numUnits, numUnitsPerVertex);
          auto unitS = vertex * numUnitsPerVertex * WgdTilePartition::kUnitSize;
          auto unitE = unitS + unitsThisVertex * WgdTilePartition::kUnitSize;

          #if DEBUG_PRINT >= 3
          std::cout << "numUnits " << numUnits << std::endl;
          std::cout << "tile : " << tile << " : " << vertex;
          std::cout << "  [" << zogS << "][" << zigS << "][0][0][";
          std::cout <<  unitS << " : " << unitE << "]\n";
          std::cout << "[" << (zogS + zogThisTile) << "][";
          std::cout << (zigS + zigThisTile) << "][3][3][";
          std::cout <<  unitS << " : " << unitE << "]\n";
          #endif

          Tensor inp =
            weights.slice(
            {
             zogS, zigS, 0, 0, 0, 0
            },
            {
             zogS + zogThisTile, zigS + zigThisTile, tp.kernelY, tp.kernelX,
             tp.zoc, tp.zic
            }).reshape(
               {
                zogThisTile * zigThisTile * tp.kernelY * tp.kernelX,
                tp.zic * tp.zoc
               }).slice(
                  {0, unitS},
                  {zogThisTile * zigThisTile * tp.kernelY * tp.kernelX,
                   unitE});

          Tensor out =
            kTf.slice(
              {
               0, 0, 0, 0, 0, 0
              },
              {
               zogThisTile, zigThisTile, tp.patchSizeY, tp.patchSizeX,
               tp.zoc, tp.zic
              }).reshape(
                 {
                  zogThisTile * zigThisTile * tp.patchSizeY * tp.patchSizeX,
                  tp.zic * tp.zoc
                 }).slice(
                    {0, unitS},
                    {zogThisTile * zigThisTile * tp.patchSizeY * tp.patchSizeX,
                         unitE});

          auto v = graph.addVertex(
                      cs,
                      templateVertex("WgdKernelTransform", tp.dType,
                                     tp.patchSizeX, tp.patchSizeY,
                                     tp.kernelX, tp.kernelY));

          graph.connect(inp, v["wIn"]);
          graph.connect(v["wTf"], out);
          graph.setFieldSize(v["wIn"],
                         tp.kernelX * tp.kernelY
                         * zogThisTile * zigThisTile);
          graph.setFieldSize(v["wTf"],
                         tp.patchSizeX * tp.patchSizeY
                         * zogThisTile * zigThisTile);
          graph.setTileMapping(v, tile);

          numUnits -= unitsThisVertex;
        }
        numPatches -= patchesThisTile;
      }
      numZog -= zogThisTile;
    }
    numZig -= zigThisTile;
  }
  return Sequence(Execute(cs));
}


static Program kernelTransform(Graph &graph,
                        const WgdTilePartition &tp,
                        const std::string layerName,
                        Tensor weights,
                        Tensor kernelTf) {
  unsigned numUnits = (tp.zi * tp.zo + WgdTilePartition::kUnitSize - 1)
                      / WgdTilePartition::kUnitSize;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();

  const unsigned numTiles = deviceInfo.getNumTiles();
  const unsigned numWorkers = deviceInfo.numWorkerContexts;

  ComputeSet cs = graph.createComputeSet(layerName + ".kernelTrf");

  assert(tp.zic % WgdTilePartition::kUnitSize == 0);

  unsigned unitsPerTile = (numUnits + numTiles - 1)/numTiles;

  for (unsigned tile = 0; tile < numTiles && numUnits; ++tile) {
    auto unitsThisTile = std::min(numUnits, unitsPerTile);
    const auto unitsPerVertex = (unitsPerTile + numWorkers - 1)/numWorkers;
    numUnits -= unitsThisTile;

    /* split units assigned to this tile over vertices */
    for (unsigned vertex = 0; vertex < numWorkers && unitsThisTile; ++vertex) {
      const auto unitsThisVertex =  std::min(unitsPerVertex, unitsThisTile);

      /* allocate units to this worker */
      auto v = graph.addVertex(cs,
                            templateVertex("WgdKernelTransform", tp.dType,
                                           tp.patchSizeX, tp.patchSizeY,
                                           tp.kernelX, tp.kernelY));
      graph.setFieldSize(v["wIn"],
                         unitsThisVertex * tp.kernelX * tp.kernelY);
      graph.setFieldSize(v["wTf"],
                         unitsThisVertex * tp.patchSizeX * tp.patchSizeY);
      graph.setTileMapping(v, tile);


      for (unsigned unit = 0; unit < unitsThisVertex; ++unit) {
        const auto sUnit = (tile * unitsPerTile
                            + vertex * unitsPerVertex + unit)
                            * WgdTilePartition::kUnitSize;

        const auto ig = (sUnit / (tp.zic * tp.zoc)) % tp.zig;
        const auto og = sUnit / (tp.zoc * tp.zic * tp.zig);


        const auto slS = sUnit - sUnit/(tp.zic * tp.zoc) * (tp.zic * tp.zoc);
        const auto slE = slS + WgdTilePartition::kUnitSize;
        for (unsigned y = 0; y < tp.kernelY; ++y) {
          for (unsigned x = 0; x < tp.kernelX; ++x) {
            graph.connect(weights[og][ig][y][x].flatten().slice(slS, slE),
                          v["wIn"][unit * tp.kernelX * tp.kernelY
                                   + y * tp.kernelX + x]);
          }
        }

        for (unsigned y = 0; y < tp.patchSizeY; ++y) {
          for (unsigned x = 0; x < tp.patchSizeX; ++x) {
            graph.connect(v["wTf"][unit * tp.patchSizeX * tp.patchSizeY
                                   + y * tp.patchSizeX + x],
                          kernelTf[og][ig][y][x].flatten().slice(slS, slE));
            graph.setTileMapping(
                     kernelTf[og][ig][y][x].flatten().slice(slS, slE), tile);
          }
        }

        #if DEBUG_PRINT >= 2
        const auto oc = (sUnit / tp.zic) % tp.zoc;
        const auto ic = sUnit % tp.zic;
        std::cout << "(KTF)unit " << sUnit  << " [" << og << "][" << ig << "][";
        std::cout << oc << "][" << ic << "] " << slS << " : " << slE;
        std::cout << " on tile " << tile << ":";
        std::cout << vertex << std::endl;
        #endif
      }

      unitsThisTile -= unitsThisVertex;
    }
  }
  return Sequence(Execute(cs));
}


static std::vector<Tensor> allocateKernelTfTensor(
                                  Graph &graph,
                                  const WgdTilePartition &tp) {
  std::vector<Tensor> kernelTf;

  if (!tp.replicateKTf) {

    kernelTf.resize(1);
    kernelTf[0] = graph.addTensor(tp.dType,
                                    {
                                      tp.zog,
                                      tp.zig,
                                      tp.patchSizeY,
                                      tp.patchSizeX,
                                      tp.zoc,
                                      tp.zic
                                    },
                                    "WgdKernelTransform");

  } else {
    kernelTf.resize(graph.getDevice().getDeviceInfo().getNumTiles());
  }

  return kernelTf;
}


static Program computeKernelTransform(Graph &graph,
                                  const WgdTilePartition &tp,
                                  const std::string layerName,
                                  Tensor weights,
                                  std::vector<Tensor> &kernelTf) {
  return tp.replicateKTf ?
            kernelTransform(graph, tp, layerName, weights, kernelTf) :
            kernelTransform(graph, tp, layerName, weights, kernelTf[0]);
}



static Program dataTransform(Graph &graph,
                        const WgdTilePartition &tp,
                        const std::string layerName,
                        Tensor in,
                        std::vector<Tensor> &dTfMapping) {

  ComputeSet cs = graph.createComputeSet(layerName + ".dataTrf");
  ComputeSet zCs = graph.createComputeSet(layerName + ".zeros");

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const unsigned numWorkers = deviceInfo.numWorkerContexts;

  unsigned numZig = tp.zig;
  for (unsigned zigTile = 0; zigTile < tp.tilesForZig; ++zigTile) {
    unsigned numZog = tp.zog;
    const unsigned zigThisTile = std::min(numZig, tp.zigPerTile);

    for (unsigned zogTile = 0; zogTile < tp.tilesForZog; ++zogTile) {
      unsigned numPatches = tp.getNumPatches();
      const unsigned zogThisTile = std::min(numZog, tp.zogPerTile);

      for (unsigned pTile = 0; pTile < tp.tilesForPatches; ++pTile) {
        const auto tile = zigTile * tp.tilesForZog * tp.tilesForPatches
                    + zogTile * tp.tilesForPatches
                    + pTile;

        /* number assigned this tile */
        const auto patchesThisTile = std::min(numPatches, tp.patchesPerTile);

        if (!patchesThisTile)
          continue;

        /* start indices */
        const auto zigS = zigTile * tp.zigPerTile;
        const auto patchS = pTile * tp.patchesPerTile;

        Tensor dTf = graph.addTensor(
                            tp.dType,
                            {
                              zigThisTile,
                              patchesThisTile,
                              tp.patchSizeY,
                              tp.patchSizeX,
                              tp.zic
                            },
                            "WgdDataTf" + std::to_string(tile));
        assert(tile < dTfMapping.size());
        dTfMapping[tile] = dTf;
        graph.setTileMapping(dTf, tile);

        auto numUnits = (zigThisTile * tp.zic * patchesThisTile +
                         WgdTilePartition::dUnitSize - 1)
                        / WgdTilePartition::dUnitSize;

        auto numUnitsPerVertex = (numUnits + numWorkers - 1) / numWorkers;
        bool zeroTensorCreated = false;
        Tensor zeroVec;

        for (unsigned vertex = 0; vertex < numWorkers && numUnits; ++vertex) {
          auto unitsThisVertex = std::min(numUnitsPerVertex, numUnits);

          #if DEBUG_PRINT >= 2
          std::cout << "Tile : " << tile << "  vertex : " << vertex << "\n";
          #endif

          auto v = graph.addVertex(cs,
                                   templateVertex("WgdDataTransform", tp.dType,
                                       tp.patchSizeX, tp.patchSizeY,
                                       tp.kernelX, tp.kernelY));

          graph.setFieldSize(v["dIn"],
                             unitsThisVertex * tp.patchSizeX * tp.patchSizeY);

          graph.setFieldSize(v["dTf"],
                             unitsThisVertex * tp.patchSizeX * tp.patchSizeY);

          graph.setTileMapping(v, tile);

          for (unsigned unit = 0; unit < unitsThisVertex; ++unit) {
            const auto unitS = (vertex * numUnitsPerVertex + unit)
                               * WgdTilePartition::dUnitSize;

            const auto ig = unitS / (tp.zic * patchesThisTile);
            const auto p = (unitS / tp.zic) % patchesThisTile;


            #if DEBUG_PRINT >= 2
            std::cout <<"unit : "<< unit<<" patch : "<< (p + patchS) << "\n";
            #endif

            unsigned patchX, patchY, prepadX, postpadX, prepadY, postpadY;
            std::tie(patchX, patchY) = tp.getPatchIdx(patchS + p);
            std::tie(prepadX, postpadX) = tp.getPaddingX(patchX);
            std::tie(prepadY, postpadY) = tp.getPaddingY(patchY);

            if ((prepadX || prepadY || postpadX || postpadY)
                   && !zeroTensorCreated) {
              zeroVec = graph.addTensor(tp.dType,
                                        {WgdTilePartition::dUnitSize},
                                        "zero");
              graph.setTileMapping(zeroVec, tile);

              auto vZ = graph.addVertex(zCs,
                                        templateVertex("Zero", tp.dType));
              graph.setInitialValue(vZ["dataPathWidth"],
                                    deviceInfo.dataPathWidth);

              graph.connect(vZ["out"], zeroVec);
              graph.setTileMapping(vZ, tile);
              zeroTensorCreated = true;
            }

            const auto slS = unitS % tp.zic;
            const auto slE = slS + WgdTilePartition::dUnitSize;


            auto inPosY = tp.getInpPosY(patchY);
            for (unsigned y = 0; y < tp.patchSizeY; ++y) {
              bool zeroY  = y < prepadY || (y >= tp.patchSizeY - postpadY);
              auto inPosX = tp.getInpPosX(patchX);
              for (unsigned x = 0; x < tp.patchSizeX; ++x) {
                bool zeroX = x < prepadX || (x >= tp.patchSizeX - postpadX);
                Tensor iPart = (zeroX || zeroY) ?
                                zeroVec :
                                in[zigS + ig][inPosY]
                                  [inPosX].flatten().slice(slS, slE);

                #if DEBUG_PRINT >= 2
                std::cout <<" ig "<<(zigS + ig)<<"  " << slS <<":"<<slE<<"\n";
                std::cout <<"inPosX: "<< inPosX<<" inPosY: "<< inPosY << "\n";
                #endif

                inPosX += !zeroX;
                const auto idx = unit * tp.patchSizeX * tp.patchSizeY
                                 + y * tp.patchSizeX + x;

                graph.connect(iPart, v["dIn"][idx]);

                Tensor oPart = dTf[ig][p][y][x].flatten().slice(slS, slE);

                graph.connect(v["dTf"][idx], oPart);
              }
              inPosY += !zeroY;
            }
          }
          numUnits -= unitsThisVertex;
        }
        numPatches -= patchesThisTile;
      }
      numZog -= zogThisTile;
    }
    numZig -= zigThisTile;
  }
  return Sequence(Execute(zCs), Execute(cs));
}



static Program dataTransform(Graph &graph,
                                const WgdTilePartition &tp,
                                const std::string layerName,
                                Tensor in,
                                Tensor dataTf) {
  unsigned numUnits = (tp.zi * tp.getNumPatches()
                       + WgdTilePartition::dUnitSize - 1)
                      / WgdTilePartition::dUnitSize;

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const unsigned numTiles = deviceInfo.getNumTiles();
  const unsigned numWorkers = deviceInfo.numWorkerContexts;

  ComputeSet dCs = graph.createComputeSet(layerName + ".dataTrf");
  ComputeSet zCs = graph.createComputeSet(layerName + ".zeros");

  unsigned unitsPerTile = (numUnits + numTiles - 1)/numTiles;

  for (unsigned tile = 0; tile < numTiles && numUnits; ++tile) {
    auto unitsThisTile = std::min(numUnits, unitsPerTile);
    const auto unitsPerVertex = (unitsPerTile + numWorkers - 1)/numWorkers;
    numUnits -= unitsThisTile;

    bool zeroTensorCreated = false;
    Tensor zeroVec;

    /* split units assigned to this tile over vertices */
    for (unsigned vertex = 0; vertex < numWorkers && unitsThisTile; ++vertex) {
      const auto unitsThisVertex =  std::min(unitsPerVertex, unitsThisTile);

      auto v = graph.addVertex(
                        dCs,
                        templateVertex("WgdDataTransform", tp.dType,
                                       tp.patchSizeX, tp.patchSizeY,
                                       tp.kernelX, tp.kernelY));
      graph.setFieldSize(v["dIn"],
                         unitsThisVertex * tp.patchSizeX * tp.patchSizeY);

      graph.setFieldSize(v["dTf"],
                         unitsThisVertex * tp.patchSizeX * tp.patchSizeY);

      graph.setTileMapping(v, tile);

      for (unsigned unit = 0; unit < unitsThisVertex; ++unit) {
        const auto sUnit = (tile * unitsPerTile + vertex * unitsPerVertex
                            + unit) * WgdTilePartition::dUnitSize;
        const auto ic = sUnit % tp.zic;
        const auto patch = (sUnit / tp.zic) % tp.getNumPatches();
        unsigned patchX, patchY, prepadX, postpadX, prepadY, postpadY;
        std::tie(patchX, patchY) = tp.getPatchIdx(patch);
        std::tie(prepadX, postpadX) = tp.getPaddingX(patchX);
        std::tie(prepadY, postpadY) = tp.getPaddingY(patchY);
        const auto ig = sUnit / (tp.zic * tp.getNumPatches());

        #if DEBUG_PRINT >= 2
        std::cout << "(DTF) unit " << sUnit  << " [" << ig << "][";
        std::cout << patchY << "][";
        std::cout << patchX << "][" << ic << "] on tile " << tile << ":";
        std::cout << vertex << std::endl;
        #endif


        if ((prepadX || prepadY || postpadX || postpadY)
             && !zeroTensorCreated) {
          zeroVec = graph.addTensor(tp.dType,
                                    {WgdTilePartition::dUnitSize},
                                    "zero");
          graph.setTileMapping(zeroVec, tile);

          auto v = graph.addVertex(zCs, templateVertex("Zero", tp.dType));
          graph.setInitialValue(v["dataPathWidth"],
                                deviceInfo.dataPathWidth);

          graph.connect(v["out"], zeroVec);
          graph.setTileMapping(v, tile);
          zeroTensorCreated = true;
        }

        unsigned slS = ic;
        unsigned slE = ic + WgdTilePartition::dUnitSize;


        auto inPosY = tp.getInpPosY(patchY);
        for (unsigned y = 0; y < tp.patchSizeY; ++y) {
          bool zeroY  = y < prepadY || (y >= tp.patchSizeY - postpadY);
          auto inPosX = tp.getInpPosX(patchX);
          for (unsigned x = 0; x < tp.patchSizeX; ++x) {
            bool zeroX = x < prepadX || (x >= tp.patchSizeX - postpadX);
            Tensor iPart = (zeroX || zeroY) ?
                            zeroVec :
                            in[ig][inPosY][inPosX].flatten().slice(slS, slE);

            inPosX += !zeroX;
            auto idx = unit * tp.patchSizeX * tp.patchSizeY
                       + y * tp.patchSizeX + x;
            graph.connect(iPart, v["dIn"][idx]);

            Tensor oPart = dataTf[ig][patch][y][x].flatten().slice(slS, slE);

            graph.connect(v["dTf"][idx], oPart);
            graph.setTileMapping(oPart, tile);
          }
          inPosY += !zeroY;
        }
      }
      unitsThisTile -= unitsThisVertex;
    }
  }
  return Sequence(Execute(zCs), Execute(dCs));
}


static std::vector<Tensor> allocateDataTfTensor(
                                  Graph &graph,
                                  const WgdTilePartition &tp) {
  std::vector<Tensor> dataTf;

  assert(tp.zic % WgdTilePartition::dUnitSize == 0);


  if (!tp.replicateDTf) {

    dataTf.resize(1);
    dataTf[0] = graph.addTensor(tp.dType,
                                {
                                 tp.zig,
                                 tp.getNumPatches(),
                                 tp.patchSizeY,
                                 tp.patchSizeX,
                                 tp.zic
                                },
                                "WgdDataTransform");
  } else {
    dataTf.resize(graph.getDevice().getDeviceInfo().getNumTiles());
  }

  return dataTf;
}


static Program computeDataTransform(Graph &graph,
                                  const WgdTilePartition &tp,
                                  const std::string layerName,
                                  Tensor prevAct,
                                  std::vector<Tensor> &dataTf) {

  return tp.replicateDTf ?
          dataTransform(graph, tp, layerName, prevAct, dataTf) :
          dataTransform(graph, tp, layerName, prevAct, dataTf[0]);
}


static Program accum(Graph &graph,
                     const WgdTilePartition &tp,
                     const std::string layerName,
                     std::vector<Tensor> &dataTf,
                     std::vector<Tensor> &kernelTf,
                     Tensor acc) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const unsigned numWorkers = deviceInfo.numWorkerContexts;

  ComputeSet cs = graph.createComputeSet(layerName + ".accum");
  ComputeSet zeroCS = graph.createComputeSet(layerName + ".zeroAccum");

  const char *baseClass = deviceInfo.sharedConvWeights ?
                                     "poplar::SupervisorVertex" :
                                     "poplar::Vertex";
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(tp.dType == "float");

  const auto numConvUnits = tp.dType == "float" ?
                              deviceInfo.fp32AccumConvUnitsPerTile :
                              deviceInfo.fp16AccumConvUnitsPerTile;

  unsigned numZig = tp.zig;
  for (unsigned zigTile = 0; zigTile < tp.tilesForZig; ++zigTile) {
    unsigned numZog = tp.zog;
    const unsigned zigThisTile = std::min(numZig, tp.zigPerTile);

    for (unsigned zogTile = 0; zogTile < tp.tilesForZog; ++zogTile) {
      unsigned numPatches = tp.getNumPatches();
      const unsigned zogThisTile = std::min(numZog, tp.zogPerTile);

      for (unsigned pTile = 0; pTile < tp.tilesForPatches; ++pTile) {
        const auto tile = zigTile * tp.tilesForZog * tp.tilesForPatches
                    + zogTile * tp.tilesForPatches
                    + pTile;

        /* number assigned this tile */
        const auto patchesThisTile = std::min(numPatches, tp.patchesPerTile);

        /* start indices */
        auto zigS = zigTile * tp.zigPerTile;
        auto zogS = zogTile * tp.zogPerTile;
        auto patchS = pTile * tp.patchesPerTile;

        /* we use accumulation of partials only if there is more than one input
         * group on a tile
         */
        bool accumulatePartials = zigThisTile > 1;

        if (accumulatePartials) {
          /* total elements to zero */
          auto numZeroElems = tp.patchSizeX * tp.patchSizeY * patchesThisTile
                              * zogThisTile;
          const auto numElemsPerVertex = (numZeroElems + numWorkers - 1)
                                          / numWorkers;

          /* divide zeroing over number of vertices */
          const auto reshapeRows = patchesThisTile * zogThisTile
                                   * tp.patchSizeY * tp.patchSizeX;
          Tensor zeroTen = acc.slice(
                                     {zogS, zigTile, patchS, 0, 0, 0},
                                     {zogS + zogThisTile, zigTile + 1,
                                      patchS + patchesThisTile,
                                      tp.patchSizeY, tp.patchSizeX,
                                      tp.zoc}).reshape({reshapeRows,
                                                        tp.zoc});

          for (unsigned vertex = 0; vertex < numWorkers &&  numZeroElems; ++vertex){
            const auto slS = vertex * numElemsPerVertex;

            const auto elemsThisVertex = std::min(numElemsPerVertex,
                                                numZeroElems);
            const auto slE = slS + elemsThisVertex;

            auto vZ = graph.addVertex(zeroCS,
                                      templateVertex("Zero2D", tp.dType),
                                      {{"out", zeroTen.slice(slS, slE)}});
            graph.setTileMapping(vZ, tile);
            const auto dataPathWidth = deviceInfo.dataPathWidth;
            graph.setInitialValue(vZ["dataPathWidth"], dataPathWidth);
            numZeroElems -= elemsThisVertex;
         }
        }

        for (unsigned og = zogS; og < zogS + zogThisTile; ++og) {
          /* all patches assigned to the same input channel group
           * and output channel groups can be processed together.
           * ceate a vertex for all patches (each patch has
           * patchSize * patchSize elements)
           */

          for (unsigned pencil = 0; pencil < tp.patchSizeX * tp.patchSizeY;
                        ++pencil) {
            const auto ptY = pencil / tp.patchSizeX;
            const auto ptX = pencil % tp.patchSizeX;

            /* create vertex for each pencil */
            auto v = graph.addVertex(
                                     cs,
                                     templateVertex("WgdPartials",
                                                     baseClass,
                                                     tp.dType,
                                                     accumulatePartials?
                                                          "true" : "false"));
            graph.setInitialValue(v["numWorkers"], numWorkers);
            graph.setInitialValue(v["numConvUnits"], numConvUnits);
            graph.setInitialValue(v["weightsPerConvUnit"],
                                    weightsPerConvUnit);
            graph.setFieldSize(v["wTf"], zigThisTile);
            graph.setFieldSize(v["dTf"], patchesThisTile * zigThisTile);
            graph.setFieldSize(v["partials"], patchesThisTile);
            graph.setTileMapping(v, tile);

            if (tp.replicateKTf) {
              graph.connect(
                kernelTf[tile].slice(
                {og - zogS, 0, ptY, ptX, 0, 0},
                {og - zogS + 1, zigThisTile, ptY + 1, ptX+1, tp.zoc, tp.zic}).
                reshape({zigThisTile, tp.zoc * tp.zic}), v["wTf"]);

            } else {
              graph.connect(
                kernelTf[0].slice(
                {og, zigS, ptY, ptX, 0, 0},
                {og + 1, zigS + zigThisTile, ptY + 1, ptX+1, tp.zoc, tp.zic}).
                reshape({zigThisTile, tp.zoc * tp.zic}), v["wTf"]);
            }

            #if DEBUG_PRINT >= 2
            std::cout << "(Accum) tile: " << tile <<   " ig: " << ig;
            std::cout << " og: " << og << " patch: " << patchS;
            std::cout  << ":" << patchesThisTile << std::endl;
            #endif

            if (tp.replicateDTf) {
              graph.connect(
                dataTf[tile].slice(
                  {0, 0, ptY, ptX, 0},
                  {zigThisTile, patchesThisTile, ptY + 1, ptX + 1, tp.zic}).
                  reshape({zigThisTile * patchesThisTile, tp.zic}), v["dTf"]);

            } else {
              graph.connect(
                dataTf[0].slice(
                  {zigS, patchS, ptY, ptX, 0},
                  {zigS + zigThisTile, patchS + patchesThisTile, ptY + 1,
                   ptX + 1, tp.zic}).
                  reshape({zigThisTile * patchesThisTile, tp.zic}), v["dTf"]);
            }

            Tensor out = acc.slice(
              {og, zigTile, patchS, ptY, ptX, 0},
              {og + 1, zigTile + 1, patchS + patchesThisTile, ptY + 1,
               ptX + 1, tp.zoc}).
              reshape({patchesThisTile, tp.zoc});

            graph.connect(v["partials"], out);

            graph.setTileMapping(out, tile);
          }
        }
        numPatches -= patchesThisTile;
      }
      numZog -= zogThisTile;
    }
    numZig -= zigThisTile;
  }

  Sequence prog;
  if (!graph.getComputeSet(zeroCS).empty()) {
    prog.add(Execute(zeroCS));
  }
  prog.add(Execute(cs));
  return prog;

}


static Program reduce(Graph &graph,
               const WgdTilePartition &tp,
               const std::string layerName,
               Tensor acc,
               Tensor red) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const unsigned numWorkers = deviceInfo.numWorkerContexts;

  ComputeSet cs = graph.createComputeSet(layerName + ".reduce");

  for (unsigned tile = 0, patch = 0; patch < tp.getTotalOutputPatches();
               ++tile, patch += tp.outPatchesPerTile) {
    auto patchesThisTile = std::min(tp.outPatchesPerTile,
                                    tp.getTotalOutputPatches() - patch);

    auto depth = std::min(tp.zocOut, tp.zoc);
    auto zFactor = tp.zocOut/depth;

    /* each element is of size depth */
    auto totalElems = patchesThisTile * zFactor
                      * tp.patchSizeX * tp.patchSizeY;

    auto elemsPerVertex = (totalElems + numWorkers - 1)
                          / numWorkers;

    for (unsigned vertex = 0; vertex < numWorkers && totalElems; ++vertex) {
      auto elemsThisVertex = std::min(elemsPerVertex, totalElems);

      auto v = graph.addVertex(cs,
                               templateVertex("WgdReduce", tp.dType,
                                              tp.patchSizeX, tp.patchSizeY));
      graph.setTileMapping(v, tile);
      graph.setFieldSize(v["inPartial"], elemsThisVertex * tp.tilesForZig);
      graph.setFieldSize(v["outPartial"], elemsThisVertex);

      #if DEBUG_PRINT >= 2
      std::cout << "Reduce:: tile : "<< tile << "   vertex : ";
      std::cout << vertex << " elems this vertex :";
      std::cout << elemsThisVertex << std::endl;

      #endif

      for (unsigned elem = 0; elem < elemsThisVertex; ++elem) {
        const auto thisElem = patch * zFactor *  tp.patchSizeX * tp.patchSizeY
                              + vertex * elemsPerVertex + elem;

        const auto absPatch = thisElem /
                             (zFactor * tp.patchSizeX * tp.patchSizeY);
        const auto thisPatch = absPatch % tp.getNumPatches();
        const auto thisOutCh  = (thisElem /
                                          (tp.patchSizeX * tp.patchSizeY
                                           * tp.getNumPatches() * zFactor))
                                * tp.zocOut
                                + (thisElem % zFactor) * depth;

        const auto ogOut = thisOutCh / tp.zocOut;
        const auto ogIn = thisOutCh / tp.zoc;
        const auto slSIn = thisOutCh % depth;
        const auto slEIn = slSIn + depth;
        const auto slSOut = thisOutCh % tp.zocOut;
        const auto slEOut = slSOut + depth;
        const auto patchElem = (thisElem / zFactor)
                               % (tp.patchSizeX * tp.patchSizeY);
        const auto y = patchElem / tp.patchSizeX;
        const auto x = patchElem % tp.patchSizeX;

        for (unsigned ig = 0; ig < tp.tilesForZig; ++ig) {

          #if DEBUG_PRINT >= 2
          std::cout << "Input: [" << ogIn << "][" << ig;
          std::cout << "][" << thisPatch << "][" << y << "][" << x;
          std::cout << "] " << slSIn << ":" << slEIn << " -> ";
          std::cout << (elem * ig) << std::endl;
          #endif
          graph.connect(acc[ogIn][ig][thisPatch][y][x].flatten().slice(
                        slSIn, slEIn),
                        v["inPartial"][elem * tp.tilesForZig + ig]);

        }


        #if DEBUG_PRINT >= 2

        std::cout << "Output: " << elem << " <- [" << ogOut;
        std::cout << "][" << thisPatch << "][" << y << "][" << x;
        std::cout << "]" << slSOut << ":" << slEOut << std::endl;

        #endif

        graph.connect(v["outPartial"][elem],
                      red[ogOut][thisPatch][y][x].flatten().slice(
                      slSOut, slEOut));
        graph.setTileMapping(red[ogOut][thisPatch][y][x].flatten().slice(
                      slSOut, slEOut), tile);

      }
      totalElems -= elemsThisVertex;
    }
  }
  return Execute(cs);
}


static Program inverseTransform(Graph &graph,
                                const WgdTilePartition &tp,
                                const std::string layerName,
                                Tensor in,
                                Tensor out) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const unsigned numWorkers = deviceInfo.numWorkerContexts;

  ComputeSet cs = graph.createComputeSet(layerName + ".invTransform");

  for (unsigned tile = 0, patch = 0; patch < tp.getTotalOutputPatches();
               ++tile, patch += tp.outPatchesPerTile) {
    auto tuplesPerZoc = tp.zocOut / WgdTilePartition::iUnitSize;

    auto tuplesThisTile = std::min(tp.outPatchesPerTile,
                                    tp.getTotalOutputPatches() - patch)
                          * tuplesPerZoc;

    /* split across number of workers */

    const auto tuplesPerVertex = (tuplesThisTile + numWorkers - 1)
                                            / numWorkers;
    for (unsigned vertex = 0; vertex < numWorkers && tuplesThisTile;
                  ++vertex) {

      auto tuplesThisVertex = std::min(tuplesPerVertex, tuplesThisTile);
      auto v = graph.addVertex(cs,
                               templateVertex("WgdInverseTransform", tp.dType,
                                              tp.patchSizeX, tp.patchSizeY,
                                              tp.kernelX, tp.kernelY));
      graph.setFieldSize(v["dTf"], tp.patchSizeX * tp.patchSizeY
                                   * tuplesThisVertex);
      graph.setFieldSize(v["dOut"], tp.getNumOutputsPerPatchY()
                                    * tp.getNumOutputsPerPatchX()
                                    * tuplesThisVertex);
      graph.setTileMapping(v, tile);


      for (unsigned tuple = 0; tuple < tuplesThisVertex; ++tuple) {
        auto thisTuple = patch * tuplesPerZoc + vertex * tuplesPerVertex
                         + tuple;

        auto absPatch = thisTuple / tuplesPerZoc;
        auto thisPatch = absPatch % tp.getNumPatches();
        auto ogOut = absPatch/tp.getNumPatches();


        for (unsigned y = 0; y < tp.patchSizeY; ++y) {
          for (unsigned x = 0; x < tp.patchSizeX; ++x) {
            auto idxIn = tuple * tp.patchSizeX * tp.patchSizeY +
                         y * tp.patchSizeX + x ;
            auto slS = (thisTuple % tuplesPerZoc) * WgdTilePartition::iUnitSize;
            auto slE = slS + WgdTilePartition::iUnitSize;
            graph.connect(in[ogOut][thisPatch][y][x].flatten().slice(slS, slE),
                              v["dTf"][idxIn]);
            #if DEBUG_PRINT >= 2
            std::cout << "Inv: tile : "<<tile<< " vertex : "<<vertex<<std::endl;
            std::cout << "input: [" << ogOut <<"]["<<thisPatch<<"]["<<y<<"][";
            std::cout << x<<"] -> " << idxIn << std::endl;
            #endif
          }
        }

        for (unsigned y = 0; y < tp.getNumOutputsPerPatchY(); ++y) {
          for (unsigned x = 0; x < tp.getNumOutputsPerPatchX(); ++x) {
            auto idxOut = tuple * tp.getNumOutputsPerPatchY()
                            * tp.getNumOutputsPerPatchX()
                          + y * tp.getNumOutputsPerPatchX() + x;

            #if DEBUG_PRINT >= 2
            std::cout << "output: [" << ogOut <<"]["<<thisPatch<<"]["<<y<<"][";
            std::cout << x<<"] <- " << idxOut << std::endl;
            #endif
            auto slS = (thisTuple % tuplesPerZoc)*4;
            auto slE = slS + 4;

            graph.connect(v["dOut"][idxOut],
                             out[ogOut]
                                [thisPatch][y][x].flatten().slice(slS, slE));

            graph.setTileMapping(out[ogOut]
                                    [thisPatch]
                                    [y][x].flatten().slice(slS, slE), tile);
          }
        }
      }
      tuplesThisTile -= tuplesThisVertex;
    }
  }
  return Execute(cs);
}


static Program complete(Graph &graph,
                        const WgdTilePartition &tp,
                        const std::string layerName,
                        Tensor in,
                        Tensor act,
                        Tensor bias,
                        NonLinearityType nonLinearityType) {
  ComputeSet cs = graph.createComputeSet(layerName + ".complete");

  for (unsigned tile = 0, patch = 0; patch < tp.getTotalOutputPatches();
               ++tile, patch += tp.outPatchesPerTile) {
    auto patchesThisTile = std::min(tp.outPatchesPerTile,
                                    tp.getTotalOutputPatches() - patch);


    for (unsigned p = 0; p < patchesThisTile; ++p) {
      auto absPatch = patch + p;
      auto thisPatch = absPatch % tp.getNumPatches();
      auto ogOut = absPatch/tp.getNumPatches();

      auto v = graph.addVertex(cs, templateVertex("WgdConvComplete", tp.dType));
      graph.setTileMapping(v, tile);

      /* keep track of number of elements as some may be skipped */
      unsigned elem = 0;

      unsigned xPosS, yPosS;
      std::tie(xPosS, yPosS) = tp.getOutIdxForPatch(thisPatch);

      auto xPosE = std::min(xPosS + tp.getNumOutputsPerPatchX(),
                            tp.getOutputSizeX());
      auto yPosE = std::min(yPosS + tp.getNumOutputsPerPatchY(),
                            tp.getOutputSizeY());

      #if DEBUG_PRINT >= 2
      std::cout << "(complete) tile: " << tile << " patch : " << thisPatch;
      std::cout << " xPosS:xPosE " << xPosS << " " << xPosE;
      std::cout << " yPosS:yPosE " << yPosS << " " << yPosE << std::endl;
      #endif


      for (unsigned y = yPosS; y < yPosE; ++y) {
        for (unsigned x = xPosS; x < xPosE; ++x) {

          #if DEBUG_PRINT >= 2
          std::cout << "in[" << ogOut << "][" << thisPatch << "][";
          std::cout << (y-yPosS) << "][" << (x-xPosS) << "] -> ";
          std::cout << elem << std::endl;
          #endif

          #if DEBUG_PRINT >= 2
          std::cout << "act[" << ogOut << "][" << y << "][";
          std::cout << x << "] -> " << elem << std::endl;
          #endif

          graph.connect(in[ogOut][thisPatch][y-yPosS][x-xPosS].flatten(),
                        v["dIn"][elem]);
          graph.connect(v["act"][elem], act[ogOut][y][x].flatten());
          ++elem;
        }
      }

      #if DEBUG_PRINT >= 2
      std::cout << "bias " << (ogOut * tp.zocOut) << ":";
      std::cout << ((ogOut + 1) * tp.zocOut) << " -> " << p << std::endl;
      #endif
      graph.connect(bias.slice(ogOut * tp.zocOut, (ogOut + 1) * tp.zocOut),
                    v["bias"]);

      graph.setFieldSize(v["dIn"], elem);
      graph.setFieldSize(v["act"], elem);
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
    }
  }

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

#if DEBUG_PRINT >= 1
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

  /* assumption that number of input channels per group must be same 
   * for input activations and weights 
   */
  assert(in.dim(0) == weights.dim(1));
  assert(in.dim(3) == weights.dim(5));


  WgdTilePartition tp(padding, padding,
                      xDim, yDim,
                      patchSizeX, patchSizeY,
                      kernelSize, kernelSize,
                      weights.dim(1) * weights.dim(5),
                      weights.dim(0) * weights.dim(4),
                      dType);

  tp.tilePartition(weights.dim(5),
                   weights.dim(4),
                   activations.dim(3),
                   nonLinearityType,
                   graph.getDevice().getDeviceInfo());

  auto prog = Sequence();

  const auto layerName = "Wgd Conv" + std::to_string(kernelSize) 
                         + "x" + std::to_string(kernelSize) + ".fwd";


  std::vector<Tensor> dataTf = allocateDataTfTensor(graph, tp);
  prog.add(computeDataTransform(graph, tp, layerName, in, dataTf));


  std::vector<Tensor> kernelTf = allocateKernelTfTensor(graph, tp);
  prog.add(computeKernelTransform(graph, tp, layerName, weights, kernelTf));

  /* accumulate across tiles */
  Tensor accumTen = graph.addTensor(dType,
                                   {
                                     tp.zog,
                                     tp.tilesForZig,
                                     tp.getNumPatches(),
                                     patchSizeY,
                                     patchSizeX,
                                     tp.zoc
                                   },
                                   "WgdAccumulate");

  prog.add(accum(graph, tp, layerName, dataTf, kernelTf, accumTen));


  Tensor invTfIn = graph.addTensor(dType,
                                   {
                                     tp.zo/tp.zocOut,
                                     tp.getNumPatches(),
                                     patchSizeY,
                                     patchSizeX,
                                     tp.zocOut
                                   },
                                   "WgdInvTrfIn");

  prog.add(reduce(graph, tp, layerName, accumTen, invTfIn));


  Tensor invTfOut = graph.addTensor(dType,
                                   {
                                     tp.zo/tp.zocOut,
                                     tp.getNumPatches(),
                                     tp.getNumOutputsPerPatchY(),
                                     tp.getNumOutputsPerPatchX(),
                                     tp.zocOut
                                   },
                                   "WgdInvTrfOut");

  prog.add(inverseTransform(graph, tp, layerName, invTfIn, invTfOut));

  prog.add(complete(graph, tp, layerName, invTfOut, activations, biases,
                    nonLinearityType));

  return prog;
}

} // namespace conv
