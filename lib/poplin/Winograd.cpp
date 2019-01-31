#include "Winograd.hpp"
#include "poplin/Convolution.hpp"
#include "poplin/ConvUtil.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_support/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <iostream>
#include <array>
#include <numeric>
#include <utility>


#define DEBUG_PRINT 0


using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace poplin {

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
                  unsigned zi, unsigned zo, Type dType,
                  Type partialsType) :

                  padX(padX), padY(padY), dimInX(dimInX), dimInY(dimInY),
                  patchSizeX(patchSizeX), patchSizeY(patchSizeY),
                  kernelX(kernelX), kernelY(kernelY),
                  zi(zi), zo(zo), dType(dType), partialsType(partialsType) {
  }
  static constexpr unsigned dUnitSize = 4;
  static constexpr unsigned kUnitSize = 4;
  static constexpr unsigned iUnitSize = 4;

  const unsigned padX, padY;
  const unsigned dimInX, dimInY;
  const unsigned patchSizeX, patchSizeY;
  const unsigned kernelX, kernelY;
  const unsigned zi, zo;
  const Type dType;
  const Type partialsType;


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
                     const ConvOptions &options,
                     const Target &target);

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

  /* Tile mapping for input patch
   */
  unsigned getTileForInputPatch(
            unsigned zigTile,
            unsigned zogTile,
            unsigned pTile) const {

    return zigTile * tilesForZog * tilesForPatches
            + zogTile * tilesForPatches
            + pTile;
  }


  std::pair<unsigned, unsigned> getOutPatchInfo(unsigned tile) const {
    /*
     * Input patches are mapped such that all patches for a some input and
     * output channel groups are mapped first
     */
    const auto inputTile = tile % tilesForPatches;
    const auto patchS = inputTile * patchesPerTile;
    const auto patchE = std::min(getNumPatches(), patchS + patchesPerTile);

    const auto patches = patchE - patchS;

    /* the input patches per tile are sub-divided such that they are distributed
     * across tilesForZig.
     */
    const auto outPatchesPerTile = (patches + tilesForZig - 1)/tilesForZig;

    /* get to which tile group the output patch is assigned to */
    const auto tileGroup = tile/(tilesForZog * tilesForPatches);

    /* "patches" are distributed evenly with constraint the there can be
     * at most "outPatchesPerTile patches assigned
     */
    const auto numPatchesInTile =
          std::min(patches - std::min(tileGroup * outPatchesPerTile, patches),
                   outPatchesPerTile);

    /* it is possible that this gives the wrong result but the number of
     * patches in tile will be zero in such a case
     */
    const auto startPatchInTile = patchS + outPatchesPerTile * tileGroup;

    return std::make_pair(startPatchInTile, numPatchesInTile);
  }

  std::pair<unsigned, unsigned> getOutZogInfo(unsigned tile) const {
    const auto zogS = std::min(
                        ((tile / tilesForPatches) % tilesForZog) * zogPerTile,
                        zog);
    const auto numZog = std::min(zog - zogS, zogPerTile);

    return std::make_pair(zogS, numZog);
  }
};

uint64_t WgdTilePartition::tilePartition(
              unsigned inpZic,
              unsigned weightsZoc,
              unsigned outZoc,
              const ConvOptions &options,
              const Target &target) {

  const unsigned numTiles = options.getNumTiles();
  const unsigned numWorkers = target.getNumWorkerContexts();
  const auto numPatches = getNumPatches();
  const auto isFloat = dType == FLOAT;
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

  /* exchange transfer bytes per cycle */
  const unsigned eBytesPerCycle = target.getExchangeBytesPerCycle();

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

        /* size in bytes of each element of untransformed kernel */
        const unsigned kAtomSize   = isFloat ? 4 : 2;

        /* size in bytes of each element of transformed kernel :
         * numkTfElems represent the number of transforms. A group of elements
         * of size kUnitSize is called an unit
         */
        const unsigned kTfAtomSize = isFloat ? 4 : 2;
        const unsigned numKTfElems = zigPerTile * zogPerTile * zic * zoc;
        unsigned numKTfBlocks1 = (numKTfElems + numWorkers - 1)/numWorkers;

        Cost ecKTf1 = (numKTfElems * kernelX * kernelY * kAtomSize)
                      / eBytesPerCycle;
        Cost ccKTf1 = getWgdKernelTransformCycles(
                                                  numKTfBlocks1,
                                                  isFloat)
                      * numWorkers;

        const unsigned numKTfUnits = (zi * zo/kUnitSize
                                       + numTiles - 1)/numTiles;
        const unsigned numKTfBlocks2 = (numKTfUnits + numWorkers - 1)
                                       / numWorkers;
        Cost ecKTf2 = (numKTfElems * patchSizeX * patchSizeY * kTfAtomSize)
                      / eBytesPerCycle;
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

        /* Size in bytes of each element of transformed data */
        const unsigned dTfAtomSize = isFloat ? 4 : 2;

        /* Size in bytes of each element of input data*/
        const unsigned dAtomSize = isFloat ? 4 : 2;

        /* numDtfElems is the number of transforms and a group containing
         * dUnitSize is called an unit
         */
        const unsigned numDTfElems = patchesPerTile * zigPerTile * zic;
        const unsigned numDTfBlocks1 = (numDTfElems + numWorkers - 1)
                                       / numWorkers;
        const unsigned numDTfUnits = (numPatches * zi/dUnitSize
                                       + numTiles - 1)/numTiles;
        const unsigned numDTfBlocks2 = (numDTfUnits + numWorkers -1)
                                        / numWorkers;

        Cost ecDTf1 = (numDTfElems * patchSizeX * patchSizeY * dAtomSize)
                       / eBytesPerCycle;
        Cost ccDTf1 = getWgdDataTransformCycles(numDTfBlocks1, isFloat)
                      * numWorkers;
        Cost ecDTf2 = (numDTfElems * patchSizeX * patchSizeY * dTfAtomSize
                      + dUnitSize * numDTfUnits * patchSizeX * patchSizeY
                        * dAtomSize) / eBytesPerCycle;
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
        Cost ecAcc = 0;
        const auto weightsPerConvUnit =
                                target.getWeightsPerConvUnit(isFloat);
        const auto numConvUnits = isFloat ?
                                target.getFp32InFp32OutConvUnitsPerTile() :
                                target.getFp16InFp16OutConvUnitsPerTile();
        const auto convUnitCoeffLoadBytesPerCycle =
                              target.getConvUnitCoeffLoadBytesPerCycle();

        Cost ccAcc;
        unsigned numBlocks = patchSizeX * patchSizeY * zogPerTile
                             * zigPerTile;
        ccAcc = getWgdAccumCycles(numBlocks, patchesPerTile, zic,
                                  zoc, numWorkers, numConvUnits,
                                  weightsPerConvUnit,
                                  convUnitCoeffLoadBytesPerCycle,
                                  isFloat);

        std::get<ACCUM>(cc) = ccAcc;
        std::get<ACCUM>(ec) = ecAcc;

        #if DEBUG_PRINT >= 3
        std::cout << "Accum cost: ec: " << ecAcc;
        std::cout << " cc: " << ccAcc << std::endl;
        #endif

        /* patches  on each tile are redistributed across number of tiles
         * over which inout groups are spread
         */
          const auto outPatchesPerTileMin = patchesPerTile/tilesForZig;
          const auto outPatchesPerTileMax = (patchesPerTile + tilesForZig - 1)
                                             /tilesForZig;


        /* exchange cost and compute cost is zero if all input channel groups
         * are allocated on  single tile
         */
        Cost ccRed = 0;
        Cost ecRed = 0;
        if (tilesForZig > 1) {

          ecRed = zoc * zogPerTile
                       * (patchesPerTile - outPatchesPerTileMin)
                       * patchSizeY * patchSizeY * dAtomSize/eBytesPerCycle;


          unsigned numRedBlocks = (outPatchesPerTileMax * zogPerTile
                                   * patchSizeX
                                   * patchSizeY + numWorkers - 1) / numWorkers;
          ccRed = getWgdReduceCycles(numRedBlocks * zoc, tilesForZig,
                                     partialsType == FLOAT) * numWorkers;
        }

        std::get<REDUCTION>(cc) = ccRed;
        std::get<REDUCTION>(ec) = ecRed;

        #if DEBUG_PRINT >= 3
        std::cout << "Red cost: ec: " << ecRed;
        std::cout << " cc: " << ccRed << std::endl;
        #endif

        /* Inverse kernel transform doesn't require exchange */
        /* size in bytes of inverse transformed data */
        //const unsigned iTfAtomSize = isFloat ? 4 : 2;
        Cost ecITf = 0;
        const unsigned numITfUnits = outPatchesPerTileMax * zoc * zogPerTile
                                     / iUnitSize;
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


        /* size in bytes of layer output */
        const unsigned cAtomSize = isFloat ? 4 : 2;
        Cost ecComp = (outPatchesPerTileMax * zocOut
                       * getOverlapX() * getOverlapY() * cAtomSize)
                       / eBytesPerCycle;

        std::get<COMPLETE>(ec) = ecComp;

        Cost ccComp = outPatchesPerTileMax
                      * getWgdCompleteCycles(
                              zocOut * getOverlapX() * getOverlapY(),
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


static void wgdMapWeights(
              Graph &graph,
              const ConvOptions &options,
              WgdTilePartition &tp,
              Tensor weights) {
  unsigned numUnits = (tp.zi * tp.zo + WgdTilePartition::kUnitSize - 1)
                      / WgdTilePartition::kUnitSize;

  const unsigned numTiles = options.getNumTiles();

  assert(tp.zic % WgdTilePartition::kUnitSize == 0);

  unsigned unitsPerTile = (numUnits + numTiles - 1)/numTiles;

  for (unsigned tile = 0; tile < numTiles && numUnits; ++tile) {
    auto unitsThisTile = std::min(numUnits, unitsPerTile);


    numUnits -= unitsThisTile;
    for (auto unit = 0U; unit < unitsThisTile; ++unit) {
      auto thisUnit = (tile * unitsPerTile + unit)
                      * WgdTilePartition::kUnitSize;

      const auto ig = (thisUnit / (tp.zic * tp.zoc)) % tp.zig;
      const auto og = thisUnit / (tp.zoc * tp.zic * tp.zig);

      const auto slS = thisUnit
                       - thisUnit/(tp.zic * tp.zoc) * (tp.zic * tp.zoc);
      const auto oc  = (slS / tp.zic) % tp.zoc;

      const auto icS = slS % tp.zic;

      Tensor wPart = weights.slice(
          {og, ig, 0, 0, oc, icS},
          {og + 1, ig + 1, tp.kernelY, tp.kernelX, oc + 1,
            icS + WgdTilePartition::kUnitSize});

      graph.setTileMapping(wPart, tile);

    }
  }
}

static Program kernelTransform(
              Graph &graph,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor weights,
              std::vector<Tensor> &kTfMapping) {

  ComputeSet cs = graph.addComputeSet(layerName + "/KernelTrf");
  const auto &target = graph.getTarget();
  const unsigned numWorkers = target.getNumWorkerContexts();

  unsigned numZig = tp.zig;
  for (unsigned zigTile = 0; zigTile < tp.tilesForZig; ++zigTile) {

    unsigned numZog = tp.zog;
    const unsigned zigThisTile = std::min(numZig, tp.zigPerTile);

    for (unsigned zogTile = 0; zogTile < tp.tilesForZog; ++zogTile) {

      unsigned numPatches = tp.getNumPatches();
      const unsigned zogThisTile = std::min(numZog, tp.zogPerTile);

      for (unsigned pTile = 0; pTile < tp.tilesForPatches; ++pTile) {

        const auto tile = tp.getTileForInputPatch(zigTile, zogTile, pTile);

        /* number assigned this tile */
        const auto patchesThisTile = std::min(numPatches, tp.patchesPerTile);

        if (!patchesThisTile)
          continue;

        /* start indices */
        const auto zigS = zigTile * tp.zigPerTile;
        const auto zogS = zogTile * tp.zogPerTile;

        auto numUnits = (tp.zic * tp.zoc + WgdTilePartition::kUnitSize - 1)
                        / WgdTilePartition::kUnitSize;

        Tensor kTf = graph.addVariable(tp.dType,
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
                      templateVertex("poplin::WgdKernelTransform", tp.dType,
                                     tp.patchSizeX, tp.patchSizeY,
                                     tp.kernelX, tp.kernelY));

          graph.connect(v["wIn"], inp);
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


static Program kernelTransform(
              Graph &graph,
              const ConvOptions &options,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor weights,
              Tensor kernelTf) {
  unsigned numUnits = (tp.zi * tp.zo + WgdTilePartition::kUnitSize - 1)
                      / WgdTilePartition::kUnitSize;
  const auto &target = graph.getTarget();

  const unsigned numTiles = options.getNumTiles();
  const unsigned numWorkers = target.getNumWorkerContexts();

  ComputeSet cs = graph.addComputeSet(layerName + "/KernelTrf");

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
                            templateVertex("poplin::WgdKernelTransform",
                                           tp.dType,
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
            graph.connect(v["wIn"][unit * tp.kernelX * tp.kernelY
                                   + y * tp.kernelX + x],
                          weights[og][ig][y][x].flatten().slice(slS, slE));
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
    kernelTf[0] = graph.addVariable(tp.dType,
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
    kernelTf.resize(graph.getTarget().getNumTiles());
  }

  return kernelTf;
}


static Program computeKernelTransform(
              Graph &graph,
              const ConvOptions &options,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor weights,
              std::vector<Tensor> &kernelTf) {
  return tp.replicateKTf ?
            kernelTransform(graph, tp, layerName, weights, kernelTf) :
            kernelTransform(graph, options, tp, layerName, weights,
                            kernelTf[0]);
}



static Program dataTransform(
              Graph &graph,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor in,
              std::vector<Tensor> &dTfMapping) {

  ComputeSet cs = graph.addComputeSet(layerName + "/DataTrf");
  ComputeSet zCs = graph.addComputeSet(layerName + "/Zeros");

  const auto &target = graph.getTarget();
  const unsigned numWorkers = target.getNumWorkerContexts();

  unsigned numZig = tp.zig;
  for (unsigned zigTile = 0; zigTile < tp.tilesForZig; ++zigTile) {
    unsigned numZog = tp.zog;
    const unsigned zigThisTile = std::min(numZig, tp.zigPerTile);

    for (unsigned zogTile = 0; zogTile < tp.tilesForZog; ++zogTile) {
      unsigned numPatches = tp.getNumPatches();
      const unsigned zogThisTile = std::min(numZog, tp.zogPerTile);

      for (unsigned pTile = 0; pTile < tp.tilesForPatches; ++pTile) {

        const auto tile = tp.getTileForInputPatch(zigTile, zogTile, pTile);


        /* number assigned this tile */
        const auto patchesThisTile = std::min(numPatches, tp.patchesPerTile);

        if (!patchesThisTile)
          continue;

        /* start indices */
        const auto zigS = zigTile * tp.zigPerTile;
        const auto patchS = pTile * tp.patchesPerTile;

        Tensor dTf = graph.addVariable(
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
                                   templateVertex("poplin::WgdDataTransform",
                                       tp.dType,
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
              zeroVec = graph.addVariable(tp.dType,
                                          {WgdTilePartition::dUnitSize},
                                          "zero");
              graph.setTileMapping(zeroVec, tile);

              auto vZ = graph.addVertex(zCs,
                                        templateVertex("popops::Zero",
                                                       tp.dType));
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

                graph.connect(v["dIn"][idx], iPart);

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


static Program dataTransform(
              Graph &graph,
              const ConvOptions &options,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor in,
              Tensor dataTf) {
  unsigned numUnits = (tp.zi * tp.getNumPatches()
                       + WgdTilePartition::dUnitSize - 1)
                      / WgdTilePartition::dUnitSize;

  const auto &target = graph.getTarget();
  const unsigned numTiles = options.getNumTiles();
  const unsigned numWorkers = target.getNumWorkerContexts();

  ComputeSet dCs = graph.addComputeSet(layerName + "/DataTrf");
  ComputeSet zCs = graph.addComputeSet(layerName + "/Zeros");

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
                        templateVertex("poplin::WgdDataTransform", tp.dType,
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
          zeroVec = graph.addVariable(tp.dType,
                                      {WgdTilePartition::dUnitSize},
                                      "zero");
          graph.setTileMapping(zeroVec, tile);

          auto v = graph.addVertex(zCs, templateVertex("popops::Zero",
                                                       tp.dType));

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
            graph.connect(v["dIn"][idx], iPart);

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
    dataTf[0] = graph.addVariable(tp.dType,
                                  {
                                   tp.zig,
                                   tp.getNumPatches(),
                                   tp.patchSizeY,
                                   tp.patchSizeX,
                                   tp.zic
                                  },
                                  "WgdDataTransform");
  } else {
    dataTf.resize(graph.getTarget().getNumTiles());
  }

  return dataTf;
}


static Program computeDataTransform(
              Graph &graph,
              const ConvOptions &options,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor prevAct,
              std::vector<Tensor> &dataTf) {

  return tp.replicateDTf ?
          dataTransform(graph, tp, layerName, prevAct, dataTf) :
          dataTransform(graph, options, tp, layerName, prevAct, dataTf[0]);
}


static Program accum(
              Graph &graph,
              const WgdTilePartition &tp,
              const std::string layerName,
              std::vector<Tensor> &dataTf,
              std::vector<Tensor> &kernelTf,
              Tensor acc) {
  ComputeSet cs = graph.addComputeSet(layerName + "/Accum");

  unsigned numZig = tp.zig;
  for (unsigned zigTile = 0; zigTile < tp.tilesForZig; ++zigTile) {
    unsigned numZog = tp.zog;
    const unsigned zigThisTile = std::min(numZig, tp.zigPerTile);

    for (unsigned zogTile = 0; zogTile < tp.tilesForZog; ++zogTile) {
      unsigned numPatches = tp.getNumPatches();
      const unsigned zogThisTile = std::min(numZog, tp.zogPerTile);

      for (unsigned pTile = 0; pTile < tp.tilesForPatches; ++pTile) {

        const auto tile = tp.getTileForInputPatch(zigTile, zogTile, pTile);

        /* number assigned this tile */
        const auto patchesThisTile = std::min(numPatches, tp.patchesPerTile);

        /* start indices */
        auto zigS = zigTile * tp.zigPerTile;
        auto zogS = zogTile * tp.zogPerTile;
        auto patchS = pTile * tp.patchesPerTile;

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
                                     templateVertex("poplin::WgdPartials",
                                                     tp.dType));

            graph.setFieldSize(v["wTf"], zigThisTile);
            graph.setFieldSize(v["dTf"], patchesThisTile * zigThisTile);
            graph.setFieldSize(v["partials"], patchesThisTile);
            graph.setTileMapping(v, tile);

            if (tp.replicateKTf) {
              graph.connect(v["wTf"],
                kernelTf[tile].slice(
                {og - zogS, 0, ptY, ptX, 0, 0},
                {og - zogS + 1, zigThisTile, ptY + 1, ptX+1, tp.zoc, tp.zic}).
                reshape({zigThisTile, tp.zoc * tp.zic}));

            } else {
              graph.connect(v["wTf"],
                kernelTf[0].slice(
                {og, zigS, ptY, ptX, 0, 0},
                {og + 1, zigS + zigThisTile, ptY + 1, ptX+1, tp.zoc, tp.zic}).
                reshape({zigThisTile, tp.zoc * tp.zic}));
            }

            #if DEBUG_PRINT >= 2
            std::cout << "(Accum) tile: " << tile <<   " ig: " << ig;
            std::cout << " og: " << og << " patch: " << patchS;
            std::cout  << ":" << patchesThisTile << std::endl;
            #endif

            if (tp.replicateDTf) {
              graph.connect(v["dTf"],
                dataTf[tile].slice(
                  {0, 0, ptY, ptX, 0},
                  {zigThisTile, patchesThisTile, ptY + 1, ptX + 1, tp.zic}).
                  reshape({zigThisTile * patchesThisTile, tp.zic}));

            } else {
              graph.connect(v["dTf"],
                dataTf[0].slice(
                  {zigS, patchS, ptY, ptX, 0},
                  {zigS + zigThisTile, patchS + patchesThisTile, ptY + 1,
                   ptX + 1, tp.zic}).
                  reshape({zigThisTile * patchesThisTile, tp.zic}));
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
  prog.add(Execute(cs));
  return prog;
}


static Program reduce(
              Graph &graph,
              const ConvOptions &options,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor acc,
              Tensor red) {
  const auto &target = graph.getTarget();
  const unsigned numWorkers = target.getNumWorkerContexts();

  ComputeSet cs = graph.addComputeSet(layerName + "/Reduce");

  for (unsigned tile = 0; tile < options.getNumTiles(); ++tile) {

    /* get information on patches assigned to this tile */
    unsigned patchS, patchesThisTile, zogS, numZog;
    std::tie(patchS, patchesThisTile) = tp.getOutPatchInfo(tile);
    std::tie(zogS, numZog) = tp.getOutZogInfo(tile);


    /* each element is of size depth */
    auto totalElems = patchesThisTile * tp.patchSizeX * tp.patchSizeY * numZog;

    const auto elemsPerVertex = (totalElems + numWorkers - 1) / numWorkers;

    for (unsigned vertex = 0; vertex < numWorkers && totalElems; ++vertex) {
      const auto elemsThisVertex = std::min(elemsPerVertex, totalElems);

      auto v = graph.addVertex(cs,
                               templateVertex("poplin::WgdReduce", tp.dType,
                                              tp.patchSizeX, tp.patchSizeY));
      graph.setTileMapping(v, tile);
      graph.setFieldSize(v["inPartial"], elemsThisVertex * tp.tilesForZig);
      graph.setFieldSize(v["outPartial"], elemsThisVertex);

      #if DEBUG_PRINT >= 2
      std::cout << "Reduce:: tile : "<< tile << "   vertex : ";
      std::cout << vertex << " elems this vertex :";
      std::cout << elemsThisVertex << " #inputs " << tp.tilesForZig<<std::endl;
      #endif

      for (unsigned elem = 0; elem < elemsThisVertex; ++elem) {
        const auto thisElem = vertex * elemsPerVertex + elem;

        const auto thisPatch = patchS +
                (thisElem / (tp.patchSizeX * tp.patchSizeY)) % patchesThisTile;

        const auto thisZog  = thisElem / (tp.patchSizeX * tp.patchSizeY
                                          * patchesThisTile);

        const auto patchElem = thisElem % (tp.patchSizeX * tp.patchSizeY);
        const auto y = patchElem / tp.patchSizeX;
        const auto x = patchElem % tp.patchSizeX;

        for (unsigned ig = 0; ig < tp.tilesForZig; ++ig) {

          graph.connect(v["inPartial"][elem * tp.tilesForZig + ig],
                        acc[zogS + thisZog][ig][thisPatch][y][x].flatten());

        }

        Tensor redPart = red[zogS + thisZog][thisPatch][y][x].flatten();

        graph.connect(v["outPartial"][elem], redPart);

        graph.setTileMapping(redPart, tile);

      }
      totalElems -= elemsThisVertex;
    }
  }
  return Execute(cs);
}


static Program inverseTransform(
              Graph &graph,
              const ConvOptions &options,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor in,
              Tensor out) {
  const auto &target = graph.getTarget();
  const unsigned numWorkers = target.getNumWorkerContexts();

  ComputeSet cs = graph.addComputeSet(layerName + "/InvTransform");

  for (unsigned tile = 0; tile < options.getNumTiles(); ++tile) {

    unsigned patchS, patchesThisTile, zogS, numZog;
    std::tie(patchS, patchesThisTile) = tp.getOutPatchInfo(tile);
    std::tie(zogS, numZog) = tp.getOutZogInfo(tile);


    auto tuplesThisTile = numZog * patchesThisTile * tp.zoc
                          / WgdTilePartition::iUnitSize;

    /* split across number of workers */
    const auto tuplesPerVertex = (tuplesThisTile + numWorkers - 1)
                                            / numWorkers;
    for (unsigned vertex = 0; vertex < numWorkers && tuplesThisTile;
                  ++vertex) {

      auto tuplesThisVertex = std::min(tuplesPerVertex, tuplesThisTile);
      auto v = graph.addVertex(cs,
                               templateVertex("poplin::WgdInverseTransform",
                                              tp.dType,
                                              tp.patchSizeX, tp.patchSizeY,
                                              tp.kernelX, tp.kernelY));
      graph.setFieldSize(v["dTf"], tp.patchSizeX * tp.patchSizeY
                                   * tuplesThisVertex);
      graph.setFieldSize(v["dOut"], tp.getNumOutputsPerPatchY()
                                    * tp.getNumOutputsPerPatchX()
                                    * tuplesThisVertex);
      graph.setTileMapping(v, tile);


      for (unsigned tuple = 0; tuple < tuplesThisVertex; ++tuple) {
        auto thisTuple = (vertex * tuplesPerVertex + tuple)
                         * WgdTilePartition::iUnitSize;
        auto patch = (thisTuple / tp.zoc) % patchesThisTile;
        auto og = thisTuple / (tp.zoc * patchesThisTile);



        for (unsigned y = 0; y < tp.patchSizeY; ++y) {
          for (unsigned x = 0; x < tp.patchSizeX; ++x) {
            auto idxIn = tuple * tp.patchSizeX * tp.patchSizeY +
                         y * tp.patchSizeX + x ;
            auto slS = thisTuple % tp.zoc;
            auto slE = slS + WgdTilePartition::iUnitSize;
            graph.connect(v["dTf"][idxIn],
                          in[og + zogS][patchS + patch][y][x].flatten().slice(
                            slS, slE));
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
            auto slS = thisTuple % tp.zoc;
            auto slE = slS + 4;

            Tensor outPart = out[zogS + og]
                                [patchS + patch][y]
                                [x].flatten().slice(slS, slE);


            graph.connect(v["dOut"][idxOut], outPart);

            graph.setTileMapping(outPart, tile);
          }
        }
      }
      tuplesThisTile -= tuplesThisVertex;
    }
  }
  return Execute(cs);
}



static Program complete(
              Graph &graph,
              const ConvOptions &options,
              const WgdTilePartition &tp,
              const std::string layerName,
              Tensor in,
              Tensor act) {
  ComputeSet cs = graph.addComputeSet(layerName + "/Complete");
  const auto &target = graph.getTarget();
  const unsigned numWorkers = target.getNumWorkerContexts();

  for (unsigned tile = 0; tile < options.getNumTiles(); ++tile) {

    unsigned patchS, patchesThisTile, zogS, numZog;
    std::tie(patchS, patchesThisTile) = tp.getOutPatchInfo(tile);
    std::tie(zogS, numZog) = tp.getOutZogInfo(tile);

    #if DEBUG_PRINT >= 2
    std::cout << "tile " << tile << " patches: " << patchS << ":";
    std::cout << patchesThisTile << " zog :" << zogS;
    std::cout << ":" << numZog << std::endl;
    #endif

    assert(std::max(tp.zoc, tp.zocOut) % std::min(tp.zoc, tp.zocOut) == 0);
    const auto depth =  std::min(tp.zoc, tp.zocOut);
    const auto zFactor = tp.zoc <= tp.zocOut ? 1 : tp.zocOut/tp.zoc;

    auto totalUnits = zFactor * patchesThisTile * numZog;

    const auto unitsPerVertex = (totalUnits + numWorkers - 1)/numWorkers;

    for (auto vertex = 0U; vertex < numWorkers && totalUnits; ++vertex) {
      const auto unitsThisVertex = std::min(unitsPerVertex, totalUnits);

      auto v = graph.addVertex(cs, templateVertex("poplin::WgdConvComplete",
                                                  tp.dType));
      graph.setTileMapping(v, tile);

      unsigned elem = 0;

      for (auto unit = 0U; unit < unitsThisVertex; ++unit) {
        const auto thisUnit = vertex * unitsPerVertex + unit;
        const auto patch = (thisUnit/zFactor) % patchesThisTile;
        const auto thisPatch = patchS + patch;
        const auto oc = (zogS + thisUnit/(patchesThisTile * zFactor)) * tp.zoc
                        + (thisUnit % zFactor) * depth;

        #if DEBUG_PRINT == 2
        std::cout << "unit " << thisUnit << " " << tp.zoc << " ";
        std::cout << tp.zocOut << " " << depth << " " << zFactor;
        std::cout << " oc " << oc << std::endl;
        #endif

        auto ogIn = oc / tp.zoc;
        auto ocIn = oc % tp.zoc;
        auto ogOut = oc / tp.zocOut;
        auto ocOut = oc % tp.zocOut;


        unsigned xPosS, yPosS;
        std::tie(xPosS, yPosS) = tp.getOutIdxForPatch(thisPatch);

        auto xPosE = std::min(xPosS + tp.getNumOutputsPerPatchX(),
                              tp.getOutputSizeX());
        auto yPosE = std::min(yPosS + tp.getNumOutputsPerPatchY(),
                              tp.getOutputSizeY());


        for (unsigned y = yPosS; y < yPosE; ++y) {
          for (unsigned x = xPosS; x < xPosE; ++x) {

            #if DEBUG_PRINT >= 2
            std::cout << "in[" << ogOut << "][" << thisPatch << "][";
            std::cout << (y-yPosS) << "][" << (x-xPosS) << "] -> ";
            std::cout << elem << std::endl;
            #endif

            #if DEBUG_PRINT >= 2
            std::cout << "act[" << ogOut << "][" << y << "][";
            std::cout << x << "][" << ocOut << ":" << (ocOut+depth) << "] <- ";
            std::cout << elem << std::endl;
            #endif

            graph.connect(v["dIn"][elem],
                          in[ogIn][thisPatch][y-yPosS][x-xPosS].flatten().slice(
                              ocIn, ocIn + depth));
            graph.connect(v["act"][elem],
                          act[ogOut][y][x].flatten().slice(ocOut,
                                                           ocOut + depth));
            ++elem;
          }
        }
      }

      graph.setFieldSize(v["dIn"], elem);
      graph.setFieldSize(v["act"], elem);
      totalUnits -= unitsThisVertex;
    }
  }
  return Execute(cs);
}


extern Program winogradConvolution(Graph &graph,
            const ConvOptions &options,
            const std::vector<unsigned> &stride,
            const std::vector<unsigned> &paddingLower,
            const std::vector<unsigned> &paddingUpper,
            unsigned xDim, unsigned yDim,
            unsigned outNumChans, unsigned patchSizeX, unsigned patchSizeY,
            const Type &dType, const Type &partialsType,
            Tensor in, Tensor weights, Tensor activations,
            const std::string &debugPrefix) {

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

  std::cout << "activations.dim(3) :" << activations.dim(3) << std::endl;

#endif

  /* assumption that number of input channels per group must be same
   * for input activations and weights
   */
  assert(in.dim(0) == weights.dim(1));
  assert(in.dim(3) == weights.dim(5));

  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  WgdTilePartition tp(paddingLower[1], paddingLower[0],
                      xDim, yDim,
                      patchSizeX, patchSizeY,
                      kernelSizeX, kernelSizeY,
                      weights.dim(1) * weights.dim(5),
                      weights.dim(0) * weights.dim(4),
                      dType, partialsType);

  tp.tilePartition(weights.dim(5),
                   weights.dim(4),
                   activations.dim(3),
                   options,
                   graph.getTarget());

  auto prog = Sequence();

  const auto layerName = debugPrefix + "/WgdConv" + std::to_string(kernelSizeX)
                         + "x" + std::to_string(kernelSizeY) + "/Fwd";

  wgdMapWeights(graph, options, tp, weights);

  std::vector<Tensor> dataTf = allocateDataTfTensor(graph, tp);
  prog.add(computeDataTransform(graph, options, tp, layerName, in, dataTf));


  std::vector<Tensor> kernelTf = allocateKernelTfTensor(graph, tp);
  prog.add(computeKernelTransform(graph, options, tp, layerName, weights,
                                  kernelTf));

  /* accumulate across tiles */
  Tensor accumTen = graph.addVariable(partialsType,
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


  Tensor invTfIn = graph.addVariable(dType,
                                     {
                                       tp.zog,
                                       tp.getNumPatches(),
                                       patchSizeY,
                                       patchSizeX,
                                       tp.zoc
                                     },
                                     "WgdInvTrfIn");

  prog.add(reduce(graph, options, tp, layerName, accumTen, invTfIn));


  Tensor invTfOut = graph.addVariable(dType,
                                      {
                                        tp.zog,
                                        tp.getNumPatches(),
                                        tp.getNumOutputsPerPatchY(),
                                        tp.getNumOutputsPerPatchX(),
                                        tp.zoc
                                      },
                                      "WgdInvTrfOut");

  prog.add(inverseTransform(graph, options, tp, layerName, invTfIn, invTfOut));

  prog.add(complete(graph, options, tp, layerName, invTfOut, activations));

  return prog;
}


Program winogradConvolution(Graph &graph,
            const ConvParams &params,
            const ConvOptions &options,
            const Tensor &in, const Tensor &weights,
            const Tensor &out,
            unsigned patchSizeX, unsigned patchSizeY,
            const Type &partialsType,
            const std::string &debugPrefix) {
  Sequence prog;
  const auto batchSize = in.dim(0);
  const auto dType = in.elementType();
  // Perform each element of the batch serially
  for (unsigned b = 0; b < batchSize; ++b) {
    prog.add(winogradConvolution(graph,
                                 options,
                                 params.outputTransform.stride,
                                 params.inputTransform.paddingLower,
                                 params.inputTransform.paddingUpper,
                                 in.dim(3), in.dim(2),
                                 out.dim(1) * out.dim(4), patchSizeX,
                                 patchSizeY, dType, partialsType,
                                 in[b], weights, out[b],
                                 debugPrefix));
  }
  return prog;
}

} // namespace poplin
