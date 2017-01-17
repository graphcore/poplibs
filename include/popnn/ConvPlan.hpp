#ifndef __ConvPlan_hpp__
#define __ConvPlan_hpp__
#include <string>
#include <poplar/Graph.hpp>

namespace conv {

// Switches to control the planning of Convolutional layers
class PlanControl {
public:
  bool forceAOPForWU = true;
};

struct Plan {
  unsigned tilesPerXAxis;
  unsigned tilesPerYAxis;
  unsigned tilesPerZAxis;
  unsigned verticesPerTilePerYAxis;
  unsigned tilesPerInZGroupAxis;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  unsigned batchesPerGroup;
  bool floatPartials;
  bool useConvolutionInstructions;
  bool flattenXY;

  Plan() = default;
  Plan(unsigned tilesPerXAxis,
       unsigned tilesPerYAxis,
       unsigned tilesPerZAxis,
       unsigned verticesPerTilePerYAxis,
       unsigned tilesPerInZGroupAxis,
       unsigned inChansPerGroup,
       unsigned partialChansPerGroup,
       unsigned batchesPerGroup,
       bool floatPartials,
       bool useConvolutionInstructions) :
    tilesPerXAxis(tilesPerXAxis),
    tilesPerYAxis(tilesPerYAxis),
    tilesPerZAxis(tilesPerZAxis),
    verticesPerTilePerYAxis(verticesPerTilePerYAxis),
    tilesPerInZGroupAxis(tilesPerInZGroupAxis),
    inChansPerGroup(inChansPerGroup),
    partialChansPerGroup(partialChansPerGroup),
    batchesPerGroup(batchesPerGroup),
    floatPartials(floatPartials),
    useConvolutionInstructions(useConvolutionInstructions) {}
  const char *getPartialType() const {
    return floatPartials ? "float" : "half";
  }
};

class PlannerCache;
class Planner {
  std::unique_ptr<PlannerCache> cache;
  unsigned percentageCyclesExcessForMemOptim;
public:
  Plan createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  unsigned kernelSizeY, unsigned kernelSizeX,
                  unsigned strideY, unsigned strideX,
                  unsigned paddingY, unsigned paddingX,
                  unsigned numChannels, unsigned batchSize,
                  std::string dType,
                  std::string partialsType, bool isFractional,
                  const poplar::Graph &graph,
                  const conv::PlanControl &planControl);
  Plan createWeightUpdatePlan(unsigned inDimY, unsigned inDimX,
                              unsigned inNumChans,
                              unsigned actChansPerGroup,
                              unsigned deltasChansPerGroup,
                              unsigned weightOutChansPerGroup,
                              unsigned kernelSizeY, unsigned kernelSizeX,
                              unsigned strideY, unsigned strideX,
                              unsigned paddingY, unsigned paddingX,
                              unsigned numChannels, unsigned batchSize,
                              std::string dType,
                              std::string partialsType, bool isFractional,
                              const poplar::Graph &graph,
                              const PlanControl &planControl);
  Planner(unsigned percentageCyclesExcessForMemOptim = 0);
  ~Planner();
};

std::ostream& operator<<(std::ostream &os, const Plan &p);

}
#endif // __ConvPlan_hpp__
