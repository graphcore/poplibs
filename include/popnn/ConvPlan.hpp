#ifndef __ConvPlan_hpp__
#define __ConvPlan_hpp__
#include <string>
#include <poplar/Graph.hpp>

namespace conv {

struct Plan {
  unsigned tilesPerXAxis;
  unsigned tilesPerYAxis;
  unsigned tilesPerZAxis;
  unsigned verticesPerTilePerYAxis;
  unsigned tilesPerInZGroupAxis;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  bool floatPartials;
  bool useConvolutionInstructions;

  unsigned batchesPerGroup;
  unsigned numBatchGroups;
  bool flattenXY;

  Plan() = default;
  Plan(unsigned tilesPerXAxis,
       unsigned tilesPerYAxis,
       unsigned tilesPerZAxis,
       unsigned verticesPerTilePerYAxis,
       unsigned tilesPerInZGroupAxis,
       unsigned inChansPerGroup,
       unsigned partialChansPerGroup,
       bool floatPartials,
       bool useConvolutionInstructions) :
    tilesPerXAxis(tilesPerXAxis),
    tilesPerYAxis(tilesPerYAxis),
    tilesPerZAxis(tilesPerZAxis),
    verticesPerTilePerYAxis(verticesPerTilePerYAxis),
    tilesPerInZGroupAxis(tilesPerInZGroupAxis),
    inChansPerGroup(inChansPerGroup),
    partialChansPerGroup(partialChansPerGroup),
    floatPartials(floatPartials),
    useConvolutionInstructions(useConvolutionInstructions) {}
  const char *getPartialType() const {
    return floatPartials ? "float" : "half";
  }
};

class PlannerCache;
class Planner {
  std::unique_ptr<PlannerCache> cache;
public:
  Plan createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  unsigned kernelSizeY, unsigned kernelSizeX,
                  unsigned strideY, unsigned strideX,
                  unsigned paddingY, unsigned paddingX,
                  unsigned numChannels, unsigned batchSize,
                  std::string dType,
                  std::string partialsType, bool isFractional,
                  bool isWeightUpdate,
                  const poplar::Graph &graph);
  Planner();
  ~Planner();
};

}
#endif // __ConvPlan_hpp__
