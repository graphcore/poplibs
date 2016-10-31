#ifndef __ConvPlan_hpp__
#define __ConvPlan_hpp__
#include <string>
#include <poplar/Graph.hpp>

namespace conv {

struct Partition {
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

  Partition() = default;
  Partition(unsigned tilesPerXAxis,
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

struct ConvPlan {
public:
  Partition fwdPartition;
  Partition bwdPartition;
  Partition wuPartition;
  bool flattenXY;
};

class PlannerCache;
class Planner {
  std::unique_ptr<PlannerCache> cache;
public:
  ConvPlan createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                      unsigned kernelSizeY, unsigned kernelSizeX,
                      unsigned strideY, unsigned strideX,
                      unsigned paddingY, unsigned paddingX,
                      unsigned numChannels, unsigned batchSize,
                      std::string dType,
                      const poplar::Graph &graph, bool forwardOnly);
  Planner();
  ~Planner();
};

}
#endif // __ConvPlan_hpp__
