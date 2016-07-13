#ifndef __ConvPlan_hpp__
#define __ConvPlan_hpp__
#include <string>
#include <DeviceInfo.hpp>

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

  Partition() = default;
  Partition(unsigned tilesPerXAxis,
           unsigned tilesPerYAxis,
           unsigned tilesPerZAxis,
           unsigned verticesPerTilePerYAxis,
           unsigned tilesPerInZGroupAxis,
           unsigned inChansPerGroup,
           unsigned partialChansPerGroup,
           bool floatPartials) :
    tilesPerXAxis(tilesPerXAxis),
    tilesPerYAxis(tilesPerYAxis),
    tilesPerZAxis(tilesPerZAxis),
    verticesPerTilePerYAxis(verticesPerTilePerYAxis),
    tilesPerInZGroupAxis(tilesPerInZGroupAxis),
    inChansPerGroup(inChansPerGroup),
    partialChansPerGroup(partialChansPerGroup),
    floatPartials(floatPartials) {}
  const char *getPartialType() const {
    return floatPartials ? "float" : "half";
  }  
};

struct ConvPlan {
public:
  Partition fwdPartition;
  bool useConvolutionInstruction;
};

ConvPlan createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSize, unsigned stride, unsigned padding,
                    unsigned numChannels,
                    std::string dType,
                    const DeviceInfo &deviceInfo);


}
#endif // __ConvPlan_hpp__
