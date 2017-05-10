#ifndef __popconv_ConvPlan_hpp__
#define __popconv_ConvPlan_hpp__
#include <popconv/Convolution.hpp>
#include <string>
#include <poplar/Graph.hpp>
#include <iosfwd>

namespace popconv {

struct Plan {
  unsigned tilesPerXAxis;
  unsigned tilesPerYAxis;
  unsigned tilesPerZAxis;
  unsigned tilesPerKernelYAxis;
  unsigned tilesPerInZGroupAxis;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  unsigned batchesPerGroup;
  bool floatPartials;
  bool useConvolutionInstructions;
  bool flattenXY = false;
  bool useWinograd = false;
  unsigned winogradPatchSize;
  enum AmpWUMethod {
    DELTAS_AS_COEFFICENTS,
    ACTIVATIONS_AS_COEFFICENTS,
  } ampWUMethod = DELTAS_AS_COEFFICENTS;

  Plan() = default;
  Plan(unsigned tilesPerXAxis,
       unsigned tilesPerYAxis,
       unsigned tilesPerZAxis,
       unsigned tilesPerKernelYAxis,
       unsigned tilesPerInZGroupAxis,
       unsigned inChansPerGroup,
       unsigned partialChansPerGroup,
       unsigned batchesPerGroup,
       bool floatPartials,
       bool useConvolutionInstructions) :
    tilesPerXAxis(tilesPerXAxis),
    tilesPerYAxis(tilesPerYAxis),
    tilesPerZAxis(tilesPerZAxis),
    tilesPerKernelYAxis(tilesPerKernelYAxis),
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

Plan getPlan(const poplar::Graph &graph,
             std::string dType,
             unsigned batchSize,
             unsigned inDimY, unsigned inDimX, unsigned inNumChans,
             std::vector<std::size_t> weightsShape,
             std::vector<unsigned> stride,
             std::vector<unsigned> paddingLower,
             std::vector<unsigned> paddingUpper,
             bool isFractional, ConvOptions options);

Plan getWeightUpdatePlan(const poplar::Graph &graph,
                         const poplar::Tensor &activations,
                         const poplar::Tensor &deltas,
                         std::vector<std::size_t> weightsShape,
                         std::vector<unsigned> stride,
                         std::vector<unsigned> paddingLower,
                         std::vector<unsigned> paddingUpper,
                         bool isFractional,
                         ConvOptions options);

std::ostream& operator<<(std::ostream &os, const Plan &p);

}
#endif // __popconv_ConvPlan_hpp__
