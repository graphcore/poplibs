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
  /// Grain size to use when splitting the x-axis across tiles.
  unsigned xAxisGrainSize;
  bool floatPartials;
  bool useConvolutionInstructions;
  bool flattenXY = false;
  bool useWinograd = false;
  enum class LinearizeTileOrder {
    STANDARD,
    FC_WU,
    FC_BWD_AS_CONV
  } linearizeTileOrder = LinearizeTileOrder::STANDARD;
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
       unsigned xAxisGrainSize,
       bool floatPartials,
       bool useConvolutionInstructions,
       Plan::LinearizeTileOrder linearizeTileOrder) :
    tilesPerXAxis(tilesPerXAxis),
    tilesPerYAxis(tilesPerYAxis),
    tilesPerZAxis(tilesPerZAxis),
    tilesPerKernelYAxis(tilesPerKernelYAxis),
    tilesPerInZGroupAxis(tilesPerInZGroupAxis),
    inChansPerGroup(inChansPerGroup),
    partialChansPerGroup(partialChansPerGroup),
    batchesPerGroup(batchesPerGroup),
    xAxisGrainSize(xAxisGrainSize),
    floatPartials(floatPartials),
    useConvolutionInstructions(useConvolutionInstructions),
    linearizeTileOrder(linearizeTileOrder) {}
  const char *getPartialType() const {
    return floatPartials ? "float" : "half";
  }
};

Plan getPlan(const poplar::Graph &graph, const ConvParams &params,
             ConvOptions options);

Plan getWeightUpdatePlan(const poplar::Graph &graph,
                         const poplar::Tensor &activations,
                         const poplar::Tensor &deltas,
                         const ConvParams &params,
                         ConvOptions options);

ConvParams
weightUpdateByAmpTransformParams(const ConvParams &params,
                                 const poplar::DeviceInfo &deviceInfo,
                                 const Plan &plan);

std::ostream& operator<<(std::ostream &os, const Plan &p);

}
#endif // __popconv_ConvPlan_hpp__
