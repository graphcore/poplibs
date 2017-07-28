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
  bool flattenXY = false;
  bool useWinograd = false;
  enum class Method {
    // Direction convolution using the MAC instruction.
    MAC,
    // Direction convolution using the AMP instruction.
    AMP,
    // Compute the convolution using the AMP instruction. Data is rearranged
    // such that the AMP units accumulate over the x-axis of the field.
    AMP_ACCUMULATE_OVER_FIELD
  } method;
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
       Plan::Method method,
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
    method(method),
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

std::ostream& operator<<(std::ostream &os, const Plan::Method m);
std::ostream& operator<<(std::ostream &os, const Plan &p);

}
#endif // __popconv_ConvPlan_hpp__
