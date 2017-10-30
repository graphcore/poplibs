#ifndef __popconv_ConvPlan_hpp__
#define __popconv_ConvPlan_hpp__
#include <popconv/Convolution.hpp>
#include <string>
#include <poplar/Graph.hpp>
#include <iosfwd>

namespace popconv {

struct Plan {
  std::vector<unsigned> tilesPerFieldAxis;
  unsigned tilesPerBatchAxis;
  unsigned tilesPerZAxis;
  unsigned tilesPerKernelYAxis;
  unsigned tilesPerInZGroupAxis;
  // tiles over which group of a grouped convolution is spread
  unsigned tilesPerConvGroups;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  /// Grain size to use when splitting the axes across tiles.
  std::vector<unsigned> fieldAxisGrainSize;
  bool floatPartials;
  bool swapOperands = false;
  // Spatial dimensions that should be expanded by taking the activations
  // multiplied by each weight in each position of the filter in this axis and
  // turning them into different input channels.
  std::vector<unsigned> expandDims;
  // Spatial dimensions that should be flattened into the output channels of the
  // kernel.
  std::vector<unsigned> outChanFlattenDims;
  // Dimensions that should be flattened. The dimensions are numbered such that
  // the batch is dimension 0 and the spatial dimensions start at 1.
  // The dimensions are flattened into the last dimension in reverse order.
  std::vector<unsigned> flattenDims;
  bool useWinograd = false;
  enum class Method {
    // Direction convolution using the MAC instruction.
    MAC,
    // Direction convolution using the AMP instruction.
    AMP,
    // Outer product of two vectors.
    OUTER_PRODUCT,
  } method;
  enum class LinearizeTileOrder {
    STANDARD,
    FC_WU,
    FC_BWD_AS_CONV
  } linearizeTileOrder = LinearizeTileOrder::STANDARD;
  unsigned winogradPatchSize;

  Plan() = default;
  Plan(std::vector<unsigned> tilesPerFieldAxis_,
       unsigned tilesPerBatchAxis_,
       unsigned tilesPerZAxis_,
       unsigned tilesPerKernelYAxis_,
       unsigned tilesPerInZGroupAxis_,
       unsigned tilesPerConvGroups_,
       unsigned inChansPerGroup_,
       unsigned partialChansPerGroup_,
       std::vector<unsigned> fieldAxisGrainSize_,
       bool floatPartials_,
       Plan::Method method_,
       Plan::LinearizeTileOrder linearizeTileOrder_) :
      tilesPerFieldAxis(std::move(tilesPerFieldAxis_)),
      tilesPerBatchAxis(tilesPerBatchAxis_),
      tilesPerZAxis(tilesPerZAxis_),
      tilesPerKernelYAxis(tilesPerKernelYAxis_),
      tilesPerInZGroupAxis(tilesPerInZGroupAxis_),
      tilesPerConvGroups(tilesPerConvGroups_),
      inChansPerGroup(inChansPerGroup_),
      partialChansPerGroup(partialChansPerGroup_),
      fieldAxisGrainSize(std::move(fieldAxisGrainSize_)),
      floatPartials(floatPartials_),
      method(method_),
      linearizeTileOrder(linearizeTileOrder_) {
    assert(tilesPerFieldAxis.size() == fieldAxisGrainSize.size());
  }
  const char *getPartialType() const {
    return floatPartials ? "float" : "half";
  }
};

Plan getPlan(const poplar::Graph &graph, const ConvParams &params,
             ConvOptions options);

/// Return whether expanding the specified spatial dimension involves
/// expanding the activations or the weights.
bool expandDimExpandActs(ConvParams &params, unsigned dim);

void swapOperands(ConvParams &params);

std::uint64_t getNumberOfMACs(const ConvParams &params);

std::ostream& operator<<(std::ostream &os, const Plan::Method m);
std::ostream& operator<<(std::ostream &os, const Plan &p);

}
#endif // __popconv_ConvPlan_hpp__
