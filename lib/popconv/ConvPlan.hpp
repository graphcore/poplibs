#ifndef __popconv_ConvPlan_hpp__
#define __popconv_ConvPlan_hpp__
#include <popconv/Convolution.hpp>
#include <string>
#include <poplar/Graph.hpp>
#include <iosfwd>

namespace popconv {

struct Plan {
  // For each spatial dimension the number of sections the input is split into
  // in that dimension to balance across tiles.
  std::vector<unsigned> fieldTileSplit;
  // The number of sections the batch axis is split into balance across tiles.
  unsigned batchTileSplit;
  // The number of sections the output channel axis is split to balance across
  // tiles.
  unsigned outChanTileSplit;
  // For each spatial dimension the number of sections the kernel is split into
  // in that dimension to balance across tiles.
  std::vector<unsigned> kernelTileSplit;
  // The number of sections the input channel axis is split to balance across
  // tiles.
  unsigned inChanTileSplit;
  // The number of sections the convolution group axis is split to balance
  // across tiles.
  unsigned convGroupTileSplit;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  /// Grain size to use when splitting the axes across tiles.
  std::vector<unsigned> fieldAxisGrainSize;
  bool floatPartials;
  // The number of additional size 1 dimensions to insert at the front.
  unsigned extraFieldDims = 0;
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
  Plan(std::vector<unsigned> fieldTileSplit_,
       unsigned batchTileSplit_,
       unsigned outChanTileSplit_,
       std::vector<unsigned> kernelTileSplit_,
       unsigned inChanTileSplit_,
       unsigned convGroupTileSplit_,
       unsigned inChansPerGroup_,
       unsigned partialChansPerGroup_,
       std::vector<unsigned> fieldAxisGrainSize_,
       bool floatPartials_,
       Plan::Method method_,
       Plan::LinearizeTileOrder linearizeTileOrder_) :
      fieldTileSplit(std::move(fieldTileSplit_)),
      batchTileSplit(batchTileSplit_),
      outChanTileSplit(outChanTileSplit_),
      kernelTileSplit(std::move(kernelTileSplit_)),
      inChanTileSplit(inChanTileSplit_),
      convGroupTileSplit(convGroupTileSplit_),
      inChansPerGroup(inChansPerGroup_),
      partialChansPerGroup(partialChansPerGroup_),
      fieldAxisGrainSize(std::move(fieldAxisGrainSize_)),
      floatPartials(floatPartials_),
      method(method_),
      linearizeTileOrder(linearizeTileOrder_) {
    assert(fieldTileSplit.size() == fieldAxisGrainSize.size());
  }
  poplar::Type getPartialType() const {
    return floatPartials ? poplar::FLOAT : poplar::HALF;
  }
};

Plan getPlan(const poplar::Graph &graph, const ConvParams &params,
             ConvOptions options);

/// Insert the specified number of dimensions of size 1 at the front.
void addExtraDims(ConvParams &params, unsigned extraDims);

void swapOperands(ConvParams &params);

std::uint64_t getNumberOfMACs(const ConvParams &params);

std::ostream& operator<<(std::ostream &os, const Plan::Method m);
std::ostream& operator<<(std::ostream &os, const Plan &p);

}
#endif // __popconv_ConvPlan_hpp__
