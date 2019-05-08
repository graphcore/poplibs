// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplin_internal_ConvPlan_hpp
#define poplin_internal_ConvPlan_hpp
#include <poplin/Convolution.hpp>
#include <string>
#include <poplar/Graph.hpp>
#include <iosfwd>

namespace poplin {

struct ConvOptions;

struct Partition {
  // For each spatial dimension the number of parts the input is split into.
  std::vector<unsigned> fieldSplit;
  // The number of parts the batch axis is split into.
  unsigned batchSplit;
  // The number of parts the output channel axis is split into.
  unsigned outChanSplit;
  // For each spatial dimension the number of parts the kernel is split into.
  std::vector<unsigned> kernelSplit;
  // The number of parts the input channel axis is split into.
  unsigned inChanSplit;
  // The number of parts the convolution group axis is split into.
  unsigned convGroupSplit;
  // Grain size to use when splitting each spatial dimension.
  std::vector<unsigned> fieldAxisGrainSize;
  // Grain size to use when splitting the input channels.
  unsigned inChanGrainSize;
  // Grain size to use when splitting the output channels.
  unsigned outChanGrainSize;

  Partition() = default;
  Partition(std::vector<unsigned> fieldSplit_,
            unsigned batchSplit_,
            unsigned outChanSplit_,
            std::vector<unsigned> kernelSplit_,
            unsigned inChanSplit_,
            unsigned convGroupSplit_,
            std::vector<unsigned> fieldAxisGrainSize_,
            unsigned inChanGrainSize_,
            unsigned outChanGrainSize_) :
    fieldSplit(std::move(fieldSplit_)),
    batchSplit(batchSplit_),
    outChanSplit(outChanSplit_),
    kernelSplit(std::move(kernelSplit_)),
    inChanSplit(inChanSplit_),
    convGroupSplit(convGroupSplit_),
    fieldAxisGrainSize(std::move(fieldAxisGrainSize_)),
    inChanGrainSize(inChanGrainSize_),
    outChanGrainSize(outChanGrainSize_) { }
};

std::ostream& operator<<(std::ostream &os, const Partition &p);

struct ConvTransform {
  // The number of additional size 1 dimensions to insert at the front.
  unsigned extraFieldDims = 0;
  // Dimensions for which input dilation should be applied after the convolution
  // instead of before.
  std::vector<unsigned> dilatePostConv;
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
};

std::ostream& operator<<(std::ostream &os, const ConvTransform &t);

struct ConvTypes {
  /// Type to use for intermediate calculations.
  poplar::Type partialType;
  /// Type to use for the result.
  poplar::Type resultType;

  ConvTypes() = default;

  ConvTypes(poplar::Type partialType, poplar::Type resultType) :
    partialType(partialType),
    resultType(resultType) {}
};

std::ostream& operator<<(std::ostream &os, const ConvTypes &t);

struct Plan {
  // Description of how the convolution is transformed at each level of the
  // hierarchy.
  std::vector<ConvTransform> transforms;
  // Description of how each level of the hierarchy is partitioned.
  std::vector<Partition> partitions;
  // The types to use at each level of the hierarchy.
  std::vector<ConvTypes> types;

  // The number of parts to split out channels to serially execute
  unsigned outChanSerialSplit;

  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
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
  unsigned startTile;
  // True if there is no weight rearrangement between the forward / backward /
  // weight update passes, in which case the plan for all passes can be
  // determined from the plan for one pass.
  bool isJointPlan;

  Plan() = default;
  Plan(std::vector<Partition> partitions_,
       std::vector<ConvTypes> types_,
       unsigned inChansPerGroup_,
       unsigned partialChansPerGroup_,
       Plan::Method method_,
       Plan::LinearizeTileOrder linearizeTileOrder_,
       unsigned startTile_,
       bool isJointPlan) :
      partitions(std::move(partitions_)),
      types(std::move(types_)),
      inChansPerGroup(inChansPerGroup_),
      partialChansPerGroup(partialChansPerGroup_),
      method(method_),
      linearizeTileOrder(linearizeTileOrder_),
      startTile(startTile_),
      isJointPlan(isJointPlan) {}
};

std::ostream& operator<<(std::ostream &os, const Plan::Method m);
std::ostream& operator<<(std::ostream &os, const Plan &p);

std::vector<unsigned> getTileHierarchy(const poplar::Target &target,
                                       const ConvOptions &options);

// plan for a convolution with the specified parameters. \a target's
// virtual-graph dependent fields are not used.
Plan getPlan(const poplar::Target &target, const ConvParams &params,
             const ConvOptions &options, PlanningCache *cache);

/// Insert the specified number of dimensions of size 1 at the front.
void addExtraDims(ConvParams &params, unsigned extraDims);

ConvParams
calculateParamsWithDeferredDilation(
    const ConvParams &params,
    const std::vector<unsigned> &dilatePostConv
);

void swapOperands(ConvParams &params);

std::uint64_t getNumberOfMACs(const ConvParams &params);

/// Add plans for the specified convolutions to the cache
void preplanConvolutionsImpl(
    const poplar::Target &target,
    const std::set<std::pair<ConvParams, ConvOptions>> &convs,
    PlanningCache &cache);

/// Expose an estimator of the cycle and memory cost of a convolution
std::pair<std::uint64_t, std::uint64_t>
estimateConvCost(const poplar::Target &target,
                 const ConvParams &params,
                 const ConvOptions &options,
                 PlanningCache *cache,
                 const Plan &plan);

}
#endif // poplin_internal_ConvPlan_hpp
