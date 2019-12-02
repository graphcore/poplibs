// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplin_internal_ConvPlan_hpp
#define poplin_internal_ConvPlan_hpp
#include <iosfwd>
#include <poplar/Graph.hpp>
#include <poplin/Convolution.hpp>
#include <string>

namespace poplin {

class ConvOptions;

// some dimensions support being split both in parallel and serially.
template <typename T> struct Split {
  T serial;
  T parallel;
};

struct Partition {
  // For each spatial dimension the number of parts the input is split into.
  std::vector<unsigned> fieldSplit;
  // The number of parts the batch axis is split into.
  unsigned batchSplit;
  // The number of parts the output channel axis is split into.
  Split<unsigned> outChanSplit;
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
  Partition(std::vector<unsigned> fieldSplit_, unsigned batchSplit_,
            Split<unsigned> outChanSplit_, std::vector<unsigned> kernelSplit_,
            unsigned inChanSplit_, unsigned convGroupSplit_,
            std::vector<unsigned> fieldAxisGrainSize_,
            unsigned inChanGrainSize_, unsigned outChanGrainSize_)
      : fieldSplit(std::move(fieldSplit_)), batchSplit(batchSplit_),
        outChanSplit(outChanSplit_), kernelSplit(std::move(kernelSplit_)),
        inChanSplit(inChanSplit_), convGroupSplit(convGroupSplit_),
        fieldAxisGrainSize(std::move(fieldAxisGrainSize_)),
        inChanGrainSize(inChanGrainSize_), outChanGrainSize(outChanGrainSize_) {
  }

  unsigned totalParallelSplit() const {
    return std::accumulate(fieldSplit.begin(), fieldSplit.end(), unsigned(1),
                           std::multiplies<unsigned>()) *
           batchSplit * outChanSplit.parallel *
           std::accumulate(kernelSplit.begin(), kernelSplit.end(), unsigned(1),
                           std::multiplies<unsigned>()) *
           inChanSplit * convGroupSplit;
  }
  unsigned totalSerialSplit() const { return outChanSplit.serial; }
};

std::ostream &operator<<(std::ostream &os, const Partition &p);

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
  // Depthwise convolutions often result in 1 input channel per conv group, this
  // can result in sub-optimal padding of the activations tensor. Therefore
  // this transformation will pad the conv groups up to the factor specified by
  // this unsigned, F and then transform the params so that we have N/F conv
  // groups, each with, F input channels and Fx output channels.
  // By padding the weights to only be non-zero if they are for
  // the same conv groups this will produce the same result as before.
  unsigned combineConvGroupsFactor = 1;
};

std::ostream &operator<<(std::ostream &os, const ConvTransform &t);

// There are several types that exist during a convolution:
//   a. Input type (this is ConvParams::inputType),
//   b. Intermediate type of an on-tile convolution,
//   c. Output type of the on-tile convolution,
//   d. Input type of the intra-IPU reduction (this is the same as c),
//   e. Accumulator type of the intra-IPU reduction,
//   f. Result type of the intra-IPU reduction,
//   g. Input type of the inter-IPU reduction (this is the same as f),
//   h. Accumulator type of the inter-IPU reduction,
//   i. The result type.
// A vector, indexed by hierarchy, of ConvTypes struct describes the types
// b-i like so:
//   ConvTypes[tileLevel] = {.partialsType = b, .resultType = c},
//   ConvTypes[ipuLevel] = {.partialsType = e, .resultType = f},
//   ConvTypes[systemLevel] = {.partialsType = h, .resultType = i}
// For convolutions over a single IPU the types g-i do not exist and f is the
// final result type.
struct ConvTypes {
  // Type to use for intermediate calculations.
  poplar::Type partialType;
  // Type to use for the result. Depending on the partials type this may mean we
  // need to downcast the final partials into the result before the convolution
  // is complete.
  poplar::Type resultType;

  ConvTypes() = default;

  ConvTypes(poplar::Type partialType, poplar::Type resultType)
      : partialType(partialType), resultType(resultType) {}
};

std::ostream &operator<<(std::ostream &os, const ConvTypes &t);

struct Plan {
  // Description of how the convolution is transformed at each level of the
  // hierarchy.
  std::vector<ConvTransform> transforms;
  // Description of how each level of the hierarchy is partitioned.
  std::vector<Partition> partitions;
  // The types to use at each level of the hierarchy.
  std::vector<ConvTypes> types;

  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
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
  unsigned startTile;
  // True if there is no weight rearrangement between the forward / backward /
  // weight update passes, in which case the plan for all passes can be
  // determined from the plan for one pass.
  bool isJointPlan;

  Plan() = default;
  Plan(std::vector<Partition> partitions_, std::vector<ConvTypes> types_,
       unsigned inChansPerGroup_, unsigned partialChansPerGroup_,
       Plan::Method method_, Plan::LinearizeTileOrder linearizeTileOrder_,
       unsigned startTile_, bool isJointPlan)
      : partitions(std::move(partitions_)), types(std::move(types_)),
        inChansPerGroup(inChansPerGroup_),
        partialChansPerGroup(partialChansPerGroup_), method(method_),
        linearizeTileOrder(linearizeTileOrder_), startTile(startTile_),
        isJointPlan(isJointPlan) {}
};

std::ostream &operator<<(std::ostream &os, const Plan::Method &m);
std::istream &operator>>(std::istream &is, Plan::Method &m);
std::ostream &operator<<(std::ostream &os, const Plan &p);

std::vector<unsigned> getTileHierarchy(const poplar::Target &target);

class CanonicalConvParams;

// plan for a convolution with the specified parameters. \a target's
// virtual-graph dependent fields are not used.
Plan getPlan(const poplar::Target &target, const CanonicalConvParams &params,
             const ConvOptions &options, PlanningCache *cache);

/// Insert the specified number of dimensions of size 1 at the front.
void addExtraDims(ConvParams &params, unsigned extraDims);

ConvParams calculateParamsWithDeferredDilation(
    const ConvParams &params, const std::vector<unsigned> &dilatePostConv);

void swapOperands(ConvParams &params);
void combineConvGroups(const unsigned factor, ConvParams &params);

// this factor is how much we reduce the number of groups by and increase the
// channel dimensions by when applying the combineConvGroup transformation.
unsigned convGroupCombineFactor(const unsigned factor,
                                unsigned inputChannelsPerConvGroup);

std::uint64_t getNumberOfMACs(const ConvParams &params);

using ConvPlanKey = std::pair<ConvParams, ConvOptions>;
/// Add plans for the specified convolutions to the cache
void preplanConvolutionsImpl(const poplar::Target &target,
                             const std::set<ConvPlanKey> &convs,
                             PlanningCache &cache);

/// Expose an estimator of the cycle and memory cost of a convolution
std::pair<std::uint64_t, std::uint64_t>
estimateConvCost(const poplar::Target &target, const ConvParams &params,
                 const ConvOptions &options, PlanningCache *cache,
                 const Plan &plan);

} // namespace poplin
#endif // poplin_internal_ConvPlan_hpp
