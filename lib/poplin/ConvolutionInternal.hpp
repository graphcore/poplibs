// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _ConvolutionInternal_hpp_
#define _ConvolutionInternal_hpp_

#include "CanonicalConvParams.hpp"
#include "ConvOptions.hpp"
#include "ConvPlan.hpp"
#include <poplin/Convolution.hpp>

namespace poplin {

struct ConvProgramTree;

struct ConvIndices {
  unsigned cg;
  unsigned b;
  std::vector<unsigned> out;
  unsigned oc;
  unsigned ic;
  std::vector<unsigned> kernel;
};

struct ConvSlice {
  unsigned cgBegin, cgEnd;
  unsigned batchBegin, batchEnd;
  std::vector<unsigned> outFieldBegin, outFieldEnd;
  unsigned outChanBegin, outChanEnd;
  unsigned inChanBegin, inChanEnd;
  std::vector<unsigned> kernelBegin, kernelEnd;

  unsigned getNumFieldDims() const { return outFieldBegin.size(); }
  unsigned getNumConvGroups() const { return cgEnd - cgBegin; }
  unsigned getBatchSize() const { return batchEnd - batchBegin; }
  unsigned getNumOutputChans() const { return outChanEnd - outChanBegin; }
  unsigned getNumInputChans() const { return inChanEnd - inChanBegin; }
  unsigned getOutputSize(unsigned dim) const {
    return outFieldEnd[dim] - outFieldBegin[dim];
  }
  unsigned getKernelSize(unsigned dim) const {
    return kernelEnd[dim] - kernelBegin[dim];
  }
};

poplar::Tensor createInput(poplar::Graph &graph, const Plan &plan,
                           const CanonicalConvParams &params,
                           const poplar::Tensor &metadata,
                           const poplar::DebugNameAndId &dnai,
                           const ConvOptions &options);

poplar::Tensor createWeights(poplar::Graph &graph, const Plan &plan,
                             const CanonicalConvParams &params,
                             const poplar::Tensor &metadata,
                             const poplar::DebugNameAndId &dnai,
                             const ConvOptions &options);

poplar::Tensor convolution(poplar::Graph &graph, const poplar::Tensor &in,
                           const poplar::Tensor &weights, const Plan &plan,
                           const CanonicalConvParams &params,
                           bool transposeAndFlipWeights, ConvProgramTree &cpt,
                           const poplar::DebugNameAndId &dnai,
                           const ConvOptions &options);

void weightsTransposeChansFlipXY(poplar::Graph &graph,
                                 const poplar::Tensor &weightsInUnGrouped,
                                 const poplar::Tensor &weightsOutUnGrouped,
                                 ConvProgramTree &cpt,
                                 const poplar::DebugNameAndId &dnai);

poplar::Tensor
calculateWeightDeltas(poplar::Graph &graph, const poplar::Tensor &zDeltas_,
                      const poplar::Tensor &activations_, const Plan &wuPlan,
                      const CanonicalConvParams &wuParams, ConvProgramTree &cpt,
                      const poplar::DebugNameAndId &dnai,
                      const ConvOptions &wuOptions);

void convolutionWeightUpdate(poplar::Graph &graph,
                             const poplar::Tensor &zDeltas,
                             const poplar::Tensor &weights,
                             const poplar::Tensor &activations,
                             const Plan &plan, CanonicalConvParams params,
                             const poplar::Tensor &scale, ConvProgramTree &cpt,
                             const poplar::DebugNameAndId &dnai,
                             const ConvOptions &options);

void convolutionWeightUpdate(poplar::Graph &graph,
                             const poplar::Tensor &zDeltas,
                             const poplar::Tensor &weights,
                             const poplar::Tensor &activations,
                             const Plan &plan, CanonicalConvParams params,
                             float scale, ConvProgramTree &cpt,
                             const poplar::DebugNameAndId &dnai,
                             const ConvOptions &options);

// Required for expand dims planning, we don't intend to use graph as mutable,
// but it is deemed safe because it only adds extra compute sets. (we don't
// provide rearrange prog to add them to). We also are careful to pass a copy of
// the plan to this function.
// TODO: make version of this without requiring a mutable graph/plan
CanonicalConvParams
convolutionPreprocess(poplar::Graph &graph, const ConvParams &params,
                      const ConvOptions &options, Plan &plan, unsigned level,
                      const std::vector<Split<ConvIndices>> &indices,
                      bool serial);

CanonicalConvParams getSubConvolution(const ConvSlice &slice,
                                      const CanonicalConvParams &originalParams,
                                      poplar::Tensor *in,
                                      poplar::Tensor *weights);

void iteratePartitionParallel(
    const CanonicalConvParams &params, const Partition &partition,
    const std::function<void(const ConvIndices &, const ConvSlice &)> &f);

poplar::Tensor sliceOutput(const poplar::Tensor &out, const ConvSlice &slice,
                           const unsigned convGroupsPerGroup,
                           const unsigned outChansPerGroup);

} // namespace poplin

#endif // _ConvolutionInternal_hpp_
