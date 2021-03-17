// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef poplibs_test_ctc_inference_update_hpp
#define poplibs_test_ctc_inference_update_hpp

#include <poplar/Graph.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>

namespace poplibs_test {
namespace ctc {

template <typename PartialsType>
std::pair<BeamHistory, std::vector<BeamProbability<PartialsType>>>
runUpdateCodelet(poplar::Graph &graph, poplibs_support::TestDevice &device,
                 poplibs_support::DeviceType deviceType, poplar::Type inType,
                 poplar::Type partialsType,
                 const std::vector<Candidate<PartialsType>> &candidates,
                 unsigned timestep, const BeamHistory &beamHistory,
                 const std::vector<BeamProbability<PartialsType>> &beamProbs,
                 unsigned blankClass, bool profile);

} // namespace ctc
} // namespace poplibs_test

#endif
