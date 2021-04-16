// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef poplibs_test_ctc_inference_merge_candidates_hpp
#define poplibs_test_ctc_inference_merge_candidates_hpp

#include <poplar/Graph.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>

namespace poplibs_test {
namespace ctc {

template <typename PartialsType>
std::vector<Candidate<PartialsType>> runMergeCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type inType,
    poplar::Type partialsType,
    const std::vector<Candidate<PartialsType>> &candidates,
    const Candidate<PartialsType> &copyCandidate, unsigned timestep,
    unsigned blankClass, const BeamHistory &beamHistory, bool profile);

} // namespace ctc
} // namespace poplibs_test

#endif
