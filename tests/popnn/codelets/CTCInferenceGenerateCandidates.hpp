// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef poplibs_test_ctc_inference_generate_candidates_hpp
#define poplibs_test_ctc_inference_generate_candidates_hpp

#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>

#include <boost/multi_array.hpp>

namespace poplibs_test {
namespace ctc {

template <typename InputType, typename PartialsType>
std::vector<Candidate<PartialsType>> runGenerateCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type inType,
    poplar::Type partialsType,
    const boost::multi_array<InputType, 2> &logProbsIn, unsigned timestep,
    const std::vector<BeamProbability<PartialsType>> &beamProbs,
    const BeamHistory &beamHistory, unsigned classToMakeAddend,
    unsigned blankClass, bool testGenerateCopyVertex, bool profile);

} // namespace ctc
} // namespace poplibs_test

#endif
