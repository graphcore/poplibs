// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef poplibs_test_ctc_inference_select_candidates_hpp
#define poplibs_test_ctc_inference_select_candidates_hpp

#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>

#include <boost/multi_array.hpp>

namespace poplibs_test {
namespace ctc {

template <typename PartialsType>
std::vector<Candidate<PartialsType>> runSelectCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<PartialsType>> &candidates, unsigned beamwidth,
    bool profile);

} // namespace ctc
} // namespace poplibs_test

#endif
