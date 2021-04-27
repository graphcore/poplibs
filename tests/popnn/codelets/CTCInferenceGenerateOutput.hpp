// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef poplibs_test_ctc_inference_generate_output_hpp
#define poplibs_test_ctc_inference_generate_output_hpp

#include <poplar/Graph.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>

namespace poplibs_test {
namespace ctc {

std::vector<unsigned> runGenerateOutputCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, unsigned timestep,
    const BeamHistory &beamHistory, unsigned beamOutLength, unsigned outputBeam,
    unsigned numClassesIncBlank, bool profile);

} // namespace ctc
} // namespace poplibs_test

#endif
