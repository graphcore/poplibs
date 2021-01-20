// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_ctc_inference_hpp
#define poplibs_test_ctc_inference_hpp

#include "CTCUtil.hpp"
#include "LogArithmetic.hpp"

#include <boost/multi_array.hpp>

#include <vector>

namespace poplibs_test {
namespace ctc {

// placeholder to represent no change to output sequence
inline constexpr unsigned voidSymbol = std::numeric_limits<unsigned>::max();

template <typename FPType>
std::tuple<FPType, std::vector<unsigned>>
infer(const boost::multi_array<FPType, 2> &input, unsigned blankSymbol,
      unsigned beamwidth, bool useLog, bool verbose = false);

//------------------------------------------------------------------------------
// Exhaustive path inference functions. Coded to look simple and be divided
// into lots of individually verifiable steps and help with debug.
boost::multi_array<unsigned, 2> findAllInputPaths(unsigned timeSteps,
                                                  unsigned sequenceLength);

template <typename FPType>
boost::multi_array<FPType, 1>
findAllInputPathProbabilities(const boost::multi_array<FPType, 2> &sequence,
                              const boost::multi_array<unsigned, 2> &inputPaths,
                              bool isLog);

boost::multi_array<unsigned, 2>
inputToOutputPath(const boost::multi_array<unsigned, 2> &inputPath,
                  unsigned blankSymbol);

template <typename FPType>
std::tuple<std::vector<unsigned>, std::vector<FPType>,
           std::vector<std::vector<unsigned>>>
mergePaths(boost::multi_array<unsigned, 2> &outPaths,
           boost::multi_array<FPType, 1> &probabilities, bool isLog);

} // namespace ctc
} // namespace poplibs_test

#endif // poplibs_test_ctc_inference_hpp
