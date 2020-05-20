// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_Multirate_hpp
#define poplibs_test_Multirate_hpp

#include <boost/multi_array.hpp>

namespace poplibs_test {

//
// Upsample multi-array by a given factor
//
// The flattened field dimensions are located in the innermost multi-array
// dimension
//
void upsample(const std::vector<std::size_t> &inFieldShape,
              const unsigned samplingRate,
              const boost::multi_array<double, 3> &input,
              boost::multi_array<double, 3> &output);

//
// Downsample multi-array by a given factor
//
// The flattened field dimensions are located in the innermost multi-array
// dimension
//
void downsample(const std::vector<std::size_t> &outFieldShape,
                const unsigned samplingRate,
                const boost::multi_array<double, 3> &input,
                boost::multi_array<double, 3> &output);

} // namespace poplibs_test

#endif // poplibs_test_Multirate_hpp
