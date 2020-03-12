// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <boost/multi_array.hpp>

namespace poplibs_test {
namespace embedding {

void multiSlice(const boost::multi_array<double, 2> &embeddingMatrix,
                const std::vector<unsigned> &indices,
                boost::multi_array<double, 2> &result);

void multiUpdateAdd(const boost::multi_array<double, 2> &deltas,
                    const std::vector<unsigned> &indices, const double scale,
                    boost::multi_array<double, 2> &embeddingMatrix);

} // namespace embedding
} // namespace poplibs_test
