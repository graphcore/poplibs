// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <boost/multi_array.hpp>

namespace poplibs_test {
namespace embedding {

template <typename FPType>
void multiSlice(const boost::multi_array<FPType, 2> &embeddingMatrix,
                const std::vector<unsigned> &indices,
                boost::multi_array<FPType, 2> &result);

template <typename FPType>
void multiUpdateAdd(const boost::multi_array<FPType, 2> &deltas,
                    const std::vector<unsigned> &indices, const FPType scale,
                    boost::multi_array<FPType, 2> &embeddingMatrix);

} // namespace embedding
} // namespace poplibs_test
