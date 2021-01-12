// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_ctc_loss_hpp
#define poplibs_test_ctc_loss_hpp

#include "CTCUtil.hpp"

#include <boost/multi_array.hpp>

namespace poplibs_test {
namespace ctc {

template <typename FPType>
boost::multi_array<FPType, 2>
alpha(const boost::multi_array<FPType, 2> &sequence,
      const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
      unsigned validTimesteps, bool logValues);

template <typename FPType>
boost::multi_array<FPType, 2>
beta(const boost::multi_array<FPType, 2> &sequence,
     const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
     unsigned validTimesteps, bool logValues);

template <typename FPType>
boost::multi_array<FPType, 2>
expandedGrad(const boost::multi_array<FPType, 2> &sequence,
             const boost::multi_array<FPType, 2> &alpha,
             const boost::multi_array<FPType, 2> &beta,
             const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
             unsigned validTimesteps, bool logValues);

template <typename FPType>
boost::multi_array<FPType, 2>
grad(const boost::multi_array<FPType, 2> &sequence,
     const boost::multi_array<FPType, 2> &alpha,
     const boost::multi_array<FPType, 2> &beta,
     const std::vector<unsigned> &paddedSequence, unsigned symbolsIncBlank,
     unsigned blankIndex, unsigned validTimesteps, bool logValues);

} // namespace ctc
} // namespace poplibs_test

#endif // poplibs_test_ctc_loss_hpp
