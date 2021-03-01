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
      unsigned validTimesteps);

template <typename FPType>
boost::multi_array<FPType, 2>
beta(const boost::multi_array<FPType, 2> &sequence,
     const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
     unsigned validTimesteps);

template <typename FPType>
FPType loss(const boost::multi_array<FPType, 2> &sequence,
            const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
            unsigned validTimesteps);

template <typename FPType>
boost::multi_array<FPType, 2>
expandedGrad(const boost::multi_array<FPType, 2> &sequence,
             const boost::multi_array<FPType, 2> &alpha,
             const boost::multi_array<FPType, 2> &beta,
             const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
             unsigned validTimesteps);

// TODO: Not really CTC grad, as this returns Prob * ctcGrad
template <typename FPType>
boost::multi_array<FPType, 2>
ctcGrad(const boost::multi_array<FPType, 2> &sequence,
        const boost::multi_array<FPType, 2> &alpha,
        const boost::multi_array<FPType, 2> &beta,
        const std::vector<unsigned> &paddedSequence, unsigned symbolsIncBlank,
        unsigned blankIndex, unsigned validTimesteps);

template <typename FPType>
boost::multi_array<FPType, 2>
grad(const boost::multi_array<FPType, 2> &sequence,
     const boost::multi_array<FPType, 2> &logProbs,
     const boost::multi_array<FPType, 2> &alpha,
     const boost::multi_array<FPType, 2> &beta,
     const std::vector<unsigned> &paddedSequence, unsigned symbolsIncBlank,
     unsigned blankIndex, unsigned validTimesteps,
     bool testReducedCodeletGradient);

} // namespace ctc
} // namespace poplibs_test

#endif // poplibs_test_ctc_loss_hpp
