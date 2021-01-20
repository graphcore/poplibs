// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplibs_test/CTCLoss.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/LogArithmetic.hpp>

#include <iostream>

namespace poplibs_test {
namespace ctc {

// When computing alpha we work through in time t0...tmax .
// When computing beta we work through in time tmax..t0 .
// Parents are those states in the trellis previous timestep.
// The child state is at the current timestep.  A child can have 1 to 3 parent
// states based on the rules below (Described for alpha)
// Subject to the bounds of the sequence we can always:
// a) Remain in the same state (parent = child)
// b) Progress 1 state (parent = child-1)
// and sometimes:
// c) Progress 2 states (parent = child-2 )
// At the `top` we of course can't come from a previous state that's negative
// Transition c) is only allowed if it doesn't skip a non-blank and doesn't
// skip a blank that is surrounded with identical characters.
//
// The above describes alpha, beta is the same in reverse:
// a) Remain in the same state (parent = child)
// b) Progress 1 state (parent = child+1)
// and sometimes:
// c) Progress 2 states (parent = child-+ )
// At the `bottom` we of course can't come from a previous state
// Transition c) is only allowed if it doesn't skip a non-blank and doesn't
// skip a blank that is surrounded with identical characters.

unsigned numberOfParents(const std::vector<unsigned> &paddedSequence,
                         unsigned index, unsigned blankIndex,
                         bool findAlphaParents) {
  unsigned parents = 0;
  // We can always maintain state
  parents++;

  if (findAlphaParents) {
    if (index > 1) {
      // We can skip a blank (index-1), if those either side of it
      // are not blanks (index and index-2)
      if (paddedSequence[index] != paddedSequence[index - 2] &&
          paddedSequence[index - 1] == blankIndex) {
        parents++;
      }
    }
    if (index != 0) {
      // We can always progress one state
      parents++;
    }
  } else {
    if (index != paddedSequence.size() - 1) {
      // We can always progress up one state
      parents++;
    }
    if (index < paddedSequence.size() - 2) {
      // We can skip a blank (index+1), if those either side of it
      // are not blanks (index and index+2)
      if (paddedSequence[index] != paddedSequence[index + 2] &&
          paddedSequence[index + 1] == blankIndex) {
        parents++;
      }
    }
  }
  // Return the number of valid parents - Implied that they are the same as this
  // index and parents-1 of those BEFORE for alpha, and AFTER for beta
  return parents;
}

// Compute the full alpha matrix, populating every time step.
//
// Each time step we consider the allowed paths into each state.
// The probability of reaching that state is given by:
// sum(probability(inputPaths)*probability(This symbol)
template <typename FPType>
boost::multi_array<FPType, 2>
alpha(const boost::multi_array<FPType, 2> &sequence,
      const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
      unsigned validTimesteps, bool logValues) {
  boost::multi_array<FPType, 2> alphas(
      boost::extents[sequence.size()][sequence[0].size()]);

  // Populate the first timestep alphas which are just the input probabilities
  alphas[0][0] = sequence[0][0];
  alphas[1][0] = sequence[1][0];
  for (unsigned j = 2; j < sequence.size(); j++) {
    alphas[j][0] = logValues ? log::min : 0;
  }

  // Iterate per column, starting with the second
  for (unsigned t = 1; t < validTimesteps; t++) {
    for (unsigned j = 0; j < sequence.size(); j++) {
      auto numParents = numberOfParents(paddedSequence, j, blankIndex, true);
      FPType sum = logValues ? log::min : 0;
      for (unsigned k = 0; k < numParents; k++) {
        const unsigned parent = j - k;
        if (logValues) {
          sum = log::add(sum, alphas[parent][t - 1]);
        } else {
          sum += alphas[parent][t - 1];
        }
      }
      if (logValues) {
        // Note that we are defining alpha as stored below, i.e:
        // alpha = sum(parentProbabilities) * probability
        // But if calculating gradient we could say that given:
        // grad = alpha * beta / probability
        // grad = sum(parentProbabilities) * beta
        // So we avoid the divide
        alphas[j][t] = log::mul(sum, sequence[j][t]);
      } else {
        alphas[j][t] = sum * sequence[j][t];
      }
    }
  }
  return alphas;
}

// Beta - basically the same as alpha but tracking legal paths from the last
// time slice to the first
template <typename FPType>
boost::multi_array<FPType, 2>
beta(const boost::multi_array<FPType, 2> &sequence,
     const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
     unsigned validTimesteps, bool logValues) {
  boost::multi_array<FPType, 2> betas(
      boost::extents[sequence.size()][sequence[0].size()]);

  const auto lastT = validTimesteps - 1;
  const auto lastL = sequence.size() - 1;
  // Populate the last timestep betas (beta starting point)
  betas[lastL][lastT] = sequence[lastL][lastT];
  betas[lastL - 1][lastT] = sequence[lastL - 1][lastT];
  for (unsigned j = 0; j < lastL - 1; j++) {
    betas[j][lastT] = logValues ? log::min : 0;
  }

  // Iterate per column, starting with the second to last
  for (unsigned i = 1; i < validTimesteps; i++) {
    auto t = validTimesteps - 1 - i;
    for (unsigned j = 0; j < sequence.size(); j++) {
      auto numParents = numberOfParents(paddedSequence, j, blankIndex, false);
      FPType sum = logValues ? log::min : 0;
      for (unsigned k = 0; k < numParents; k++) {
        const auto parent = j + k;
        if (logValues) {
          sum = log::add(sum, betas[parent][t + 1]);
        } else {
          sum += betas[parent][t + 1];
        }
      }
      if (logValues) {
        // Note that we are defining beta as stored below, i.e:
        // beta = sum(cparentProbabilities) * probability
        // But if calculating gradient we could say that given:
        // grad = alpha * beta / probability
        // grad = sum(parentProbabilities) * alpha
        // So we avoid the divide
        betas[j][t] = log::mul(sum, sequence[j][t]);
      } else {
        betas[j][t] = sum * sequence[j][t];
      }
    }
  }
  return betas;
}

// Note - not an accumulated gradient, the full input shape
template <typename FPType>
boost::multi_array<FPType, 2>
expandedGrad(const boost::multi_array<FPType, 2> &sequence,
             const boost::multi_array<FPType, 2> &alpha,
             const boost::multi_array<FPType, 2> &beta,
             const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
             unsigned validTimesteps, bool logValues) {

  // Result is the same shape as sequence, alphas and betas
  boost::multi_array<FPType, 2> gradient(
      boost::extents[sequence.shape()[0]][sequence.shape()[1]]);

  for (unsigned t = 0; t < validTimesteps; t++) {
    for (unsigned i = 0; i < sequence.shape()[0]; i++) {
      if (logValues) {
        auto alphaBeta = log::mul(alpha[i][t], beta[i][t]);
        gradient[i][t] = log::div(alphaBeta, sequence[i][t]);
      } else {
        auto alphaBeta = alpha[i][t] * beta[i][t];
        // To avoid numeric error
        gradient[i][t] = alphaBeta / (sequence[i][t] + 1e-50);
      }
    }
  }
  return gradient;
}

// Accumulated gradient - reduction of the results with the same symbol
template <typename FPType>
boost::multi_array<FPType, 2>
grad(const boost::multi_array<FPType, 2> &sequence,
     const boost::multi_array<FPType, 2> &alpha,
     const boost::multi_array<FPType, 2> &beta,
     const std::vector<unsigned> &paddedSequence, unsigned symbolsIncBlank,
     unsigned blankIndex, unsigned validTimesteps, bool logValues) {

  // A result: 1 row per valid symbol, 1 column per timestep
  // Initialise to zero as some symbols may be unused.
  boost::multi_array<FPType, 2> gradient(
      boost::extents[symbolsIncBlank][sequence.shape()[1]]);
  std::fill(gradient.data(), gradient.data() + gradient.num_elements(),
            logValues ? log::min : 0);

  for (unsigned t = 0; t < validTimesteps; t++) {
    for (unsigned i = 0; i < sequence.shape()[0]; i++) {
      if (logValues) {
        auto alphaBeta = log::mul(alpha[i][t], beta[i][t]);
        alphaBeta = log::div(alphaBeta, sequence[i][t]);
        gradient[paddedSequence[i]][t] =
            log::add(gradient[paddedSequence[i]][t], alphaBeta);
      } else {
        auto alphaBeta = alpha[i][t] * beta[i][t];
        // To avoid numeric error
        gradient[paddedSequence[i]][t] += alphaBeta / (sequence[i][t] + 1e-50);
      }
    }
  }
  return gradient;
}

template boost::multi_array<float, 2>
alpha(const boost::multi_array<float, 2> &sequence,
      const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
      unsigned validTimesteps, bool logValues);

template boost::multi_array<double, 2>
alpha(const boost::multi_array<double, 2> &sequence,
      const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
      unsigned validTimesteps, bool logValues);

template boost::multi_array<float, 2>
beta(const boost::multi_array<float, 2> &sequence,
     const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
     unsigned validTimesteps, bool logValues);

template boost::multi_array<double, 2>
beta(const boost::multi_array<double, 2> &sequence,
     const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
     unsigned validTimesteps, bool logValues);

template boost::multi_array<float, 2>
expandedGrad(const boost::multi_array<float, 2> &sequence,
             const boost::multi_array<float, 2> &alpha,
             const boost::multi_array<float, 2> &beta,
             const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
             unsigned validTimesteps, bool logValues);

template boost::multi_array<double, 2>
expandedGrad(const boost::multi_array<double, 2> &sequence,
             const boost::multi_array<double, 2> &alpha,
             const boost::multi_array<double, 2> &beta,
             const std::vector<unsigned> &paddedSequence, unsigned blankIndex,
             unsigned validTimesteps, bool logValues);

template boost::multi_array<float, 2>
grad(const boost::multi_array<float, 2> &sequence,
     const boost::multi_array<float, 2> &alpha,
     const boost::multi_array<float, 2> &beta,
     const std::vector<unsigned> &paddedSequence, unsigned symbolsIncBlank,
     unsigned blankIndex, unsigned validTimesteps, bool logValues);

template boost::multi_array<double, 2>
grad(const boost::multi_array<double, 2> &sequence,
     const boost::multi_array<double, 2> &alpha,
     const boost::multi_array<double, 2> &beta,
     const std::vector<unsigned> &paddedSequence, unsigned symbolsIncBlank,
     unsigned blankIndex, unsigned validTimesteps, bool logValues);

} // namespace ctc
} // namespace poplibs_test
