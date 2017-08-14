#ifndef __poprand_RandomGen_hpp__
#define __poprand_RandomGen_hpp__

#include "popstd/exceptions.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <utility>
#include <string>
#include <cstdint>
#include <array>

namespace poprand {

// Each of the random generation functions can be given a seed.
// If a seed is given the behaviour is as follows:
//  - If mode is ALWAYS_REPEATABLE, given the same seed will generate the same
//    data across multiple calls of a given generation function
//  - In SYSTEM_REPEATABLE and NOT_REPEATABLE modes, uncorrelated data is
//    generated across calls
//  - Even if the same seed is given for different generation functions,
//    uncorrelated dats is generated regardless of the mode
//
// If a seed is not given
//  - If mode is ALWAYS_REPEATABLE, data generated for a given random generation
//    function will be different across calls but data is guaranteed to be the
//    same for the same call
//  - For other modes, data generated is uncorrelated across calls

enum RandomGenMode {
  /// Numbers generated are always repeatable regardless of the system
  /// on which the generation is executed.
  /// This mode is the slowest of the modes
  ALWAYS_REPEATABLE,
  /// Numbers generated are repeatable on a fixed system. There is no guarantee
  /// that numbers generated are repeatable across runs on different systems
  /// This mode is faster than ALWAYS_REPEATABLE but slower than NOT_REPEATABLE
  SYSTEM_REPEATABLE,
  /// Generates random but data which is not repeatable across
  /// runs This mode is the fastest amongst all the modes
  NOT_REPEATABLE
};

class Random {
  RandomGenMode mode = NOT_REPEATABLE;
  uint64_t seed = 0xAA99330FCC55E1B4ULL;
  std::vector<uint16_t> callCount;
public:
  Random(RandomGenMode mode, uint64_t seed);
  Random();
  Random(RandomGenMode mode);
  RandomGenMode getMode() const { return mode; }

  /// Uniform distribution in a given interval with maxVal >
  /// minVal The mode determines whether the numbers generated are
  /// repeatable across systems
  ///
  /// The tensor A may be of type "float", "half" or "int". It generates data
  /// with uniform distribution in the interval [minVal, maxVal]. For "int",
  /// data is generated in interval [minVal, maxVal] with uniform probability if
  /// maxVa - minVal is a power of 2. Otherwise there will be a small bias in
  /// the probability generated with the bias directly proportional to the ratio
  /// maxVal-minVal+1 / 2^32.
  void
  uniform(poplar::Graph &graph, poplar::Tensor &A, double minVal, double maxVal,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "");

  /// Bernoulli with probablility of 1 = "prob"
  /// The mode determines whether the numbers generated are
  /// repeatable across systems
  /// Tensor types supported are "float", "half" and "int"
  void
  bernoulli(poplar::Graph &graph, poplar::Tensor &A, double prob,
            poplar::program::Sequence &prog,
            const std::string &debugPrefix = "");

  /// Normal distribution with given mean and standard deviation
  /// The mode determines whether the numbers generated are
  /// repeatable across systems
  /// The tensor A may be of type "half" and "float"
  void
  normal(poplar::Graph &graph, poplar::Tensor &A, double mean, double stdDev,
         poplar::program::Sequence &prog, const std::string &debugPrefix = "");

  /// Truncated normal distribution derived from a normal
  /// distribution with mean "mean" and standard deviation
  /// "stdDev". This normal distribution is truncated
  /// symmetrically about the mean at:
  ///   (mean - alpha * stdDev) and (mean + alpha * stdDev)
  /// The mode determines whether the numbers generated are
  /// repeatable across systems
  /// The tensor A may be of type "half" and "float"
  void
  truncatedNormal(poplar::Graph &graph, poplar::Tensor &A, double mean,
                  double stdDev, double alpha, poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");
};

}// namespace poprand

#endif // __poprand_RandomGen_hpp__
