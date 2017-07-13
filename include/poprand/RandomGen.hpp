#ifndef __poprand_RandomGen_hpp__
#define __poprand_RandomGen_hpp__

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <utility>
#include <string>

namespace poprand {

enum RandomGenMode {
  // Numbers generated are always repeatable regardless of the system
  // on which the generation is executed.
  // This mode is not yet supported
  ALWAYS_REPEATABLE,
  // Numbers generated are repeatable on a fixed system. There is no guarantee
  // that numbers generated are repeatable across runs on different systems
  PLATFORM_REPEATABLE
};

/// Uniform distribution in [minVal, maxVal] with maxVal > minVal
/// The mode determines whether the numbers generated are
/// repeatable across systems
void uniform(poplar::Graph &graph, poplar::Tensor &A, float minVal,
             float maxVal, uint64_t seed, RandomGenMode mode,
             poplar::program::Sequence &prog,
             const std::string &debugPrefix = "");

/// Uniform distribution in [minVal, maxVal] with maxVal > minVal
/// Repeatability is not guaranteed
void uniform(poplar::Graph &graph, poplar::Tensor &A, float minVal,
             float maxVal, poplar::program::Sequence &prog,
             const std::string &debugPrefix = "");

/// Bernoulli with probablility of 1 = "prob"
/// The mode determines whether the numbers generated are
/// repeatable across systems
void bernoulli(poplar::Graph &graph, poplar::Tensor &A, float prob,
               uint64_t seed, RandomGenMode mode,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");

/// Bernoulli with probablility of 1 = "prob"
/// Repeatability is not guaranteed
void bernoulli(poplar::Graph &graph, poplar::Tensor &A, float prob,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");

/// Normal distribution with given mean and standard deviation
/// The mode determines whether the numbers generated are
/// repeatable across systems
void normal(poplar::Graph &graph, poplar::Tensor &A, float mean, float stdDev,
            uint64_t seed, RandomGenMode mode,
            poplar::program::Sequence &prog,
            const std::string &debugPrefix = "");

/// Normal distribution with given mean and standard deviation
/// Repeatability is not guaranteed
void normal(poplar::Graph &graph, poplar::Tensor &A, float mean, float stdDev,
            poplar::program::Sequence &prog,
            const std::string &debugPrefix = "");

/// Truncated normal distribution derived from a normal
/// distribution with mean "mean" and standard deviation
/// "stdDev". This normal distribution is truncated
/// symmetrically about the mean at:
///   (mean - alpha * stdDev) and (mean + alpha * stdDev)
/// The mode determines whether the numbers generated are
/// repeatable across systems
void truncatedNormal(poplar::Graph &graph, poplar::Tensor &A, float mean,
                     float stdDev, float alpha, uint64_t seed,
                     poplar::program::Sequence &prog,
                     const std::string &debugPrefix = "");

/// Truncated normal distribution derived from a normal
/// distribution with mean "mean" and standard deviation
/// "stdDev". This normal distribution is truncated
/// symmetrically about the mean at:
///   (mean - alpha * stdDev) and (mean + alpha * stdDev)
/// Repeatability is not guaranteed
void truncatedNormal(poplar::Graph &graph, poplar::Tensor &A, float mean,
                     float stdDev, float alpha, poplar::program::Sequence &prog,
                     const std::string &debugPrefix = "");
} // namespace poprand

#endif // __poprand_RandomGen_hpp__
