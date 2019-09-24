// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poprand_RandomGen_hpp
#define poprand_RandomGen_hpp

#include "poputil/exceptions.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <utility>
#include <string>
#include <cstdint>
#include <array>

namespace poprand {

/// Apply dropout to input tensor
///
/// The \a input tensor is multiplied by a sequence of 1 or 0. The
/// probability of the dropout is P(1) = \a dropoutProbability. The reference
/// tensor must be of the same shape as the input. The layout of the output is
/// the same as the reference to guarantee that if the same seed and
/// \a seedModifier is given then the same mask is generated.
/// The scale factor scales the input tensor and should
/// typically be the inverse of the dropout probability.
/// If \a seed is not null, the seed which is a tensor of shape {2} is used to
/// seed the generator to generate the dropout mask. The h/w random generator
/// state at the end of dropout is restored to be the same as before it is
/// applied.
poplar::Tensor
dropout(poplar::Graph &graph,
        const poplar::Tensor *seed,
        const uint32_t seedModifier,
        const poplar::Tensor &input,
        const poplar::Tensor &reference,
        double dropoutProbability,
        double scale,
        poplar::program::Sequence &prog,
        const std::string &debugPrefix = "");


/// Uniform distribution in a given interval with maxVal > minVal
///
/// The tensor \a A may be of type "float", "half" or "int". It generates data
/// with uniform distribution in the interval [minVal, maxVal]. For "int",
/// data is generated in interval [minVal, maxVal] with uniform probability if
/// maxVal - minVal is a power of 2. Otherwise there will be a small bias in
/// the probability generated with the bias directly proportional to the ratio
/// maxVal-minVal+1 / 2^32.
///
/// The output has the same shape and mapping as the reference to guarantee that
/// the same output is generated if the \a seed and \a seedModifier is used.
///
/// If \a seed is not null, the seed which is a tensor of shape {2} is used to
/// seed the generator. The h/w random generator state at the end of generation
/// is restored to be the same as before it is applied. \a seedModifier is
/// ignored if \a seed is a nullptr.
poplar::Tensor
uniform(poplar::Graph &graph,
        const poplar::Tensor *seed,
        uint32_t seedModifier,
        const poplar::Tensor &reference,
        const poplar::Type &outType,
        double minVal,
        double maxVal,
        poplar::program::Sequence &prog,
        const std::string &debugPrefix = "");

/// Bernoulli with probablility of 1 = "prob"
/// Tensor types supported are "float", "half" and "int"
///
/// The output has the same shape and mapping as the reference to guarantee that
/// the same output is generated if the \a seed and \a seedModifier is used.
///
/// If \a seed is not null, the seed which is a tensor of shape {2} is used to
/// seed the generator. The h/w random generator state at the end of generation
/// is restored to be the same as before it is applied. \a seedModifier is
/// ignored if \a seed is a nullptr.
poplar::Tensor
bernoulli(poplar::Graph &graph,
          const poplar::Tensor *seed,
          uint32_t seedModifier,
          const poplar::Tensor &reference,
          const poplar::Type &outType,
          double prob,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "");

/// Normal distribution with given mean and standard deviation
/// The tensor A may be of type "half" and "float"
///
/// The output has the same shape and mapping as the reference to guarantee that
/// the same output is generated if the \a seed and \a seedModifier is used.
///
/// If \a seed is not null, the seed which is a tensor of shape {2} is used to
/// seed the generator. The h/w random generator state at the end of generation
/// is restored to be the same as before it is applied. \a seedModifier is
/// ignored if \a seed is a nullptr.
poplar::Tensor
normal(poplar::Graph &graph,
       const poplar::Tensor *seed,
       uint32_t seedModifier,
       const poplar::Tensor &reference,
       const poplar::Type &outType,
       double mean,
       double stdDev,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix = "");

/// Truncated normal distribution derived from a normal distribution with mean
/// \a mean and standard deviation \a stdDev. This normal distribution is
/// truncated symmetrically about the mean at:
///   (mean - alpha * stdDev) and (mean + alpha * stdDev)
///
/// The tensor A may be of type "half" and "float"
///
/// The output has the same shape and mapping as the reference to guarantee that
/// the same output is generated if the \a seed and \a seedModifier is used.
///
/// If \a seed is not null, the seed which is a tensor of shape {2} is used to
/// seed the generator. The h/w random generator state at the end of generation
/// is restored to be the samethe same as before it is applied. \a seedModifier
/// is ignored if \a seed is a nullptr.
poplar::Tensor
truncatedNormal(poplar::Graph &graph,
                const poplar::Tensor *seed,
                uint32_t seedModifier,
                const poplar::Tensor &reference,
                const poplar::Type &outType,
                double mean,
                double stdDev,
                double alpha,
                poplar::program::Sequence &prog,
                const std::string &debugPrefix = "");

/// Sets the seed on all tiles given a single tensor of shape {2} and of type
/// UNSIGNED_INT, and \a seedModifier.
void setSeed(poplar::Graph &graph,
             const poplar::Tensor &masterSeed,
             uint32_t seedModifier,
             poplar::program::Sequence &prog,
             const std::string &debugPrefix = "");

}// namespace poprand

#endif // poprand_RandomGen_hpp
