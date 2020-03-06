// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;

namespace poprand {

template <typename T> static const T &min(const T &x, const T &y) {
  return x < y ? x : y;
}

// number of warmup iterations the PRNG takes to have a random number of 0s
// and 1s in its state given any seed
#define WARMUP_ITERATIONS 4

// Rotate left a 64-bit register by k
static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// Update LFSR and return 64-bit random number
static uint64_t next(std::array<uint64_t, 2> &s) {
  uint64_t s0 = s[0];
  uint64_t s1 = s[1];

  // TODO: T12981 use the actual non linear operation used in hardware
  const uint64_t result = s0 + s1;

  s1 ^= s0;
  s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
  s[1] = rotl(s1, 36);
  return result;
}

// Initialise register with seed value
static std::array<uint64_t, 2>
initialiseAndPrime(const std::array<uint64_t, 2> &seed) {
  const unsigned primeCount = WARMUP_ITERATIONS;
  std::array<uint64_t, 2> s = seed;
  for (auto i = 0U; i != primeCount; ++i) {
    next(s);
  }
  return s;
}

// returns an uniform number in the interval [0.5,0.5]
template <typename T> static T convertToUniform(uint64_t x) {
  const unsigned shift = std::is_same<T, half>::value ? 16 : 32;
  const auto r = x & ((1ULL << shift) - 1);
  const double scale = 1.0 / static_cast<double>(1ULL << shift);
  return static_cast<double>(r) * scale - 0.5;
}

// returns an array of 4 elements each of which is a gaussian approximation
// with zero mean and standard deviation of 1
static std::array<float, 4> grand(std::array<uint64_t, 2> &s) {
  std::array<float, 4> result;
  // this is not an exact model of the actual hardware
  for (auto i = 0U; i != 4; ++i) {
    auto r = next(s);
    unsigned acc = 0;
    for (auto j = 0U; j != 12; ++j, r >>= 5) {
      acc += r & 0x1F;
    }
    const auto gr = static_cast<float>(acc) - 6 * 31;
    result[i] = gr / 32.0f;
  }
  return result;
}

// Returns an array of truncated normal distribution with the underlying
// normal distribution with zero mean and standard deviation of 1.
// A normal sample is picked for a fixed number of iterations until the
// sample is within the truncation bounds.
// The samples which exceed bounds are then filled with sample with a
// triangular probability. As an optimisation, these samples could be
// picked from an uniform distribution if alpha is less than a certain value.
static std::array<float, 4> truncNormal(std::array<uint64_t, 2> &s,
                                        unsigned iterations, float alpha) {
  std::array<float, 4> result;
  std::array<bool, 4> mask;
  result.fill(0);
  mask.fill(false);
  for (auto i = 0U; i != iterations; ++i) {
    const auto x = grand(s);
    for (auto j = 0U; j != 4; ++j) {
      if (!mask[j]) {
        result[j] = x[j];
      }
      if (result[j] > alpha || result[j] < -alpha) {
        result[j] = 0;
      } else {
        mask[j] = true;
      }
    }
  }
  // finally replace zero's by triangular or uniform distribution. For large
  // alpha it is better to use triangular. Only the worst case is shown here.
  for (auto i = 0U; i != 2; ++i) {
    uint64_t r = next(s);
    for (auto j = 0U; j != 4; ++j, r >>= 16) {
      if (mask[j] == false) {
        result[j] += alpha * static_cast<float>(convertToUniform<half>(r));
      }
    }
  }
  return result;
}

} // namespace poprand
