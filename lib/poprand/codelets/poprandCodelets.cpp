#include "poplibs_support/ExternalCodelet.hpp"
#include "print.h"
#include <array>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;

#if defined(__IPU__) && !defined(POPLIBS_DISABLE_ASM_CODELETS)
#define EXTERNAL_CODELET true
#else
#define EXTERNAL_CODELET false
#endif

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

  // TODO: use the actual non linear operation used in hardware
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

namespace poprand {

template <typename OutType> class UniformSupervisor : public SupervisorVertex {
public:
  UniformSupervisor();

  Output<Vector<OutType, SPAN, 8>> out;
  const float offset;
  const float scale;
  const unsigned int shift;

  static const bool isExternalCodelet = EXTERNAL_CODELET;

  bool compute() {
    uint32_t seed[2] = {0xDEADBEEF, 0xBEEFDEAD};
    uint32_t seedModifier = 0x900DDEED;

    uint64_t seedH = seed[0] + (static_cast<uint64_t>(seed[1]) << 32);
    uint64_t seedL = seed[1] + (static_cast<uint64_t>(seed[0]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<OutType, half>::value;
    const unsigned maxPerCall = isHalf ? 4 : 2;
    const unsigned bitsPerVal = isHalf ? 16 : 32;

    unsigned n = out.size();
    unsigned idx = 0;
    while (n) {
      const unsigned genSamples = min(n, maxPerCall);
      auto r = next(s);
      for (auto k = 0; k != genSamples; ++k, ++idx, r >>= bitsPerVal) {
        out[idx] =
            static_cast<float>(convertToUniform<OutType>(r)) * scale + offset;
      }
      n -= genSamples;
    }
    return true;
  }
};

template class UniformSupervisor<float>;
template class UniformSupervisor<half>;

// Template specialisation for int
template <> class UniformSupervisor<int> : public SupervisorVertex {
public:
  UniformSupervisor();

  Output<Vector<int, SPAN, 8>> out;
  const int offset;
  // is the range of the uniform generator. Called scale because it can also
  // be seen as a scale factor for an uniform distribution [0,1) to produce the
  // integer
  const unsigned int scale;
  const unsigned int shift;

  static const bool isExternalCodelet = EXTERNAL_CODELET;

  bool compute() {
    uint32_t seed[2] = {0xDEADBEEF, 0xBEEFDEAD};
    uint32_t seedModifier = 0x900DDEED;

    uint64_t seedH = seed[0] + (static_cast<uint64_t>(seed[1]) << 32);
    uint64_t seedL = seed[1] + (static_cast<uint64_t>(seed[0]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    const unsigned maxPerCall = 2;
    const unsigned bitsPerVal = 32;
    unsigned n = out.size();
    unsigned idx = 0;
    while (n) {
      const unsigned genSamples = min(n, maxPerCall);
      auto r = next(s);
      for (auto k = 0; k != genSamples; ++k, ++idx, r >>= bitsPerVal) {
        uint64_t rmasked = r & ((1ULL << bitsPerVal) - 1);
        // scale == 0 is the special case where whole range of int is used
        if (scale != 0) {
          rmasked = ((rmasked >> 8) * scale) >> shift;
        }
        int64_t res32 = static_cast<int64_t>(rmasked) + offset;
        out[idx] = res32;
      }
      n -= genSamples;
    }
    return true;
  }
};

template <typename OutType>
class BernoulliSupervisor : public SupervisorVertex {
public:
  BernoulliSupervisor();

  Output<Vector<OutType, SPAN, 8>> out;
  const unsigned prob;

  static const bool isExternalCodelet = EXTERNAL_CODELET;

  bool compute() {
    uint32_t seed[2] = {0xDEADBEEF, 0xBEEFDEAD};
    uint32_t seedModifier = 0x900DDEED;

    uint64_t seedH = seed[0] + (static_cast<uint64_t>(seed[1]) << 32);
    uint64_t seedL = seed[1] + (static_cast<uint64_t>(seed[0]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<OutType, half>::value;
    const unsigned maxPerCall = isHalf ? 4 : 2;
    const unsigned bitsPerVal = isHalf ? 16 : 32;

    // rmask instruction takes the probability as int16
    uint64_t probToCode = prob * (1ULL << (bitsPerVal - 16));

    unsigned n = out.size();
    unsigned idx = 0;
    while (n) {
      const unsigned genSamples = min(n, maxPerCall);
      auto r = next(s);
      for (auto k = 0; k != genSamples; ++k, ++idx, r >>= bitsPerVal) {
        const uint64_t thisVal = r & ((1ULL << bitsPerVal) - 1);
        out[idx] = (thisVal < probToCode);
      }
      n -= genSamples;
    }
    return true;
  }
};

template class BernoulliSupervisor<float>;
template class BernoulliSupervisor<half>;
template class BernoulliSupervisor<int>;

template <typename OutType> class NormalSupervisor : public SupervisorVertex {
public:
  NormalSupervisor();

  Output<Vector<OutType, SPAN, 8>> out;
  const float mean;   // mean of normal distribution
  const float stdDev; // standard deviation of normal distribution

  // SimOnlyField<bool> saveRestoreSeed;

  static const bool isExternalCodelet = EXTERNAL_CODELET;

  bool compute() {
    uint32_t seed[2] = {0xDEADBEEF, 0xBEEFDEAD};
    uint32_t seedModifier = 0x900DDEED;

    uint64_t seedH = seed[0] + (static_cast<uint64_t>(seed[1]) << 32);
    uint64_t seedL = seed[1] + (static_cast<uint64_t>(seed[0]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<OutType, half>::value;
    const unsigned maxPerCall = isHalf ? 4 : 2;
    unsigned n = out.size();
    unsigned idx = 0;
    while (n) {
      const unsigned genSamples = min(n, maxPerCall);
      const auto grandVec = grand(s);
      for (auto k = 0; k != genSamples; ++k, ++idx) {
        out[idx] = grandVec[k] * stdDev + mean;
      }
      n -= genSamples;
    }
    return true;
  }
};

template class NormalSupervisor<float>;
template class NormalSupervisor<half>;

template <typename OutType>
class TruncatedNormalSupervisor : public SupervisorVertex {
public:
  TruncatedNormalSupervisor();

  Output<Vector<OutType, SPAN, 8>> out;
  const float mean;          // mean of symmetric truncated normal distribution
  const float stdDev;        // stdDev of original normal distribution which is
                             // truncated
  const float alpha;         // truncation as a multiple of stdDev
  const unsigned iterations; // number of iterations of generate and replace

  static const bool isExternalCodelet = EXTERNAL_CODELET;

  bool compute() {
    uint32_t seed[2] = {0xDEADBEEF, 0xBEEFDEAD};
    uint32_t seedModifier = 0x900DDEED;

    uint64_t seedH = seed[0] + (static_cast<uint64_t>(seed[1]) << 32);
    uint64_t seedL = seed[1] + (static_cast<uint64_t>(seed[0]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<OutType, half>::value;
    const unsigned maxPerCall = isHalf ? 4 : 2;

    unsigned n = out.size();
    unsigned idx = 0;
    while (n) {
      const unsigned genSamples = min(n, maxPerCall);
      const auto grandVec = truncNormal(s, iterations, alpha);
      for (auto k = 0; k != genSamples; ++k, ++idx) {
        out[idx] = grandVec[k] * stdDev + mean;
      }
      n -= genSamples;
    }
    return true;
  }
};

template class TruncatedNormalSupervisor<float>;
template class TruncatedNormalSupervisor<half>;

template <typename FPType> class DropoutSupervisor : public SupervisorVertex {
public:
  DropoutSupervisor();

  Input<Vector<FPType, SPAN, 8>> in;
  Output<Vector<FPType, SPAN, 8>> out;
  const FPType scale;
  const unsigned prob;

  static const bool isExternalCodelet = EXTERNAL_CODELET;

  bool compute() {
    uint32_t seed[2] = {0xDEADBEEF, 0xBEEFDEAD};
    uint32_t seedModifier = 0x900DDEED;
    uint64_t seedL =
        (seed[0] + (static_cast<uint64_t>(seed[0]) << 32)) ^ seedModifier;
    uint64_t seedH =
        (seed[1] + (static_cast<uint64_t>(seed[1]) << 32)) ^ ~seedModifier;
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<FPType, half>::value;

    const unsigned maxPerCall = isHalf ? 4 : 2;
    const unsigned bitsPerVal = isHalf ? 16 : 32;

    unsigned n = in.size();

    unsigned idx = 0;
    while (n) {
      const unsigned genSamples = min(n, maxPerCall);
      auto r = next(s);
      for (auto k = 0; k != genSamples; ++k, ++idx, r >>= bitsPerVal) {
        const uint64_t thisVal = r & ((1ULL << 16) - 1);
        float x = (thisVal < prob) * (float)in[idx] * (float)scale;
        out[idx] = x;
      }
      n -= genSamples;
    }
    return true;
  }
};

template class DropoutSupervisor<float>;
template class DropoutSupervisor<half>;

class SetSeedSupervisor : public SupervisorVertex {
public:
  SetSeedSupervisor();

  Input<Vector<unsigned, ONE_PTR, 8>> seed;
  const uint32_t seedModifierUser;
  const uint32_t seedModifierHw;

  static const bool isExternalCodelet = EXTERNAL_CODELET;

  bool compute() { return true; }
};

} // end namespace poprand
