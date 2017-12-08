#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <array>
#include <cmath>

using namespace poplar;

template <typename T>
static const T &min(const T &x, const T &y) {
  return x < y ? x : y;
}

// number of warmup iterations the PRNG takes to have a random number of 0s
// and 1s in its state given any seed
#define WARMUP_ITERATIONS   4

// Rotate left a 64-bit register by k
static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// Update LFSR and return 64-bit random number
static uint64_t next(std::array<uint64_t, 2>  &s) {
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
template<typename T>
static T convertToUniform(uint64_t x) {
  const unsigned shift = std::is_same<T, half>::value ? 16 : 32;
  const auto r = x & ((1ULL << shift) - 1);
  const double scale = 1.0 / static_cast<double>(1ULL << shift);
  return static_cast<double>(r) * scale - 0.5;
}


// returns an array of 4 elements each of which is a gaussian approximation
// with zero mean and standard deviation of 1
static std::array<float, 4> grand(std::array<uint64_t, 2> &s) {
  std::array<float, 4> result;
  //this is not an exact model of the actual hardware
  for (auto i = 0U; i != 4; ++i) {
    auto r = next(s);
    unsigned acc = 0;
    for (auto j = 0U; j != 12; ++j, r >>= 5) {
      acc += r & 0x1F;
    }
    const auto gr = static_cast<float>(acc) - 6*31;
    result[i] = gr / 32.0;
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
static std::array<float, 4>
truncNormal(std::array<uint64_t, 2> &s, unsigned iterations, float alpha) {
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
        result[j] += alpha * convertToUniform<half>(r);
      }
    }
  }
  return result;
}


namespace poprand {

template <typename OutType>
class Uniform : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  float offset;
  float scale;
  // A separate vertex needs to be defined if save/restore of seeds is not
  // required
  Vector<uint32_t> vSeedH;
  Vector<uint32_t> vSeedL;
  SimOnlyField<unsigned> dataPathWidth;
  // It is expected that there will be two variants of vertices: one which
  // saves and restores seeds
  SimOnlyField<bool> saveRestoreSeed;

  bool compute() {
    uint64_t seedH =
      vSeedH[0] + (static_cast<uint64_t>(vSeedH[1]) << 32);
    uint64_t seedL =
      vSeedL[0] + (static_cast<uint64_t>(vSeedL[1]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<OutType, half>::value;
    const unsigned maxPerCall = isHalf ? 4 : 2;
    const unsigned bitsPerVal = isHalf ? 16 : 32;

    for (auto i = 0; i != out.size(); ++i) {
      unsigned n = out[i].size();
      unsigned idx = 0;
      while (n) {
        const unsigned genSamples =  min(n, maxPerCall);
        auto r = next(s);
        for (auto k = 0; k != genSamples; ++k, ++idx, r >>= bitsPerVal) {
          out[i][idx] = convertToUniform<OutType>(r) * scale + offset;
        }
        n -= genSamples;
      }
    }
    // save seeds
    vSeedL[0] = static_cast<uint32_t>(s[0]);
    vSeedL[1] = static_cast<uint32_t>(s[0] >> 32);
    vSeedH[0] = static_cast<uint32_t>(s[1]);
    vSeedH[1] = static_cast<uint32_t>(s[1] >> 32);
    return true;
  }
};

template class Uniform<float>;
template class Uniform<half>;


// Template specialisation for int
template <>
class Uniform<int> : public Vertex {
public:
  Vector<Output<Vector<int>>> out;
  int offset;
  // is the range of the uniform generator. Called scale because it can also
  // be seen as a scale factor for an uniform distribution [0,1) to produce the
  // integer
  unsigned int scale;
  // A separate vertex needs to be defined if save/restore of seeds is not
  // required
  Vector<uint32_t> vSeedH;
  Vector<uint32_t> vSeedL;
  SimOnlyField<unsigned> dataPathWidth;
  // It is expected that there will be two variants of vertices: one which
  // saves and restores seeds
  SimOnlyField<bool> saveRestoreSeed;
  bool compute() {
    uint64_t seedH =
      vSeedH[0] + (static_cast<uint64_t>(vSeedH[1]) << 32);
    uint64_t seedL =
      vSeedL[0] + (static_cast<uint64_t>(vSeedL[1]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    const unsigned maxPerCall = 2;
    const unsigned bitsPerVal = 32;
    for (auto i = 0; i != out.size(); ++i) {
      unsigned n = out[i].size();
      unsigned idx = 0;
      while (n) {
        const unsigned genSamples =  min(n, maxPerCall);
        auto r = next(s);
        for (auto k = 0; k != genSamples; ++k, ++idx, r >>= bitsPerVal) {
          uint64_t rmasked = r & ((1ULL << bitsPerVal) - 1);
          // scale == 0 is the special case where whole range of int is used
          if (scale != 0) {
            rmasked = (rmasked * scale) >> 32;
          }
          int64_t res32 = static_cast<int64_t>(rmasked) + offset;
          out[i][idx] = res32;;
        }
        n -= genSamples;
      }
    }
    // save seeds
    vSeedL[0] = static_cast<uint32_t>(s[0]);
    vSeedL[1] = static_cast<uint32_t>(s[0] >> 32);
    vSeedH[0] = static_cast<uint32_t>(s[1]);
    vSeedH[1] = static_cast<uint32_t>(s[1] >> 32);
    return true;
  }
};


template <typename OutType>
class Bernoulli : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  float prob;
  // A separate vertex needs to be defined if save/restore of seeds is not
  // required
  Vector<uint32_t> vSeedH;
  Vector<uint32_t> vSeedL;
  SimOnlyField<unsigned> dataPathWidth;
  // It is expected that there will be two variants of vertices: one which
  // saves and restores seeds
  SimOnlyField<bool> saveRestoreSeed;

  bool compute() {
    uint64_t seedH =
      vSeedH[0] + (static_cast<uint64_t>(vSeedH[1]) << 32);
    uint64_t seedL =
      vSeedL[0] + (static_cast<uint64_t>(vSeedL[1]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<OutType, half>::value;
    const unsigned maxPerCall = isHalf ? 4 : 2;
    const unsigned bitsPerVal = isHalf ? 16 : 32;

    uint64_t probToCode = prob * (1ULL << bitsPerVal);

    for (auto i = 0; i != out.size(); ++i) {
      unsigned n = out[i].size();
      unsigned idx = 0;
      while (n) {
        const unsigned genSamples =  min(n, maxPerCall);
        auto r = next(s);
        for (auto k = 0; k != genSamples; ++k, ++idx, r >>= bitsPerVal) {
          const uint64_t thisVal = r & ((1ULL << bitsPerVal) - 1);
          out[i][idx] = (thisVal <  probToCode);
        }
        n -= genSamples;
      }
    }
    // save seeds
    vSeedL[0] = static_cast<uint32_t>(s[0]);
    vSeedL[1] = static_cast<uint32_t>(s[0] >> 32);
    vSeedH[0] = static_cast<uint32_t>(s[1]);
    vSeedH[1] = static_cast<uint32_t>(s[1] >> 32);
    return true;
  }
};

template class Bernoulli<float>;
template class Bernoulli<half>;
template class Bernoulli<int>;

template <typename OutType>
class Normal : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  float mean;               // mean of normal distribution
  float stdDev;             // standard deviation of normal distribution
  // A separate vertex needs to be defined if save/restore of seeds is not
  // required
  Vector<uint32_t> vSeedH;
  Vector<uint32_t> vSeedL;
  SimOnlyField<unsigned> dataPathWidth;
  // It is expected that there will be two variants of vertices: one which
  // saves and restores seeds
  SimOnlyField<bool> saveRestoreSeed;

  bool compute() {
    uint64_t seedH =
      vSeedH[0] + (static_cast<uint64_t>(vSeedH[1]) << 32);
    uint64_t seedL =
      vSeedL[0] + (static_cast<uint64_t>(vSeedL[1]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<OutType, half>::value;
    const unsigned maxPerCall = isHalf ? 4 : 2;
    for (auto i = 0U; i != out.size(); ++i) {
      unsigned n = out[i].size();
      unsigned idx = 0;
      while (n) {
        const unsigned genSamples = min(n, maxPerCall);
        const auto grandVec = grand(s);
        for (auto k = 0; k != genSamples; ++k, ++idx) {
          out[i][idx] = grandVec[k] * stdDev + mean;
        }
        n -= genSamples;
      }
    }
    // save seeds
    vSeedL[0] = static_cast<uint32_t>(s[0]);
    vSeedL[1] = static_cast<uint32_t>(s[0] >> 32);
    vSeedH[0] = static_cast<uint32_t>(s[1]);
    vSeedH[1] = static_cast<uint32_t>(s[1] >> 32);
    return true;
  }
};

template class Normal<float>;
template class Normal<half>;

template <typename OutType>
class TruncatedNormal : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  unsigned iterations;   // number of iterations of generate and replace
  float mean;            // mean of symmetric truncated normal distribution
  float stdDev;          // stdDev of original normal distribution which is
                         // truncated
  float alpha;           // truncation as a multiple of stdDev
  // A separate vertex needs to be defined if save/restore of seeds is not
  // required
  Vector<uint32_t> vSeedH;
  Vector<uint32_t> vSeedL;
  SimOnlyField<unsigned> dataPathWidth;
  // It is expected that there will be two variants of vertices: one which
  // saves and restores seeds
  SimOnlyField<bool> saveRestoreSeed;

  bool compute() {
    uint64_t seedH =
      vSeedH[0] + (static_cast<uint64_t>(vSeedH[1]) << 32);
    uint64_t seedL =
      vSeedL[0] + (static_cast<uint64_t>(vSeedL[1]) << 32);
    auto s = initialiseAndPrime({seedL, seedH});
    bool isHalf = std::is_same<OutType, half>::value;
    const unsigned maxPerCall = isHalf ? 4 : 2;
    for (auto i = 0U; i != out.size(); ++i) {
      unsigned n = out[i].size();
      unsigned idx = 0;
      while (n) {
        const unsigned genSamples =  min(n, maxPerCall);
        const auto grandVec = truncNormal(s, iterations, alpha);
        for (auto k = 0; k != genSamples; ++k, ++idx) {
          out[i][idx] = grandVec[k] * stdDev + mean;
        }
        n -= genSamples;
      }
    }
    // save seeds
    seedL = s[0];
    seedH = s[1];
    return true;
  }
};

template class TruncatedNormal<float>;
template class TruncatedNormal<half>;

} // end namespace poprand
