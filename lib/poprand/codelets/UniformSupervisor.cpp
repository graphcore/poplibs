// Copyright (c) Graphcore Ltd, All rights reserved.
#include "RandomUtils.hpp"

using namespace poplar;

namespace poprand {

template <typename OutType>
class UniformSupervisor : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  UniformSupervisor();

  Output<Vector<OutType, SPAN, 8>> out;
  const float offset;
  const float scale;
  const unsigned int shift;

  IS_EXTERNAL_CODELET(true);

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
template <>
class UniformSupervisor<int> : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  UniformSupervisor();

  Output<Vector<int, SPAN, 8>> out;
  const int offset;
  // is the range of the uniform generator. Called scale because it can also
  // be seen as a scale factor for an uniform distribution [0,1) to produce the
  // integer
  const unsigned int scale;
  const unsigned int shift;

  IS_EXTERNAL_CODELET(true);

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

} // namespace poprand
