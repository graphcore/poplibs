// Copyright (c) Graphcore Ltd, All rights reserved.
#include "RandomUtils.hpp"

using namespace poplar;

namespace poprand {

template <typename FPType>
class DropoutSupervisor : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  DropoutSupervisor();

  Input<Vector<FPType, SCALED_PTR64, 8>> in;
  Output<Vector<FPType, SCALED_PTR64, 8>> out;
  const unsigned numElems;
  const FPType scale;
  const unsigned short prob;

  IS_EXTERNAL_CODELET(true);

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

    unsigned n = numElems;

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

} // namespace poprand
