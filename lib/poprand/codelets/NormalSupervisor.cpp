// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "RandomUtils.hpp"

using namespace poplar;

namespace poprand {

template <typename OutType>
class NormalSupervisor : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  NormalSupervisor();

  Output<Vector<OutType, SPAN, 8>> out;
  const float mean;   // mean of normal distribution
  const float stdDev; // standard deviation of normal distribution

  // SimOnlyField<bool> saveRestoreSeed;

  IS_EXTERNAL_CODELET(true);

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

} // namespace poprand
