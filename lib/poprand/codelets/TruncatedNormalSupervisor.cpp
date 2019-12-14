// Copyright (c) Graphcore Ltd, All rights reserved.
#include "RandomUtils.hpp"

namespace poprand {

template <typename OutType>
class TruncatedNormalSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  TruncatedNormalSupervisor();

  Output<Vector<OutType, SPAN, 8>> out;
  const float mean;          // mean of symmetric truncated normal distribution
  const float stdDev;        // stdDev of original normal distribution which is
                             // truncated
  const float alpha;         // truncation as a multiple of stdDev
  const unsigned iterations; // number of iterations of generate and replace

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

} // namespace poprand
