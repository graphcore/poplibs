#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;


#if defined(__IPU__) && !defined(POPLIBS_DISABLE_ASM_CODELETS)
#define EXTERNAL_CODELET true
#else
#define EXTERNAL_CODELET false
#endif

namespace popops {

template <class FPType>
class
AddToChannel : public SupervisorVertex {
public:
  AddToChannel();

  Input<Vector<FPType, SPAN, 8>> addend;
  InOut<Vector<FPType, ONE_PTR, 8, true>> acts;
  // actsBlockCount = acts.size() / addend.size();
  // actsBlockCountPacked = (actsBlockCount/6 << 3) | (actsBlockCount % 6)
  const uint16_t actsBlockCountPacked;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = addend.size();
    unsigned actsBlockCount = (actsBlockCountPacked >> 3) * 6
                              + (actsBlockCountPacked & 0x07);
    for (unsigned j = 0; j != actsBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        acts[j * chansPerGroup + k] += addend[k];
      }
    }
    return true;
  }
};
template class AddToChannel<float>;
template class AddToChannel<half>;

template <class FPType>
class
AddToChannel2D : public Vertex {
public:
  AddToChannel2D();

  // n is equal to addend.size(), addendLen.size(), acts.size()
  // and actsBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> addend;
  Vector<uint16_t, ONE_PTR> addendLen;
  Vector<InOut<Vector<FPType, ONE_PTR, 8, true>>, ONE_PTR> acts;
  Vector<uint16_t, ONE_PTR> actsBlockCount;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
        unsigned blockCount = actsBlockCount[i];
        unsigned len = addendLen[i];

        for (unsigned b = 0; b != blockCount; ++b) {
            for (unsigned a = 0; a != len; ++a) {
                acts[i][b * len + a] += addend[i][a];
            }
        }
    }

    return true;
  }
};
template class AddToChannel2D<float>;
template class AddToChannel2D<half>;

template <class FPType>
class
ScaledAddToChannel : public SupervisorVertex {
public:
  ScaledAddToChannel();

  Input<Vector<FPType, SPAN, 8>> addend;
  InOut<Vector<FPType, ONE_PTR, 8, true>> acts;
  // actsBlockCount = acts.size() / addend.size();
  // actsBlockCountPacked = (actsBlockCount/6 << 3) | (actsBlockCount % 6)
  const uint16_t actsBlockCountPacked;
  const FPType scale;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = addend.size();
    unsigned actsBlockCount = (actsBlockCountPacked >> 3) * 6
                              + (actsBlockCountPacked & 0x07);
    for (unsigned j = 0; j != actsBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        acts[j * chansPerGroup + k] += addend[k] * scale;
      }
    }
    return true;
  }
};

template class ScaledAddToChannel<float>;
template class ScaledAddToChannel<half>;

template <class FPType>
class
ScaledAddToChannel2D : public Vertex {
public:
  ScaledAddToChannel2D();

  // n is equal to addend.size(), addendLen.size(), acts.size()
  // and actsBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> addend;
  Vector<uint16_t, ONE_PTR> addendLen;
  Vector<InOut<Vector<FPType, ONE_PTR, 8, true>>, ONE_PTR> acts;
  Vector<uint16_t, ONE_PTR> actsBlockCount;
  const FPType scale;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
        unsigned blockCount = actsBlockCount[i];
        unsigned len = addendLen[i];

        for (unsigned b = 0; b != blockCount; ++b) {
            for (unsigned a = 0; a != len; ++a) {
                acts[i][b * len + a] += addend[i][a] * scale;
            }
        }
    }

    return true;
  }
};

template class ScaledAddToChannel2D<float>;
template class ScaledAddToChannel2D<half>;

template <class FPType>
class
[[poplar::constraint("elem(*actsIn) != elem(*actsOut)")]]
ChannelMul : public SupervisorVertex {
public:
  ChannelMul();

  Input<Vector<FPType, SPAN, 8>> scale;
  Input<Vector<FPType, ONE_PTR, 8>> actsIn;
  Output<Vector<FPType, ONE_PTR, 8>> actsOut;
  // actsBlockCount = actsIn.size() / scale.size();
  // actsBlockCountPacked = (actsBlockCount/6 << 3) | (actsBlockCount % 6)
  const uint16_t actsBlockCountPacked;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = scale.size();
    unsigned actsBlockCount = (actsBlockCountPacked >> 3) * 6
                              + (actsBlockCountPacked & 0x07);
    for (unsigned j = 0; j != actsBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        actsOut[j * chansPerGroup + k] =
            actsIn[j * chansPerGroup + k] * scale[k];
      }
    }
    return true;
  }
};

template class ChannelMul<float>;
template class ChannelMul<half>;

template <class FPType>
class
[[poplar::constraint("elem(**actsIn) != elem(**actsOut)")]]
ChannelMul2D : public Vertex {
public:
  ChannelMul2D();

  // n is equal to scale.size(), scaleLen.size(), actsIn.size(), actsOut.size()
  // and scale.size().
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> scale;
  Vector<uint16_t, ONE_PTR> scaleLen;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> actsIn;
  Vector<Output<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> actsOut;
  Vector<uint16_t, ONE_PTR> actsBlockCount;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
        unsigned blockCount = actsBlockCount[i];
        unsigned len = scaleLen[i];

        for (unsigned b = 0; b != blockCount; ++b) {
            for (unsigned a = 0; a != len; ++a) {
                actsOut[i][b * len + a] =
                    actsIn[i][b * len + a] * scale[i][a];
            }
        }
    }

    return true;
  }
};

template class ChannelMul2D<float>;
template class ChannelMul2D<half>;
} // end namespace popops
