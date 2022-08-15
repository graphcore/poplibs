// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cmath>
#include <cstdlib>
#include <poplar/HalfFloat.hpp>
#include <type_traits>

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;

namespace popops {

// The second template parameter
// nanOrInf = true  : Detect if either NaN or Inf is present
// nanOrInf = false : Detect only NaN
template <typename InType, typename OutType>
class NormaliseImage : public MultiVertex {
public:
  NormaliseImage();
  IS_EXTERNAL_CODELET((std::is_same_v<InType, unsigned char>));
  Input<Vector<InType, ONE_PTR, 4>> in;
  Output<Vector<OutType, ONE_PTR, 8>> out;
  Input<Vector<OutType, ONE_PTR, 8>> scales;
  Input<Vector<OutType, ONE_PTR, 8>> offsets;
  OutType inScale;
  unsigned packedNPixels;

  void compute(unsigned wid) {
    auto nWorkers = numWorkers();
    auto remainder = packedNPixels & 0x7;
    auto nElem = (packedNPixels >> 3) + (wid < remainder);
    auto *inPtr = &in[wid * 3];
    auto *outPtr = &out[wid * 4];
#ifdef __IPU__
    // Always write whole pixels - this ensures writes are always whole words.
    typedef OutType outV4Type __attribute__((vector_size(4 * sizeof(OutType))));
    outV4Type offset{offsets[0], offsets[1], offsets[2], 0};
    outV4Type scale{scales[0], scales[1], scales[2], 0};
    outV4Type *outPtr4 = reinterpret_cast<outV4Type *>(&out[wid * 4]);
    for (unsigned i = 0; i != nElem; ++i) {
      outV4Type inPixel{OutType(inPtr[0]), OutType(inPtr[1]),
                        OutType(inPtr[2])};
      inPtr += 3 * nWorkers;
      *outPtr4 = ((inPixel * inScale) - offset) * scale;
      outPtr4 += nWorkers;
    }
#else
    for (unsigned i = 0; i != nElem; ++i) {
      for (unsigned chan = 0; chan != 3; ++chan) {
        *outPtr++ = ((*inPtr++ * inScale) - offsets[chan]) * scales[chan];
      }
      *outPtr++ = 0.;
      outPtr += (nWorkers - 1) * 4;
      inPtr += (nWorkers - 1) * 3;
    }
#endif
  }
};

template class NormaliseImage<float, float>;
template class NormaliseImage<half, half>;
template class NormaliseImage<unsigned char, half>;

} // namespace popops
