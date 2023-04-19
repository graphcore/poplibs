// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include "poplar/Loops.hpp"
#include <cmath>
#include <poplar/Vertex.hpp>

using namespace poplar;

namespace popops {

template <typename FPType, unsigned degree> struct Basis {
  static inline FPType forward(FPType v, unsigned kMod) {
    if (degree == 1) {
      return static_cast<FPType>(1.) - v - kMod +
             static_cast<FPType>(2.) * v * kMod;
    } else if (degree == 2) {
      if (kMod == 0)
        return static_cast<FPType>(0.5) * v * v - v + static_cast<FPType>(.5);
      else if (kMod == 1)
        return -v * v + v + static_cast<FPType>(0.5);
      else
        return static_cast<FPType>(0.5) * v * v;
    } else if (degree == 3) {
      if (kMod == 0)
        return (static_cast<FPType>(1.) - v) * (static_cast<FPType>(1.) - v) *
               (static_cast<FPType>(1.) - v) / static_cast<FPType>(6.);
      else if (kMod == 1)
        return (static_cast<FPType>(3.) * v * v * v -
                static_cast<FPType>(6.) * v * v + static_cast<FPType>(4.)) /
               static_cast<FPType>(6.);
      else if (kMod == 2)
        return (static_cast<FPType>(-3.) * v * v * v +
                static_cast<FPType>(3.) * v * v + static_cast<FPType>(3.) * v +
                static_cast<FPType>(1.)) /
               static_cast<FPType>(6.);
      else
        return v * v * v / static_cast<FPType>(6.);
    } else {
      return static_cast<FPType>(-1.);
    }
  }
};

template <typename FPType, unsigned degree> class SplineBasis : public Vertex {
public:
  SplineBasis();

  Input<Vector<FPType, VectorLayout::ONE_PTR>> pseudo;
  Input<Vector<int, VectorLayout::ONE_PTR>> kernelSize;
  Input<Vector<unsigned char, VectorLayout::ONE_PTR>> isOpenSpline;

  Vector<Output<Vector<FPType>>> basis;
  Vector<Output<Vector<int, VectorLayout::ONE_PTR>>> weightIndex;

  const Vector<unsigned> offsets;
  const unsigned numDims;
  const unsigned numSplines;
  const unsigned edgeOffset;

  void compute() {
    const rptsize_t loopCountDims = numDims;
    for (unsigned i = 0; i < basis.size(); ++i) {
      const auto offset = offsets[i];
      for (unsigned j = 0; j < basis[i].size(); ++j) {
        const unsigned idx = j + offset;
        const unsigned e = idx / numSplines - edgeOffset;
        const unsigned s = idx % numSplines;

        unsigned k = s, wi = 0, wiOffset = 1;
        FPType b = static_cast<FPType>(1.);

        for (unsigned d = 0; d < loopCountDims; d++) {
          const unsigned kMod = k % (degree + 1);
          k /= degree + 1;

          FPType v = pseudo[e * numDims + d];
          v *= kernelSize[d] - degree * isOpenSpline[d];

          wi += ((static_cast<unsigned>(v) + kMod) % kernelSize[d]) * wiOffset;
          wiOffset *= kernelSize[d];

          v -= floorf(v);
          v = Basis<FPType, degree>::forward(v, kMod);
          b *= v;
        }

        basis[i][j] = b;
        weightIndex[i][j] = wi;
      }
    }
  }
};

template class SplineBasis<float, 1>;
template class SplineBasis<float, 2>;
template class SplineBasis<float, 3>;

template class SplineBasis<half, 1>;
template class SplineBasis<half, 2>;
template class SplineBasis<half, 3>;

} // namespace popops