// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "Dot.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

namespace poplin {

template <class FloatType, bool lower>
class [[poplar::constraint("elem(*a) != elem(*x)")]] TriangularSolve : Vertex {
public:
  TriangularSolve();

  const unsigned an;
  Input<Vector<FloatType, poplar::VectorLayout::SPAN, 8>> a;
  Input<Vector<FloatType>> b;
  Output<Vector<FloatType, poplar::VectorLayout::SPAN, 8>> x;

  bool compute() {
    assert(an != 0);
    assert(a.size() == an * an);
    assert(b.size() == an);
    assert(x.size() == an);

    if (lower) {
      x[0] = b[0];
    } else {
      const auto last = an - 1;
      x[last] = b[last];
    }

    std::size_t iBase = lower ? an : a.size() - an - an;
    std::size_t bBase = lower ? 1 : (an - 2);
    const auto *dotSrc1 = &a[iBase];
    if (lower) {
      for (std::size_t i = 1; i < an; ++i) {

        const auto *dotSrc2 = &x[0];
        FloatType dot = Dot<FloatType, true>::compute(dotSrc1, dotSrc2, i);
        dotSrc1 += an - i;
        x[bBase] = b[bBase] - dot;
        ++bBase;
      }
    } else {
      dotSrc1 += an - 1;
      for (std::size_t i = 1; i < an; ++i) {
        auto jBase = an - i;

        const auto *dotSrc2 = &x[jBase];
        FloatType dot = Dot<FloatType, true>::compute(dotSrc1, dotSrc2, i);
        dotSrc1 -= an + i + 1;
        x[bBase] = b[bBase] - dot;
        --bBase;
      }
    }

    return true;
  }
};

template class TriangularSolve<float, false>;
template class TriangularSolve<float, true>;
template class TriangularSolve<half, false>;
template class TriangularSolve<half, true>;

} // end namespace poplin
