// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

namespace poplin {

template <class FloatType, bool lower> class TriangularSolve : Vertex {
public:
  TriangularSolve();

  const unsigned an;
  Input<Vector<FloatType>> a;
  Input<Vector<FloatType>> b;
  Output<Vector<FloatType>> x;

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

    std::size_t base_i = lower ? an : a.size() - an - an;
    std::size_t base_b = lower ? 1 : (an - 2);
    if (lower) {
      for (std::size_t i = 1; i < an; ++i) {
        FloatType dot = 0;
        for (std::size_t j = 0; j < i; ++j) {
          dot += a[base_i++] * x[j];
        }
        x[base_b] = b[base_b] - dot;
        base_i += an - i;
        ++base_b;
      }
    } else {
      for (std::size_t i = 1; i < an; ++i) {
        FloatType dot = 0;
        auto base_j = an - i;
        for (std::size_t j = i; j--; ++base_j) {
          dot += a[base_i + base_j] * x[base_j];
        }
        x[base_b] = b[base_b] - dot;

        base_i -= an;
        --base_b;
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
