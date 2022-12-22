// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "Dot.hpp"
#include <algorithm>
#include <cassert>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

#ifdef __IPU__
// Use the IPU intrinsics
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#define NAMESPACE ipu
#else
// Use the std functions
#include <cmath>
#define NAMESPACE std
#endif

using namespace poplar;

namespace poplin {

template <class FloatType> class TriangularInverseWithTranspose : MultiVertex {
public:
  TriangularInverseWithTranspose();

  const unsigned dim;
  const unsigned colBegin;
  const unsigned colCnt;
  Input<Vector<FloatType>> in;
  InOut<Vector<FloatType>> out;

  void computeInvColBased(size_t col) {

    for (std::size_t i = 0, index = (col - colBegin) * dim; i < col;
         ++i, index += 1) {
      out[index] = 0;
    }
    const size_t oBase = (col - colBegin) * dim + col;
    for (std::size_t i = 0, iBase = col * dim + col; i < dim - col;
         ++i, iBase += dim) {
      const auto *sumSrc1 = &in[iBase];
      const auto *sumSrc2 = &out[oBase];
      auto d = FloatType(1) / in[iBase + i];
      if (i > 0) {
        const auto sum = Dot<FloatType, false>::compute(sumSrc1, sumSrc2, i);
        d = -d * sum;
      }
      out[oBase + i] = d;
    }
  }

  bool compute(unsigned wid) {
    assert(dim * dim == in.size());
    assert(in.size() == out.size());

    for (size_t col = colBegin + wid; col < colBegin + colCnt;
         col += CTXT_WORKERS) {
      computeInvColBased(col);
    }

    return true;
  }
};

template <class FloatType, bool lower> class Cholesky : Vertex {
public:
  Cholesky();

  const unsigned dim;
  InOut<Vector<FloatType>> in;

  void compute() {
    assert(dim * dim == in.size());

    if (lower) {
      in[0] = NAMESPACE::sqrt(in[0]);
      std::size_t iBase = dim;
      for (std::size_t i = 1; i < dim; ++i, iBase += dim) {
        std::size_t kBase = 0;
        for (std::size_t k = 0; k <= i; ++k, kBase += dim) {
          const auto *sumSrc1 = &in[iBase];
          const auto *sumSrc2 = &in[kBase];
          FloatType sum = Dot<FloatType, false>::compute(sumSrc1, sumSrc2, k);

          auto m = in[iBase + k] - sum;
          if (i == k) {
            in[iBase + i] = NAMESPACE::sqrt(m);
          } else {
            in[iBase + k] = m / in[kBase + k];
          }
        }
      }
    } else {
      std::size_t iBase = 0;
      for (std::size_t i = 0; i < dim; ++i, iBase += dim) {
        std::size_t kBase = 0;
        for (std::size_t k = 0; k <= i; ++k, kBase += dim) {
          FloatType sum = 0;
          std::size_t jBase = 0;
          for (std::size_t j = 0; j < k; ++j, jBase += dim) {
            sum += in[jBase + i] * in[jBase + k];
          }

          auto m = in[kBase + i] - sum;
          if (i == k) {
            in[kBase + k] = NAMESPACE::sqrt(m);
          } else {
            in[kBase + i] = m / in[kBase + k];
          }
        }
      }
    }
  }
};

template class TriangularInverseWithTranspose<float>;
template class TriangularInverseWithTranspose<half>;

template class Cholesky<float, false>;
template class Cholesky<float, true>;
template class Cholesky<half, false>;
template class Cholesky<half, true>;

} // end namespace poplin
