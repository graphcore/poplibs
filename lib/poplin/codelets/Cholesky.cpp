// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "Dot.hpp"
#include <algorithm>
#include <cassert>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

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

template <class FloatType, bool lower> class TriangularInverse : Vertex {
public:
  TriangularInverse();

  const unsigned dim;
  Input<Vector<FloatType>> in;
  Output<Vector<FloatType>> out;

  bool compute() {
    assert(dim * dim == in.size());
    assert(in.size() == out.size());

    if (lower) {
      std::size_t iBase = 0;
      for (std::size_t i = 0; i < dim; ++i, iBase += dim) {
        auto d = FloatType(1) / in[iBase + i];
        out[iBase + i] = d;

        for (std::size_t j = 0; j < i; ++j) {
          FloatType sum = 0;
          std::size_t kBase = j * dim;
          for (std::size_t k = j; k < i; ++k, kBase += dim) {
            sum += in[iBase + k] * out[kBase + j];
          }
          out[iBase + j] = -d * sum;
        }
        for (std::size_t j = i + 1; j < dim; ++j) {
          out[iBase + j] = 0;
        }
      }
    } else {
      std::size_t iBase = 0;
      for (std::size_t i = 0; i < dim; ++i, iBase += dim) {
        auto d = FloatType(1) / in[iBase + i];
        out[iBase + i] = d;

        std::size_t jBase = 0;
        for (std::size_t j = 0; j < i; ++j, jBase += dim) {
          FloatType sum = 0;
          std::size_t kBase = j * dim;
          for (std::size_t k = j; k < i; ++k, kBase += dim) {
            sum += in[kBase + i] * out[jBase + k];
          }
          out[jBase + i] = -d * sum;
        }
        jBase += dim;
        for (std::size_t j = i + 1; j < dim; ++j, jBase += dim) {
          out[jBase + i] = 0;
        }
      }
    }

    return true;
  }
};

template <class FloatType, bool lower> class Cholesky : Vertex {
public:
  Cholesky();

  const unsigned dim;
  InOut<Vector<FloatType>> in;

  bool compute() {
    assert(dim * dim == in.size());

    if (lower) {
      std::size_t iBase = 0;
      for (std::size_t i = 0; i < dim; ++i, iBase += dim) {
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

    return true;
  }
};

template class TriangularInverse<float, false>;
template class TriangularInverse<float, true>;
template class TriangularInverse<half, false>;
template class TriangularInverse<half, true>;

template class Cholesky<float, false>;
template class Cholesky<float, true>;
template class Cholesky<half, false>;
template class Cholesky<half, true>;

} // end namespace poplin
