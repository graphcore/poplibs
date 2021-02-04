// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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
      std::size_t base_i = 0;
      for (std::size_t i = 0; i < dim; ++i, base_i += dim) {
        auto d = FloatType(1) / in[base_i + i];
        out[base_i + i] = d;

        for (std::size_t j = 0; j < i; ++j) {
          FloatType sum = 0;
          std::size_t base_k = j * dim;
          for (std::size_t k = j; k < i; ++k, base_k += dim) {
            sum += in[base_i + k] * out[base_k + j];
          }
          out[base_i + j] = -d * sum;
        }
        for (std::size_t j = i + 1; j < dim; ++j) {
          out[base_i + j] = 0;
        }
      }
    } else {
      std::size_t base_i = 0;
      for (std::size_t i = 0; i < dim; ++i, base_i += dim) {
        auto d = FloatType(1) / in[base_i + i];
        out[base_i + i] = d;

        std::size_t base_j = 0;
        for (std::size_t j = 0; j < i; ++j, base_j += dim) {
          FloatType sum = 0;
          std::size_t base_k = j * dim;
          for (std::size_t k = j; k < i; ++k, base_k += dim) {
            sum += in[base_k + i] * out[base_j + k];
          }
          out[base_j + i] = -d * sum;
        }
        base_j += dim;
        for (std::size_t j = i + 1; j < dim; ++j, base_j += dim) {
          out[base_j + i] = 0;
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
      std::size_t base_i = 0;
      for (std::size_t i = 0; i < dim; ++i, base_i += dim) {
        std::size_t base_k = 0;
        for (std::size_t k = 0; k <= i; ++k, base_k += dim) {
          FloatType sum = 0;
          for (std::size_t j = 0; j < k; ++j) {
            sum += in[base_i + j] * in[base_k + j];
          }

          auto m = in[base_i + k] - sum;
          if (i == k) {
            in[base_i + i] = NAMESPACE::sqrt(m);
          } else {
            in[base_i + k] = m / in[base_k + k];
          }
        }
      }
    } else {
      std::size_t base_i = 0;
      for (std::size_t i = 0; i < dim; ++i, base_i += dim) {
        std::size_t base_k = 0;
        for (std::size_t k = 0; k <= i; ++k, base_k += dim) {
          FloatType sum = 0;
          std::size_t base_j = 0;
          for (std::size_t j = 0; j < k; ++j, base_j += dim) {
            sum += in[base_j + i] * in[base_j + k];
          }

          auto m = in[base_k + i] - sum;
          if (i == k) {
            in[base_k + k] = NAMESPACE::sqrt(m);
          } else {
            in[base_k + i] = m / in[base_k + k];
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
