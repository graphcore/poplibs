// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CheckAccuracyWhenCast.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <poplibs_support/ExternalCodelet.hpp>

using namespace poplar;

static constexpr auto SPAN = VectorLayout::SPAN;
static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;

namespace popops {

template <typename TensorType, typename ScalarType>
class ScalarMultiply1D : public MultiVertex {
public:
  ScalarMultiply1D();

  Input<Vector<TensorType, SPAN, 8>> in1;
  Input<Vector<ScalarType, ONE_PTR>> in2;
  const float tolerance;

  Output<Vector<TensorType, ONE_PTR, 8>> out;

  IS_EXTERNAL_CODELET(true);

  void compute(unsigned wid) {
    if (wid == 0) {
      if (checkAccuracyWhenCastComputeImpl<float, half>(in2[0], tolerance)) {
        for (unsigned i = 0; i < in1.size(); i++) {
          out[i] = static_cast<half>(static_cast<half>(in2[0]) *
                                     static_cast<half>(in1[i]));
        }
      } else {
        for (unsigned i = 0; i < in1.size(); i++) {
          out[i] = static_cast<half>(static_cast<float>(in2[0]) *
                                     static_cast<float>(in1[i]));
        }
      }
    }
  }
};

template <typename TensorType, typename ScalarType>
class ScalarMultiply1DInplace : public MultiVertex {
public:
  ScalarMultiply1DInplace();

  InOut<Vector<TensorType, SPAN, 8>> in1Out;
  Input<Vector<ScalarType, ONE_PTR>> in2;
  const float tolerance;

  IS_EXTERNAL_CODELET(true);

  void compute(unsigned wid) {
    if (wid == 0) {
      if (checkAccuracyWhenCastComputeImpl<float, half>(in2[0], tolerance)) {
        for (unsigned i = 0; i < in1Out.size(); i++) {
          in1Out[i] *= in2[0];
        }
      } else {
        for (unsigned i = 0; i < in1Out.size(); i++) {
          in1Out[i] = static_cast<half>(static_cast<float>(in2[0]) *
                                        static_cast<float>(in1Out[i]));
        }
      }
    }
  }
};

template <typename TensorType, typename ScalarType>
class ScalarMultiply2D : public Vertex {
public:
  ScalarMultiply2D();

  Vector<Input<Vector<TensorType, SPAN, 8>>, SPAN> in1;
  Input<Vector<ScalarType, ONE_PTR>> in2;
  const float tolerance;

  Vector<Output<Vector<TensorType, ONE_PTR, 8>>, ONE_PTR> out;

  IS_EXTERNAL_CODELET(true);

  void compute() {
    if (checkAccuracyWhenCastComputeImpl<float, half>(in2[0], tolerance)) {
      for (unsigned i = 0; i < in1.size(); i++) {
        for (unsigned j = 0; j < in1[i].size(); j++) {
          out[i][j] = static_cast<half>(static_cast<half>(in2[0]) *
                                        static_cast<half>(in1[i][j]));
        }
      }
    } else {
      for (unsigned i = 0; i < in1.size(); i++) {
        for (unsigned j = 0; j < in1[i].size(); j++) {
          out[i][j] = static_cast<half>(static_cast<float>(in2[0]) *
                                        static_cast<float>(in1[i][j]));
        }
      }
    }
  }
};

template <typename TensorType, typename ScalarType>
class ScalarMultiply2DInplace : public Vertex {
public:
  ScalarMultiply2DInplace();

  Vector<InOut<Vector<TensorType, SPAN, 8>>, SPAN> in1Out;
  Input<Vector<ScalarType, ONE_PTR>> in2;
  const float tolerance;

  IS_EXTERNAL_CODELET(true);

  void compute() {
    if (checkAccuracyWhenCastComputeImpl<float, half>(in2[0], tolerance)) {
      for (unsigned i = 0; i < in1Out.size(); i++) {
        for (unsigned j = 0; j < in1Out[i].size(); j++) {
          in1Out[i][j] *= static_cast<half>(in2[0]);
        }
      }
    } else {
      for (unsigned i = 0; i < in1Out.size(); i++) {
        for (unsigned j = 0; j < in1Out[i].size(); j++) {
          in1Out[i][j] = static_cast<half>(static_cast<float>(in2[0]) *
                                           static_cast<float>(in1Out[i][j]));
        }
      }
    }
  }
};

template class ScalarMultiply1D<half, float>;
template class ScalarMultiply1DInplace<half, float>;
template class ScalarMultiply2D<half, float>;
template class ScalarMultiply2DInplace<half, float>;

} // namespace popops
