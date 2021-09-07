// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CheckAccuracyWhenCast.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

static constexpr auto SPAN = VectorLayout::SPAN;

namespace popops {

template <typename TensorType, typename ScalarType>
class ScalarMultiply1D : public MultiVertex {
public:
  ScalarMultiply1D();

  Input<Vector<TensorType>> in1;
  Input<Vector<ScalarType>> in2;
  const float tolerance;

  Output<Vector<TensorType>> out;

  bool compute(unsigned wid) {
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
    return true;
  }
};

template <typename TensorType, typename ScalarType>
class ScalarMultiply1DInplace : public MultiVertex {
public:
  ScalarMultiply1DInplace();

  InOut<Vector<TensorType>> in1Out;
  Input<Vector<ScalarType>> in2;
  const float tolerance;

  bool compute(unsigned wid) {
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
    return true;
  }
};

template <typename TensorType, typename ScalarType>
class ScalarMultiply2D : public Vertex {
public:
  ScalarMultiply2D();

  Vector<Input<Vector<TensorType, SPAN, 8>>> in1;
  Input<Vector<ScalarType>> in2;
  const float tolerance;

  Vector<Output<Vector<TensorType, SPAN, 8>>> out;

  bool compute() {
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
    return true;
  }
};

template <typename TensorType, typename ScalarType>
class ScalarMultiply2DInplace : public Vertex {
public:
  ScalarMultiply2DInplace();

  Vector<InOut<Vector<TensorType, SPAN, 8>>> in1Out;
  Input<Vector<ScalarType>> in2;
  const float tolerance;

  bool compute() {
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
    return true;
  }
};

template class ScalarMultiply1D<half, float>;
template class ScalarMultiply1DInplace<half, float>;
template class ScalarMultiply2D<half, float>;
template class ScalarMultiply2DInplace<half, float>;

} // namespace popops
