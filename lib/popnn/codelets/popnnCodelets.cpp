#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>
#include "popnn/Loss.hpp"
#include "popnn/NonLinearity.hpp"
#include "popnn/PoolingDef.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;

// Macro to instantiate a template class for non linear operations
#define INSTANTIATE_NL(v) \
        template class v<float, \
                         popnn::NonLinearityType::SIGMOID>; \
        template class v<half, \
                         popnn::NonLinearityType::SIGMOID>; \
        template class v<float, \
                         popnn::NonLinearityType::RELU>; \
        template class v<half, \
                         popnn::NonLinearityType::RELU>; \
        template class v<float, \
                         popnn::NonLinearityType::TANH>; \
        template class v<half, \
                         popnn::NonLinearityType::TANH>;

/****************************************************************************/
/*            Auxiliary math functions                                      */
/****************************************************************************/
static float sigmoid(float x)
{
  return (1.0f / (1.0f + exp(-x)));
}

static float sigmoid_derivative(float activation)
{
  return activation * (1.0f - activation);
}

static float relu(float x)
{
  if (x > 0.0f)
    return x;
  return 0.0f;
}

static float relu_derivative(float activation)
{
  if (activation > 0.0f)
    return 1.0f;
  return 0.0f;
}

static float tanh_derivative(float activation)
{
  return 1.0f - activation * activation;
}


static float nonlinearity(popnn::NonLinearityType t, float x) {
  switch (t) {
  case popnn::NonLinearityType::SIGMOID:
    return sigmoid(x);
  case popnn::NonLinearityType::RELU:
    return relu(x);
  case popnn::NonLinearityType::TANH:
    return tanh(x);
  case popnn::NonLinearityType::SOFTMAX:
    assert(0 && "Non linearity not supported");
    return x;
  }
}

static float nonlinearity_derivative(popnn::NonLinearityType t,
                                     float activation) {
  switch (t) {
  case popnn::NonLinearityType::SIGMOID:
    return sigmoid_derivative(activation);
  case popnn::NonLinearityType::RELU:
    return relu_derivative(activation);
  case popnn::NonLinearityType::TANH:
    return tanh_derivative(activation);
  case popnn::NonLinearityType::SOFTMAX:
    assert(0 && "Non linearity not supported");
    return activation;
  }
}


/****************************************************************************/
/*            Vertices                                                      */
/****************************************************************************/

namespace popnn {
template <typename FPType, NonLinearityType nlType>
class WORKER_ALIGN NonLinearitySupervisor : public SupervisorVertex {
public:
  InOut<Vector<FPType, SCALED_PTR32>> data;
  unsigned short n;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < n; ++i) {
      data[i] = nonlinearity(nlType, float(data[i]));
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearitySupervisor)

template <typename FPType, NonLinearityType nlType>
class WORKER_ALIGN NonLinearityGradSupervisor : public SupervisorVertex {
public:
  Input<Vector<FPType, SCALED_PTR32, 8>> outGrad;
  Input<Vector<FPType, SCALED_PTR32, 8>> out;
  Output<Vector<FPType, SCALED_PTR32, 8>> inGrad;
  unsigned short n;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < n; ++i) {
      const auto derivative =
        nonlinearity_derivative(nlType, float(out[i]));
      inGrad[i] = outGrad[i] * FPType(derivative);
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearityGradSupervisor)

template <typename FPType, NonLinearityType nlType>
class NonLinearity2D : public Vertex {
public:
  InOut<VectorList<FPType, VectorListLayout::DELTAN>> data;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < data.size(); ++i) {
      for (unsigned j = 0; j < data[i].size(); ++j) {
        data[i][j] = FPType(nonlinearity(nlType, float(data[i][j])));
      }
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearity2D)

template <typename FPType, NonLinearityType nlType>
class NonLinearityGrad2D : public Vertex {
public:
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> outGrad;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> out;
  Output<VectorList<FPType, DELTAN, 8>> inGrad;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < inGrad.size(); ++i) {
      for (unsigned j = 0; j < inGrad[i].size(); ++j) {
        const auto derivative =
          nonlinearity_derivative(nlType, float(out[i][j]));
        inGrad[i][j] = outGrad[i][j] * FPType(derivative);
      }
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearityGrad2D)

template <typename FPType>
class MaxPooling : public Vertex {
  FPType identity() const
  {
    if (std::is_same<FPType, float>{}) {
      return -std::numeric_limits<FPType>::infinity();
    } else {
      // half type has no infinity so use the lowest finite value instead.
      return std::numeric_limits<FPType>::lowest();
    }
  }

public:
  IS_EXTERNAL_CODELET(true);

  Output<VectorList<FPType, DELTAN, 8>> out;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, SCALED_PTR32> in;
  Input<Vector<unsigned short, ONE_PTR>> windowSizes;

  bool compute() {
    unsigned inIndex = 0;
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned chan = 0; chan < out[i].size(); ++chan) {
        FPType val = identity();
        for (unsigned w = 0; w < windowSizes[i]; ++w) {
          if (w == 0 || val < in[inIndex + w][chan])
            val = in[inIndex + w][chan];
        }
        out[i][chan] = val;
      }
      inIndex += windowSizes[i];
    }
    return true;
  }
};

template class MaxPooling<float>;
template class MaxPooling<half>;

template <typename FPType, PoolingType PType>
class ScaledSumPooling : public Vertex {
  static_assert(PType != PoolingType::MAX,
                "MaxPooling is handled by a dedicated vertex.");

  constexpr static bool scaleOutput = PType == PoolingType::AVG;
public:
  IS_EXTERNAL_CODELET(true);

  Output<VectorList<FPType, DELTAN, 8>> out;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, SCALED_PTR32> in;
  Input<Vector<unsigned short, ONE_PTR>> windowSizes;

  bool compute() {
    unsigned inIndex = 0;
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned chan = 0; chan < out[i].size(); ++chan) {
        // May have to add an intermediate type to the vertex
        FPType val = 0;
        for (unsigned w = 0; w < windowSizes[i]; ++w) {
          val += in[inIndex + w][chan];
        }
        if (scaleOutput && windowSizes[i]) {
          val /= windowSizes[i];
        }
        out[i][chan] = val;
      }
      inIndex += windowSizes[i];
    }
    return true;
  }
};

template class ScaledSumPooling<float, PoolingType::AVG>;
template class ScaledSumPooling<float, PoolingType::SUM>;
template class ScaledSumPooling<half, PoolingType::AVG>;
template class ScaledSumPooling<half, PoolingType::SUM>;

template <typename FPType>
class MaxPoolingGrad : public Vertex {
public:
  IS_EXTERNAL_CODELET(true);

  Vector<Input<Vector<FPType, ONE_PTR, 8>>, SCALED_PTR32> out;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, SCALED_PTR32> outGrad;
  Input<VectorList<FPType, DELTAN, 8>> in;
  Input<Vector<unsigned short, SCALED_PTR32>> windowSizes;
  Output<VectorList<FPType, DELTAN, 8>> inGrad;

  bool compute() {
    unsigned inIndex = 0;
    for (unsigned i = 0; i < inGrad.size(); ++i) {
      for (unsigned chan = 0; chan < inGrad[i].size(); ++chan) {
        FPType val = 0;
        for (unsigned w = 0; w < windowSizes[i]; ++w) {
          if (in[i][chan] == out[inIndex + w][chan])
            val += outGrad[inIndex + w][chan];
        }
        inGrad[i][chan] = val;
      }
      inIndex += windowSizes[i];
    }
    return true;
  }
};

template class MaxPoolingGrad<float>;
template class MaxPoolingGrad<half>;


template <typename FPType>
class SumPoolingGrad : public Vertex {
public:
  IS_EXTERNAL_CODELET(true);

  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> outGrad;
  Vector<Output<Vector<FPType, SPAN, 8>>> inGrad;
  Input<Vector<unsigned short, ONE_PTR>> windowSizes;

  bool compute() {
    unsigned inIndex = 0;
    for (unsigned i = 0; i < inGrad.size(); ++i) {
      for (unsigned chan = 0; chan < inGrad[i].size(); ++chan) {
        FPType val = 0;
        for (auto w = 0; w < windowSizes[i]; ++w) {
          val += outGrad[inIndex + w][chan];
        }
        inGrad[i][chan] = val;
      }
      inIndex += windowSizes[i];
    }
    return true;
  }
};

template class SumPoolingGrad<float>;
template class SumPoolingGrad<half>;

template <typename FPType>
class LossSumSquaredTransform : public Vertex {
public:
  Input<Vector<FPType, SCALED_PTR32, 4>> probs;
  Input<Vector<FPType, SCALED_PTR32, 4>> expected;
  Output<Vector<FPType, SCALED_PTR32, 4>> deltas;
  Output<Vector<FPType, SCALED_PTR32, 4>> transformed;
  unsigned short size;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (std::size_t i = 0; i < size; i++) {
      FPType expect = expected[i];
      FPType actual = probs[i];
      FPType delta = (actual - expect);
      deltas[i] = delta;
      transformed[i] = FPType(0.5) * delta * delta;
    }
    return true;
  }
};

template class LossSumSquaredTransform<float>;
template class LossSumSquaredTransform<half>;

template <typename FPType>
class LossSoftmaxTransform : public Vertex {
public:
  Input<Vector<FPType, SCALED_PTR32, 4>> probs;
  Input<Vector<FPType, SCALED_PTR32, 4>> expected;
  Output<Vector<FPType, SCALED_PTR32, 4>> deltas;
  Output<Vector<FPType, SCALED_PTR32, 4>> transformed;
  unsigned short size;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (std::size_t i = 0; i < size; i++) {
      FPType expect = expected[i];
      FPType actual = probs[i];
      deltas[i] = (actual - expect);
      transformed[i] = -expect * FPType(log(float(actual)));
    }
    return true;
  }
};

template class LossSoftmaxTransform<float>;
template class LossSoftmaxTransform<half>;

// Takes a contiguous set of activations starting
// at the given index, returns the max index and
// value of these.
template <typename FPType, typename LabelType>
class ReduceMaxClassGather : public SupervisorVertex {
public:
  Input<Vector<FPType, ONE_PTR>> activations;
  LabelType index;
  Output<Vector<float, ONE_PTR>> maxValue;
  Output<Vector<LabelType, ONE_PTR>> maxIndex;
  unsigned size;
  unsigned short divisorLog2;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    // Work is split between up to N workers based on the divisor
    // and outputs to each maxValue/Index output based on this
    const auto divisor = (1u << divisorLog2);
    const auto nOutputs = (size + divisor - 1) / divisor;
    for (std::size_t i = 0; i < nOutputs; ++i) {
      LabelType maxI = divisor * i;
      FPType maxV = activations[maxI];
      const auto end = (maxI + divisor > size) ? size : maxI + divisor;
      for (std::size_t j = maxI + 1; j < end; ++j) {
        if (activations[j] > maxV) {
          maxV = activations[j];
          maxI = j;
        }
      }
      maxValue[i] = float(maxV);
      maxIndex[i] = maxI + index;
    }
    return true;
  }
};

template class ReduceMaxClassGather<float, unsigned int>;
template class ReduceMaxClassGather<half, unsigned int>;
template class ReduceMaxClassGather<float, int>;
template class ReduceMaxClassGather<half, int>;

template <typename LabelType>
class ReduceMaxClassSparse : Vertex {
public:
  Input<Vector<float>> activations;
  Input<Vector<LabelType, ONE_PTR>> labels;
  Output<float> maxValue;
  Output<LabelType> maxIndex;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    LabelType maxI = 0;
    float maxV = activations[0];
    for (std::size_t i = 1; i < activations.size(); ++i) {
      if (activations[i] > maxV) {
        maxV = activations[i];
        maxI = i;
      }
    }
    *maxValue = maxV;
    *maxIndex = labels[maxI];
    return true;
  }
};

template class ReduceMaxClassSparse<unsigned int>;
template class ReduceMaxClassSparse<int>;

template <typename LabelType>
class CalcAccuracy : public Vertex {
public:
  Input<Vector<LabelType>> maxPerBatch;
  Input<Vector<LabelType, ONE_PTR>> expected;
  InOut<unsigned> numCorrect;

  bool compute() {
    auto count = *numCorrect;
    for (std::size_t i = 0; i < maxPerBatch.size(); ++i) {
      count += (maxPerBatch[i] == expected[i]);
    }
    *numCorrect = count;
    return true;
  }
};

template class CalcAccuracy<unsigned int>;
template class CalcAccuracy<int>;

template <class InType, class PartialsType>
class BatchNormEstimates : public Vertex {
public:
  Vector<Input<Vector<InType>>, ONE_PTR> acts;
  Vector<Output<Vector<InType>>> mean;
  Vector<Output<Vector<InType, ONE_PTR>>, ONE_PTR> iStdDev;
  float eps;

  bool compute() {
    const unsigned n = mean.size();
    unsigned actsIdx = 0;
    unsigned batchSize = acts[0].size();

    for (unsigned i = 0; i != n; ++i) {
      const unsigned numActs = mean[i].size();
      for (unsigned a = 0; a != numActs; ++a) {
        PartialsType sum = 0;
        PartialsType sumOfSquares = 0;
        for (unsigned b = 0; b != batchSize; ++b) {
          sum += PartialsType(acts[actsIdx][b]);
          sumOfSquares += PartialsType(acts[actsIdx][b] * acts[actsIdx][b]);
        }
        ++actsIdx;
        PartialsType sampleMean = sum / batchSize;
        mean[i][a] = sampleMean;
        // compute unbiased sample variance
        PartialsType sampleVariance =
            sumOfSquares / (batchSize - 1) - sampleMean * sampleMean + eps;
        // machine allows only 32 bit inverse of sqrt
        float istdDevEstimate = sqrt(1.0f / static_cast<float>(sampleVariance));
        iStdDev[i][a] = istdDevEstimate;
      }
    }
    return true;
  }
};

template class BatchNormEstimates<float, float>;
template class BatchNormEstimates<half, float>;

} // end namespace popnn
