#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>
#include "popnn/Loss.hpp"
#include "popnn/NonLinearity.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;

// Macro to instantiate a template class for non linear operations
#define INSTANTIATE_NL(v) \
        template class v<float, \
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID>; \
        template class v<half, \
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID>; \
        template class v<float, \
                         popnn::NonLinearityType::NON_LINEARITY_RELU>; \
        template class v<half, \
                         popnn::NonLinearityType::NON_LINEARITY_RELU>; \
        template class v<float, \
                         popnn::NonLinearityType::NON_LINEARITY_TANH>; \
        template class v<half, \
                         popnn::NonLinearityType::NON_LINEARITY_TANH>;

/****************************************************************************/
/*            Auxiliary math functions                                      */
/****************************************************************************/
// Unsigned integer version of log2 rounded up
// Single-line constexpr form added to allow compile-time calculation.
// Could be nicer if using multi-line constexpr function (needs C++14).
constexpr static unsigned ceilLog2Aux(unsigned n) {
  return (n ? 1 + ceilLog2Aux(n >> 1) : 0);
}
// Check if power of 2 and then call to count up to most significant bit
constexpr static unsigned ceilLog2(unsigned n) {
  return ((n & (n - 1)) ? 1 : 0) + ceilLog2Aux(n >> 1);
}

static float sigmoid(float x)
{
  return (1. / (1. + exp(-x)));
}

static float sigmoid_derivative(float activation)
{
  return activation * (1. - activation);
}

static float relu(float x)
{
  if (x > 0)
    return x;
  return 0;
}

static float relu_derivative(float activation)
{
  if (activation > 0)
    return 1;
  return 0;
}

static float tanh_derivative(float activation)
{
  return 1 - activation * activation;
}


static float nonlinearity(popnn::NonLinearityType t, float x) {
  switch (t) {
  case popnn::NonLinearityType::NON_LINEARITY_SIGMOID:
    return sigmoid(x);
  case popnn::NonLinearityType::NON_LINEARITY_RELU:
    return relu(x);
  case popnn::NonLinearityType::NON_LINEARITY_TANH:
    return tanh(x);
  case popnn::NonLinearityType::NON_LINEARITY_SOFTMAX:
    assert(0 && "Non linearity not supported");
    return x;
  }
}

static float nonlinearity_derivative(popnn::NonLinearityType t,
                                     float activation) {
  switch (t) {
  case popnn::NonLinearityType::NON_LINEARITY_SIGMOID:
    return sigmoid_derivative(activation);
  case popnn::NonLinearityType::NON_LINEARITY_RELU:
    return relu_derivative(activation);
  case popnn::NonLinearityType::NON_LINEARITY_TANH:
    return tanh_derivative(activation);
  case popnn::NonLinearityType::NON_LINEARITY_SOFTMAX:
    assert(0 && "Non linearity not supported");
    return activation;
  }
}


/****************************************************************************/
/*            Vertices                                                      */
/****************************************************************************/

namespace popnn {
template <typename FPType, unsigned nlType>
class NonLinearitySupervisor : public SupervisorVertex {
public:
  InOut<Vector<FPType, SCALED_PTR32, 2>> data;
  unsigned short n;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    // Unpack size from n.
    // elementsPerChunk = num worker contexts * element vector width
    // These values are checked in the API.
    // NOTE: Because sizeof(half) can differ from getTypeSize(HALF) when
    // using IPUModel + FastHalf this is hardcoded based on type for now.
    constexpr std::size_t elementsPerChunk =
      std::is_same<FPType, half>::value ? 24 : 12;
    constexpr auto remainderBits = ceilLog2(elementsPerChunk);
    const auto size =
      ((n >> remainderBits) * elementsPerChunk) +
      (n & ((1u << remainderBits) - 1));
    for (unsigned i = 0; i < size; ++i) {
      data[i] = nonlinearity(NonLinearityType(nlType), data[i]);
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearitySupervisor)

template <typename FPType, unsigned nlType>
class NonLinearityGradSupervisor : public SupervisorVertex {
public:
  Input<Vector<FPType, SCALED_PTR32, 2>> outGrad;
  Input<Vector<FPType, SCALED_PTR32, 2>> out;
  Output<Vector<FPType, SCALED_PTR32, 2>> inGrad;
  unsigned short n;

  bool compute() {
    for (unsigned i = 0; i < n; ++i) {
      inGrad[i] = outGrad[i] *
                  nonlinearity_derivative(NonLinearityType(nlType), out[i]);
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearityGradSupervisor)

template <typename FPType, unsigned nlType>
class NonLinearity2D : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> data;

  bool compute() {
    for (unsigned i = 0; i < data.size(); ++i) {
      for (unsigned j = 0; j < data[i].size(); ++j) {
        data[i][j] = nonlinearity(NonLinearityType(nlType), data[i][j]);
      }
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearity2D)

template <typename FPType, unsigned nlType>
class NonLinearityGrad2D : public Vertex {
public:
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> outGrad;
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> out;
  Vector<Output<Vector<FPType>>> inGrad;

  bool compute() {
    for (unsigned i = 0; i < inGrad.size(); ++i) {
      for (unsigned j = 0; j < inGrad[i].size(); ++j) {
        inGrad[i][j] =
            outGrad[i][j] *
              nonlinearity_derivative(NonLinearityType(nlType),
                                                       out[i][j]);
      }
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearityGrad2D)

template <typename FPType>
class MaxPooling : public Vertex {
public:
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> in;
  Vector<Output<Vector<FPType>>> out;
  Vector<unsigned short, ONE_PTR> windowSizes;

  bool compute() {
    unsigned inIndex = 0;
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned chan = 0; chan < out[i].size(); ++chan) {
        FPType val;
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

template <typename FPType>
class ScaledSumPooling : public Vertex {
public:
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> in;
  Vector<Output<Vector<FPType>>> out;
  Vector<unsigned short, ONE_PTR> windowSizes;
  // This field may be removed if separate vertices are defined for
  // Sum Pooling and Avg pooling
  bool scaleOutput;

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

template class ScaledSumPooling<float>;
template class ScaledSumPooling<half>;

template <typename FPType>
class MaxPoolingGrad : public Vertex {
public:
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> outGrad;
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> out;
  Vector<Output<Vector<FPType>>> inGrad;
  Vector<unsigned short, ONE_PTR> windowSizes;

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
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> outGrad;
  Vector<Output<Vector<FPType>>> inGrad;
  Vector<unsigned short, ONE_PTR> windowSizes;


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
  Input<Vector<FPType>> probs;
  Input<Vector<FPType, ONE_PTR>> expected;

  Output<Vector<FPType, ONE_PTR>> deltas;
  Output<Vector<FPType, ONE_PTR>> transformed;
  bool compute() {
    for (std::size_t i = 0; i < probs.size(); i++) {
      FPType expect = expected[i];
      FPType actual = probs[i];
      FPType delta = (actual - expect);
      deltas[i] = delta;
      transformed[i] = 0.5 * delta * delta;
    }
    return true;
  }
};

template class LossSumSquaredTransform<float>;
template class LossSumSquaredTransform<half>;

template <typename FPType>
class LossSoftmaxTransform : public Vertex {
public:
  Input<Vector<FPType>> probs;
  Input<Vector<FPType, ONE_PTR>> expected;

  Output<Vector<FPType, ONE_PTR>> deltas;
  Output<Vector<FPType, ONE_PTR>> transformed;
  bool compute() {
    for (std::size_t i = 0; i < probs.size(); i++) {
      FPType expect = expected[i];
      FPType actual = probs[i];
      deltas[i] = (actual - expect);
      transformed[i] = -expect * log(actual);
    }
    return true;
  }
};

template class LossSoftmaxTransform<float>;
template class LossSoftmaxTransform<half>;

template <typename FPType, typename LabelType>
class CalcAccuracy : public Vertex {
public:
  Input<VectorList<FPType, VectorListLayout::DELTAN>> activations;
  Input<Vector<LabelType, ONE_PTR>> labels;

  InOut<unsigned> numCorrect;
  bool compute() {
    const auto batchSize = activations.size();
    for (unsigned batch = 0; batch < batchSize; ++batch) {
      auto in = activations[batch];
      FPType max = in[0];
      LabelType maxIndex = 0;
      for (LabelType i = 0; i < in.size(); i++) {
        if (in[i] > max) {
          max = in[i];
          maxIndex = i;
        }
      }
      *numCorrect += (maxIndex == labels[batch] ? 1 : 0);
    }
    return true;
  }
};

template class CalcAccuracy<float,unsigned int>;
template class CalcAccuracy<half,unsigned int>;
template class CalcAccuracy<float,int>;
template class CalcAccuracy<half,int>;

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
          sum += acts[actsIdx][b];
          sumOfSquares += acts[actsIdx][b] * acts[actsIdx][b];
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
