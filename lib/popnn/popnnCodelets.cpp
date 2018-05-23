#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>
#include "popnn/Loss.hpp"
#include "popnn/NonLinearity.hpp"

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
  InOut<Vector<FPType, SCALED_PTR32>> data;
  unsigned short n;

  bool compute() {
    for (unsigned i = 0; i < n; ++i) {
      data[i] = nonlinearity(NonLinearityType(nlType), data[i]);
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearitySupervisor)

template <typename FPType, unsigned nlType>
class NonLinearityGradSupervisor : public SupervisorVertex {
public:
  Input<Vector<FPType, SCALED_PTR32>> outGrad;
  Input<Vector<FPType, SCALED_PTR32>> out;
  Output<Vector<FPType, SCALED_PTR32>> inGrad;
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


template <typename FPType, typename LabelType>
class CalcLoss : public Vertex {
public:
  Vector<Input<Vector<FPType>>> batchIn;
  Input<Vector<LabelType, ONE_PTR>> label;

  Vector<Output<Vector<FPType, ONE_PTR>>, ONE_PTR> batchDeltaOut;
  Vector<Output<FPType>, ONE_PTR> loss;
  InOut<unsigned> numCorrect;

  Vector<FPType> probs;

  unsigned lossType;

  bool compute() {
    const auto batchSize = batchIn.size();
    for (unsigned batchNum = 0; batchNum < batchSize; ++batchNum) {
      auto in = batchIn[batchNum];
      auto deltaOut = batchDeltaOut[batchNum];
      switch (lossType) {
      case SUM_SQUARED_LOSS: {
        /* Calculate the sum-squared error and the partial derivative
           to pass back. */
        FPType sum = 0;
        for (LabelType i = 0;  i < in.size(); ++i) {
          FPType expected = (i == label[batchNum] ? 1 : 0);
          FPType actual = in[i];
          deltaOut[i] = (actual - expected);
          sum += 0.5 * (actual - expected) *  (actual - expected);
        }
        loss[batchNum] = sum;
        }
        break;
      case SOFTMAX_CROSS_ENTROPY_LOSS:
        /* Calculate the softmax probability distribution */
        for (LabelType i = 0;  i < in.size(); ++i) {
          FPType act = in[i];
          probs[i] = exp(act);
        }
        FPType sum = 0;
        for (FPType p : probs)
          sum += p;

        for (LabelType i = 0;  i < in.size(); ++i) {
          probs[i] /= sum;
        }

        /* Calculate the cross-entropy error and the partial derivative
         to pass back. */
        FPType error = 0;
        for (LabelType i = 0;  i < probs.size(); ++i) {
          FPType expected = (i == label[batchNum] ? 1 : 0);
          deltaOut[i] = (probs[i] - expected);
          error += -expected * log(probs[i]);
        }
        loss[batchNum] = error;
        break;
      }

      // Calculate the classification error for reporting test results
      // This assumes that the
      // non-linearity is monotonic, so the max output of the previous
      // layer is the max z-term of the previous layer.
      FPType max = in[0];
      LabelType maxIndex = 0;
      for (LabelType i = 0;  i < in.size(); ++i) {
        if (in[i] > max) {
          max = in[i];
          maxIndex = i;
        }
      }
      bool correct = (maxIndex == label[batchNum]);
      if (correct) {
        *numCorrect += 1;
      }
    }
    return true;
  }
};


template class CalcLoss<float,unsigned int>;
template class CalcLoss<float,int>;
template class CalcLoss<half,unsigned int>;
template class CalcLoss<half,int>;

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
        mean[i][a] = sum / batchSize;
        iStdDev[i][a] = 1.0 / std::sqrt(sumOfSquares / batchSize
                                 - mean[i][a] * mean[i][a] + eps);
      }
    }
    return true;
  }
};

template class BatchNormEstimates<float, float>;
template class BatchNormEstimates<half, float>;

} // end namespace popnn
