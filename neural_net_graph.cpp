#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <type_traits>
#include "neural_net_common.h"
#include "PerformanceEstimation.hpp"

using namespace poplar;

/****************************************************************************/
/*            Auxiliary math functions                                      */
/****************************************************************************/

static float sigmoid(float x)
{
  return (1. / (1. + exp(-x)));
}

static float sigmoid_derivative(float x)
{
  return sigmoid(x) * (1. - sigmoid(x));
}

static float relu(float x)
{
  if (x > 0)
    return x;
  return 0;
}

static float relu_derivative(float x)
{
  if (x > 0)
    return 1;
  return 0;
}

static float nonlinearity(NonLinearityType t, float x) {
  switch (t) {
  case NON_LINEARITY_SIGMOID:
    return sigmoid(x);
  case NON_LINEARITY_RELU:
    return relu(x);
  case NON_LINEARITY_NONE:
    return x;
  }
}

static float nonlinearity_derivative(NonLinearityType t, float x) {
  switch (t) {
  case NON_LINEARITY_SIGMOID:
    return sigmoid_derivative(x);
  case NON_LINEARITY_RELU:
    return relu_derivative(x);
  case NON_LINEARITY_NONE:
    return 1;
  }
}


/****************************************************************************/
/*            Vertices                                                      */
/****************************************************************************/

template <typename FPType>
class FullyConnected : public Vertex {
public:
  Input<Vector<FPType>> activationIn;
  Input<Vector<FPType>> weights;
  Input<FPType> bias;
  NonLinearityType nonLinearityType;
  Output<FPType> zOut;
  Output<FPType> activationOut;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < activationIn.size(); ++i) {
      sum += activationIn[i] * weights[i];
    }
    sum += *bias;
    *zOut = sum;
    *activationOut = nonlinearity(nonLinearityType, sum);
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    return 20 + getDenseDotProductCycles(isFloat, activationIn.size());
  }
};

template class FullyConnected<float>;
template class FullyConnected<half>;

template <typename FPType>
class FullyConnectedPartial : public Vertex {
public:
  Input<Vector<FPType>> in;
  Input<Vector<FPType>> weights;
  Output<float> out;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < in.size(); ++i) {
      sum += in[i] * weights[i];
    }
    *out = sum;
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    return getFullyConnectedPartialCycleEstimate(isFloat, in.size());
  }
};

template class FullyConnectedPartial<float>;
template class FullyConnectedPartial<half>;

template <typename FPType>
class FullyConnectedReduce : public Vertex {
public:
  Input<Vector<float>> partials;
  Input<FPType> bias;
  NonLinearityType nonLinearityType;
  Output<FPType> zOut;
  Output<FPType> activationOut;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < partials.size(); ++i) {
      sum += partials[i];
    }
    sum += *bias;
    *zOut = sum;
    *activationOut = nonlinearity(nonLinearityType, sum);
    return true;
  }

  uint64_t getCycleEstimate() const {
    return (partials.size()+1)/2+15;
  }
};

template class FullyConnectedReduce<float>;
template class FullyConnectedReduce<half>;

/**
 * Compute a sum of 1x1 convolutions over a subset of the input channels for
 * multiple output channels.
 **/
template <typename FPType>
class ConvPartial1x1: public Vertex {
public:
  Vector<Input<Vector<FPType>>> in;
  Input<Vector<FPType>> weights;
  Vector<Output<Vector<float>>> out;

  static const auto inChansPerGroup = 16;
  static const auto outChansPerGroup = 4;

  bool compute() {
    assert(out[0].size() % outChansPerGroup == 0);
    const auto outWidth = out[0].size() / outChansPerGroup;
    const auto outHeight = out.size();
    assert(in[0].size() % inChansPerGroup == 0);
    const auto inWidth = in[0].size() / inChansPerGroup;
    const auto stride = (inWidth + outWidth - 1) / outWidth;
    assert((inWidth + stride - 1) / stride == outWidth);

    const auto inHeight = (outHeight - 1) * stride + 1;
    assert(in.size() % inHeight == 0);
    unsigned numInChanGroups = in.size() / inHeight;

    assert(weights.size() ==
           numInChanGroups * outChansPerGroup * inChansPerGroup);

    for (auto &v : out) {
      for (auto &o : v) {
        o = 0.0;
      }
    }
    for (unsigned inChanGroup = 0; inChanGroup != numInChanGroups;
         ++inChanGroup) {
      for (unsigned y = 0; y != outHeight; ++y) {
        for (unsigned x = 0; x != outWidth; ++x) {
          for (unsigned inChanIndex = 0; inChanIndex != inChansPerGroup;
               ++inChanIndex) {
            for (unsigned outChanIndex = 0; outChanIndex != outChansPerGroup;
                 ++outChanIndex) {
              const auto outIndex = outChanIndex + outChansPerGroup * x;
              const auto weightIndex =
                  inChanIndex + inChansPerGroup * (
                    outChanIndex + outChansPerGroup * (
                      inChanGroup
                    )
                  );
              const auto inIndex = inChanIndex + inChansPerGroup * x * stride;
              out[y][outIndex] += weights[weightIndex] *
                                  in[y * stride][inIndex];
            }
          }
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    const auto outWidth = out[0].size() / outChansPerGroup;
    const auto height = out.size();
    const auto inWidth = in[0].size() / inChansPerGroup;
    unsigned numInChanGroups = in.size() / height;

    const auto stride = (inWidth + outWidth - 1) / outWidth;

    bool isFloat = std::is_same<FPType, float>::value;
    const auto kernelSize = 1;
    return getConvPartialCycleEstimate(isFloat, inChansPerGroup, stride,
                                       kernelSize, numInChanGroups, height,
                                       outWidth, outChansPerGroup);
  }
};

template class ConvPartial1x1<float>;
template class ConvPartial1x1<half>;

/* Compute a partial convolution for a sub-set of input channels and
 * output channels over a number of rows of the input field. */
template <typename FPType>
class ConvPartial: public Vertex {
public:
  Vector<Input<Vector<FPType>>> in;
  Vector<Input<Vector<FPType>>> weights;
  Output<Vector<float>> out;
  unsigned inChansPerGroup;
  // The amount of implicit of zero padding before the first element of the
  // input.
  unsigned padding;
  unsigned stride;

  bool compute() {
    unsigned numInRows = in.size();
    unsigned inputWidth = in[0].size() / inChansPerGroup;
    unsigned outputWidth = out.size();
    unsigned kernelSize = weights[0].size() / inChansPerGroup;
    unsigned distanceFromCentre = (kernelSize - 1) / 2;

    for (auto &o : out) {
      o = 0.0;
    }

    for (unsigned i = 0; i != numInRows; ++i) {
      FPType *row = &in[i][0];
      FPType *rowWeights = &weights[i][0];
      for (unsigned outX = 0; outX < outputWidth; outX += stride) {
        unsigned inXCentre = outX / stride + padding;
        unsigned inXBegin =
            inXCentre > distanceFromCentre ? inXCentre - distanceFromCentre :
                                             0;
        unsigned inXEnd =
            std::min(inXCentre + distanceFromCentre + 1, inputWidth);
        for (unsigned inX = inXBegin; inX != inXEnd; ++inX) {
          unsigned weightX = inX + distanceFromCentre - inXCentre;
          for (unsigned inZ = 0; inZ != inChansPerGroup; ++inZ) {
            out[outX] += row[inX * inChansPerGroup + inZ] *
                         rowWeights[weightX * inChansPerGroup + inZ];
          }
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numInRows = in.size();
    unsigned outputWidth = out.size();
    unsigned kernelSize = weights[0].size() / inChansPerGroup;
    bool isFloat = std::is_same<FPType, float>::value;
    const auto outChansPerGroup = 1;
    return getConvPartialCycleEstimate(isFloat, inChansPerGroup, stride,
                                       kernelSize, numInRows, 1, outputWidth,
                                       1);
  }
};

template class ConvPartial<float>;
template class ConvPartial<half>;

class ConvReduce : public Vertex {
public:
  Output<Vector<float>> out;
  Vector<Input<Vector<float>>> partials;

  bool compute() {
    unsigned numPartials = partials.size();
    unsigned numElem = out.size();
    for (unsigned i = 0; i < numElem; ++i) {
      float sum = 0;
      for (unsigned j = 0; j < numPartials; ++j) {
        sum += partials[j][i];
      }
      out[i] = sum;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numPartials = partials.size();
    unsigned numElem = out.size();
    return 4 + numElem * (1 + numPartials / 2);
  }
};

template <typename FPType>
class ConvComplete : public Vertex {
public:
  Vector<Input<Vector<float>>> in;
  Input<Vector<FPType>> bias;
  Output<Vector<FPType>> out;
  NonLinearityType nonLinearityType;

  bool compute() {
    unsigned outChans = bias.size();
    unsigned outCols = in[0].size();
    for (unsigned ochan = 0; ochan < outChans; ++ochan) {
      for (unsigned ocol = 0; ocol < outCols; ++ocol) {
        float sum = in[ochan][ocol];
        sum += bias[ochan];
        out[ochan * outChans + ocol] = nonlinearity(nonLinearityType, sum);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned vertexOverhead = 5;
    unsigned outChans = bias.size();
    unsigned outCols = in[0].size();
    return vertexOverhead + 2*outCols*outChans;
  }

};

template class ConvComplete<float>;
template class ConvComplete<half>;

template <typename FPType>
class ConvCompleteRes : public Vertex {
public:
  Vector<Input<Vector<float>>> in;
  Input<Vector<FPType>> bias;
  Output<Vector<FPType>> out;
  NonLinearityType nonLinearityType;
  Input<Vector<FPType>> res;

  bool compute() {
    unsigned outChans = bias.size();
    unsigned outCols = in[0].size();
    for (unsigned ochan = 0; ochan < outChans; ++ochan) {
      for (unsigned ocol = 0; ocol < outCols; ++ocol) {
        float sum = in[ochan][ocol];
        sum += bias[ochan];
        sum += res[ochan * outChans + ocol];
        out[ochan * outChans + ocol] = nonlinearity(nonLinearityType, sum);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned vertexOverhead = 5;
    unsigned outChans = bias.size();
    unsigned outCols = in[0].size();
    return vertexOverhead + 2*outCols*outChans;
  }

};

template class ConvCompleteRes<float>;
template class ConvCompleteRes<half>;

template <typename FPType>
class CopyResidual : public Vertex {
public:
  Input<Vector<FPType>> in;
  Output<Vector<FPType>> out;

  bool compute() {
    for (unsigned i = 0; i < in.size(); ++i)
      out[i] = in[i];
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO: make this more accurate
    bool isFloat = std::is_same<FPType, float>::value;
    auto copiesPerCycle = isFloat ? 2 : 4;
    auto copyCycles = (in.size() + copiesPerCycle - 1) / copiesPerCycle;
    return 4 + copyCycles;
  }
};

template class CopyResidual<float>;
template class CopyResidual<half>;


template <typename FPType>
class Zero : public Vertex {
public:
  Output<Vector<FPType>> out;

  bool compute() {
    for (unsigned i = 0; i < out.size(); ++i) {
      out[i] = 0;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO: make this more accurate
    bool isFloat = std::is_same<FPType, float>::value;
    auto zeroesPerCycle = isFloat ? 2 : 4;
    auto zeroCycles = (out.size() + zeroesPerCycle - 1) / zeroesPerCycle;
    return 4 + zeroCycles;
  }
};

template class Zero<float>;
template class Zero<half>;


template <typename FPType>
class MaxPooling : public Vertex {
public:
  Vector<Input<FPType>> activationIn;
  Output<FPType> activationOut;

  bool compute() {
    float maxVal = activationIn[0];
    unsigned receptiveFieldSize = activationIn.size();
    for (unsigned i = 0; i < receptiveFieldSize; ++i) {
      if (activationIn[i] > maxVal) {
        maxVal = activationIn[i];
      }
    }
    *activationOut = maxVal;
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 10 + activationIn.size() * 2;
  }
};

template class MaxPooling<float>;
template class MaxPooling<half>;

template <typename FPType>
class CalcLoss : public Vertex {
public:
  Input<Vector<FPType>> zIn;
  NonLinearityType nonLinearityType;
  Input<unsigned> label;
  Input<LossType> lossType;
  Output<Vector<FPType>> errorOut;
  Output<FPType> loss;
  InOut<unsigned> numCorrect;

  Vector<FPType> probs;

  bool compute() {
    switch (lossType) {
    case SUM_SQUARED_LOSS: {
      /* Calculate the sum-squared error and the partial derivative
         to pass back. */
      FPType sum = 0;
      for (unsigned i = 0;  i < zIn.size(); ++i) {

        FPType expected = (i == label ? 1 : 0);
        FPType actual = nonlinearity(nonLinearityType, zIn[i]);
        FPType nlGradient = nonlinearity_derivative(nonLinearityType, zIn[i]);
        errorOut[i] = (actual - expected) * nlGradient;
        sum += (actual - expected) *  (actual - expected);
      }
      *loss = sum;
    }
      break;
    case SOFTMAX_CROSS_ENTROPY_LOSS:
      /* Calculate the softmax probability distribution */
      for (unsigned i = 0;  i < zIn.size(); ++i) {
        FPType act = nonlinearity(nonLinearityType, zIn[i]);
        probs[i] = exp(act);
      }
      FPType sum = 0;
      for (FPType p : probs)
        sum += p;
      for (unsigned i = 0;  i < zIn.size(); ++i) {
        probs[i] /= sum;
      }

      /* Calculate the cross-entropy error and the partial derivative
         to pass back. */
      FPType error = 0;
      for (unsigned i = 0;  i < probs.size(); ++i) {
        FPType expected = (i == label ? 1 : 0);
        FPType nlGradient = nonlinearity_derivative(nonLinearityType, zIn[i]);
        errorOut[i] = (probs[i] - expected) * nlGradient;
        error += expected * log(probs[i]);
      }
      *loss = error;
      break;
    }

    // Calculate the classification error for reporting test results
    // This assumes that the
    // non-linearity is monotonic, so the max output of the previous
    // layer is the max z-term of the previous layer.
    FPType max = zIn[0];
    unsigned maxIndex = 0;
    for (unsigned i = 0;  i < zIn.size(); ++i) {
      if (zIn[i] > max) {
        max = zIn[i];
        maxIndex = i;
      }
    }
    bool correct = (maxIndex == label);
    if (correct) {
      *numCorrect += 1;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 0;
    uint64_t cycles = 5;
    switch (lossType) {
    case SUM_SQUARED_LOSS:
      cycles += zIn.size() * 30;
      break;
    case SOFTMAX_CROSS_ENTROPY_LOSS:
      cycles += zIn.size() * 50;
      break;
    }

    cycles += zIn.size() * 10;

    cycles += 5;

    return cycles;
  }
};


template class CalcLoss<float>;
template class CalcLoss<half>;
