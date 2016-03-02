#include <poplar/Vertex.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "neural_net_common.h"

#ifndef FPType
#error Need to define FPType!
#endif

using namespace poplar;


static uint64_t dense_dotproduct_cycles(unsigned size) {
  if (sizeof(FPType) == 2) {
    return (size+1)/4+2;
  } else {
    return (size+1)/2+2;
  }
}


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
    sum += bias;
    *zOut = sum;
    *activationOut = nonlinearity(nonLinearityType, sum);
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 20 + dense_dotproduct_cycles(activationIn.size());
  }
};

class FullyConnectedPartial : public Vertex {
public:
  Input<Vector<FPType>> in;
  Input<Vector<FPType>> weights;
  Output<FPType> out;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < in.size(); ++i) {
      sum += in[i] * weights[i];
    }
    *out = sum;
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 5 + dense_dotproduct_cycles(in.size());
  }
};


class FullyConnectedReduce : public Vertex {
public:
  Input<Vector<FPType>> partials;
  Input<FPType> bias;
  NonLinearityType nonLinearityType;
  Output<FPType> zOut;
  Output<FPType> activationOut;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < partials.size(); ++i) {
      sum += partials[i];
    }
    sum += bias;
    *zOut = sum;
    *activationOut = nonlinearity(nonLinearityType, sum);
    return true;
  }

  uint64_t getCycleEstimate() const {
    if (sizeof(FPType) == 2) {
      return (partials.size()+1)/4+15;
    } else {
      return (partials.size()+1)/2+15;
    }
  }
};

class Convolution : public Vertex {
public:
  Vector<Input<Vector<FPType>>> activationIn;
  Vector<Input<Vector<FPType>>> weights;
  Input<FPType> bias;
  NonLinearityType nonLinearityType;
  Output <FPType> activationOut;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < activationIn.size(); ++i) {
      for (unsigned j = 0; j < activationIn[i].size(); ++j) {
        sum += activationIn[i][j] * weights[i][j];
      }
    }
    sum += bias; // bias
    *activationOut = nonlinearity(nonLinearityType, sum);
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned N = activationIn.size();
    unsigned M = activationIn[0].size();
    unsigned vertexOverhead = 6;
    unsigned reluCycles = 3;
    return vertexOverhead + reluCycles +
           N * (1 + dense_dotproduct_cycles(M));
  }

};


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

