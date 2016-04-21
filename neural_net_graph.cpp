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
    return (size + 3) / 4 + 2;
  } else {
    return (size + 1) / 2 + 2;
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
    return 5 + dense_dotproduct_cycles(in.size());
  }
};


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
    sum += bias;
    *zOut = sum;
    *activationOut = nonlinearity(nonLinearityType, sum);
    return true;
  }

  uint64_t getCycleEstimate() const {
    return (partials.size()+1)/2+15;
  }
};

class Convolution : public Vertex {
public:
  Vector<Input<Vector<FPType>>> activationIn;
  Vector<Input<Vector<FPType>>> weights;
  Input<Vector<FPType>> bias;
  NonLinearityType nonLinearityType;
  Output<Vector<FPType>> activationOut;

  bool compute() {
    unsigned numOutputs = activationOut.size();
    unsigned wSize = activationIn.size();
    for (unsigned i = 0; i < numOutputs; ++i) {
      float sum = 0;
      for (unsigned j = 0; j < wSize; ++j) {
        for (unsigned k = 0; k < activationIn[i].size(); ++k) {
          sum += activationIn[j][k] * weights[i * wSize + j][k];
        }
      }
      sum += bias[i]; // bias
      activationOut[i] = nonlinearity(nonLinearityType, sum);
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numOutputs = activationOut.size();
    unsigned N = activationIn.size();
    unsigned M = activationIn[0].size();
    unsigned vertexOverhead = 6;
    unsigned reluCycles = 3;
    return vertexOverhead +
      numOutputs * (reluCycles +
                    N * (1 + dense_dotproduct_cycles(M)));
  }

};

/* Compute a partial convolution for a sub-set of input channels and
 * output channels over a number of rows of the input field.
 *
 * TODO: For non 3x3 convolutions, this code needs extra temporary memory
 * for partial convolutions. This needs to be accounted for.
 */
class ConvPartial: public Vertex {
public:
  Input<Vector<FPType>> in;
  Input<Vector<FPType>> weights;
  Vector<Output<Vector<float>>> out;
  unsigned kernelSize;
  unsigned stride;
  unsigned inputCols;
  unsigned chans;

  bool compute() {
    unsigned outputRows = out.size();
    unsigned outputCols = out[0].size();
    for (unsigned orow = 0; orow < outputRows; ++orow) {
      unsigned fieldHeight = kernelSize;
      if (orow + fieldHeight > outputRows)
        fieldHeight = outputRows - orow;
      for (unsigned irow = 0; irow < fieldHeight; ++irow) {
        FPType *row = &in[orow * stride * inputCols + irow * inputCols];
        FPType *rowWeights = &weights[irow * kernelSize * chans];
        for (unsigned ocol = 0; ocol < outputCols; ++ocol) {
          float sum = 0;
          FPType *field = &row[ocol * stride * chans];
          for (unsigned i = 0; i < kernelSize * chans; ++i) {
            FPType v;
            if (ocol + i < outputCols)
              v = field[i];
            else
              v = 0;
            sum += v * rowWeights[i];
          }
          if (irow == 0)
            out[orow][ocol] = sum;
          else
            out[orow][ocol] += sum;
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned outputRows = out.size();
    unsigned outputCols = out[0].size();
    unsigned vertexOverhead = 5;
    if (sizeof(FPType) == 2 && stride == 1) {
      // Each output row will have a number of passes of 3x1 convolutions
      // followed by a summation.
      unsigned numPasses = (kernelSize + 2) / 3;
      unsigned innerLoopCycles = numPasses * outputCols;
      return vertexOverhead +
             outputRows * (1 + kernelSize * (1 + innerLoopCycles));
    }  else {
      return vertexOverhead +
        (1 +  outputCols * (1 + kernelSize * (1 + dense_dotproduct_cycles(kernelSize*chans))));
    }
  }

};

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

