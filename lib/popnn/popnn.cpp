#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>
#include "popnn/NonLinearityDef.hpp"
#include "popnn/ConvDef.hpp"
#include "popnn/NetDef.hpp"
#include "PerformanceEstimation.hpp"

using namespace poplar;

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

static float nonlinearity_derivative(NonLinearityType t, float activation) {
  switch (t) {
  case NON_LINEARITY_SIGMOID:
    return sigmoid_derivative(activation);
  case NON_LINEARITY_RELU:
    return relu_derivative(activation);
  case NON_LINEARITY_NONE:
    return 1;
  }
}


/****************************************************************************/
/*            Vertices                                                      */
/****************************************************************************/

template <typename FPType>
class FullyConnectedPartial : public Vertex {
public:
  Input<Vector<FPType>> in;
  Input<Vector<FPType>> weights;
  Output<float> out;

  SimOnlyField<unsigned> dataPathWidth;

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
    return getFullyConnectedPartialCycleEstimate(isFloat, in.size(),
                                                 dataPathWidth);
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
  Output<FPType> activationOut;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < partials.size(); ++i) {
      sum += partials[i];
    }
    sum += *bias;
    *activationOut = nonlinearity(nonLinearityType, sum);
    return true;
  }

  uint64_t getCycleEstimate() const {
    const auto floatVectorWidth = dataPathWidth / 32;
    return (partials.size() + floatVectorWidth - 1) / floatVectorWidth + 15;
  }
};

template class FullyConnectedReduce<float>;
template class FullyConnectedReduce<half>;

template <typename FPType>
class FullyConnectedBwd : public Vertex {
public:
  Input<Vector<FPType>> in;
  Vector<Input<Vector<FPType>>> weights;
  Vector<Output<float>> out;

  bool compute() {
    assert(in.size() == weights.size());
    for (auto &sum : out) {
      sum = 0.0;
    }
    for (unsigned i = 0; i != in.size(); ++i) {
      for (unsigned j = 0; j != out.size(); ++j) {
        assert(weights[i].size() == out.size());
        out[j] += in[i] * weights[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return getFullyConnectedBwdCycleEstimate(weights.size());
  }
};

template class FullyConnectedBwd<float>;
template class FullyConnectedBwd<half>;

template <typename InType, typename OutType>
class FullyConnectedBwdReduce : public Vertex {
public:
  Output<OutType> out;
  Vector<Input<InType>> partials;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    *out = 0.0;
    for (const auto x : partials) {
      *out += x;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 5 + partials.size() * 2;
  }
};

template class FullyConnectedBwdReduce<float, float>;
template class FullyConnectedBwdReduce<float, half>;
template class FullyConnectedBwdReduce<half, float>;
template class FullyConnectedBwdReduce<half, half>;

template <typename FPType>
class FullyConnectedWeightUpdate : public Vertex {
public:
  Input<FPType> d;
  InOut<Vector<FPType>> weights;
  Input<Vector<FPType>> in;
  float eta;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned i = 0; i < weights.size(); ++i) {
      auto grad = *d * in[i];
      weights[i] = weights[i] - grad * eta;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (weights.size() + vectorWidth - 1) / vectorWidth;
    // Inner loop involves multiplication by (*d * eta) and addition.
    return 5 + 2 * numVectors;
  }
};

template class FullyConnectedWeightUpdate<float>;
template class FullyConnectedWeightUpdate<half>;

template <typename FPType>
class FullyConnectedBiasUpdate : public Vertex {
public:
  Vector<Input<FPType>> d;
  Vector<InOut<FPType>> bias;
  float eta;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    const auto numBiases = bias.size();
    assert(d.size() == numBiases);
    for (unsigned i = 0; i != numBiases; ++i) {
      bias[i] = bias[i] - d[i] * eta;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (bias.size() + vectorWidth - 1) / vectorWidth;
    return 5 + 2 * numVectors;
  }
};

template class FullyConnectedBiasUpdate<float>;
template class FullyConnectedBiasUpdate<half>;

/**
 * Compute nx1 convolutions and accumulate them with partial sums in memory.
 **/
template <class Base, class FPType, class AccumType, bool forward>
class ConvPartialnx1InOut: public Base {
public:
  Vector<Input<Vector<FPType>>> in;
  Vector<Input<Vector<FPType>>> weights;
  Vector<unsigned> weightReuseCount;
  Vector<InOut<Vector<AccumType>>> out;

  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> inChansPerGroup;
  SimOnlyField<unsigned> outChansPerGroup;

  bool compute() {
    assert(out.size() > 0);
    assert(in.size() % out.size() == 0);
    const auto filterHeight = in.size() / out.size();
    assert(weights.size() % filterHeight == 0);
    assert(weightReuseCount.size() % (weights.size() / filterHeight) == 0);
    const auto numContexts = weightReuseCount.size() /
                             (weights.size() / filterHeight);
    unsigned convNum = 0;
    for (unsigned w = 0; w != weights.size() / filterHeight; ++w) {
      for (unsigned c = 0; c != numContexts; ++c) {
        for (unsigned i = 0; i != weightReuseCount[w * numContexts + c]; ++i) {
          const auto outWidth = out[convNum].size() / outChansPerGroup;
          const auto inWidth = in[convNum * filterHeight].size() /
                               inChansPerGroup;
          unsigned inStride, outStride;
          if (forward) {
            if (outWidth == 1) {
              assert(inWidth == 1);
              inStride = outStride = 1;
            } else {
              assert((inWidth - 1) % (outWidth - 1) == 0);
              inStride = (inWidth - 1) / (outWidth - 1);
              outStride = 1;
            }
          } else {
            if (inWidth == 1) {
              assert(outWidth == 1);
              inStride = outStride = 1;
            } else {
              assert((outWidth - 1) % (inWidth - 1) == 0);
              outStride = (outWidth - 1) / (inWidth - 1);
              inStride = 1;
            }
          }
          const auto numOutputs = (outWidth + outStride - 1) / outStride;
          for (unsigned x = 0; x != numOutputs; ++x) {
            for (unsigned fy = 0; fy != filterHeight; ++fy) {
              for (unsigned inChanIndex = 0; inChanIndex != inChansPerGroup;
                   ++inChanIndex) {
                for (unsigned outChanIndex = 0;
                     outChanIndex != outChansPerGroup;
                     ++outChanIndex) {
                  const auto outIndex =
                      outChanIndex + outChansPerGroup * x * outStride;
                  const auto weightIndex =
                      inChanIndex + inChansPerGroup * outChanIndex;
                  const auto inIndex =
                      inChanIndex + inChansPerGroup * x * inStride;
                  out[convNum][outIndex] +=
                      weights[w * filterHeight + fy][weightIndex] *
                      in[convNum * filterHeight + fy][inIndex];
                }
              }
            }
          }
          ++convNum;
        }
      }
    }
    assert(convNum == out.size());
    return true;
  }

  std::uint64_t getCycleEstimate() const {
    bool isSupervisorVertex = std::is_same<Base, SupervisorVertex>::value;
    const auto filterHeight = in.size() / out.size();
    const auto numContexts = weightReuseCount.size() /
                             (weights.size() / filterHeight);
    const auto numConvUnitsPerTile = outChansPerGroup;
    assert(dataPathWidth % 16 == 0);
    const auto halfVectorWidth = dataPathWidth / 16;
    assert(inChansPerGroup % halfVectorWidth == 0);
    const auto convUnitPipelineDepth = inChansPerGroup / halfVectorWidth;
    if (isSupervisorVertex) {
      std::vector<std::vector<std::vector<unsigned>>>
          convolutionsByWeightAndWorker;
      unsigned convNum = 0;
      for (unsigned w = 0; w != weights.size() / filterHeight; ++w) {
        convolutionsByWeightAndWorker.emplace_back();
        auto &convolutionsByWeight = convolutionsByWeightAndWorker.back();
        for (unsigned c = 0; c != numContexts; ++c) {
          convolutionsByWeight.emplace_back();
          for (unsigned i = 0; i != weightReuseCount[w * numContexts + c];
               ++i) {
            auto convSize = out[convNum].size() / outChansPerGroup;
            if (!forward) {
              const auto outWidth = out[convNum].size() / outChansPerGroup;
              const auto inWidth = in[convNum * filterHeight].size() /
                                   inChansPerGroup;
              const auto stride = (outWidth + inWidth - 1) / inWidth;
              assert((outWidth + stride - 1) / stride == inWidth);
              convSize = convSize / stride;
            }
            convolutionsByWeight.back().push_back(convSize);
            ++convNum;
          }
        }
      }
      assert(convNum == out.size());
      return getConvPartialnx1SupervisorCycleEstimate(
        convolutionsByWeightAndWorker,
        convUnitPipelineDepth,
        numConvUnitsPerTile,
        filterHeight
      );
    }
    assert(numContexts == 1);
    std::vector<std::vector<unsigned>> convolutionsByWeight(1);
    unsigned convNum = 0;
    for (unsigned w = 0; w != weights.size() / filterHeight; ++w) {
      convolutionsByWeight.emplace_back();
      for (unsigned i = 0; i != weightReuseCount[w]; ++i) {
        auto convSize = out[convNum].size() / outChansPerGroup;
        if (!forward) {
          const auto outWidth = out[convNum].size() / outChansPerGroup;
          const auto inWidth = in[convNum * filterHeight].size() /
                               inChansPerGroup;
          const auto stride = (outWidth + inWidth - 1) / inWidth;
          assert((outWidth + stride - 1) / stride == inWidth);
          convSize = convSize / stride;
        }
        convolutionsByWeight.back().push_back(convSize);
        ++convNum;
      }
    }
    assert(convNum == out.size());
    return getConvPartialnx1CycleWorkerEstimate(convolutionsByWeight,
                                                convUnitPipelineDepth,
                                                numConvUnitsPerTile,
                                                filterHeight);
  }
};

template class ConvPartialnx1InOut<Vertex, float, half, false>;
template class ConvPartialnx1InOut<Vertex, float, float, false>;
template class ConvPartialnx1InOut<SupervisorVertex, float, half, false>;
template class ConvPartialnx1InOut<SupervisorVertex, float, float, false>;
template class ConvPartialnx1InOut<Vertex, float, half, true>;
template class ConvPartialnx1InOut<Vertex, float, float, true>;
template class ConvPartialnx1InOut<SupervisorVertex, float, half, true>;
template class ConvPartialnx1InOut<SupervisorVertex, float, float, true>;

template class ConvPartialnx1InOut<Vertex, half, half, false>;
template class ConvPartialnx1InOut<Vertex, half, float, false>;
template class ConvPartialnx1InOut<SupervisorVertex, half, half, false>;
template class ConvPartialnx1InOut<SupervisorVertex, half, float, false>;
template class ConvPartialnx1InOut<Vertex, half, half, true>;
template class ConvPartialnx1InOut<Vertex, half, float, true>;
template class ConvPartialnx1InOut<SupervisorVertex, half, half, true>;
template class ConvPartialnx1InOut<SupervisorVertex, half, float, true>;

template <typename FPType, typename AccumType>
class ConvBwd : public Vertex {
public:
  Vector<Input<Vector<FPType>>> in;
  Vector<Input<FPType>> weights;
  Vector<Output<Vector<AccumType>>> out;
  NonLinearityType nonLinearityType;
  bool debug;

  bool compute() {
    const auto outRows = out.size();
    const auto inChansPerCol = weights.size();
    const auto inRows = in.size();
    const auto stride = outRows / inRows;
    for (unsigned rowIndex = 0; rowIndex < outRows; rowIndex += stride) {
      auto &inRow = in[rowIndex/stride];
      auto &outRow = out[rowIndex];
      unsigned weightIndex = 0;
      for (unsigned outIndex = 0; outIndex < outRow.size();
           outIndex += stride) {
        float sum = 0;
        for (unsigned weightIndex = 0;
             weightIndex < weights.size();
             ++weightIndex) {
          auto inIndex = outIndex/stride * weights.size() + weightIndex;
          sum += inRow[inIndex] * weights[weightIndex];
        }
        outRow[outIndex] = sum;
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO
    return 0;
  }
};

template class ConvBwd<half, half>;
template class ConvBwd<half, float>;
template class ConvBwd<float, half>;
template class ConvBwd<float, float>;

template <typename FPType>
class ConvReduceBwd : public Vertex {
public:
  Output<Vector<FPType>> out;
  Vector<Input<Vector<FPType>>> partials;

  SimOnlyField<unsigned> dataPathWidth;

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
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    return 4 + numElem * (1 + (numPartials + vectorWidth - 1) / vectorWidth);
  }
};

template class ConvReduceBwd<float>;
template class ConvReduceBwd<half>;

template <class InType, class OutType>
class ConvCompleteBwd : public Vertex {
public:
  Vector<Input<InType>> in;
  Output<Vector<OutType>> out;

  bool compute() {
    for (unsigned i = 0; i < out.size(); ++i) {
      out[i] = in[i];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO
    return 0;
  }

};

template class ConvCompleteBwd<float, float>;
template class ConvCompleteBwd<float, half>;
template class ConvCompleteBwd<half, half>;


template <class FPType>
class ConvWeightGradCalc : public Vertex {
public:
  Vector<Input<Vector<FPType>>> acts;
  Vector<Input<Vector<FPType>>> deltas;
  Output<Vector<FPType>> weights;

  unsigned kernelSize;
  unsigned xpadding, ypadding;
  unsigned stride;
  unsigned inChansPerGroup;
  unsigned outChansPerGroup;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numInRows = acts.size();
    unsigned numOutRows = deltas.size();
    assert(acts[0].size() % inChansPerGroup == 0);
    unsigned inputWidth = acts[0].size() / inChansPerGroup;
    assert(deltas[0].size() % outChansPerGroup == 0);
    unsigned outputWidth = deltas[0].size() / outChansPerGroup;

    for (FPType &o : weights) {
      o = 0.0;
    }
    for (int wy = 0; wy < kernelSize; ++wy) {
      for (int wx = 0; wx < kernelSize; ++wx) {
        auto weightsPerKernelElement = outChansPerGroup * inChansPerGroup;
        FPType *w = &weights[(wy * kernelSize + wx) * weightsPerKernelElement];
        for (unsigned outChan = 0; outChan < outChansPerGroup; ++outChan)
        {
          int inRow = wy - static_cast<int>(ypadding);
          unsigned outRow = 0;
          while (inRow < 0) {
            inRow += stride;
            outRow += 1;
          }
          while (outRow < numOutRows && inRow < numInRows) {
            int inCol = wx - static_cast<int>(xpadding);
            unsigned outCol = 0;
            while (inCol < 0) {
              inCol += stride;
              outCol += 1;
            }
            while (outCol < outputWidth && inCol < inputWidth) {
              for (unsigned inChan = 0; inChan < inChansPerGroup; ++inChan) {
                FPType a = acts[inRow][inCol * inChansPerGroup + inChan];
                w[outChan * inChansPerGroup + inChan] +=
                    a * deltas[outRow][outCol * outChansPerGroup + outChan];
              }
              outCol += 1;
              inCol += stride;
            }
            outRow += 1;
            inRow += stride;
          }
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numInRows = acts.size();
    unsigned numOutRows = deltas.size();
    unsigned inputWidth = acts[0].size() / inChansPerGroup;
    unsigned outputWidth = deltas.size() / outChansPerGroup;
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    return getWeightGradCalcCycles(numOutRows, numInRows,
                                   outputWidth, inputWidth,
                                   outChansPerGroup, inChansPerGroup,
                                   stride, kernelSize,
                                   xpadding, ypadding,
                                   vectorWidth);

  }
};

template class ConvWeightGradCalc<float>;
template class ConvWeightGradCalc<half>;

template <typename WeightType, typename PartialsType>
class ConvWeightUpdate : public Vertex {
public:
  Vector<Input<Vector<PartialsType>>> partials;
  InOut<Vector<WeightType>> weights;

  float eta;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned w = 0; w < weights.size(); ++w) {
      float sum = 0;
      for (unsigned i = 0; i < partials.size(); ++i) {
        assert(w < partials[i].size());
        sum += partials[i][w];
      }
      weights[w] -= eta * sum;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numPartials = partials.size();
    unsigned numElem = weights.size();
    bool isFloat = std::is_same<PartialsType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    return 4 + 2 * numElem * (1 + (numPartials + vectorWidth - 1) / vectorWidth);
  }
};

template class ConvWeightUpdate<float, float>;
template class ConvWeightUpdate<half, half>;
template class ConvWeightUpdate<half, float>;


template <typename FPType>
class ConvBiasReduce: public Vertex {
public:
  Vector<Output<FPType>> biases;
  Vector<Input<FPType>> deltas;

  bool compute() {
    assert(deltas.size() % biases.size() == 0);
    auto numBiases = biases.size();
    auto deltasPerBias = deltas.size() / biases.size();

    for (unsigned bias = 0; bias < numBiases; ++bias) {
      float sum = 0;
      for (unsigned i = 0; i < deltasPerBias; ++i) {
        sum += deltas[bias * deltasPerBias + i];
      }
      biases[bias] = sum;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    auto numBiases= biases.size();
    auto deltasPerBias = deltas.size() / biases.size();
    uint64_t cycles = 10;

    for (unsigned bias = 0; bias < numBiases; ++bias) {
      cycles += deltasPerBias;
    }
    return cycles;
  }
};


template class ConvBiasReduce<float>;
template class ConvBiasReduce<half>;

template <typename FPType>
class ConvBiasUpdate: public Vertex {
public:
  InOut<FPType> bias;
  Vector<Input<FPType>> deltas;
  float eta;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < deltas.size(); ++i) {
      sum += deltas[i];
    }
    *bias -= eta * sum;
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 15 + deltas.size();
  }
};

template class ConvBiasUpdate<float>;
template class ConvBiasUpdate<half>;

template <typename FPType>
class NonLinearityBwd : public Vertex {
public:
  Input<Vector<FPType>> deltasIn;
  Input<Vector<FPType>> activations;
  Output<Vector<FPType>> deltasOut;
  NonLinearityType nonLinearityType;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(deltasIn.size() == deltasOut.size());
    assert(deltasIn.size() == activations.size());
    for (unsigned i = 0; i < deltasIn.size(); ++i) {
      deltasOut[i] = deltasIn[i] * nonlinearity_derivative(nonLinearityType,
                                                           activations[i]);
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (deltasIn.size() + vectorWidth - 1) / vectorWidth;
    return 5 + numVectors * 3;
  }
};

template class NonLinearityBwd<float>;
template class NonLinearityBwd<half>;

/**
 * Compute a sum of 1x1 convolutions over a subset of the input channels for
 * multiple output channels.
 **/
template <class Base, class FPType, class AccumType>
class ConvPartial1x1Out: public Base {
public:
  Vector<Input<Vector<FPType>>> in;
  Input<Vector<FPType>> weights;
  Vector<Output<Vector<AccumType>>> out;
  Vector<unsigned> weightReuseCount;

  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> inChansPerGroup;
  SimOnlyField<unsigned> outChansPerGroup;

  bool compute() {
    unsigned numContexts = weightReuseCount.size();
    assert(weights.size() % (inChansPerGroup * outChansPerGroup) == 0);
    const auto numInChanGroups =
        weights.size() / (inChansPerGroup * outChansPerGroup);
    assert(in.size() == out.size());
    unsigned conv = 0;
    for (unsigned inChanGroup = 0; inChanGroup != numInChanGroups;
         ++inChanGroup) {
      for (unsigned context = 0; context < numContexts; ++context) {
        unsigned endConv = conv + weightReuseCount[context];
        for (;conv != endConv; ++conv) {
          assert(out[conv].size() % outChansPerGroup == 0);
          const auto outWidth = out[conv].size() / outChansPerGroup;
          assert(in[conv].size() % inChansPerGroup == 0);
          const auto inWidth = in[conv].size() / inChansPerGroup;
          unsigned inStride;
          if (outWidth == 1) {
            assert(inWidth == 1);
            inStride = 0;
          } else {
            assert((inWidth - 1) % (outWidth - 1) == 0);
            inStride = (inWidth - 1) / (outWidth - 1);
          }
          for (unsigned x = 0; x != outWidth; ++x) {
            for (unsigned outChanIndex = 0; outChanIndex != outChansPerGroup;
                 ++outChanIndex) {
              const auto outIndex =
                  outChanIndex + outChansPerGroup * x;
              if (inChanGroup == 0)
                out[conv][outIndex] = 0;
              float sum = 0;
              for (unsigned inChanIndex = 0; inChanIndex != inChansPerGroup;
                   ++inChanIndex) {
                const auto weightIndex =
                    inChanIndex + inChansPerGroup * (
                      outChanIndex + outChansPerGroup * (
                        inChanGroup
                      )
                    );
                const auto inIndex =
                    inChanIndex + inChansPerGroup * x * inStride;
                sum += weights[weightIndex] * in[conv][inIndex];
              }
              out[conv][outIndex] += sum;
            }
          }
        }
      }
    }
    assert(conv == out.size());
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isSupervisorVertex = std::is_same<Base, SupervisorVertex>::value;
    const auto numContexts = weightReuseCount.size();
    const auto numConvUnitsPerTile = outChansPerGroup;
    assert(dataPathWidth % 16 == 0);
    const auto halfVectorWidth = dataPathWidth / 16;
    assert(inChansPerGroup % halfVectorWidth == 0);
    const auto convUnitPipelineDepth = inChansPerGroup / halfVectorWidth;
    if (isSupervisorVertex) {
      std::vector<std::vector<std::vector<unsigned>>>
          convolutionsByWeightAndWorker;
      const auto numInChanGroups =
          weights.size() / (inChansPerGroup * outChansPerGroup);
      unsigned convNum = 0;
      for (unsigned inChanGroup = 0; inChanGroup != numInChanGroups;
           ++inChanGroup) {
        convolutionsByWeightAndWorker.emplace_back();
        auto &convolutionsByWeight = convolutionsByWeightAndWorker.back();
        for (unsigned c = 0; c != numContexts; ++c) {
          convolutionsByWeight.emplace_back();
          for (unsigned i = 0; i != weightReuseCount[c];
               ++i) {
            auto convSize = out[convNum].size() / outChansPerGroup;
            convolutionsByWeight.back().push_back(convSize);
            ++convNum;
          }
        }
      }
      assert(convNum == out.size());
      return getConvPartialnx1SupervisorCycleEstimate(
        convolutionsByWeightAndWorker,
        convUnitPipelineDepth,
        numConvUnitsPerTile,
        1
      );
    }
    assert(numContexts == 1);
    std::vector<std::vector<unsigned>> convolutionsByWeight(1);
    unsigned convNum = 0;
    for (unsigned w = 0; w != weights.size(); ++w) {
      convolutionsByWeight.emplace_back();
      for (unsigned i = 0; i != weightReuseCount[w]; ++i) {
        auto convSize = out[convNum].size() / outChansPerGroup;
        convolutionsByWeight.back().push_back(convSize);
        ++convNum;
      }
    }
    assert(convNum == out.size());
    return getConvPartialnx1CycleWorkerEstimate(convolutionsByWeight,
                                                convUnitPipelineDepth,
                                                numConvUnitsPerTile,
                                                1);
  }
};

template class ConvPartial1x1Out<Vertex, float, float>;
template class ConvPartial1x1Out<Vertex, float, half>;
template class ConvPartial1x1Out<SupervisorVertex, float, float>;
template class ConvPartial1x1Out<SupervisorVertex, float, half>;
template class ConvPartial1x1Out<Vertex, half, float>;
template class ConvPartial1x1Out<Vertex, half, half>;
template class ConvPartial1x1Out<SupervisorVertex, half, float>;
template class ConvPartial1x1Out<SupervisorVertex, half, half>;

/* Compute a partial convolution for a sub-set of input channels and
 * output channels over a number of rows of the input field. */
template <typename InType, typename AccumType>
class ConvPartial: public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Input<Vector<InType>>> weights;
  Output<Vector<AccumType>> out;
  unsigned inChansPerGroup;
  // The amount of implicit of zero padding before the first element of the
  // input.
  unsigned padding;
  unsigned stride;

  SimOnlyField<unsigned> dataPathWidth;

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
      auto *row = &in[i][0];
      auto *rowWeights = &weights[i][0];
      for (unsigned outX = 0; outX < outputWidth; ++outX) {
        int inXBegin = static_cast<int>(outX * stride) - padding;
        unsigned inXEnd = std::min(inXBegin + kernelSize,
                                   inputWidth);
        unsigned weightShift = 0;
        if (inXBegin < 0) {
          weightShift = -inXBegin;
          inXBegin = 0;
        }
        for (unsigned inX = inXBegin; inX != inXEnd; ++inX) {
          unsigned weightX = inX - inXBegin + weightShift;
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
    bool isFloat = std::is_same<InType, float>::value;
    return getConvPartialByDotProductCycleEstimate(isFloat, inChansPerGroup,
                                                   kernelSize, numInRows,
                                                   outputWidth,
                                                   dataPathWidth, 1);
  }
};

template class ConvPartial<float, float>;
template class ConvPartial<half, float>;
template class ConvPartial<half, half>;

template <typename FPType>
class ConvReduce : public Vertex {
public:
  Vector<Output<Vector<FPType>>> out;
  Vector<Input<Vector<FPType>>> partials;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        float sum = 0;
        for (unsigned j = 0; j < numPartials; ++j) {
          sum += partials[r * numPartials + j][i];
        }
        out[r][i] = sum;
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned cycles = 4;
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      cycles += 1 + numElem * (1 + (numPartials + vectorWidth - 1) / vectorWidth);
    }

    return cycles;
  }
};

template class ConvReduce<float>;
template class ConvReduce<half>;

template <class InType, class OutType>
class ConvComplete : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Input<Vector<OutType>>> bias;
  Vector<Output<Vector<OutType>>> out;
  NonLinearityType nonLinearityType;
  Vector<unsigned> outputChanGroupsPerBias;
  Vector<Input<Vector<OutType>>> res;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numOut = out.size();
    unsigned outChans = bias[0].size();
    unsigned chunkSize = in[0].size();
    unsigned biasIndex = 0;
    unsigned biasCount = outputChanGroupsPerBias[0];
    unsigned inIndex = 0;
    for (unsigned o = 0; o < numOut; ++o) {
      unsigned outCols = out[o].size() / outChans;
      for (unsigned ocol = 0; ocol < outCols; ++ocol) {
        for (unsigned ochan = 0; ochan < outChans; ++ochan) {
          auto outIndex = ocol * outChans + ochan;
          float sum = in[inIndex / chunkSize][inIndex % chunkSize];
          ++inIndex;
          sum += bias[biasIndex][ochan];
          // The outputs are ordered in a way such that residuals may
          // only be needed for outputs at the beginning of the sequence
          // (or none at all) if the residual has fewer channels than
          // the output.
          if (o < res.size())
            sum += res[o][outIndex];
          out[o][outIndex] = nonlinearity(nonLinearityType, sum);
        }
      }
      --biasCount;
      if (biasCount == 0) {
        ++biasIndex;
        if (biasIndex < outputChanGroupsPerBias.size())
          biasCount = outputChanGroupsPerBias[biasIndex];
      }
    }
    assert(biasIndex == outputChanGroupsPerBias.size());
    assert(inIndex == in.size() * chunkSize);
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool inIsFloat = std::is_same<InType, float>::value;
    bool outIsFloat = std::is_same<InType, float>::value;
    assert(!outIsFloat || inIsFloat && "Output is wider than input");
    const auto inVectorWidth = dataPathWidth / (inIsFloat ? 32 : 16);
    unsigned numOut = out.size();
    unsigned outChans = bias[0].size();
    unsigned chunkSize = in[0].size();
    unsigned numCycles = 5;
    for (unsigned o = 0; o < numOut; ++o) {
      unsigned outCols = out[o].size() / outChans;
      for (unsigned ocol = 0; ocol < outCols; ++ocol) {
        assert(outChans % chunkSize == 0);
        for (unsigned chunk = 0; chunk != outChans / chunkSize; ++chunk) {
          // load input, load bias and add
          // - dual loads, dual issue = 2 vectors in 2 cycles
          numCycles += (chunkSize + inVectorWidth - 1) / inVectorWidth;
          if (o < res.size())
            // Load residual and add.
            numCycles += (chunkSize + inVectorWidth - 1) / inVectorWidth;
          numCycles += (chunkSize + inVectorWidth - 1) / inVectorWidth; // RELU
        }
      }
    }
    return numCycles;
  }
};

template class ConvComplete<float, float>;
template class ConvComplete<float, half>;
template class ConvComplete<half, half>;

template <typename FPType>
class CopyResidual : public Vertex {
public:
  Input<Vector<FPType>> in;
  Output<Vector<FPType>> out;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned i = 0; i < in.size(); ++i)
      out[i] = in[i];
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO: make this more accurate
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    auto copiesPerCycle = vectorWidth;
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

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned i = 0; i < out.size(); ++i) {
      out[i] = 0;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO: make this more accurate
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    auto zeroCycles = (out.size() + vectorWidth - 1) / vectorWidth;
    return 4 + zeroCycles;
  }
};

template class Zero<float>;
template class Zero<half>;

template <typename FPType>
class Zero2D : public Vertex {
public:
  Vector<Output<Vector<FPType>>> out;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (auto &row : out) {
      for (auto &x : row) {
        x = 0;
      }
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const {
    // TODO: make this more accurate
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    std::uint64_t cycles = 4;
    for (auto &row : out) {
      auto zeroCycles = (row.size() + vectorWidth - 1) / vectorWidth;
      cycles += 1 + zeroCycles;
    }
    return cycles;
  }
};

template class Zero2D<float>;
template class Zero2D<half>;

template <typename FPType>
class MaxPooling : public Vertex {
public:
  Vector<Input<Vector<FPType>>> activationIn;
  Output<Vector<FPType>> activationOut;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    auto outChansPerGroup = activationOut.size();
    auto chunkSize = activationIn[0].size();
    assert(outChansPerGroup % chunkSize == 0);
    auto chunksPerGroup = outChansPerGroup / chunkSize;
    assert(activationIn.size() % chunksPerGroup == 0);
    unsigned receptiveFieldSize = activationIn.size() / chunksPerGroup;
    for (unsigned chunk = 0; chunk != chunksPerGroup; ++chunk) {
      for (unsigned i = 0; i != receptiveFieldSize; ++i) {
        for (unsigned chanInChunk = 0; chanInChunk != chunkSize;
             ++chanInChunk) {
          auto chan = chunk * chunkSize + chanInChunk;
          auto in = activationIn[chunk * receptiveFieldSize + i][chanInChunk];
          if (i == 0 || activationOut[chan] < in)
            activationOut[chan] = in;
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numCycles = 10;
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    //numChunks = inputFieldSize * chunksPerGroup
    auto numChunks = activationIn.size();
    auto chunkSize = activationIn[0].size();
    numCycles += numChunks * (1 + (chunkSize + vectorWidth - 1) / vectorWidth);
    return numCycles;
  }
};

template class MaxPooling<float>;
template class MaxPooling<half>;

template <typename FPType>
class MaxPoolingBwd : public Vertex {
public:
  Vector<Input<Vector<FPType>>> actOut;
  Input<Vector<FPType>> actIn;
  Vector<Input<Vector<FPType>>> errIn;
  Output<Vector<FPType>> errOut;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    // Update errOut with the sum of errors from all kernels that it updated
    auto nOutChannels = errOut.size();
    auto chunkSize = errIn[0].size();
    auto nextFieldSize = errIn.size();
    assert(chunkSize == nOutChannels);
    assert(nOutChannels % chunkSize == 0);
    auto nChunks = nOutChannels / chunkSize;
    for (unsigned i = 0; i != nextFieldSize; ++i) {
      for (unsigned chan = 0; chan != nOutChannels; ++chan) {
        if (i == 0)
          errOut[chan] = 0;
        if (actIn[chan] == actOut[i][chan])
          errOut[chan] += errIn[i][chan];
      }
    }

    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numCycles = 10;
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    auto nChanGroups = (errOut.size() + vectorWidth - 1) / vectorWidth;
    auto nextFieldSize = errIn.size();
    // Expected implementation per group:
    // load group of actIn
    // for fieldsize:
    // load actOut
    //  compare
    //  res<<=14 (covert to 0.5/0)
    //  mac
    // getacc
    // double
    // store
    numCycles += ((3 * nextFieldSize) + 5) * nChanGroups;
    return numCycles;
  }
};

template class MaxPoolingBwd<float>;
template class MaxPoolingBwd<half>;

template <typename FPType>
class CalcLoss : public Vertex {
public:
  Vector<Input<Vector<FPType>>> batchIn;
  Input<unsigned> label;

  Vector<Output<Vector<FPType>>> batchDeltaOut;
  Output<FPType> loss;
  InOut<unsigned> numCorrect;

  Vector<FPType> probs;

  LossType lossType;

  bool compute() {
    const auto batchSize = batchIn.size();
    assert(batchIn.size() == batchDeltaOut.size());
    for (unsigned batchNum = 0; batchNum < batchSize; ++batchNum) {
      auto in = batchIn[batchNum];
      auto deltaOut = batchDeltaOut[batchNum];
      assert(in.size() == deltaOut.size());
      assert(probs.size() == in.size());
      switch (lossType) {
      case SUM_SQUARED_LOSS: {
        /* Calculate the sum-squared error and the partial derivative
           to pass back. */
        FPType sum = 0;
        for (unsigned i = 0;  i < in.size(); ++i) {
          FPType expected = (i == label ? 1 : 0);
          FPType actual = in[i];
          deltaOut[i] = (actual - expected);
          sum += 0.5 * (actual - expected) *  (actual - expected);
        }
        *loss = sum;
        }
        break;
      case SOFTMAX_CROSS_ENTROPY_LOSS:
        /* Calculate the softmax probability distribution */
        for (unsigned i = 0;  i < in.size(); ++i) {
          FPType act = in[i];
          probs[i] = exp(act);
        }
        FPType sum = 0;
        for (FPType p : probs)
          sum += p;

        for (unsigned i = 0;  i < in.size(); ++i) {
          probs[i] /= sum;
        }

        /* Calculate the cross-entropy error and the partial derivative
         to pass back. */
        FPType error = 0;
        for (unsigned i = 0;  i < probs.size(); ++i) {
          FPType expected = (i == label ? 1 : 0);
          deltaOut[i] = (probs[i] - expected);
          error += expected * log(probs[i]);
        }
        *loss = error;
        break;
      }

      // Calculate the classification error for reporting test results
      // This assumes that the
      // non-linearity is monotonic, so the max output of the previous
      // layer is the max z-term of the previous layer.
      FPType max = in[0];
      unsigned maxIndex = 0;
      for (unsigned i = 0;  i < in.size(); ++i) {
        if (in[i] > max) {
          max = in[i];
          maxIndex = i;
        }
      }
      bool correct = (maxIndex == label);
      if (correct) {
        *numCorrect += 1;
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 0;
  }
};


template class CalcLoss<float>;
template class CalcLoss<half>;


template <class InType, class OutType>
class RegroupChans : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<OutType>>> out;
  unsigned outChans;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numOut = out.size();
    unsigned chunkSize = in[0].size();
    unsigned inIndex = 0;
    for (unsigned o = 0; o < numOut; ++o) {
      unsigned outCols = out[o].size() / outChans;
      for (unsigned ocol = 0; ocol < outCols; ++ocol) {
        for (unsigned ochan = 0; ochan < outChans; ++ochan) {
          auto outIndex = ocol * outChans + ochan;
          out[o][outIndex] = in[inIndex / chunkSize][inIndex % chunkSize];
          ++inIndex;
        }
      }
    }
    assert(inIndex == in.size() * chunkSize);
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool inIsFloat = std::is_same<InType, float>::value;
    bool outIsFloat = std::is_same<InType, float>::value;
    assert(!outIsFloat || inIsFloat && "Output is wider than input");
    const auto inVectorWidth = dataPathWidth / (inIsFloat ? 32 : 16);
    unsigned numOut = out.size();
    unsigned chunkSize = in[0].size();
    unsigned numCycles = 5;
    for (unsigned o = 0; o < numOut; ++o) {
      unsigned outCols = out[o].size() / outChans;
      for (unsigned ocol = 0; ocol < outCols; ++ocol) {
        assert(outChans % chunkSize == 0);
        for (unsigned chunk = 0; chunk != outChans / chunkSize; ++chunk) {
          // This assumes the loop is pipelined such that loads and stores
          // of data being transposed happen in the same cycle. An additional
          // cycle is required to load the pointer to the chunk.
          numCycles += 1 + (chunkSize + inVectorWidth - 1) / inVectorWidth;
        }
      }
    }
    return numCycles;
  }
};

template class RegroupChans<float, float>;
template class RegroupChans<float, half>;
template class RegroupChans<half, half>;


template <class Type>
class DimShuffle : public Vertex {
public:
  Vector<Input<Vector<Type>>> in;
  Output<Vector<Type>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numOut = out.size();
    assert(numOut % in.size() == 0);
    unsigned chunkSize = numOut / in.size();
    unsigned numChunks = numOut / chunkSize;
    for (unsigned i = 0; i != numChunks; ++i) {
      for (unsigned j = 0; j != chunkSize; ++j) {
        out[i * chunkSize + j] = in[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<Type, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numOut = out.size();
    unsigned chunkSize = numOut / in.size();
    unsigned numChunks = numOut / chunkSize;
    return 5 + numChunks * (1 + (chunkSize + vectorWidth - 1) / vectorWidth);
  }
};

template class DimShuffle<half>;
template class DimShuffle<float>;

template <class FPType>
class ConvTransformWeights : public Vertex {
public:
  Vector<Input<FPType>> in;
  Vector<Output<Vector<FPType>>> out;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned inIndex = 0;
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned j = 0; j < out[i].size(); ++j) {
        out[i][j] = in[inIndex++];
      }
    }
    assert(inIndex == in.size());
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 5 + 3 * out.size();
  }
};

template class ConvTransformWeights<float>;
template class ConvTransformWeights<half>;



template <class FPType, unsigned patchSizeX, unsigned patchSizeY,
          unsigned kernelX, unsigned kernelY>
class WgdDataTransform : public Vertex {

  /* Set this to true if transform is stored in transposed order */
  static constexpr bool transpose = true;

  FPType rdIn(unsigned base, unsigned row, unsigned col, unsigned el) const {
    return dIn[base + col * patchSizeX + row][el];
  }

  FPType& wrTf(unsigned base, unsigned row, unsigned col, unsigned el) {
    if (!transpose) {
      return dTf[base + col * patchSizeX + row][el];
    }
    else
    {
      return dTf[base + row * patchSizeY + col][el];
    }
  }

public:
  /* The input is an array of one dimensional vectors each of size equal
   * to a number of independent input channels. This implementation differs from
   * assembler implementation in that it assumes a vector for every X,Y point 
   * and doesn't require the vector length to be a multiple of 4.
   * The assembler implementation assumes a pointer to every Y with known
   * dim(X)*dim(Z_in_partial).
   */
  Vector<Input<Vector<FPType>>> dIn;

  /* Exactly same implementation details as input vector dIn
   */
  Vector<Output<Vector<FPType>>> dTf;

  bool compute() {

    assert(patchSizeX == 4);
    assert(patchSizeY == 4);
    assert(kernelX == 3);
    assert(kernelY == 3);
    const unsigned numInpCols = patchSizeY;
    const unsigned numInpRows = patchSizeX;
    const unsigned numOutCols = patchSizeY;
    const unsigned numOutRows = patchSizeX;

    const unsigned nPatches = dIn.size() / (numInpRows * numInpCols);

    for (auto patch = 0; patch < nPatches; ++patch) {
      /* patch Base */
      unsigned pBase = patch * numInpCols * numInpRows;

      const unsigned depth = dIn[0].size();

      for (int elem = 0; elem < depth; ++elem) {
        FPType dTemp[numOutCols][numOutCols];

        /* First stage: input tile must be square */
        for (unsigned row = 0; row < numInpRows; ++row) {
          dTemp[row][0] = rdIn(pBase, row, 0, elem) - rdIn(pBase, row, 2, elem);
          dTemp[row][1] = rdIn(pBase, row, 1, elem) + rdIn(pBase, row, 2, elem);

          dTemp[row][2] = rdIn(pBase, row, 2, elem) - rdIn(pBase, row, 1, elem);
          dTemp[row][3] = rdIn(pBase, row, 1, elem) - rdIn(pBase, row, 3, elem);
        }

        /* Final stage: rows==columns for outputs */
        for (unsigned col = 0; col < numOutCols; ++col) {
          wrTf(pBase, 0, col, elem) = dTemp[0][col] - dTemp[2][col];
          wrTf(pBase, 1, col, elem) = dTemp[1][col] + dTemp[2][col];
          wrTf(pBase, 2, col, elem) = dTemp[2][col] - dTemp[1][col];
          wrTf(pBase, 3, col, elem) = dTemp[1][col] - dTemp[3][col];
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const unsigned numInpRows = patchSizeX;
    const unsigned numInpCols = patchSizeY;

    const unsigned nPatches = dIn.size() / (numInpCols * numInpRows);

    return getWgdDataTransformCycles(nPatches * dIn[0].size(), isFloat);
  }
};

template class WgdDataTransform<float, 4, 4, 3, 3>;
template class WgdDataTransform<half, 4, 4, 3, 3>;



template <class FPType, unsigned patchSizeX, unsigned patchSizeY,
          unsigned kernelX, unsigned kernelY>
class WgdKernelTransform : public Vertex {

  /* Set this to true if transform is stored in transposed order */
  static constexpr bool transpose = true;

  /* storage depends on whether transpose or normal form of transform is 
   * stored 
   */
  FPType& wrTf(const unsigned base, const unsigned row, const unsigned col, 
               const unsigned elem) {
    return transpose ? wTf[base + row * patchSizeY + col][elem] :
                           wTf[base + col * patchSizeX + row][elem];
  }

  FPType rdIn(unsigned base, unsigned row, unsigned col, unsigned elem) const {
    return wIn[base + col * kernelX + row][elem];
  }

public:
  /* Each input is a 1D vector of independent channels which may be a mix of 
   * input and output channels. Therefore kernelCols*kernelRow vectors are 
   * required to have all elements of a kernel. The 1D vectors are stored in row 
   * order
   */
  Vector<Input<Vector<FPType>>> wIn;

  /* Same as wIn except that numOutCols*numOutRows vectors each of dimension 
   * 1xdepth are stored
   */
  Vector<Output<Vector<FPType>>> wTf;


  bool compute() {
    const unsigned numOutCols = patchSizeY;
    const unsigned numOutRows = patchSizeX;
    const unsigned nGroups = wTf.size() / (numOutCols * numOutRows);
    assert(numOutCols == 4);
    assert(numOutRows == 4);
    assert(kernelX == 3);
    assert(kernelY == 3);

    for (int group = 0; group < nGroups; ++group) {
      unsigned gBaseIn  = kernelY * kernelX * group;
      unsigned gBaseOut = numOutRows * numOutCols * group;

      const unsigned depth = wIn[0].size();

      for (unsigned elem = 0; elem < depth; ++elem) {
        FPType g[kernelX][kernelY];

        for (unsigned row = 0; row < kernelX; ++row) {
          for (unsigned col = 0; col < kernelY; ++col) {
            g[row][col] = rdIn(gBaseIn, row, col, elem);
          }
        }

        FPType A = (g[0][0] + g[0][1] + g[0][2]) * 0.5;
        FPType B = (g[0][0] - g[0][1] + g[0][2]) * 0.5;

        FPType C = (g[0][0] + g[1][0] + g[2][0]) * 0.5;
        FPType F = (g[0][0] - g[1][0] + g[2][0]) * 0.5;

        FPType D = (g[2][0] + g[2][1] + g[2][2]) * 0.5;
        FPType E = (g[2][0] - g[2][1] + g[2][2]) * 0.5;

        FPType G = (g[1][0] + g[1][1] + g[1][2]) * 0.5;
        FPType H = (g[1][0] - g[1][1] + g[1][2]) * 0.5;

        FPType I = (g[0][2] + g[1][2] + g[2][2]) * 0.5;
        FPType J = (g[0][2] - g[1][2] + g[2][2]) * 0.5;

        wrTf(gBaseOut, 0, 0, elem) = g[0][0];
        wrTf(gBaseOut, 0, 1, elem) = A;
        wrTf(gBaseOut, 0, 2, elem) = B;
        wrTf(gBaseOut, 0, 3, elem) = g[0][2];

        wrTf(gBaseOut, 1, 0, elem) = C;
        wrTf(gBaseOut, 1, 1, elem) = (A + G + D) * 0.5;
        wrTf(gBaseOut, 1, 2, elem) = (B + H + E) * 0.5;
        wrTf(gBaseOut, 1, 3, elem) = I;

        wrTf(gBaseOut, 2, 0, elem) = F;
        wrTf(gBaseOut, 2, 1, elem) = (A - G + D) * 0.5;
        wrTf(gBaseOut, 2, 2, elem) = (B - H + E) * 0.5;
        wrTf(gBaseOut, 2, 3, elem) = J;

        wrTf(gBaseOut, 3, 0, elem) = g[2][0];
        wrTf(gBaseOut, 3, 1, elem) = D;
        wrTf(gBaseOut, 3, 2, elem) = E;
        wrTf(gBaseOut, 3, 3, elem) = g[2][2];
      }
    }
    return true;
  }


  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const unsigned numOutRows = patchSizeX;
    const unsigned numOutCols = patchSizeY;

    const unsigned nGroups = wTf.size() / (numOutCols * numOutRows);

    return getWgdInvTransformCycles(wIn[0].size() * nGroups, isFloat);
  }
};

template class WgdKernelTransform<float, 4, 4, 3, 3>;
template class WgdKernelTransform<half, 4, 4, 3, 3>;


template <class Base, class FPType, bool accumulate>
class WgdPartials : public Base {

public:
  /* data transform vectors. Each vector is a 1D vector of length inpChanDepth.
   * Every input vector shares the same weight vector.
   * A total of nGroups 1D vectors may be provided.
   */
  Vector<Input<Vector<FPType>>> dTf;

  /* kernel transform vector. Each vector is of length inpChanDepth*outChanDepth
   * The same input data is used to generate outChanDepth outputs for each input
   * vector
   */
  Vector<Input<Vector<FPType>>> wTf;

  /* Output for each of the nGroups 1D vectors. Each input vector results in a 
   * 1xoutChanDepth vector.
   */
  Vector<InOut<Vector<FPType>>> partials;


  SimOnlyField<unsigned> numWorkers;
  SimOnlyField<unsigned> weightsPerConvUnit;
  SimOnlyField<unsigned> numConvUnits;

  bool compute() {

    const unsigned outChanDepth = partials[0].size();
    const unsigned inpChanDepth = dTf[0].size();
    const unsigned numInpGroups = wTf.size();
    const unsigned comPencils = partials.size();



    /* all feature elements share the same weights */
    assert(wTf[0].size() == inpChanDepth * outChanDepth);

    for (unsigned ig = 0; ig < numInpGroups; ++ig) {
      for (unsigned gr = 0; gr < comPencils; ++gr) {
        for (unsigned oc = 0; oc < outChanDepth; ++oc) {
          FPType acc{0};

          for (unsigned ic = 0; ic < inpChanDepth; ++ic) {
            const auto idx = ig * comPencils + gr;
            acc += dTf[idx][ic] * wTf[ig][oc * inpChanDepth + ic];
          }
          if (accumulate) {
            if (ig == 0)
              partials[gr][oc] = acc;
            else
              partials[gr][oc] += acc;              
          } else {
            partials[gr][oc] = acc;
          }
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isSupervisorVertex = std::is_same<Base, SupervisorVertex>::value;
    bool isFloat = std::is_same<FPType, float>::value;
    const unsigned outChanDepth = partials[0].size();
    const unsigned inpChanDepth = dTf[0].size();
    const unsigned comPencils = partials.size();
    const unsigned numInpGroups = wTf.size();


    return getWgdAccumCycles(
                      isSupervisorVertex,
                      numInpGroups,
                      comPencils,
                      inpChanDepth,
                      outChanDepth,
                      numWorkers,
                      numConvUnits,
                      weightsPerConvUnit,
                      isFloat);
  }

};

template class WgdPartials<poplar::SupervisorVertex, float, true>;
template class WgdPartials<poplar::Vertex, float, true>;
template class WgdPartials<poplar::SupervisorVertex, float, false>;
template class WgdPartials<poplar::Vertex, float, false>;

template class WgdPartials<poplar::SupervisorVertex, half, true>;
template class WgdPartials<poplar::Vertex, half, true>;
template class WgdPartials<poplar::SupervisorVertex, half, false>;
template class WgdPartials<poplar::Vertex, half, false>;


template <class FPType, unsigned patchSizeX, unsigned patchSizeY>
class WgdReduce: public Vertex {

public:
  /* The vector of partial contains 1D vectors of length inpLength. The 
   * partialSumLen 1D vectors are summed to produce a single output vector of 
   * the same length as the input vector. Several such operations may be 
   * performed to produce nGroups vectors of 1D vectors.
   */
  Vector<Input<Vector<FPType>>> inPartial;

  /*
   * The output may be a sum of all partials to produce partial sum or a full 
   * sum
   */
  Vector<Output<Vector<FPType>>> outPartial;

  bool compute() {
    const unsigned numOutRows = patchSizeX;
    const unsigned numOutCols = patchSizeY;
    const unsigned numElems = outPartial.size();
    const unsigned numOutChans = outPartial[0].size();
    const unsigned numInpChans = inPartial.size() / numElems;



    for (unsigned elem = 0; elem < numElems ; ++elem) {

      auto inIdx = elem * numInpChans;

      for (unsigned oc = 0; oc < numOutChans; ++oc) {

        FPType acc {0};

        for (unsigned ic = 0; ic < numInpChans; ++ic) {
          acc += inPartial[inIdx + ic][oc];
        }

        outPartial[elem][oc] = acc;
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const unsigned numOutCols = patchSizeY;
    const unsigned numOutRows = patchSizeX;

    const unsigned numElems = outPartial.size();
    const unsigned numOutChans = outPartial[0].size();
    const unsigned numInpChans = inPartial.size() / numElems;

    return getWgdReduceCycles(
                   numElems * numOutChans,
                   numInpChans,
                   isFloat
                   );
  }
};

template class WgdReduce<float, 4, 4>;
template class WgdReduce<half, 4, 4>;



template <class FPType, unsigned patchSizeX, unsigned patchSizeY,
          unsigned kernelX, unsigned kernelY>
class WgdInverseTransform : public Vertex {

  /* Set this to true if transform is stored in transposed order */
  static constexpr bool transpose = true;

  FPType rdTf(const unsigned base, const unsigned row, const unsigned col, 
              const unsigned el) const {
    return dTf[base+col*patchSizeX+row][el];
  }

  FPType& wrOut(const unsigned base,  unsigned row, const unsigned col, 
                const unsigned el) {
    const unsigned numOutCols = patchSizeY - kernelY + 1;
    const unsigned numOutRows = patchSizeX - kernelX + 1;
    if (!transpose) {
      return dOut[base + col * numOutRows + row][el];
    }
    else
    {
      return dOut[base + row * numOutCols + col][el];
    }
  }

public:
  /* The data transform vector dTf is an array of vectors each of length 
   * depthDim. The 1D vectors are stacked to have 16 elements called a group 
   * which are rows and columns needed to compute the inverse transform.
   */
  Vector<Input<Vector<FPType>>> dTf;

  /* Each output vector in the array of vectors is of length depthDim. 
   * numOutCols*numOutRows vectors are produced for each group
   */
  Vector<Output<Vector<FPType>>> dOut;

  bool compute() {

    const unsigned numInCols = patchSizeY;
    const unsigned numInRows = patchSizeX;
    const unsigned numOutCols = patchSizeY - kernelY + 1;
    const unsigned numOutRows = patchSizeX - kernelX + 1;

    assert(numInCols == 4);
    assert(numInRows == 4);
    assert(kernelX == 3);
    assert(kernelY == 3);

    const unsigned nGroups = dTf.size() / (numInCols * numInRows);

    for (unsigned gr = 0; gr < nGroups; ++gr) {
      unsigned grInOff = gr * numInCols * numInRows;
      unsigned grOutOff = gr * numOutCols * numOutRows;
      const unsigned depthDim = dTf[0].size();

      for (unsigned elem = 0; elem < depthDim; ++elem) {
        FPType e = rdTf(grInOff, 0, 0, elem) + rdTf(grInOff, 0, 1, elem) 
                                             + rdTf(grInOff, 0, 2, elem);
        FPType f = rdTf(grInOff, 0, 1, elem) - rdTf(grInOff, 0, 2, elem) 
                                             - rdTf(grInOff, 0, 3, elem);

        FPType a = rdTf(grInOff, 1, 0, elem) + rdTf(grInOff, 1, 1, elem) 
                                             + rdTf(grInOff, 1, 2, elem);
        FPType c = rdTf(grInOff, 1, 1, elem) - rdTf(grInOff, 1, 2, elem) 
                                             - rdTf(grInOff, 1, 3, elem);

        FPType b = rdTf(grInOff, 2, 0, elem) + rdTf(grInOff, 2, 1, elem) 
                                             + rdTf(grInOff, 2, 2, elem);
        FPType d = rdTf(grInOff, 2, 1, elem) - rdTf(grInOff, 2, 2, elem) 
                                             - rdTf(grInOff, 2, 3, elem);

        FPType g = rdTf(grInOff, 3, 0, elem) + rdTf(grInOff, 3, 1, elem) 
                                             + rdTf(grInOff, 3, 2, elem);
        FPType h = rdTf(grInOff, 3, 1, elem) - rdTf(grInOff, 3, 2, elem) 
                                             - rdTf(grInOff, 3, 3, elem);

        wrOut(grOutOff, 0, 0, elem) = a + b + e;
        wrOut(grOutOff, 1, 0, elem) = a - b - g;
        wrOut(grOutOff, 0, 1, elem) = c + d + f;
        wrOut(grOutOff, 1, 1, elem) = c - d - h;
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const unsigned numInCols = patchSizeY;
    const unsigned numInRows = patchSizeX;

    const unsigned nGroups = dTf.size() / (numInCols * numInRows);
    const unsigned depthDim = dOut[0].size();

    return getWgdInvTransformCycles(nGroups * depthDim, isFloat);
  }
};

template class WgdInverseTransform<float, 4, 4, 3, 3>;
template class WgdInverseTransform<half, 4, 4, 3, 3>;


template <class FPType>
class WgdConvComplete : public Vertex {

public:
  /* Each input vector is a of length "vecLen"
   */
  Vector<Input<Vector<FPType>>> dIn;

  /* Each bias vector is of length "vecLen"
   */
  Vector<Input<Vector<FPType>>> bias;

  /* The output activation once non-linearity is applied
   */
  Vector<Output<Vector<FPType>>> act;

  /* Non linearity applied to compute activation */
  NonLinearityType nonLinearityType;

  bool compute() {
    const unsigned nGroups = dIn.size();
    const unsigned vecLen = dIn[0].size();

    for (unsigned gr = 0; gr < nGroups; ++gr) {
      for (unsigned el = 0; el < vecLen; ++el) {
        act[gr][el] = nonlinearity(nonLinearityType, bias[gr][el]+dIn[gr][el]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const unsigned nGroups = dIn.size();    
    const unsigned vecLen = dIn[0].size();
    return getWgdCompleteCycles(
                               vecLen * nGroups,
                               nonLinearityType,
                               isFloat);
  }
};

template class WgdConvComplete<float>;
template class WgdConvComplete<half>;
