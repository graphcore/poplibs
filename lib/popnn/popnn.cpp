#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>
#include "popnn/NonLinearityDef.hpp"
#include "popnn/ResidualDef.hpp"
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


namespace popnn {

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
class NonLinearityFwd : public Vertex {
public:
  Input<Vector<FPType>> activationIn;
  NonLinearityType nonLinearityType;
  Output<Vector<FPType>> activationOut;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned i = 0; i < activationIn.size(); ++i) {
      activationOut[i] = nonlinearity(nonLinearityType, activationIn[i]);
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    return getNonLinearityCycles(activationIn.size(), nonLinearityType,
                                 isFloat, dataPathWidth);

  }
};

template class NonLinearityFwd<float>;
template class NonLinearityFwd<half>;

template <typename FPType>
class FullyConnectedReduce : public Vertex {
public:
  Input<Vector<float>> partials;
  Input<FPType> bias;
  Output<FPType> activationOut;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < partials.size(); ++i) {
      sum += partials[i];
    }
    sum += *bias;
    *activationOut = sum;
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
  Vector<Input<FPType>> d;
  InOut<Vector<FPType>> weights;
  Vector<Input<Vector<FPType>>> in;
  float eta;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    const auto batchSize = d.size();
    for (unsigned i = 0; i < weights.size(); ++i) {
      float grad = 0;
      for (unsigned b = 0; b < batchSize; ++b) {
        grad += d[b] * in[b][i];
      }
      weights[i] = weights[i] - grad * eta;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    const auto batchSize = d.size();
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (weights.size() + vectorWidth - 1) / vectorWidth;

    if (batchSize == 1) {
      // Assume a specialized version that accumulates directly into the
      // weight vector.
      // Inner loop involves multiplication by (*d * eta) and addition.
      return 5 + 2 * numVectors;
    } else if (batchSize <= 4) {
      // Assume a specialized version where each delta is loaded to a register
      auto deltaLoadCycles = batchSize * 3; // Load, conversion and multiply
                                            // by eta for each delta
      // Unrolled inner loop  involves multiplication by (*d * eta) and
      // addition for each element in batch
      return 5 + deltaLoadCycles + 2 * numVectors * batchSize;
    } else {
      // Use broadcast mac
      // 5 cycles to load/store accumulators in outer loop
      // Inner loop requires 2 cycles per mac to load vector and scalar
      // and  convert scalar to 32 bits in the case of halves.
      return 5 + numVectors * (5 + 2 * batchSize);
    }
  }
};

template class FullyConnectedWeightUpdate<float>;
template class FullyConnectedWeightUpdate<half>;

template <typename FPType>
class FullyConnectedBiasUpdate : public Vertex {
public:
  Vector<Input<Vector<FPType>>> d;
  Vector<InOut<FPType>> bias;
  float eta;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    const auto batchSize = d.size();
    const auto numBiases = bias.size();
    assert(d[0].size() == numBiases);
    for (unsigned i = 0; i != numBiases; ++i) {
      float grad = 0;
      for (unsigned b = 0; b < batchSize; ++b) {
        grad += d[b][i];
      }
      bias[i] = bias[i] - grad * eta;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    const auto batchSize = d.size();
    return 5 + bias.size() * (2 + batchSize * 1);

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
template <class FPType, class AccumType, bool isFractional>
class ConvPartialnx1InOut: public SupervisorVertex {
public:
  Vector<Input<Vector<FPType>>> in;
  Vector<Input<Vector<FPType>>> weights;
  Vector<unsigned> weightReuseCount;
  Vector<InOut<Vector<AccumType>>> out;

  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> inChansPerGroup;
  SimOnlyField<unsigned> outChansPerGroup;
  SimOnlyField<unsigned> convUnitCoeffLoadBytesPerCycle;

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
          if (!isFractional) {
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
    const auto filterHeight = in.size() / out.size();
    const auto numContexts = weightReuseCount.size() /
                             (weights.size() / filterHeight);
    const auto numConvUnitsPerTile = outChansPerGroup;
    const auto bitWidth = std::is_same<FPType, float>::value ? 32 : 16;
    assert(dataPathWidth % bitWidth == 0);
    const auto vectorWidth = dataPathWidth / bitWidth;
    assert(inChansPerGroup % vectorWidth == 0);
    const auto convUnitPipelineDepth = inChansPerGroup / vectorWidth;
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
          if (isFractional) {
            if (!in[convNum * filterHeight].empty() ) {
              const auto outWidth = out[convNum].size() / outChansPerGroup;
              const auto inWidth = in[convNum * filterHeight].size() /
                                   inChansPerGroup;
              const auto stride = (outWidth + inWidth - 1) / inWidth;
              assert((outWidth + stride - 1) / stride == inWidth);
              convSize = convSize / stride;
            } else {
              //nothing for this worker thread
              convSize = 0;
            }
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
      convUnitCoeffLoadBytesPerCycle,
      filterHeight
    );
  }
};

template class ConvPartialnx1InOut<float, half, false>;
template class ConvPartialnx1InOut<float, float, false>;
template class ConvPartialnx1InOut<float, half, true>;
template class ConvPartialnx1InOut<float, float, true>;
template class ConvPartialnx1InOut<half, half, false>;
template class ConvPartialnx1InOut<half, float, false>;
template class ConvPartialnx1InOut<half, half, true>;
template class ConvPartialnx1InOut<half, float, true>;

template <class InputType, class PartialTypes>
class ConvWeightGradAop : public Vertex {
public:
  Vector<Input<Vector<InputType>>> acts;
  Vector<Input<Vector<InputType>>> deltas;
  Vector<InOut<Vector<PartialTypes>>> weightDeltas;
  Vector<unsigned> weightReuseCount;

  unsigned inChansPerGroup;
  unsigned outChansPerGroup;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numWeightDeltas = weightDeltas.size();
    assert(weightReuseCount.size() == numWeightDeltas);
    assert(acts.size() == deltas.size());

    unsigned i = 0;
    for (unsigned w = 0; w != numWeightDeltas; ++w) {
      for (unsigned pass = 0; pass != weightReuseCount[w]; ++pass, ++i) {
        assert(i < acts.size());
        assert(acts[i].size() % inChansPerGroup == 0);
        assert(deltas[i].size() % outChansPerGroup == 0);
        const auto actsWidth = acts[i].size() / inChansPerGroup;
        const auto deltasWidth = deltas[i].size() / outChansPerGroup;
        unsigned actsStride;
        if (deltasWidth == 1) {
          assert(actsWidth == 1);
          actsStride = 0;
        } else {
          assert((actsWidth - 1) % (deltasWidth - 1) == 0);
          actsStride = (actsWidth - 1) / (deltasWidth - 1);
        }
        for (unsigned x = 0; x != deltasWidth; ++x) {
          for (unsigned oz = 0; oz != outChansPerGroup; ++oz) {
            for (unsigned iz = 0; iz != inChansPerGroup; ++iz) {
              weightDeltas[w][iz + oz * inChansPerGroup] +=
                  acts[i][x * actsStride * inChansPerGroup + iz] *
                  deltas[i][x * outChansPerGroup + oz];
            }
          }
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool floatInput = std::is_same<InputType, float>::value;
    bool floatPartials = std::is_same<InputType, float>::value;
    unsigned numWeightDeltas = weightDeltas.size();
    unsigned i = 0;
    std::vector<std::vector<unsigned>> shape;
    for (unsigned w = 0; w != numWeightDeltas; ++w) {
      shape.emplace_back();
      for (unsigned pass = 0; pass != weightReuseCount[w]; ++pass, ++i) {
        const auto deltasWidth = deltas[i].size() / outChansPerGroup;
        shape.back().push_back(deltasWidth);
      }
    }
    return
      getWeightGradAopCycles(floatInput, floatPartials, dataPathWidth,
                             inChansPerGroup, outChansPerGroup, shape);
  }
};

template class ConvWeightGradAop<float, float>;
template class ConvWeightGradAop<half, float>;
template class ConvWeightGradAop<half, half>;

template <typename WeightType>
class ConvWeightUpdate : public Vertex {
public:
  Input<Vector<WeightType>> weightDeltas;
  InOut<Vector<WeightType>> weights;

  float eta;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned w = 0; w < weights.size(); ++w) {
      weights[w] -= eta * weightDeltas[w];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numElem = weights.size();
    bool isFloat = std::is_same<WeightType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    // Inner loop uses the axpy instruction.
    return 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
  }
};

template class ConvWeightUpdate<float>;
template class ConvWeightUpdate<half>;


template <typename FPType>
class ConvBiasReduce1: public Vertex {
public:
  Output<Vector<FPType>> out;
  Vector<Input<Vector<FPType>>> in;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    auto numBiases = out.size();
    for (unsigned bias = 0; bias < numBiases; ++bias) {
      float sum = 0;
      for (unsigned d = 0; d < in.size(); d++) {
        assert(in[d].size() % out.size() == 0);
        auto deltasPerBias = in[d].size() / out.size();
        for (unsigned i = 0; i < deltasPerBias; ++i) {
          sum += in[d][i * numBiases + bias];
        }
      }
      out[bias] = sum;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (out.size() + vectorWidth - 1) / vectorWidth;

    uint64_t cycles = 5;
    for (unsigned d = 0; d < in.size(); d++) {
      cycles += 5;
      auto deltasPerBias = in[d].size() / out.size();
      cycles += numVectors * (2 + deltasPerBias);
    }
    return cycles;
  }
};

template class ConvBiasReduce1<float>;
template class ConvBiasReduce1<half>;

template <typename FPType>
class ConvBiasReduce2: public Vertex {
public:
  Vector<Output<FPType>> out;
  Vector<Input<FPType>> in;

  bool compute() {
    assert(in.size() % out.size() == 0);
    auto numBiases = out.size();
    auto deltasPerBias = in.size() / out.size();

    for (unsigned bias = 0; bias < numBiases; ++bias) {
      float sum = 0;
      for (unsigned i = 0; i < deltasPerBias; ++i) {
        sum += in[bias * deltasPerBias + i];
      }
      out[bias] = sum;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    auto numBiases= out.size();
    auto deltasPerBias = in.size() / out.size();
    uint64_t cycles = 10;

    for (unsigned bias = 0; bias < numBiases; ++bias) {
      cycles += deltasPerBias;
    }
    return cycles;
  }
};

template class ConvBiasReduce2<float>;
template class ConvBiasReduce2<half>;

template <typename FPType>
class ConvBiasUpdate: public Vertex {
public:
  InOut<FPType> bias;
  Vector<Input<FPType>> partials; // partial sums of the bias gradient
  float eta;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < partials.size(); ++i) {
      sum += partials[i];
    }
    *bias -= eta * sum;
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 15 + partials.size();
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
    return getBwdNonlinearityDerivativeCycles(deltasIn.size(),
                                              nonLinearityType,
                                              isFloat,
                                              dataPathWidth);
  }
};

template class NonLinearityBwd<float>;
template class NonLinearityBwd<half>;

/**
 * Compute a sum of 1x1 convolutions over a subset of the input channels for
 * multiple output channels.
 **/
template <class FPType, class AccumType>
class ConvPartial1x1Out: public SupervisorVertex {
public:
  Vector<Input<Vector<FPType>>> in;
  Input<Vector<FPType>> weights;
  Vector<Output<Vector<AccumType>>> out;
  Vector<unsigned> weightReuseCount;

  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> inChansPerGroup;
  SimOnlyField<unsigned> outChansPerGroup;
  SimOnlyField<unsigned> convUnitCoeffLoadBytesPerCycle;

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
    const auto numContexts = weightReuseCount.size();
    const auto numConvUnitsPerTile = outChansPerGroup;
    const auto bitWidth = std::is_same<FPType, float>::value ? 32 : 16;
    assert(dataPathWidth % bitWidth == 0);
    const auto vectorWidth = dataPathWidth / bitWidth;
    assert(inChansPerGroup % vectorWidth == 0);
    const auto convUnitPipelineDepth = inChansPerGroup / vectorWidth;
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
      convUnitCoeffLoadBytesPerCycle,
      1
    );
  }
};

template class ConvPartial1x1Out<float, float>;
template class ConvPartial1x1Out<float, half>;
template class ConvPartial1x1Out<half, float>;
template class ConvPartial1x1Out<half, half>;

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

template <typename OutType, typename PartialsType>
class ConvReduce : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType>>> partials;

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
    bool isPartialsFloat = std::is_same<PartialsType, float>::value;
    bool isOutTypeFloat = std::is_same<OutType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isPartialsFloat ? 32 : 16);
    bool conversionCyles = isPartialsFloat != isOutTypeFloat;
    unsigned cycles = 4;
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      auto numVectors = (numElem + vectorWidth - 1) / vectorWidth;
      cycles += 1 + numPartials * (1 + numVectors)
                + conversionCyles * numVectors;
    }
    return cycles;
  }
};

template class ConvReduce<float, float>;
template class ConvReduce<half, float>;
template class ConvReduce<half, half>;

template <class InType, class OutType>
class ConvComplete : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Input<Vector<OutType>>> bias;
  Vector<Output<Vector<OutType>>> out;
  Vector<unsigned> outputChanGroupsPerBias;

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

          out[o][outIndex] = sum;
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
      assert(outChans % chunkSize == 0);
      // load input, load bias and add
      // - dual loads, dual issue = 2 vectors in 2 cycles
      numCycles += (chunkSize + inVectorWidth - 1) / inVectorWidth
                   * (outChans / chunkSize)
                   * outCols;
    }
    return numCycles;
  }
};

template class ConvComplete<float, float>;
template class ConvComplete<float, half>;
template class ConvComplete<half, half>;

// AddTensors
// Sum the input tensors into the output
// \a out and \a in1 must have the same sizes. \a in2 may be smaller than the
// output, missing elements are treated as zero
template <class InType, class OutType>
class AddTensors : public Vertex {
public:
  Input<Vector<InType>> in0;
  Input<Vector<InType>> in1;
  Output<Vector<OutType>> out;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned outSize = out.size();
    unsigned in1Size = in1.size();
    for (unsigned o = 0; o < in1Size; ++o) {
      out[o] = in0[o] + in1[o];
    }
    // elements not present in \a in2
    for (unsigned o = in1Size; o < outSize; ++o) {
        out[o] = in0[o];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    assert(out.size() >= in1.size());
    assert(out.size() == in0.size());
    bool inIsFloat = std::is_same<InType, float>::value;
    bool outIsFloat = std::is_same<InType, float>::value;
    assert(!outIsFloat || inIsFloat && "Output is wider than input");
    const auto inVectorWidth = dataPathWidth / (inIsFloat ? 32 : 16);
    const auto outVectorWidth = dataPathWidth / (outIsFloat ? 32 : 16);
    unsigned numOut = out.size();
    unsigned numIn2 = in1.size();
    // 2*loads + 1*store per operation, memory accesses will dominate,
    // assume common memory element
    unsigned numCycles = 15
      + (numOut + inVectorWidth - 1) / inVectorWidth   //1*load
      + (numOut + outVectorWidth - 1) / outVectorWidth //1*store
      + (numIn2 + inVectorWidth - 1) / inVectorWidth;  //1*load

    return numCycles;
  }
};
template class AddTensors<float, float>;
template class AddTensors<float, half>;
template class AddTensors<half, half>;

// Acumulate
// Accumulate the input tensors with the output, striding through the output
// vector. Only min(inOut channels, in1 channels) are combined
template <class InType, class OutType>
class Accumulate : public Vertex {
public:
  InOut<Vector<OutType>> outIn0;
  Input<Vector<InType>> in1;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned outSize = std::min(outIn0.size(), in1.size());
    for (unsigned o = 0; o < outSize; ++o) {
      outIn0[o] += in1[o];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool inIsFloat = std::is_same<InType, float>::value;
    bool outIsFloat = std::is_same<InType, float>::value;
    assert(outIn0.size() >= in1.size());
    assert(!outIsFloat || inIsFloat && "Output is wider than input");
    const auto inVectorWidth = dataPathWidth / (inIsFloat ? 32 : 16);
    const auto outVectorWidth = dataPathWidth / (outIsFloat ? 32 : 16);
    unsigned numOut = std::min(outIn0.size(), in1.size());
    // 2*loads + 1*store per operation, memory accesses will dominate,
    // assume common memory element
    unsigned numCycles = 15
      + (numOut + inVectorWidth - 1) / inVectorWidth   //1*load
      + (numOut + outVectorWidth - 1) / outVectorWidth //1*store
      + (numOut + inVectorWidth - 1) / inVectorWidth;  //1*load

    return numCycles;
  }
};
template class Accumulate<float, float>;
template class Accumulate<float, half>;
template class Accumulate<half, half>;

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

template <typename FPType, typename LabelType>
class CalcLoss : public Vertex {
public:
  Vector<Input<Vector<FPType>>> batchIn;
  Input<Vector<LabelType>> label;

  Vector<Output<Vector<FPType>>> batchDeltaOut;
  Vector<Output<FPType>> loss;
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

  uint64_t getCycleEstimate() const {
    return 0;
  }
};


template class CalcLoss<float,unsigned int>;
template class CalcLoss<float,int>;
template class CalcLoss<half,unsigned int>;
template class CalcLoss<half,int>;


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


template <class FPType>
class WgdPartials : public SupervisorVertex {

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
  SimOnlyField<unsigned> convUnitCoeffLoadBytesPerCycle;

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

          if (ig == 0) {
            partials[gr][oc] = acc;
          } else {
            partials[gr][oc] += acc;
          }
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const unsigned outChanDepth = partials[0].size();
    const unsigned inpChanDepth = dTf[0].size();
    const unsigned comPencils = partials.size();
    const unsigned numInpGroups = wTf.size();


    return getWgdAccumCycles(
                      numInpGroups,
                      comPencils,
                      inpChanDepth,
                      outChanDepth,
                      numWorkers,
                      numConvUnits,
                      weightsPerConvUnit,
                      convUnitCoeffLoadBytesPerCycle,
                      isFloat);
  }
};

template class WgdPartials<float>;
template class WgdPartials<half>;


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
                               isFloat);
  }
};

template class WgdConvComplete<float>;
template class WgdConvComplete<half>;

template <typename SrcType, typename DstType>
class Cast : public Vertex {
public:
  Input<Vector<SrcType>> src;
  Output<Vector<DstType>> dst;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned i = 0; i < dst.size(); ++i) {
      dst[i] = static_cast<DstType>(src[i]);
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    const auto floatVectorWidth = dataPathWidth / 32;
    return (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
  }
};

template class Cast<float, half>;
template class Cast<half, float>;
template class Cast<float, float>;
template class Cast<half, half>;

template <typename SrcType, typename DstType>
class Cast2D : public Vertex {
public:
  Vector<Input<Vector<SrcType>>> src;
  Vector<Output<Vector<DstType>>> dst;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(src.size() == dst.size());
    for (unsigned i = 0; i != dst.size(); ++i) {
      assert(src[i].size() == dst[i].size());
      for (unsigned j = 0; j != dst[i].size(); ++j) {
        dst[i][j] = static_cast<DstType>(src[i][j]);
      }
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const {
    const auto floatVectorWidth = dataPathWidth / 32;
    std::uint64_t cycles = 5;
    for (unsigned i = 0; i != dst.size(); ++i) {
      // Estimate based on 6 cycles of loop overhead per src / dst pointer pair:
      //
      // 1: load src
      // 2: load dst
      // 3: load length
      // 4: load src[0]
      // 5: { load src[1] ; convert src[0] }
      // 6: repeat
      cycles += 6 + (dst[i].size() + floatVectorWidth - 1) / floatVectorWidth;
    }
    return cycles;
  }
};

template class Cast2D<float, half>;
template class Cast2D<half, float>;
template class Cast2D<float, float>;
template class Cast2D<half, half>;

template <typename T>
class Transpose2D : public Vertex {
public:
  Vector<Input<Vector<T>>> src;
  Vector<Output<Vector<T>>> dst;
  // TODO specialize the vertex based on the value of this field to avoid extra
  // memory usage.
  unsigned numSrcColumns;

  bool compute() {
    assert(src.size() == dst.size());
    const auto numTranspositions = src.size();
    for (unsigned i = 0; i != numTranspositions; ++i) {
      assert(src[i].size() == dst[i].size());
      const auto numElements = src[i].size();
      assert(numElements % numSrcColumns == 0);
      const auto numSrcRows = numElements / numSrcColumns;
      for (unsigned x = 0; x != numSrcColumns; ++x) {
        for (unsigned y = 0; y != numSrcRows; ++y) {
          dst[i][x * numSrcRows + y] = src[i][y * numSrcColumns + x];
        }
      }
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<T, float>::value;
    std::uint64_t cycles = 2 + // Run instruction.
                           6;  // Vertex overhead.
    const auto numTranspositions = src.size();
    for (unsigned i = 0; i != numTranspositions; ++i) {
      const auto numElements = src[i].size();
      cycles += 2; // Load src and dst pointers.
      if (isFloat) {
        cycles += 1; // 1 cycle latency before first value is written to memory.
        cycles += numElements;
      } else {
        // Cycle count taken from transpose16x8 microbenchmark.
        assert(numElements % numSrcColumns == 0);
        const auto numSrcRows = numElements / numSrcColumns;
        const auto middleIterations = (numSrcColumns + 3) / 4;
        const auto innerIterations = (numSrcRows + 1) / 2;
        cycles += 3 + middleIterations * (3 + innerIterations * 6);
      }
    }
    return cycles;
  }
};

template class Transpose2D<float>;
template class Transpose2D<half>;

} // end namespace popnn
