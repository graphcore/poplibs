#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>
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

  SimOnlyField<unsigned> dataPathWidth;

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
    return 20 + getDenseDotProductCycles(isFloat, activationIn.size(),
                                         dataPathWidth);
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
  Output<FPType> zOut;
  Output<FPType> activationOut;

  SimOnlyField<unsigned> dataPathWidth;

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
    const auto floatVectorWidth = dataPathWidth / 32;
    return (partials.size() + floatVectorWidth - 1) / floatVectorWidth + 15;
  }
};

template class FullyConnectedReduce<float>;
template class FullyConnectedReduce<half>;

/**
 * Compute 1x1 convolutions and accumulate them with partial sums in memory.
 **/
template <class Base, class AccumType>
class ConvPartial1x1InOut: public Base {
public:
  Vector<Input<Vector<half>>> in;
  Vector<Input<Vector<half>>> weights;
  Vector<unsigned> weightReuseCount;
  Vector<InOut<Vector<AccumType>>> out;

  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> inChansPerGroup;
  SimOnlyField<unsigned> outChansPerGroup;

  bool compute() {
    assert(out.size() > 0);
    assert(out.size() == in.size());
    assert(weightReuseCount.size() % weights.size() == 0);
    const auto numContexts = weightReuseCount.size() / weights.size();
    unsigned convNum = 0;
    for (unsigned w = 0; w != weights.size(); ++w) {
      for (unsigned c = 0; c != numContexts; ++c) {
        for (unsigned i = 0; i != weightReuseCount[w * numContexts + c]; ++i) {
          const auto outWidth = out[convNum].size() / outChansPerGroup;
          const auto inWidth = in[convNum].size() / inChansPerGroup;
          const auto stride = (inWidth + outWidth - 1) / outWidth;
          assert((inWidth + stride - 1) / stride == outWidth);
          for (unsigned x = 0; x != outWidth; ++x) {
            for (unsigned inChanIndex = 0; inChanIndex != inChansPerGroup;
                 ++inChanIndex) {
              for (unsigned outChanIndex = 0; outChanIndex != outChansPerGroup;
                   ++outChanIndex) {
                const auto outIndex = outChanIndex + outChansPerGroup * x;
                const auto weightIndex =
                    inChanIndex + inChansPerGroup * outChanIndex;
                const auto inIndex = inChanIndex + inChansPerGroup * x * stride;
                out[convNum][outIndex] += weights[w][weightIndex] *
                                          in[convNum][inIndex];
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
    const auto numContexts = weightReuseCount.size() / weights.size();
    const auto numConvUnitsPerTile = outChansPerGroup;
    assert(dataPathWidth % 16 == 0);
    const auto halfVectorWidth = dataPathWidth / 16;
    assert(inChansPerGroup % halfVectorWidth == 0);
    const auto convUnitPipelineDepth = inChansPerGroup / halfVectorWidth;
    if (isSupervisorVertex) {
      std::vector<std::vector<std::vector<unsigned>>>
          convolutionsByWeightAndWorker;
      unsigned convNum = 0;
      for (unsigned w = 0; w != weights.size(); ++w) {
        convolutionsByWeightAndWorker.emplace_back();
        auto &convolutionsByWeight = convolutionsByWeightAndWorker.back();
        for (unsigned c = 0; c != numContexts; ++c) {
          convolutionsByWeight.emplace_back();
          for (unsigned i = 0; i != weightReuseCount[w * numContexts + c];
               ++i) {
            convolutionsByWeight.back().push_back(out[convNum].size() /
                                                  outChansPerGroup);
            ++convNum;
          }
        }
      }
      assert(convNum == out.size());
      return getConvPartial1x1SupervisorCycleEstimate(
        convolutionsByWeightAndWorker,
        convUnitPipelineDepth,
        numConvUnitsPerTile
      );
    }
    assert(numContexts == 1);
    std::vector<std::vector<unsigned>> convolutionsByWeight(1);
    unsigned convNum = 0;
    for (unsigned w = 0; w != weights.size(); ++w) {
      convolutionsByWeight.emplace_back();
      for (unsigned i = 0; i != weightReuseCount[w]; ++i) {
        convolutionsByWeight.back().push_back(out[convNum].size() /
                                              outChansPerGroup);
        ++convNum;
      }
    }
    assert(convNum == out.size());
    return getConvPartial1x1CycleEstimate(convolutionsByWeight,
                                          convUnitPipelineDepth,
                                          numConvUnitsPerTile);
  }
};

template class ConvPartial1x1InOut<Vertex, half>;
template class ConvPartial1x1InOut<Vertex, float>;
template class ConvPartial1x1InOut<SupervisorVertex, half>;
template class ConvPartial1x1InOut<SupervisorVertex, float>;


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

template <typename FPType>
class ConvPartialWeightUpdate : public Vertex {
public:
  Input<FPType> d;
  Output<Vector<FPType>> weightUpdates;
  Vector<Input<FPType>> in;

  bool compute() {
    for (unsigned i = 0; i < weightUpdates.size(); ++i) {
      weightUpdates[i] = *d * in[i];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO
    return 0;
  }
};

template class ConvPartialWeightUpdate<float>;
template class ConvPartialWeightUpdate<half>;

template <typename FPType>
class ConvWeightUpdateReduce: public Vertex {
public:
  InOut<FPType> weight;
  Vector<Input<FPType>> partials;
  float eta;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < partials.size(); ++i) {
      sum += partials[i];
    }
    *weight = *weight - sum * eta;
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO
    return 0;
  }
};

template class ConvWeightUpdateReduce<float>;
template class ConvWeightUpdateReduce<half>;

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
    *bias -= sum * eta;
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO
    return 0;
  }
};

template class ConvBiasUpdate<float>;
template class ConvBiasUpdate<half>;

template <typename FPType>
class NonLinearityBwd : public Vertex {
public:
  Input<Vector<FPType>> deltasIn;
  Input<Vector<FPType>> z;
  Output<Vector<FPType>> deltasOut;
  NonLinearityType nonLinearityType;

  bool compute() {
    assert(deltasIn.size() == deltasOut.size());
    assert(deltasIn.size() == z.size());
    for (unsigned i = 0; i < deltasIn.size(); ++i) {
      deltasOut[i] = deltasIn[i] * nonlinearity_derivative(nonLinearityType,
                                                           z[i]);
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO
    return 0;
  }
};

template class NonLinearityBwd<float>;
template class NonLinearityBwd<half>;

template <typename FPType>
class FullyConnectedBwd : public Vertex {
public:
  Input<Vector<FPType>> in;
  Vector<Input<FPType>> weights;
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
    // TODO
    return 0;
  }
};

template class FullyConnectedBwd<float>;
template class FullyConnectedBwd<half>;

template <typename FPType>
class FullyConnectedWeightUpdate : public Vertex {
public:
  Input<FPType> d;
  InOut<Vector<FPType>> weights;
  Input<Vector<FPType>> in;
  InOut<FPType> bias;
  float eta;

  bool compute() {
    for (unsigned i = 0; i < weights.size(); ++i) {
      auto grad = *d * in[i];
      weights[i] = weights[i] - grad * eta;
    }
    *bias = *bias - *d * eta;
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO
    return 0;
  }
};

template class FullyConnectedWeightUpdate<float>;
template class FullyConnectedWeightUpdate<half>;

/**
 * Compute a sum of 1x1 convolutions over a subset of the input channels for
 * multiple output channels.
 **/
template <class Base, class AccumType>
class ConvPartial1x1Out: public Base {
public:
  Vector<Input<Vector<half>>> in;
  Input<Vector<half>> weights;
  Vector<Output<Vector<AccumType>>> out;

  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> inChansPerGroup;
  SimOnlyField<unsigned> outChansPerGroup;

  bool compute() {
    assert(out[0].size() % outChansPerGroup == 0);
    const auto outWidth = out[0].size() / outChansPerGroup;
    const auto outHeight = out.size();
    assert(in[0].size() % inChansPerGroup == 0);
    const auto inWidth = in[0].size() / inChansPerGroup;
    const auto stride = (inWidth + outWidth - 1) / outWidth;
    assert((inWidth + stride - 1) / stride == outWidth);

    assert(in.size() % outHeight == 0);
    unsigned numInChanGroups = in.size() / outHeight;

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
                                  in[y][inIndex];
            }
          }
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    assert(out[0].size() % outChansPerGroup == 0);
    const auto outWidth = out[0].size() / outChansPerGroup;
    const auto outHeight = out.size();
    assert(in[0].size() % inChansPerGroup == 0);
    const auto inWidth = in[0].size() / inChansPerGroup;
    const auto stride = (inWidth + outWidth - 1) / outWidth;
    assert((inWidth + stride - 1) / stride == outWidth);

    assert(in.size() % outHeight == 0);
    unsigned numInChanGroups = in.size() / outHeight;

    bool isSupervisorVertex = std::is_same<Base, SupervisorVertex>::value;

    const auto numConvUnitsPerTile = outChansPerGroup;
    assert(dataPathWidth % 16 == 0);
    const auto halfVectorWidth = dataPathWidth / 16;
    assert(inChansPerGroup % halfVectorWidth == 0);
    const auto convUnitPipelineDepth = inChansPerGroup / halfVectorWidth;
    return getConvPartial1x1CycleEstimate(1 /*kernelWidth*/,
                                          numInChanGroups,
                                          outHeight,
                                          outWidth,
                                          convUnitPipelineDepth,
                                          numConvUnitsPerTile,
                                          isSupervisorVertex);
  }
};

template class ConvPartial1x1Out<Vertex, float>;
template class ConvPartial1x1Out<Vertex, half>;
template class ConvPartial1x1Out<SupervisorVertex, float>;
template class ConvPartial1x1Out<SupervisorVertex, half>;

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
    const auto outChansPerGroup = 1;
    return getConvPartialByDotProductCycleEstimate(isFloat, inChansPerGroup,
                                                   kernelSize, numInRows, 1,
                                                   outputWidth, 1,
                                                   dataPathWidth);
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
  Vector<Output<Vector<OutType>>> z;
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
          z[o][outIndex] = sum;
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
          if (i == 0 || activationOut[chan] > in)
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
  Input<FPType> actOut;
  Input<FPType> actIn;
  Input<FPType> errIn;
  Output<FPType> errOut;

  bool compute() {
    if (*actIn == *actOut) {
      *errOut = *errIn;
    } else {
      *errOut = 0;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO
    return 0;
  }
};

template class MaxPoolingBwd<float>;
template class MaxPoolingBwd<half>;

template <typename FPType>
class CalcLoss : public Vertex {
public:
  Input<Vector<FPType>> zIn;
  NonLinearityType nonLinearityType;
  Input<unsigned> label;
  Input<LossType> lossType;
  Output<Vector<FPType>> deltaOut;
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
        deltaOut[i] = (actual - expected);
        sum += 0.5 * (actual - expected) *  (actual - expected);
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
