#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>
#include "popops/EncodingConstants.hpp"
#include "popnn/Loss.hpp"
#include "popnn/NonLinearity.hpp"
#include "popnn/PoolingDef.hpp"
#include "poplibs_support/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;
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
  case popnn::NonLinearityType::SOFTMAX_STABLE:
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
  case popnn::NonLinearityType::SOFTMAX_STABLE:
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
  NonLinearitySupervisor();

  InOut<Vector<FPType, SCALED_PTR32>> data;
  const unsigned short n;

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
  NonLinearityGradSupervisor();

  Input<Vector<FPType, SCALED_PTR32, 8>> outGrad;
  Input<Vector<FPType, SCALED_PTR32, 8>> out;
  Output<Vector<FPType, SCALED_PTR32, 8>> inGrad;
  const unsigned short n;

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
  NonLinearity2D();

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
  NonLinearityGrad2D();

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
class WORKER_ALIGN MaxPooling : public SupervisorVertex {
  static FPType identity()
  {
    if (std::is_same<FPType, float>{}) {
      return -std::numeric_limits<FPType>::infinity();
    } else {
      // half type has no infinity so use the lowest finite value instead.
      return std::numeric_limits<FPType>::lowest();
    }
  }

  static FPType max(FPType lhs, FPType rhs) {
    return lhs > rhs ? lhs : rhs;
  }

public:
  MaxPooling();

  IS_EXTERNAL_CODELET(true);

  Vector<Output<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> out;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> in;
  // starting position within vector list for each context. The number
  // to process can be found from the difference from previous
  Input<Vector<unsigned short, SCALED_PTR32>> startPos;
  // Base offset for each entry in list
  //  - Kept as a pair with even entry for output and odd entry for input
  Input<Vector<unsigned short, SCALED_PTR32>> offsetBase;
  // Each worklist entry describes work to be done per partial row and
  // contains input, output offsets, and number of elements for each kernal
  // position if it exists.
  // The following associated information is kept
  // - Starting position of the worklist entry for a context. There are always
  //   as many entries as the number of contexts. This also gives the number to
  //   worklist entries to be processed by each context.
  // - Base for input begin offset for each worklist entry. The actual offset is
  //   this + the entry in the worklist
  // - Base for output begin offset for each worklist entry. The actual offset
  //   is this + the entry in the worklist
  // The base for the offsets contains alternating values for output and input
  // respectively (i.e. output offset bases are at even and input offset bases
  // are at odd positions
  Input<VectorList<unsigned short, DELTAN>> workList;
  const unsigned short initInfo;
  const unsigned short numChanGroupsM1;
  // the following are scaled by the amount of FPType we can fit into 64-bits.
  const unsigned short chansPerGroupD;
  const unsigned inStrideD;
  const unsigned outStrideD;

  bool compute() {
    const auto scaleFactor = std::is_same<FPType, half>::value ? 4 : 2;
    const auto numChanGroups = numChanGroupsM1 + 1;
    const auto chansPerGroup = chansPerGroupD * scaleFactor;
    const auto inStride = inStrideD * scaleFactor;
    const auto outStride = outStrideD * scaleFactor;

    // initialise output
    for (unsigned cg = 0; cg != numChanGroups; ++cg) {
      for (unsigned i = 0; i != initInfo * chansPerGroup; ++i) {
        out[cg][i] = identity();
      }
    }

    // do pooling operation
    for (unsigned ctxtM1 = 0; ctxtM1 != NUM_WORKERS; ++ctxtM1) {
      const unsigned numRows =
          ctxtM1 == 0 ? startPos[0] : startPos[ctxtM1] - startPos[ctxtM1 - 1];
      const unsigned sPos =
          ctxtM1 == 0 ? 0 : startPos[ctxtM1 - 1];

      // the first 4 loops are completely independent from each other so order
      // them in such a way that the more intermediate work required by a step
      // is the outer most loop.
      for (unsigned row = 0; row != numRows; ++row) {
        const auto pos = sPos + row;
        const auto outOffsetBase = offsetBase[2 * pos];
        const auto inOffsetBase = offsetBase[2 * pos + 1];
        const auto numWorkItems = workList[pos].size();
        // there are always three items for each work vector
        for (unsigned w = 0; w != numWorkItems; w += 3) {
          const auto outBeginOffset = workList[pos][w + 0] + outOffsetBase;
          const auto inBeginOffset = workList[pos][w + 1] + inOffsetBase;
          const auto numElements = workList[pos][w + 2] + 1;
          for (unsigned cg = 0; cg != numChanGroups; ++cg) {
            const auto in_ = in[cg];
            auto out_ = out[cg];
            for (unsigned c = 0; c != chansPerGroup/2; ++c) {
              unsigned outPos = (chansPerGroup * outBeginOffset) + c*2;
              unsigned inPos = (chansPerGroup * inBeginOffset) + c*2;
              for (unsigned f = 0; f != numElements; ++f) {
                out_[outPos] = max(out_[outPos], in_[inPos]);
                out_[outPos+1] = max(out_[outPos+1], in_[inPos+1]);

                outPos += outStride;
                inPos += inStride;
              }
            }
          }
        }
      }
    }
    return true;
  }
};

template class MaxPooling<float>;
template class MaxPooling<half>;

template <typename FPType>
class WORKER_ALIGN MaxPoolingGradientScale : public SupervisorVertex {
public:
  MaxPoolingGradientScale();

  Vector<Output<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> out;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> fwdActsOut;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> in;
  // starting position within vector list for each context. The number
  // to process can be found from the difference from previous
  Input<Vector<unsigned short, SCALED_PTR32>> startPos;
  // Base offset for each entry in list
  //  - Kept as a pair with even entry for output and odd entry for input
  Input<Vector<unsigned short, SCALED_PTR32>> offsetBase;
  // Each worklist entry describes work to be done per partial row and
  // contains input, output offsets, and number of elements for each kernal
  // position if it exists.
  // The following associated information is kept
  // - Starting position of the worklist entry for a context. There are always
  //   as many entries as the number of contexts. This also gives the number to
  //   worklist entries to be processed by each context.
  // - Base for input begin offset for each worklist entry. The actual offset is
  //   this + the entry in the worklist
  // - Base for output begin offset for each worklist entry. The actual offset
  //   is this + the entry in the worklist
  // The base for the offsets contains alternating values for output and input
  // respectively (i.e. output offset bases are at even and input offset bases
  // are at odd positions
  Input<VectorList<unsigned short, DELTAN>> workList;
  const unsigned short initInfo;
  const unsigned short numChanGroupsM1;
  // the following are scaled by the amount of FPType we can fit into 64-bits.
  const unsigned short chansPerGroupD;
  const unsigned inStrideD;
  const unsigned outStrideD;

  bool compute() {
    const auto scaleFactor = std::is_same<FPType, half>::value ? 4 : 2;
    const auto numChanGroups = numChanGroupsM1 + 1;
    const auto chansPerGroup = chansPerGroupD * scaleFactor;
    const auto inStride = inStrideD * scaleFactor;
    const auto outStride = outStrideD * scaleFactor;

    // initialise output
    for (unsigned cg = 0; cg != numChanGroups; ++cg) {
      for (unsigned i = 0; i != initInfo * chansPerGroup; ++i) {
        out[cg][i] = 0;
      }
    }

    // do pooling operation
    for (unsigned ctxtM1 = 0; ctxtM1 != NUM_WORKERS; ++ctxtM1) {
      const unsigned numRows =
          ctxtM1 == 0 ? startPos[0] : startPos[ctxtM1] - startPos[ctxtM1 - 1];
      const unsigned sPos =
          ctxtM1 == 0 ? 0 : startPos[ctxtM1 - 1];

      // the first 4 loops are completely independent from each other so order
      // them in such a way that the more intermediate work required by a step
      // is the outer most loop.
      for (unsigned row = 0; row != numRows; ++row) {
        const auto pos = sPos + row;
        const auto outOffsetBase = offsetBase[2 * pos];
        const auto inOffsetBase = offsetBase[2 * pos + 1];
        const auto numWorkItems = workList[pos].size();
        // there are always three items for each work vector
        for (unsigned w = 0; w != numWorkItems; w += 3) {
          const auto outBeginOffset = workList[pos][w + 0] + outOffsetBase;
          const auto inBeginOffset = workList[pos][w + 1] + inOffsetBase;
          const auto numElements = workList[pos][w + 2] + 1;
          for (unsigned cg = 0; cg != numChanGroups; ++cg) {
            const auto in_ = in[cg];
            auto out_ = out[cg];
            auto fwdOut = fwdActsOut[cg];
            for (unsigned c = 0; c != chansPerGroup/2; ++c) {
              unsigned outPos = (chansPerGroup * outBeginOffset) + c*2;
              unsigned inPos = (chansPerGroup * inBeginOffset) + c*2;
              for (unsigned f = 0; f != numElements; ++f) {
                out_[outPos] += fwdOut[outPos] == in_[inPos];
                out_[outPos+1] += fwdOut[outPos+1] == in_[inPos+1];
                outPos += outStride;
                inPos += inStride;
              }
            }
          }
        }
      }
    }
    // Compute scale
    for (unsigned cg = 0; cg != numChanGroups; ++cg) {
      for (unsigned i = 0; i != initInfo * chansPerGroup; ++i) {
        if (out[cg][i])
          out[cg][i] = 1.0f / static_cast<float>(out[cg][i]);
      }
    }
    return true;
  }
};
template class MaxPoolingGradientScale<float>;
template class MaxPoolingGradientScale<half>;

template <typename FPType>
class WORKER_ALIGN SumPooling : public SupervisorVertex {
  IS_EXTERNAL_CODELET(true);

  Vector<Output<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> out;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> in;
  // starting position within vector list for each context. The number
  // to process can be found from the difference from previous
  Input<Vector<unsigned short, SCALED_PTR32>> startPos;
  // Base offset for each entry in list
  //  - Kept as a pair with even entry for output and odd entry for input
  Input<Vector<unsigned short, SCALED_PTR32>> offsetBase;
  Input<VectorList<unsigned short, DELTAN>> workList;

public:
  SumPooling();

  const unsigned short initInfo;
  const unsigned short numChanGroupsM1;
  // the following are scaled by the amount of FPType we can fit into 64-bits.
  const unsigned short chansPerGroupD;
  const unsigned inStrideD;
  const unsigned outStrideD;
  const FPType scale;


  bool compute() {
    const auto scaleFactor = std::is_same<FPType, half>::value ? 4 : 2;
    const auto numChanGroups = numChanGroupsM1 + 1;
    const auto chansPerGroup = chansPerGroupD * scaleFactor;
    const auto inStride = inStrideD * scaleFactor;
    const auto outStride = outStrideD * scaleFactor;

    // initialise output
    for (unsigned cg = 0; cg != numChanGroups; ++cg) {
      for (unsigned i = 0; i != initInfo * chansPerGroup; ++i) {
        out[cg][i] = 0;
      }
    }

    // do pooling operation
    for (unsigned ctxtM1 = 0; ctxtM1 != NUM_WORKERS; ++ctxtM1) {
      const unsigned numRows =
          ctxtM1 == 0 ? startPos[0] : startPos[ctxtM1] - startPos[ctxtM1 - 1];
      const unsigned sPos =
          ctxtM1 == 0 ? 0 : startPos[ctxtM1 - 1];

      for (unsigned row = 0; row != numRows; ++row) {
        const auto pos = sPos + row;
        const auto outOffsetBase = offsetBase[2 * pos] * chansPerGroup;
        const auto inOffsetBase = offsetBase[2 * pos + 1] * chansPerGroup;
        const auto numWorkItems = workList[pos].size();
        // there are always three items for each work vector
        for (unsigned w = 0; w != numWorkItems; w += 3) {
          const auto outBeginOffset =
            workList[pos][w + 0] * chansPerGroup + outOffsetBase;
          const auto inBeginOffset =
            workList[pos][w + 1] * chansPerGroup + inOffsetBase;
          const auto numElements = workList[pos][w + 2] + 1;
          for (unsigned cg = 0; cg != numChanGroups; ++cg) {
            const auto in_ = in[cg];
            auto out_ = out[cg];
            for (unsigned c = 0; c != chansPerGroup; ++c) {
              unsigned outPos = outBeginOffset + c;
              unsigned inPos = inBeginOffset + c;
              for (unsigned f = 0; f != numElements; ++f) {
                out_[outPos] += scale * in_[inPos];

                outPos += outStride;
                inPos += inStride;
              }
            }
          }
        }
      }
    }
    return true;
  }
};

template class SumPooling<float>;
template class SumPooling<half>;

template <typename FPType>
class SelectiveScaling : public SupervisorVertex {
public:
  SelectiveScaling();

  IS_EXTERNAL_CODELET(false);
  Input<VectorList<unsigned short, DELTAN>> scaleWorklist;
  Vector<InOut<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> inOut;
  const unsigned short numChanGroups;
  const unsigned short chansPerGroup;

  bool compute() {
    // Scale output
    for (unsigned ctxt = 0; ctxt != NUM_WORKERS; ++ctxt) {
      for (unsigned w = 0; w != scaleWorklist[ctxt].size(); w += 3) {
        for (unsigned f = 0; f !=  scaleWorklist[ctxt][w + 1]; ++f) {
          for (unsigned cg = 0; cg != numChanGroups; ++cg) {
            for (unsigned c = 0; c != chansPerGroup; ++c) {
              unsigned outPos =
                  (f + scaleWorklist[ctxt][w]) * chansPerGroup + c;
              FPType scale = static_cast<FPType>(scaleWorklist[ctxt][w + 2]);
              inOut[cg][outPos] /= scale;
            }
          }
        }
      }
    }
    return true;
  }
};

template class SelectiveScaling<float>;
template class SelectiveScaling<half>;


template <typename FPType>
class MaxPoolingGrad : public SupervisorVertex {
public:
  MaxPoolingGrad();

  IS_EXTERNAL_CODELET(true);

  Vector<Output<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> out;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> in;
  // starting position within vector list for each context. The number
  // to process can be found from the difference from previous
  Input<Vector<unsigned short, SCALED_PTR32>> startPos;
  // Base offset for each entry in list
  //  - Kept as a pair with even entry for output and odd entry for input
  Input<Vector<unsigned short, SCALED_PTR32>> offsetBase;
  Input<VectorList<unsigned short, DELTAN>> workList;
  const unsigned short initInfo;
  const unsigned short numChanGroupsM1;
  // the following are scaled by the amount of FPType we can fit into 64-bits.
  const unsigned short chansPerGroupD;
  const unsigned inStrideD;
  const unsigned outStrideD;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> fwdActsIn;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> fwdActsOut;

  bool compute() {
    const auto scaleFactor = std::is_same<FPType, half>::value ? 4 : 2;
    const auto numChanGroups = numChanGroupsM1 + 1;
    const auto chansPerGroup = chansPerGroupD * scaleFactor;
    const auto inStride = inStrideD * scaleFactor;
    const auto outStride = outStrideD * scaleFactor;

    // initialise output
    for (unsigned cg = 0; cg != numChanGroups; ++cg) {
      for (unsigned i = 0; i != initInfo * chansPerGroup; ++i) {
        out[cg][i] = 0;
      }
    }

    // do pooling operation
    for (unsigned ctxtM1 = 0; ctxtM1 != NUM_WORKERS; ++ctxtM1) {
      const unsigned numRows =
          ctxtM1 == 0 ? startPos[0] : startPos[ctxtM1] - startPos[ctxtM1 - 1];
      const unsigned sPos =
          ctxtM1 == 0 ? 0 : startPos[ctxtM1 - 1];

      for (unsigned row = 0; row != numRows; ++row) {
        const auto pos = sPos + row;
        const auto outOffsetBase = offsetBase[2 * pos] * chansPerGroup;
        const auto inOffsetBase = offsetBase[2 * pos + 1] * chansPerGroup;
        const auto numWorkItems = workList[pos].size();
        // there are always three items for each work vector
        for (unsigned w = 0; w != numWorkItems; w += 3) {
          const auto outBeginOffset =
            workList[pos][w + 0] * chansPerGroup + outOffsetBase;
          const auto inBeginOffset =
            workList[pos][w + 1] * chansPerGroup + inOffsetBase;
          const auto numElements = workList[pos][w + 2] + 1;
          for (unsigned cg = 0; cg != numChanGroups; ++cg) {
            const auto fwdActsIn_ = fwdActsIn[cg];
            const auto fwdActsOut_ = fwdActsOut[cg];
            const auto in_ = in[cg];
            auto out_ = out[cg];
            for (unsigned c = 0; c != chansPerGroup; ++c) {
              unsigned outPos = outBeginOffset + c;
              unsigned inPos = inBeginOffset + c;
              for (unsigned f = 0; f != numElements; ++f) {
                if (fwdActsIn_[outPos] == fwdActsOut_[inPos]) {
                  out_[outPos] += in_[inPos];
                }

                outPos += outStride;
                inPos += inStride;
              }
            }
          }
        }
      }
    }
    return true;
  }
};

template class MaxPoolingGrad<float>;
template class MaxPoolingGrad<half>;

template <typename FPType>
class LossSumSquaredTransform : public Vertex {
public:
  LossSumSquaredTransform();

  Input<Vector<FPType, SCALED_PTR32, 4>> probs;
  Input<Vector<FPType, SCALED_PTR32, 4>> expected;
  Output<Vector<FPType, SCALED_PTR32, 4>> deltas;
  Output<Vector<FPType, SCALED_PTR32, 4>> transformed;
  const unsigned short size;

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
class LossCrossEntropyTransform : public Vertex {
public:
  LossCrossEntropyTransform();

  Input<Vector<FPType, SCALED_PTR32, 4>> probs;
  Input<Vector<FPType, SCALED_PTR32, 4>> expected;
  Output<Vector<FPType, SCALED_PTR32, 4>> deltas;
  Output<Vector<FPType, SCALED_PTR32, 4>> transformed;
  const unsigned short size;

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

template class LossCrossEntropyTransform<float>;
template class LossCrossEntropyTransform<half>;

// Takes a contiguous set of activations starting
// at the given index, returns the max index and
// value of these.
template <typename FPType, typename LabelType>
class ReduceMaxClassGather : public SupervisorVertex {
public:
  ReduceMaxClassGather();

  Input<Vector<FPType, ONE_PTR>> activations;
  const LabelType index;
  Output<Vector<float, ONE_PTR>> maxValue;
  Output<Vector<LabelType, ONE_PTR>> maxIndex;
  const unsigned size;
  const unsigned short divisorLog2;

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
  ReduceMaxClassSparse();

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
  CalcAccuracy();

  Input<Vector<LabelType>> maxPerBatch;
  Input<Vector<LabelType, ONE_PTR>> expected;
  InOut<unsigned> numCorrect;

  bool compute() {
    auto count = *numCorrect;
    for (std::size_t i = 0; i < maxPerBatch.size(); ++i) {
      if (expected[i] != MASKED_LABEL_CODE) {
        count += (maxPerBatch[i] == expected[i]);
      }
    }
    *numCorrect = count;
    return true;
  }
};

template class CalcAccuracy<unsigned int>;
template class CalcAccuracy<int>;

} // end namespace popnn
