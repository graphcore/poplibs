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
  case popnn::NonLinearityType::SOFTMAX_SCALED:
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
  case popnn::NonLinearityType::SOFTMAX_SCALED:
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
  Input<FPType> deltasScale;
  Input<FPType> modelOutputScaling;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    float eps =
        std::is_same<FPType, float>() ? EPS_LOG_N_FLOAT : EPS_LOG_N_HALF;
    const FPType scale = *deltasScale / *modelOutputScaling;
    const FPType logModelOutputScaling =
                 FPType(log(float(*modelOutputScaling)));
    for (std::size_t i = 0; i < size; i++) {
      FPType expect = expected[i];
      FPType actual = probs[i];
      // Returned deltas are scaled by deltasScale to
      // maintain accuracy (actual is already assumed to be scaled by
      // modelOutputScaling)

      deltas[i] = scale * (actual - expect * (*modelOutputScaling));
      // Returned transformed is adjusted to no longer be scaled
      transformed[i] = -expect * (FPType(log(float(actual) + eps)) -
                       logModelOutputScaling);
    }
    return true;
  }
};

template class LossCrossEntropyTransform<float>;
template class LossCrossEntropyTransform<half>;

// Takes a contiguous set of activations starting
// at the given index, returns the max index and
// value of these.
template <typename InType, typename LabelType>
class ReduceMaxClassGather : public SupervisorVertex {
  constexpr static bool isIntegralIn = std::is_integral<InType>::value;
  using OutType = typename std::conditional<isIntegralIn, InType, float>::type;
public:
  ReduceMaxClassGather();

  Input<Vector<InType, ONE_PTR>> activations;
  const LabelType index;
  Output<Vector<OutType, ONE_PTR>> maxValue;
  Output<Vector<LabelType, ONE_PTR>> maxIndex;
  const unsigned size;
  const unsigned short divisorLog2;

  IS_EXTERNAL_CODELET(!isIntegralIn);
  bool compute() {
    // Work is split between up to N workers based on the divisor
    // and outputs to each maxValue/Index output based on this
    const auto divisor = (1u << divisorLog2);
    const auto nOutputs = (size + divisor - 1) / divisor;
    for (std::size_t i = 0; i < nOutputs; ++i) {
      LabelType maxI = divisor * i;
      InType maxV = activations[maxI];
      const auto end = (maxI + divisor > size) ? size : maxI + divisor;
      for (std::size_t j = maxI + 1; j < end; ++j) {
        if (activations[j] > maxV) {
          maxV = activations[j];
          maxI = j;
        }
      }
      maxValue[i] = OutType(maxV);
      maxIndex[i] = maxI + index;
    }
    return true;
  }
};

template class ReduceMaxClassGather<float, unsigned int>;
template class ReduceMaxClassGather<half, unsigned int>;
template class ReduceMaxClassGather<int, unsigned int>;
template class ReduceMaxClassGather<unsigned int, unsigned int>;

template class ReduceMaxClassGather<float, int>;
template class ReduceMaxClassGather<half, int>;
template class ReduceMaxClassGather<int, int>;
template class ReduceMaxClassGather<unsigned int, int>;

template <typename InOutType, typename LabelType>
class ReduceMaxClassSparse : Vertex {
  constexpr static bool ext = !std::is_integral<InOutType>::value;
public:
  ReduceMaxClassSparse();

  Input<Vector<InOutType>> activations;
  Input<Vector<LabelType, ONE_PTR>> labels;
  Output<InOutType> maxValue;
  Output<LabelType> maxIndex;

  IS_EXTERNAL_CODELET(ext);
  bool compute() {
    LabelType maxI = 0;
    InOutType maxV = activations[0];
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

template class ReduceMaxClassSparse<float, unsigned int>;
template class ReduceMaxClassSparse<float, int>;

template class ReduceMaxClassSparse<unsigned int, unsigned int>;
template class ReduceMaxClassSparse<unsigned int, int>;

template class ReduceMaxClassSparse<int, unsigned int>;
template class ReduceMaxClassSparse<int, int>;

// Takes a contiguous set of activations starting
// at the given index, returns the min index and
// value of these.
template <typename InType, typename LabelType>
class ReduceMinClassGather : public SupervisorVertex {
  constexpr static bool isIntegralIn = std::is_integral<InType>::value;
  using OutType = typename std::conditional<isIntegralIn, InType, float>::type;
public:
  ReduceMinClassGather();

  Input<Vector<InType, ONE_PTR>> activations;
  const LabelType index;
  Output<Vector<OutType, ONE_PTR>> minValue;
  Output<Vector<LabelType, ONE_PTR>> minIndex;
  const unsigned size;
  const unsigned short divisorLog2;

  IS_EXTERNAL_CODELET(!isIntegralIn);
  bool compute() {
    // Work is split between up to N workers based on the divisor
    // and outputs to each minValue/Index output based on this
    const auto divisor = (1u << divisorLog2);
    const auto nOutputs = (size + divisor - 1) / divisor;
    for (std::size_t i = 0; i < nOutputs; ++i) {
      LabelType minI = divisor * i;
      InType minV = activations[minI];
      const auto end = (minI + divisor > size) ? size : minI + divisor;
      for (std::size_t j = minI + 1; j < end; ++j) {
        if (activations[j] < minV) {
          minV = activations[j];
          minI = j;
        }
      }
      minValue[i] = OutType(minV);
      minIndex[i] = minI + index;
    }
    return true;
  }
};

template class ReduceMinClassGather<float, unsigned int>;
template class ReduceMinClassGather<half, unsigned int>;
template class ReduceMinClassGather<int, unsigned int>;
template class ReduceMinClassGather<unsigned int, unsigned int>;

template class ReduceMinClassGather<float, int>;
template class ReduceMinClassGather<half, int>;
template class ReduceMinClassGather<int, int>;
template class ReduceMinClassGather<unsigned int, int>;

template <typename InOutType, typename LabelType>
class ReduceMinClassSparse : Vertex {
  constexpr static bool ext = !std::is_integral<InOutType>::value;
public:
  ReduceMinClassSparse();

  Input<Vector<InOutType>> activations;
  Input<Vector<LabelType, ONE_PTR>> labels;
  Output<InOutType> minValue;
  Output<LabelType> minIndex;

  IS_EXTERNAL_CODELET(ext);
  bool compute() {
    LabelType minI = 0;
    InOutType minV = activations[0];
    for (std::size_t i = 1; i < activations.size(); ++i) {
      if (activations[i] < minV) {
        minV = activations[i];
        minI = i;
      }
    }
    *minValue = minV;
    *minIndex = labels[minI];
    return true;
  }
};

template class ReduceMinClassSparse<float, unsigned int>;
template class ReduceMinClassSparse<float, int>;

template class ReduceMinClassSparse<unsigned int, unsigned int>;
template class ReduceMinClassSparse<unsigned int, int>;

template class ReduceMinClassSparse<int, unsigned int>;
template class ReduceMinClassSparse<int, int>;

namespace {

/*
  MinHeapView takes a reference to container type and treats it like a heap. The
  container is assumed to be allocated with the current size being given by a
  parameter to the push/replace functions. This is designed to be used by the
  topK codelets which want to replace the smallest value at the top so we can
  replace inplace by removing the 0th element (the smallest) and then rotate
  down to repair the tree.
 */
template <typename IndexVectorType, typename DataVectorType, typename DataType>
class MinHeapView {
public:
  MinHeapView(IndexVectorType &vec, DataVectorType &underlayingStorage)
      : partialBucket(vec), resourceVector(underlayingStorage) {}

  // Reference to the underlaying storage which we are treating as a min heap.
  IndexVectorType &partialBucket;

  DataVectorType &resourceVector;

  // Return the parent of the binary heap.
  inline int GetParent(int i) const { return (i - 1) / 2; }

  // Returns true if this value is larger than the smallest node in the heap.
  inline bool IsLargerThanSmallest(DataType newVal) const {
    return newVal > resourceVector[partialBucket[0]];
  }

  void ReplaceValue(DataType value, int ind, const size_t size) {
    int index = ind;
    partialBucket[index] = value;

    if (index == 0)
      return;
    // For the worst log(n) case with early exit.
    std::size_t parentIndex = index;
    do {
      parentIndex = GetParent(index);

      // If we are in the correct position in the tree, exit.
      if (resourceVector[partialBucket[parentIndex]] < resourceVector[value])
        break;

      // Otherwise we should continue rotating up.
      // Swap the values.
      partialBucket[index] = partialBucket[parentIndex];
      partialBucket[parentIndex] = value;
      index = parentIndex;
    } while (parentIndex != 0);
  }

  // Push to the binary heap by rotating up the values.
  inline void Push(DataType value, const size_t size) {
    // Since the array has been preallocated we can push by "replacing" the
    // value at the end of the logical size. Should save on code.
    ReplaceValue(value, size, size + 1);
  }

  // Pop a value from the binary heap.
  DataType Pop(const size_t size) {
    if (size == 0) {
      return partialBucket[0];
    }
    const size_t newSize = size - 1;

    DataType valueToReturn = partialBucket[0];

    // Swap the smallest element at the top for the element at the bottom.
    std::swap(partialBucket[0], partialBucket[newSize]);

    // Repair the tree now we have broken the heap condition.
    RepairTree(newSize);
    return valueToReturn;
  }

  // Replace the smallest value in the heap and then repair the heap.
  void ReplaceAndRepair(DataType value, const size_t size) {
    if (size == 0) {
      partialBucket[0] = value;
      return;
    }

    // Replace the smallest element.
    partialBucket[0] = value;

    // Repair the tree now we have (maybe) broken the heap condition.
    RepairTree(size);
  }

  void RepairTree(const size_t size, const size_t offset = 0) {
    int index = 0;
    // For the worst log(n) case with early exit.
    do {

      std::size_t left = 2 * index + 1;
      std::size_t right = 2 * index + 2;
      bool largerThanLeft = left < size
                                ? resourceVector[partialBucket[index]] >
                                      resourceVector[partialBucket[left]]
                                : false;
      bool largerThanRight = right < size
                                 ? resourceVector[partialBucket[index]] >
                                       resourceVector[partialBucket[right]]
                                 : false;

      // If we are smaller than both our children we are in the right place.
      if (!(largerThanLeft || largerThanRight)) {
        break;
      }

      // Otherwise we should continue rotating down. Swap the right unless we
      // can swap the left and it is smaller than the right.
      if (largerThanRight &&
          !(largerThanLeft && resourceVector[partialBucket[right]] >
                                  resourceVector[partialBucket[left]])) {
        std::swap(partialBucket[index], partialBucket[right]);
        index = right;
      } else if (largerThanLeft) {
        std::swap(partialBucket[index], partialBucket[left]);
        index = left;
      }
    } while (index < size);
  }

    void Sort(size_t size) {
    // Pop each element off. This involves moving the smallest (the top) to the
    // back of the array.
    for (int i = size; i >= 0; --i) {
      this->Pop(i);
    }
  }
};

} // Namespace
/*
  Calcuate the top |numK| values of the given |activations| and store them in
  the tensor |maxValues|. The algorithm works as follows.

  1. Create a Heap of the maxValues. We use a HeapView which treats the values
  as if they are a min heap and operates in place.
  2. Init the Heap by pushing the first |numK| values to it.
  3. For all the values after |numK| we check if they are larger than the
  smallest previously added. If so we add them to the heap and remove the
  smallest at the same time. We just overwrite the smallest then repair the
  tree.
  4. Repeat 3  until we have exhausted the input |activations|.
  5. If the Sort template parameter has been given we use pop to shuffle the
  heap into array sorted order before returning it.
*/
template <typename DataType, bool Sort = false>
class ReduceMaxNClassSparse : Vertex {
public:
  ReduceMaxNClassSparse();
  Input<Vector<DataType>> activations;

  Output<Vector<DataType>> maxValues;

  Output<Vector<unsigned, ONE_PTR>> maxValuesIndices;

  Input<Vector<unsigned, ONE_PTR>> labels;

  const bool shouldSort;

  unsigned numK;
  IS_EXTERNAL_CODELET(false);
  bool compute() {
    // Create an inplace view of the maxValues array as a min heap.
    MinHeapView<decltype(maxValuesIndices), decltype(activations), DataType>
        heapView{maxValuesIndices, activations};

    heapView.Push(0, 0);

    for (std::size_t i = 1; i < activations.size(); ++i) {
      if (i < numK) {
        // Initialize the heap with the first "k" values.
        heapView.Push(i, i);
      } else if (heapView.IsLargerThanSmallest(activations[i])) {
        // Replace the smallest value in the heap with this value then shuffle
        // it to the correct place in the heap.
        heapView.ReplaceAndRepair(i, numK);
      }
    }

    // Sort if template parameter Sort is true and the runtime flag is set. If
    // the runtime flag will never be set (I.E compile time Sort=false) this
    // should be trivially eliminated by DCE.
    if (Sort && shouldSort) {
      heapView.Sort(numK);
    }

    for (int i = 0; i < numK; ++i) {
      maxValues[i] = activations[maxValuesIndices[i]];

      // Assign the max index its actual label. "i" is in the range of 0-size
      // where 0-size are relative to the activation context. A.K.A are a
      // subarray of the actual activations.
      maxValuesIndices[i] = labels[maxValuesIndices[i]];
    }

    return true;
  }
};

// Unsorted.
template class ReduceMaxNClassSparse<float>;
template class ReduceMaxNClassSparse<int>;
template class ReduceMaxNClassSparse<unsigned int>;

// Sorted outputs.
template class ReduceMaxNClassSparse<float, true>;
template class ReduceMaxNClassSparse<int, true>;
template class ReduceMaxNClassSparse<unsigned int, true>;

/*
  See the description of ReduceMaxNClassSparse for the general algorithm for
  calculating the top |numK| from the given |activations|. This version is only
  different in that it works on multiple batches of input at a time.
*/
template <typename FPType, bool Sort = false>
class ReduceMaxNClassGather : public SupervisorVertex {
public:
  ReduceMaxNClassGather();

  Input<Vector<FPType, ONE_PTR>> activations;
  const unsigned index;

  Output<Vector<FPType, ONE_PTR>> maxValues;

  Output<Vector<unsigned, ONE_PTR>> maxValuesIndices;

  unsigned numK;
  const unsigned size;
  const unsigned short divisorLog2;
  const bool shouldSort;
  IS_EXTERNAL_CODELET(false);
  bool compute() {
    // Work is split between up to N workers based on the divisor
    // and outputs to each maxValue/Index output based on this
    const auto divisor = (1u << divisorLog2);
    const auto nOutputs = (size + divisor - 1) / divisor;
    for (std::size_t i = 0; i < nOutputs; ++i) {
      std::size_t offset = divisor * i;
      const auto end = (offset + divisor > size) ? size : offset + divisor;

      // Smallest value in the MaxHeap. Aka the smallest of all the largest
      // numbers.
      FPType smallest = activations[offset];
      int smallestIndex = -1;

      // To avoid having the extra vertex data associated with vector of vectors
      // and vector lists we just have one vector and treat it as a 2D vector.
      int topKIndex = numK * i;
      unsigned *currentPartialBucket = &maxValuesIndices[topKIndex];
      FPType *currentPartialBucketData = &maxValues[topKIndex];

      // Create an inplace view of the maxValues array as a min heap.
      MinHeapView<decltype(currentPartialBucket), decltype(activations), FPType>
          heapView{currentPartialBucket, activations};

      heapView.Push(offset, 0);

      size_t elements_in_heap = 1;
      for (std::size_t j = offset + 1; j < end; ++j) {

        if (elements_in_heap < numK) {
          heapView.Push(j, elements_in_heap);
          elements_in_heap++;
        } else if (heapView.IsLargerThanSmallest(activations[j])) {
          // Replace the smallest value in the heap with this value then shuffle
          // it to the correct place in the heap.
          heapView.ReplaceAndRepair(j, numK);
        }
      }

      // If this gather is uneven I.E the NumOuput*topK is greater than the size
      // we have to adjust the numK so we don't sort the bit at the end as well.
      // How much remains in the last iteration.
      unsigned remainder = end - offset;

      // If there is more numK than remaining elements
      unsigned adjustedNumK = numK > remainder ? remainder : numK;

      // If the remainder is smaller than numK is larger than topK we have to
      // fill the remaining numK slots with the smallest possible value for that
      // type.
      if (adjustedNumK != numK) {
        for (int k = adjustedNumK; k < numK; ++k) {
          currentPartialBucketData[k] = std::numeric_limits<FPType>::lowest();
          currentPartialBucket[k] = std::numeric_limits<unsigned>::max();
        }
      }
      // Sort if template parameter Sort is true and the runtime flag is set. If
      // the runtime flag will never be set (I.E compile time Sort=false) this
      // should be trivially eliminated by DCE.
      if (Sort && shouldSort) {
        heapView.Sort(adjustedNumK);
      }

      for (int k = 0; k < adjustedNumK; ++k) {
        currentPartialBucketData[k] = activations[currentPartialBucket[k]];

        // Add the index base to get the actual index.
        currentPartialBucket[k] += index;
      }
    }
    return true;
  }
};

// Unsorted.
template class ReduceMaxNClassGather<float>;
template class ReduceMaxNClassGather<int>;
template class ReduceMaxNClassGather<unsigned int>;


// Sorted outputs.
template class ReduceMaxNClassGather<float, true>;
template class ReduceMaxNClassGather<int, true>;
template class ReduceMaxNClassGather<unsigned int, true>;

template <typename LabelType> class CalcAccuracy : public Vertex {
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
