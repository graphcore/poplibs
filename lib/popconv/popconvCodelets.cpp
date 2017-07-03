#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popconv {

/**
 * Compute nx1 convolutions and accumulate them with partial sums in memory.
 **/
template <class FPType, class AccumType, bool inOut, bool useDeltasForEdges>
class ConvPartialnx1: public SupervisorVertex {
public:
  Vector<Input<Vector<FPType>>> in;
  Vector<Input<Vector<FPType>>> weights;
  Vector<unsigned> weightReuseCount;
  Vector<typename std::conditional<inOut,
                                   InOut<Vector<AccumType>>,
                                   Output<Vector<AccumType>>>::type> out;
  Vector<bool> zeroOut;
  unsigned inStride;
  unsigned outStride;

  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> inChansPerGroup;
  SimOnlyField<unsigned> outChansPerGroup;
  SimOnlyField<unsigned> convUnitCoeffLoadBytesPerCycle;

  bool compute() {
    assert(out.size() > 0);
    assert(inOut || zeroOut.size() == out.size());
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
          assert((outWidth - 1) * inStride == (inWidth - 1) * outStride);
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
                  if (!inOut && fy == 0 && inChanIndex == 0 &&
                      zeroOut[convNum]) {
                    out[convNum][outIndex] = 0.0;
                  }
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
          auto outWidth = out[convNum].size() / outChansPerGroup;
          const auto numOutputs = (outWidth + outStride - 1) / outStride;
          convolutionsByWeight.back().push_back(numOutputs);
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
      filterHeight,
      useDeltasForEdges
    );
  }
};

template class ConvPartialnx1<float, float, true, true>;
template class ConvPartialnx1<half, half, true, true>;
template class ConvPartialnx1<half, float, true, true>;
template class ConvPartialnx1<float, float, true, false>;
template class ConvPartialnx1<half, half, true, false>;
template class ConvPartialnx1<half, float, true, false>;
template class ConvPartialnx1<float, float, false, true>;
template class ConvPartialnx1<half, half, false, true>;
template class ConvPartialnx1<half, float, false, true>;
template class ConvPartialnx1<float, float, false, false>;
template class ConvPartialnx1<half, half, false, false>;
template class ConvPartialnx1<half, float, false, false>;

template <class InputType, class PartialTypes, bool useDeltasForEdges>
class ConvWeightGradAop : public Vertex {
public:
  Vector<Input<Vector<InputType>>> acts;
  Vector<Input<Vector<InputType>>> deltas;
  Vector<InOut<Vector<PartialTypes>>> weightDeltas;
  Vector<unsigned> weightReuseCount;

  unsigned inChansPerGroup;
  unsigned outChansPerGroup;

  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> numAopAccumulators;

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
    const unsigned vectorWidth = dataPathWidth / (floatInput ? 32 : 16);
    assert(numAopAccumulators % vectorWidth == 0);
    assert(numAopAccumulators / vectorWidth <= vectorWidth);
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
                             inChansPerGroup, outChansPerGroup, shape,
                             useDeltasForEdges, numAopAccumulators);
  }
};

template class ConvWeightGradAop<float, float, false>;
template class ConvWeightGradAop<half, float, false>;
template class ConvWeightGradAop<half, half, false>;
template class ConvWeightGradAop<float, float, true>;
template class ConvWeightGradAop<half, float, true>;
template class ConvWeightGradAop<half, half, true>;

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
  Vector<unsigned> numInputsPerBias;

  bool compute() {
    auto numBiases = out.size();
    unsigned inIndex = 0;
    for (unsigned bias = 0; bias < numBiases; ++bias) {
      float sum = 0;
      for (unsigned i = 0; i < numInputsPerBias[bias]; ++i) {
        sum += in[inIndex++];
      }
      out[bias] = sum;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    auto numBiases= out.size();
    uint64_t cycles = 10;

    for (unsigned bias = 0; bias < numBiases; ++bias) {
      cycles += numInputsPerBias[bias];
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

/**
 * Compute a sum of 1x1 convolutions over a subset of the input channels for
 * multiple output channels.
 **/
template <class FPType, class AccumType, bool useDeltasForEdges>
class ConvPartial1x1Out: public SupervisorVertex {
public:
  Vector<Input<Vector<FPType>>> in;
  Input<Vector<FPType>> weights;
  Vector<Output<Vector<AccumType>>> out;
  Vector<unsigned> weightReuseCount;

  SimOnlyField<unsigned> inChansPerGroup;
  SimOnlyField<unsigned> outChansPerGroup;
  SimOnlyField<unsigned> dataPathWidth;
  SimOnlyField<unsigned> weightsPerConvUnit;
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
    assert(weightsPerConvUnit % vectorWidth == 0);
    const auto convUnitPipelineDepth = weightsPerConvUnit / vectorWidth;
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
      1,
      useDeltasForEdges
    );
  }
};

template class ConvPartial1x1Out<float, float, true>;
template class ConvPartial1x1Out<float, half, true>;
template class ConvPartial1x1Out<half, float, true>;
template class ConvPartial1x1Out<half, half, true>;
template class ConvPartial1x1Out<float, float, false>;
template class ConvPartial1x1Out<float, half, false>;
template class ConvPartial1x1Out<half, float, false>;
template class ConvPartial1x1Out<half, half, false>;

/* Perform a series of 1x1 convolutions using the MAC instruction were the
 * axis of accumulation is across the vector. */
template <typename InType, typename AccumType>
class ConvPartialHorizontalMac: public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Input<Vector<InType>>> weights;
  Vector<InOut<Vector<AccumType>>> out;

  unsigned inStride;
  unsigned outStride;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numInRows = in.size();
    assert(weights.size() == numInRows);
    assert(out.size() == numInRows);

    for (unsigned i = 0; i != numInRows; ++i) {
      const auto outWidth = out[i].size();
      const auto numChans = weights[i].size();
      assert(in[i].size() % numChans == 0);
      const auto inWidth = in[i].size() / numChans;
      assert((outWidth - 1) * inStride == (inWidth - 1) * outStride);
      for (unsigned outX = 0, inX = 0; outX < outWidth; outX += outStride,
                                                        inX += inStride) {
        for (unsigned chan = 0; chan != numChans; ++chan) {
          out[i][outX] += in[i][inX * numChans + chan] * weights[i][chan];
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    unsigned numInRows = in.size();
    assert(numInRows != 0);
    const auto numChans = weights[0].size();
    std::vector<unsigned> convSizes;
    convSizes.reserve(numInRows);
    for (unsigned i = 0; i != numInRows; ++i) {
      const auto outWidth = out[i].size();
      const auto numOutputs = (outWidth + outStride - 1) / outStride;
      convSizes.push_back(numOutputs);
    }
    bool isFloat = std::is_same<InType, float>::value;
    return getConvPartialHorizontalMacCycleEstimate(isFloat, numChans,
                                                    convSizes, dataPathWidth);
  }
};

template class ConvPartialHorizontalMac<float, float>;
template class ConvPartialHorizontalMac<half, float>;
template class ConvPartialHorizontalMac<half, half>;

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

  /* The output activation once non-linearity is applied
   */
  Vector<Output<Vector<FPType>>> act;

  bool compute() {
    const unsigned nGroups = dIn.size();
    const unsigned vecLen = dIn[0].size();

    for (unsigned gr = 0; gr < nGroups; ++gr) {
      for (unsigned el = 0; el < vecLen; ++el) {
        act[gr][el] = dIn[gr][el];
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

template <class FPType>
class AddBias : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> acts;
  Vector<Input<Vector<FPType>>> biases;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned n = acts.size();
    assert(biases.size() == n);
    for (unsigned i = 0; i != n; ++i) {
      unsigned chansPerGroup = biases[i].size();
      assert(acts[i].size() % chansPerGroup == 0);
      unsigned len = acts[i].size() / chansPerGroup;
      for (unsigned j = 0; j != len; ++j) {
        for (unsigned k = 0; k != chansPerGroup; ++k) {
          acts[i][j * chansPerGroup + k] += biases[i][k];
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned n = acts.size();
    unsigned numCycles = 5;
    for (unsigned i = 0; i != n; ++i) {
      unsigned chansPerGroup = biases[i].size();
      assert(acts[i].size() % chansPerGroup == 0);
      unsigned len = acts[i].size() / chansPerGroup;
      numCycles += 2; // Load bias and act pointers.
      numCycles += 1; // Warmup.
      // Add biases to acts using add + dual load + store.
      numCycles += (len * chansPerGroup + vectorWidth - 1) / vectorWidth;
    }
    return numCycles;
  }
};

template class AddBias<float>;
template class AddBias<half>;

template <class FPType>
class AddToScaledChannel : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> acts;
  Vector<Input<Vector<FPType>>> addend;
  float scale;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned n = acts.size();
    assert(addend.size() == n);
    for (unsigned i = 0; i != n; ++i) {
      unsigned chansPerGroup = addend[i].size();
      assert(acts[i].size() % chansPerGroup == 0);
      unsigned len = acts[i].size() / chansPerGroup;
      for (unsigned j = 0; j != len; ++j) {
        for (unsigned k = 0; k != chansPerGroup; ++k) {
          acts[i][j * chansPerGroup + k] += scale * addend[i][k];
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned n = acts.size();
    unsigned numCycles = 6;
    for (unsigned i = 0; i != n; ++i) {
      unsigned chansPerGroup = addend[i].size();
      assert(acts[i].size() % chansPerGroup == 0);
      unsigned len = acts[i].size() / chansPerGroup;
      numCycles += 2; // Load addend and act pointers.
      numCycles += 1; // Warmup.
      // Add addend to acts using axpby + dual load + store.
      numCycles += (len * chansPerGroup + vectorWidth - 1) / vectorWidth;
    }
    return numCycles;
  }
};

template class AddToScaledChannel<float>;
template class AddToScaledChannel<half>;

template <class FPType>
class ChannelMul : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> actsIn;
  Vector<InOut<Vector<FPType>>> actsOut;
  Vector<Input<Vector<FPType>>> scale;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(actsIn.size() == actsOut.size());
    unsigned n = actsIn.size();
    assert(scale.size() == n);
    for (unsigned i = 0; i != n; ++i) {
      unsigned chansPerGroup = scale[i].size();
      assert(actsIn[i].size() % chansPerGroup == 0);
      assert(actsOut[i].size() % chansPerGroup == 0);
      unsigned len = actsIn[i].size() / chansPerGroup;
      for (unsigned j = 0; j != len; ++j) {
        for (unsigned k = 0; k != chansPerGroup; ++k) {
          actsOut[i][j * chansPerGroup + k] =
            actsIn[i][j * chansPerGroup + k] * scale[i][k];
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned n = actsIn.size();
    unsigned numCycles = 5;
    for (unsigned i = 0; i != n; ++i) {
      unsigned chansPerGroup = scale[i].size();
      assert(actsIn[i].size() % chansPerGroup == 0);
      assert(actsOut[i].size() % chansPerGroup == 0);
      unsigned len = actsIn[i].size() / chansPerGroup;
      numCycles += 3; // Load scale and act pointers.
      numCycles += 1; // Warmup.
      // multiply scale by acts using mul + dual load + store.
      numCycles += (len * chansPerGroup + vectorWidth - 1) / vectorWidth;
    }
    return numCycles;
  }
};

template class ChannelMul<float>;
template class ChannelMul<half>;


template <typename InType, typename OutType>
class ConvBNReduce: public Vertex {
public:
  Output<Vector<OutType>> sum;
  Vector<Input<Vector<InType>>> in;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    auto numEstimates = sum.size();
    for (unsigned est = 0; est < numEstimates; ++est) {
      float s = 0;
      for (unsigned d = 0; d < in.size(); d++) {
        assert(in[d].size() % sum.size() == 0);
        auto samplesPerEst = in[d].size() / sum.size();
        for (unsigned i = 0; i < samplesPerEst; ++i) {
          auto x = in[d][i * numEstimates + est];
          s += x;
        }
      }
      sum[est] = s;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<InType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (sum.size() + vectorWidth - 1) / vectorWidth;

    uint64_t cycles = 5;
    for (unsigned d = 0; d < in.size(); d++) {
      cycles += 5;
      auto samplesPerEst = in[d].size() / sum.size();
      cycles += numVectors * (2 + samplesPerEst);
    }
    return cycles;
  }
};

template class ConvBNReduce<float, float>;
template class ConvBNReduce<half, float>;


template <typename InType, typename OutType>
class ConvBNReduceSquare: public Vertex {
public:
  Output<Vector<OutType>> sum;
  Vector<Input<Vector<InType>>> in;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    auto numEstimates = sum.size();
    for (unsigned est = 0; est < numEstimates; ++est) {
      float s = 0;
      for (unsigned d = 0; d < in.size(); d++) {
        assert(in[d].size() % sum.size() == 0);
        auto samplesPerEst = in[d].size() / sum.size();
        for (unsigned i = 0; i < samplesPerEst; ++i) {
          auto x = in[d][i * numEstimates + est];
          s += x * x;
        }
      }
      sum[est] = s;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<InType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (sum.size() + vectorWidth - 1) / vectorWidth;

    uint64_t cycles = 5;
    for (unsigned d = 0; d < in.size(); d++) {
      cycles += 5;
      auto samplesPerEst = in[d].size() / sum.size();
      cycles += numVectors * (2 + samplesPerEst);
    }
    return cycles;
  }
};

template class ConvBNReduceSquare<float, float>;
template class ConvBNReduceSquare<half, float>;

template <typename InType, typename OutType>
class ConvBNReduceAndScale: public Vertex {
public:
  Output<OutType> out;
  Vector<Input<InType>> sum; // partial running sum
  float scale;

  bool compute() {
    float sumTotal = 0;
    for (unsigned i = 0; i < sum.size(); ++i) {
      sumTotal += sum[i];
    }
    float m = scale * sumTotal;
    *out = m;
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 15 + sum.size();
  }
};

template class ConvBNReduceAndScale<float, float>;
template class ConvBNReduceAndScale<float, half>;
template class ConvBNReduceAndScale<half, half>;


template <class FPType>
class BatchNorm : public Vertex {
public:
  Vector<Input<Vector<FPType>>> actsIn;
  Vector<Input<Vector<FPType>>> stdDev;
  Vector<Input<Vector<FPType>>> gamma;
  Vector<Input<Vector<FPType>>> beta;
  Vector<Output<Vector<FPType>>> actsOut;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned n = actsIn.size();
    assert(actsOut.size() == n);
    assert(stdDev.size() == n);
    assert(gamma.size() == n);
    assert(beta.size() == n);

    for (unsigned i = 0; i != n; ++i) {
      unsigned chansPerGroup = stdDev[i].size();
      assert(actsIn[i].size() % chansPerGroup == 0);
      unsigned len = actsIn[i].size() / chansPerGroup;
      for (unsigned j = 0; j != len; ++j) {
        for (unsigned k = 0; k != chansPerGroup; ++k) {
          const auto idx = j * chansPerGroup + k;
          FPType whitenedAct = actsIn[i][idx] / stdDev[i][k];
          actsOut[i][idx] = gamma[i][k] * whitenedAct + beta[i][k];
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned n = actsIn.size();
    unsigned numCycles = 5;
    for (unsigned i = 0; i != n; ++i) {
      unsigned chansPerGroup = stdDev[i].size();
      assert(actsIn[i].size() % chansPerGroup == 0);
      unsigned len = actsIn[i].size() / chansPerGroup;
      numCycles += 2; //
      numCycles += 1; // Warmup.
      // Add biases to acts using add + dual load + store.
      numCycles += (len * chansPerGroup + vectorWidth - 1) / vectorWidth;
    }
    return numCycles;
  }
};


template class BatchNorm<float>;
template class BatchNorm<half>;

} // end namespace popconv
