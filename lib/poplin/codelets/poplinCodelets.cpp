#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;

#if defined(__IPU__) && !defined(POPLIBS_DISABLE_ASM_CODELETS)
#define EXTERNAL_CODELET true
#else
#define EXTERNAL_CODELET false
#endif

namespace poplin {

/**
 * Compute nx1 convolutions and accumulate them with partial sums in memory.
 * useLimitedVer is "true" if there are constraints imposed on
 * - size of strides are bounded to strides supported by ISA
 * - worklists-offsets are bounded to fit 16-bits
 * - worklists-number of elements <= maximum count supported by rpt instruction
 **/
template <class FPType, class AccumType, bool useLimitedVer, bool use128BitLoad>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartialnx1
    : public SupervisorVertex {
public:
  ConvPartialnx1();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType = typename std::conditional<useLimitedVer, short, int>::type;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, SCALED_PTR64, weightsAlign, use128BitLoad>>,
         ONE_PTR>
      weights;
  Vector<Output<Vector<AccumType, SCALED_PTR64, 8, true>>, ONE_PTR> out;
  const unsigned zerosInfo;
  Input<VectorList<WorkListType, VectorListLayout::DELTAN>> worklists;
  const UnsignedType numOutGroupsM1;
  const UnsignedType numInGroups;
  const UnsignedType kernelOuterSizeM1;
  const UnsignedType kernelInnerElementsM1;

  // This value is
  // (inStrideX - 1 - (ampKernelHeight - 1) * inRowStride)
  //      * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  const SignedType transformedInStride;
  // This output stride also encodes the flip parameter and is given as
  // -6 + outChansPerGroup * (actual output stride) if flipOut = false
  // -6 - outChansPerGroup * (actual output stride) if flipOut = true
  const SignedType transformedOutStride;
  const UnsignedType numConvGroupsM1;
  // The number of kernel elements we accumulate across within the AMP unit
  const UnsignedType ampKernelHeightM1;
  // The actual coding of this is
  //  (inRowSride - 1) * inChansPerGroup / convInputLoadElems + 1
  const SignedType transformedInRowStride;
  const UnsignedType outChansPerGroup;
  const UnsignedType inChansPerGroup;

  static const bool isExternalCodelet =
      (EXTERNAL_CODELET) &&
      !(std::is_same<AccumType, half>() && std::is_same<FPType, float>()) &&
      useLimitedVer == true;

  bool compute() {
    const unsigned numWorkers = NUM_WORKERS;
    const unsigned convInputLoadElems = std::is_same<FPType, float>::value
                                            ? CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT
                                            : CONV_UNIT_INPUT_LOAD_ELEMS_HALF;
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned ampKernelHeight = ampKernelHeightM1 + 1;
    const unsigned kernelOuterSize = kernelOuterSizeM1 + 1;
    const unsigned kernelInnerElements = kernelInnerElementsM1 + 1;

    int inRowStride =
        (transformedInRowStride - 1) * convInputLoadElems / inChansPerGroup + 1;

    const int inStride =
        (transformedInStride - 1) * convInputLoadElems / inChansPerGroup + 1 +
        (ampKernelHeight - 1) * inRowStride;

    const auto usedContexts =
        worklists.size() / (kernelOuterSize * kernelInnerElements);
    const auto outStrideThresh = std::is_same<AccumType, float>() ? -6 : -4;

    const auto flipOut = transformedOutStride < outStrideThresh;
    const int outStride =
        flipOut ? (-transformedOutStride + outStrideThresh) / outChansPerGroup
                : (transformedOutStride - outStrideThresh) / outChansPerGroup;

    const unsigned numElems = zerosInfo;

    for (unsigned cg = 0; cg != numConvGroups; ++cg) {
      for (unsigned og = 0; og != numOutGroups; ++og) {
        for (unsigned i = 0; i != numElems; ++i)
          out[cg * numOutGroups + og][i] = 0;
      }
    }
    for (unsigned cg = 0; cg < numConvGroups; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups + (numOutGroups - 1 - og)];
          for (unsigned ky = 0; ky < kernelOuterSize; ++ky) {
            for (unsigned kx = 0; kx < kernelInnerElements; ++kx) {
              for (unsigned context = 0; context < usedContexts; ++context) {
                const auto k = (ky * kernelInnerElements + kx);
                const auto &wl = worklists[k * usedContexts + context];
                unsigned wi = 0;
                while (wi < wl.size()) {
                  auto outOffset = wl[wi];
                  auto numFieldElems = wl[wi + 1];
                  auto inOffset = wl[wi + 2];

                  wi += 3;
                  for (unsigned i = 0; i < numFieldElems; ++i) {
                    for (unsigned outChan = 0; outChan < outChansPerGroup;
                         ++outChan) {
                      const auto outIndex =
                          (outOffset + (flipOut ? -i : i) * outStride) *
                              outChansPerGroup +
                          outChan;
                      AccumType sum = out[cg * numOutGroups + og][outIndex];
                      for (unsigned ak = 0; ak < ampKernelHeight; ++ak) {
                        for (unsigned inChan = 0; inChan < inChansPerGroup;
                             ++inChan) {
                          const auto inIndex =
                              (inOffset + i * inStride) * inChansPerGroup +
                              ak * inRowStride * inChansPerGroup + inChan;
                          const auto weightIndex =
                              ky * ampKernelHeight * kernelInnerElements *
                                  outChansPerGroup * inChansPerGroup +
                              kx * outChansPerGroup * inChansPerGroup +
                              ak * kernelInnerElements * outChansPerGroup *
                                  inChansPerGroup +
                              outChan * inChansPerGroup + inChan;

                          sum += AccumType(in[cg * numInGroups + ig][inIndex] *
                                           w[weightIndex]);
                        }
                      }
                      out[cg * numOutGroups + og][outIndex] = sum;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return true;
  }
};

template class ConvPartialnx1<float, float, true, false>;
template class ConvPartialnx1<half, half, true, false>;
template class ConvPartialnx1<half, float, true, false>;
template class ConvPartialnx1<float, float, false, false>;
template class ConvPartialnx1<half, half, false, false>;
template class ConvPartialnx1<half, float, false, false>;

template class ConvPartialnx1<float, float, true, true>;
template class ConvPartialnx1<half, half, true, true>;
template class ConvPartialnx1<half, float, true, true>;
template class ConvPartialnx1<float, float, false, true>;
template class ConvPartialnx1<half, half, false, true>;
template class ConvPartialnx1<half, float, false, true>;

/**
 * Compute a sum of 1x1 convolutions over a subset of the input channels for
 * multiple output channels.
 * useLimitedVer is "true" if there are constraints imposed on
 * - size of strides are bounded to strides supported by ISA
 * - worklists-offsets are bounded to fit 16-bits
 * - worklists-number of elements <= maximum count supported by rpt instruction
 **/
template <class FPType, class AccumType, bool useLimitedVer, bool use128BitLoad>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartial1x1Out
    : public SupervisorVertex {
public:
  ConvPartial1x1Out();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType = typename std::conditional<useLimitedVer, short, int>::type;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> in;
  Vector<Input<Vector<FPType, SCALED_PTR64, weightsAlign, use128BitLoad>>,
         SCALED_PTR32>
      weights;
  Vector<Output<Vector<AccumType, SCALED_PTR64, 16, true>>, SCALED_PTR32> out;
  Input<Vector<WorkListType, SCALED_PTR32>> worklists;
  const UnsignedType numConvGroupsM1;
  // Actual value is 1 more than this
  const UnsignedType numOutGroupsM1;
  const UnsignedType numInGroups;
  // This value is
  // (inStrideX - 1) * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  const SignedType transformedInStride;
  const UnsignedType outChansPerGroup;
  // This stride encodes the flip out parameter
  const SignedType transformedOutStride;
  const UnsignedType inChansPerGroup;

  static const bool isExternalCodelet =
      (EXTERNAL_CODELET) &&
      !(std::is_same<AccumType, half>() && std::is_same<FPType, float>()) &&
      useLimitedVer == true;

  bool compute() {
    const unsigned convInputLoadElems = std::is_same<FPType, float>::value
                                            ? CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT
                                            : CONV_UNIT_INPUT_LOAD_ELEMS_HALF;
    const auto usedContexts = NUM_WORKERS;
    // modify to set actual values used by vertex
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const int inStride =
        (transformedInStride - 1) * convInputLoadElems / inChansPerGroup + 1;
    bool flipOut =
        transformedOutStride < (std::is_same<AccumType, float>() ? -6 : -4);

    for (unsigned cg = 0; cg < numConvGroups; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups + (numOutGroups - 1 - og)];
          for (unsigned context = 0; context < usedContexts; ++context) {
            auto outOffset = worklists[3 * context];
            auto numFieldElems = worklists[3 * context + 1];
            auto inOffset = worklists[3 * context + 2];

            for (unsigned i = 0; i < numFieldElems; ++i) {
              for (unsigned outChan = 0; outChan < outChansPerGroup;
                   ++outChan) {
                const auto outIndex =
                    (outOffset + (flipOut ? -i : i)) * outChansPerGroup +
                    outChan;
                if (ig == 0)
                  out[cg * numOutGroups + og][outIndex] = 0;
                float sum = 0;
                for (unsigned inChan = 0; inChan < inChansPerGroup; ++inChan) {
                  const auto inIndex =
                      (inOffset + i * inStride) * inChansPerGroup + inChan;
                  const auto weightIndex = outChan * inChansPerGroup + inChan;
                  sum += float(in[cg * numInGroups + ig][inIndex]) *
                         float(w[weightIndex]);
                }
                out[cg * numOutGroups + og][outIndex] += sum;
              }
            }
          }
        }
      }
    }
    return true;
  }
};

template class ConvPartial1x1Out<half, half, true, false>;
template class ConvPartial1x1Out<half, float, true, false>;
template class ConvPartial1x1Out<float, half, true, false>;
template class ConvPartial1x1Out<float, float, true, false>;
template class ConvPartial1x1Out<half, half, false, false>;
template class ConvPartial1x1Out<half, float, false, false>;
template class ConvPartial1x1Out<float, half, false, false>;
template class ConvPartial1x1Out<float, float, false, false>;

template class ConvPartial1x1Out<half, half, true, true>;
template class ConvPartial1x1Out<half, float, true, true>;
template class ConvPartial1x1Out<float, half, true, true>;
template class ConvPartial1x1Out<float, float, true, true>;
template class ConvPartial1x1Out<half, half, false, true>;
template class ConvPartial1x1Out<half, float, false, true>;
template class ConvPartial1x1Out<float, half, false, true>;
template class ConvPartial1x1Out<float, float, false, true>;

/* Perform a series of 1x1 convolutions using the MAC instruction where the
 * axis of accumulation is across the vector.
 * useLimitedVer is "true" if there are constraints imposed on
 * - The number of input channels is a multiple of 2
 * - worklists items are bounded to fit 16-bits
 * - number of input channels per group divided by 2 or 4 depending on whether
 *   those channels are multiple of 2 and 4 respectively
 *    <= maximum count supported by rpt instruction
 */
template <class FPType, class AccumType, bool useLimitedVer>
class [[poplar::constraint(
    "elem(**in) != elem(**weights)")]] ConvPartialHorizontalMac
    : public SupervisorVertex {
public:
  ConvPartialHorizontalMac();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, ONE_PTR> weights;
  Vector<Output<Vector<AccumType, SCALED_PTR64, 8>>, ONE_PTR> out;
  const unsigned zerosInfo;
  Input<VectorList<WorkListType, VectorListLayout::DELTAN>> worklists;
  const UnsignedType numOutGroupsM1;

  // transformedInStride =  ("actual input stride" - 1) * inChansPerGroup
  const unsigned transformedInStride;
  // transformedOutStride =
  //   = (-1 * "actual output stride" - 1 * outChansPerGroup (if flip output)
  //   = +1 * "actual output stride" * outChansPerGroup
  const int transformedOutStride;
  const UnsignedType numInGroups;
  const UnsignedType kernelSizeM1;
  const UnsignedType numConvGroupsM1;
  const UnsignedType outChansPerGroup;
  const UnsignedType inChansPerGroup;

  static const bool isExternalCodelet = (EXTERNAL_CODELET) &&
                                        std::is_same<AccumType, float>() &&
                                        useLimitedVer == true;
  bool compute() {
    const unsigned numWorkers = NUM_WORKERS;
    const unsigned kernelSize = kernelSizeM1 + 1;
    const auto usedContexts = worklists.size() / kernelSize;
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const auto outStride =
        transformedOutStride / static_cast<int>(outChansPerGroup) + 1;
    const auto inStride = transformedInStride / inChansPerGroup;
    const unsigned numElems = zerosInfo;

    for (unsigned cg = 0; cg != numConvGroups; ++cg) {
      for (unsigned og = 0; og != numOutGroups; ++og) {
        for (unsigned i = 0; i != numElems; ++i)
          out[cg * numOutGroups + og][i] = 0;
      }
    }

    for (unsigned cg = 0; cg != numConvGroups; ++cg) {
      for (unsigned og = 0; og != numOutGroups; ++og) {
        for (unsigned ig = 0; ig != numInGroups; ++ig) {
          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups + (numOutGroups - 1 - og)];

          for (unsigned k = 0; k != kernelSize; ++k) {
            for (unsigned context = 0; context < usedContexts; ++context) {
              const auto &wl = worklists[k * usedContexts + context];
              unsigned wi = 0;
              while (wi < wl.size()) {
                auto outOffset = wl[wi];
                auto numConv = wl[wi + 1];
                auto inOffset = wl[wi + 2];
                wi += 3;
                for (unsigned i = 0; i != numConv; ++i) {
                  for (unsigned oc = 0; oc != outChansPerGroup; ++oc) {
                    const auto outIndex =
                        (outOffset + i * outStride) * outChansPerGroup + oc;
                    AccumType sum = out[cg * numOutGroups + og][outIndex];
                    for (unsigned ic = 0; ic != inChansPerGroup; ++ic) {
                      const auto inIndex =
                          (inOffset + i * inStride) * inChansPerGroup + ic;
                      const auto weightIndex =
                          k * outChansPerGroup * inChansPerGroup +
                          oc * inChansPerGroup + ic;
                      sum += AccumType(in[cg * numInGroups + ig][inIndex] *
                                       w[weightIndex]);
                    }
                    out[cg * numOutGroups + og][outIndex] = sum;
                  }
                }
              }
            }
          }
        }
      }
    }
    return true;
  }
};
template class ConvPartialHorizontalMac<float, float, true>;
template class ConvPartialHorizontalMac<float, float, false>;
template class ConvPartialHorizontalMac<half, float, true>;
template class ConvPartialHorizontalMac<half, float, false>;
template class ConvPartialHorizontalMac<half, half, true>;
template class ConvPartialHorizontalMac<half, half, false>;

template <typename T>
class [[poplar::constraint("elem(**src) != elem(**dst)")]] Transpose2d
    : public Vertex {
public:
  Transpose2d();

  Vector<Input<Vector<T, ONE_PTR, 8>>> src;
  Vector<Output<Vector<T, ONE_PTR, 8>>, ONE_PTR> dst;
  // TODO specialize the vertex based on the value of this field to avoid extra
  // memory usage.
  const unsigned short numSrcRows;
  const unsigned short numSrcColumns;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const auto numTranspositions = src.size();
    for (unsigned i = 0; i != numTranspositions; ++i) {
      for (unsigned x = 0; x != numSrcColumns; ++x) {
        for (unsigned y = 0; y != numSrcRows; ++y) {
          dst[i][x * numSrcRows + y] = src[i][y * numSrcColumns + x];
        }
      }
    }
    return true;
  }
};

template class Transpose2d<float>;
template class Transpose2d<half>;

template <typename T>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Transpose
    : public Vertex {
public:
  Transpose();

  Input<Vector<T, SCALED_PTR64, 8>> src;
  Output<Vector<T, SCALED_PTR64, 8>> dst;
  const unsigned short numSrcRowsD4;
  const unsigned short numSrcColumnsD4;
  const unsigned short numTranspositionsM1;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const unsigned numTranspositions = numTranspositionsM1 + 1;
    const unsigned numSrcColumns = numSrcColumnsD4 * 4;
    const unsigned numSrcRows = numSrcRowsD4 * 4;
    for (unsigned t = 0; t != numTranspositions; ++t) {
      for (unsigned x = 0; x != numSrcColumns; ++x) {
        for (unsigned y = 0; y != numSrcRows; ++y) {
          dst[t * numSrcRows * numSrcColumns + x * numSrcRows + y] =
              src[t * numSrcRows * numSrcColumns + y * numSrcColumns + x];
        }
      }
    }
    return true;
  }
};

template class Transpose<half>;

template <typename T>
class WORKER_ALIGN
    [[poplar::constraint("elem(*src) != elem(*dst)")]] TransposeSupervisor
    : public SupervisorVertex {
public:
  TransposeSupervisor();

  Input<Vector<T, SCALED_PTR64, 8>> src;
  Output<Vector<T, SCALED_PTR64, 8>> dst;
  const unsigned short numSrcRowsD4;
  const unsigned short numSrcColumnsD4;
  // There will be 'workerCount' workers (1 <= workerCount <= 6) transposing
  // 'numTranspositions' matrices ('numTranspositions' always >0) plus
  // (6-workerCount) workers transposing (numTranspositions-1) matrices.
  // Note that (6-workerCount) and/or (numTranspositions-1) could be zero.
  const unsigned short numTranspositions;
  const unsigned short workerCount;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned totalTranspositions =
        workerCount * numTranspositions +
        (CTXT_WORKERS - workerCount) * (numTranspositions - 1);

    const unsigned numSrcColumns = numSrcColumnsD4 * 4;
    const unsigned numSrcRows = numSrcRowsD4 * 4;
    for (unsigned t = 0; t != totalTranspositions; ++t) {
      for (unsigned x = 0; x != numSrcColumns; ++x) {
        for (unsigned y = 0; y != numSrcRows; ++y) {
          dst[t * numSrcRows * numSrcColumns + x * numSrcRows + y] =
              src[t * numSrcRows * numSrcColumns + y * numSrcColumns + x];
        }
      }
    }
    return true;
  }
};

template class TransposeSupervisor<half>;

template <class MeanType, class PowerType, class OutType>
class InverseStdDeviation : public Vertex {
public:
  InverseStdDeviation();

  // inner loop will process two elements at a time;
  // output can be written as a single fp32 or pair of f16 plus a trailing
  // aligned element
  Vector<Input<Vector<MeanType, SPAN, sizeof(MeanType) * 2>>> mean;
  Vector<Input<Vector<PowerType, ONE_PTR, sizeof(PowerType) * 2>>, ONE_PTR>
      power;
  Vector<Output<Vector<OutType, ONE_PTR, 4>>, ONE_PTR> iStdDev;
  const float scaleVar;
  const float eps;

  bool compute() {
    for (unsigned i = 0; i != mean.size(); ++i) {
      for (unsigned j = 0; j != mean[i].size(); ++j) {
        float elem = float(mean[i][j]);
        float varianceEst = float(power[i][j]) - elem * elem;
        // rounding can cause this estimate to become negative
        if (varianceEst < 0.0f)
          varianceEst = 0.0f;
        varianceEst += eps;
        varianceEst *= scaleVar;
        float invStdDev = 1.0f / sqrt(varianceEst);
        iStdDev[i][j] = invStdDev;
      }
    }
    return true;
  }
};

template class InverseStdDeviation<float, float, float>;
template class InverseStdDeviation<float, float, half>;
template class InverseStdDeviation<half, float, half>;
template class InverseStdDeviation<half, half, half>;

template <class T>
class [[poplar::constraint("elem(*weights) != elem(**out)")]] OuterProduct
    : public Vertex {
public:
  OuterProduct();

  Input<Vector<T>> in;
  Input<Vector<T, ONE_PTR, 8>> weights;
  Vector<Output<Vector<T, ONE_PTR, 8>>> out;
  const unsigned chansPerGroup;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    const auto width = in.size();
    const auto numChanGroups = out.size();

    for (unsigned g = 0; g != numChanGroups; ++g) {
      for (unsigned chanInGroup = 0; chanInGroup != chansPerGroup;
           ++chanInGroup) {
        const auto c = chanInGroup + g * chansPerGroup;
        for (unsigned x = 0; x != width; ++x) {
          out[g][chanInGroup + x * chansPerGroup] = in[x] * weights[c];
        }
      }
    }
    return true;
  }
};

template class OuterProduct<float>;
template class OuterProduct<half>;

template <typename OutType, typename PartialsType>
class ReduceAdd : public SupervisorVertex {
public:
  ReduceAdd();

  Vector<Input<Vector<PartialsType, ONE_PTR, 8, false>>, SCALED_PTR32> partials;
  Output<Vector<OutType, SCALED_PTR32, 8>> out;
  const unsigned short numPartials;
  const unsigned short numElems;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < numElems; ++i) {
      float sum = 0;
      for (unsigned j = 0; j < numPartials; ++j) {
        sum += float(partials[j][i]);
      }
      out[i] = sum;
    }
    return true;
  }
};

template class ReduceAdd<float, float>;
template class ReduceAdd<half, float>;
template class ReduceAdd<float, half>;
template class ReduceAdd<half, half>;

} // end namespace poplin
