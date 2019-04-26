#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
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
class
[[poplar::constraint("elem(**in) != elem(**out)")]]
ConvPartialnx1: public SupervisorVertex {
public:
  ConvPartialnx1();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType =
      typename std::conditional<useLimitedVer, short, int>::type;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, SCALED_PTR64, weightsAlign,
                      use128BitLoad>>, ONE_PTR> weights;
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

  static const bool isExternalCodelet = (EXTERNAL_CODELET) &&
                                        !(std::is_same<AccumType, half>() &&
                                          std::is_same<FPType, float>()) &&
                                        useLimitedVer == true;

  bool compute() {
    const unsigned numWorkers = NUM_WORKERS;
    const unsigned convInputLoadElems =
        std::is_same<FPType, float>::value ? CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT :
                                             CONV_UNIT_INPUT_LOAD_ELEMS_HALF;
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned ampKernelHeight = ampKernelHeightM1 + 1;
    const unsigned kernelOuterSize = kernelOuterSizeM1 + 1;
    const unsigned kernelInnerElements = kernelInnerElementsM1 + 1;

    int inRowStride =
        (transformedInRowStride - 1) * convInputLoadElems/ inChansPerGroup + 1;

    const int inStride =
        (transformedInStride - 1) * convInputLoadElems / inChansPerGroup + 1 +
        (ampKernelHeight - 1) * inRowStride;

    const auto usedContexts = worklists.size() / (kernelOuterSize *
                                                  kernelInnerElements);
    const auto outStrideThresh =  std::is_same<AccumType, float>() ? -6 : -4;

    const auto flipOut = transformedOutStride < outStrideThresh;
    const int outStride =
        flipOut ? (-transformedOutStride + outStrideThresh) / outChansPerGroup :
                  (transformedOutStride - outStrideThresh) / outChansPerGroup;

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
                                  ig * numOutGroups +
                                  (numOutGroups - 1 - og)];
          for (unsigned ky = 0; ky < kernelOuterSize; ++ky) {
            for (unsigned kx = 0; kx < kernelInnerElements; ++kx) {
              for (unsigned context = 0; context < usedContexts; ++context) {
                const auto k = (ky * kernelInnerElements + kx);
                const auto &wl = worklists[k * usedContexts + context];
                unsigned wi = 0;
                while (wi < wl.size()) {
                  auto outOffset  = wl[wi];
                  auto numFieldElems   = wl[wi + 1];
                  auto inOffset   = wl[wi + 2];

                  wi += 3;
                  for (unsigned i = 0; i < numFieldElems; ++i) {
                    for (unsigned outChan = 0;
                         outChan < outChansPerGroup;
                         ++outChan) {
                      const auto outIndex =
                          (outOffset + (flipOut ? -i : i) * outStride)
                          * outChansPerGroup + outChan;
                      AccumType sum = out[cg * numOutGroups + og][outIndex];
                      for (unsigned ak = 0; ak < ampKernelHeight; ++ak) {
                        for (unsigned inChan = 0;
                             inChan < inChansPerGroup;
                             ++inChan) {
                          const auto inIndex =
                              (inOffset + i * inStride) * inChansPerGroup +
                              ak * inRowStride * inChansPerGroup +
                              inChan;
                          const auto weightIndex =
                              ky * ampKernelHeight * kernelInnerElements *
                                   outChansPerGroup * inChansPerGroup +
                              kx * outChansPerGroup * inChansPerGroup +
                              ak * kernelInnerElements * outChansPerGroup *
                                   inChansPerGroup +
                              outChan * inChansPerGroup +
                              inChan;

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
class
[[poplar::constraint("elem(**in) != elem(**out)")]]
ConvPartial1x1Out: public SupervisorVertex {
public:
  ConvPartial1x1Out();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType =
      typename std::conditional<useLimitedVer, short, int>::type;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> in;
  Vector<Input<Vector<FPType, SCALED_PTR64,
                      weightsAlign, use128BitLoad>>, SCALED_PTR32> weights;
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

  static const bool isExternalCodelet = (EXTERNAL_CODELET) &&
                                        !(std::is_same<AccumType, half>() &&
                                          std::is_same<FPType, float>()) &&
                                        useLimitedVer == true;

  bool compute() {
    const unsigned convInputLoadElems =
        std::is_same<FPType, float>::value ? CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT :
                                             CONV_UNIT_INPUT_LOAD_ELEMS_HALF;
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
        for (unsigned ig = 0; ig < numInGroups ; ++ig) {
          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups +
                                  (numOutGroups - 1 - og)];
          for (unsigned context = 0; context < usedContexts; ++context) {
            auto outOffset  = worklists[3 * context];
            auto numFieldElems = worklists[3 * context + 1];
            auto inOffset   = worklists[3 * context + 2];

            for (unsigned i = 0; i < numFieldElems; ++i) {
              for (unsigned outChan = 0; outChan < outChansPerGroup;++outChan) {
                const auto outIndex =
                    (outOffset + (flipOut ? -i : i)) * outChansPerGroup
                    + outChan;
                if (ig == 0)
                  out[cg * numOutGroups + og][outIndex] = 0;
                float sum = 0;
                for (unsigned inChan = 0; inChan < inChansPerGroup; ++inChan) {
                  const auto inIndex =
                      (inOffset + i * inStride) * inChansPerGroup + inChan;
                  const auto weightIndex =
                      outChan * inChansPerGroup + inChan;
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
class
[[poplar::constraint("elem(**in) != elem(**weights)")]]
ConvPartialHorizontalMac : public SupervisorVertex {
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
                                  ig * numOutGroups +
                                  (numOutGroups - 1 - og)];

          for (unsigned k = 0; k != kernelSize; ++k) {
            for (unsigned context = 0; context < usedContexts; ++context) {
              const auto &wl =
                  worklists[k * usedContexts + context];
              unsigned wi = 0;
              while (wi < wl.size()) {
                auto outOffset  = wl[wi];
                auto numConv   = wl[wi + 1];
                auto inOffset   = wl[wi + 2];
                wi += 3;
                for (unsigned i = 0; i != numConv; ++i) {
                  for (unsigned oc = 0; oc != outChansPerGroup; ++oc) {
                    const auto outIndex =
                      (outOffset +  i * outStride) * outChansPerGroup + oc;
                    AccumType sum = out[cg * numOutGroups + og][outIndex];
                    for (unsigned ic = 0; ic != inChansPerGroup; ++ic) {
                      const auto inIndex =
                        (inOffset + i * inStride) * inChansPerGroup + ic;
                      const auto weightIndex =
                            k * outChansPerGroup * inChansPerGroup +
                            oc * inChansPerGroup + ic;
                      sum += AccumType(in[cg * numInGroups + ig][inIndex]
                                       * w[weightIndex]);
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
  Vector<Input<Vector<FPType>>, ONE_PTR> wIn;

  /* Same as wIn except that numOutCols*numOutRows vectors each of dimension
   * 1xdepth are stored
   */
  Vector<Output<Vector<FPType, ONE_PTR>>> wTf;


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

        FPType A = (g[0][0] + g[0][1] + g[0][2]) * FPType(0.5);
        FPType B = (g[0][0] - g[0][1] + g[0][2]) * FPType(0.5);

        FPType C = (g[0][0] + g[1][0] + g[2][0]) * FPType(0.5);
        FPType F = (g[0][0] - g[1][0] + g[2][0]) * FPType(0.5);

        FPType D = (g[2][0] + g[2][1] + g[2][2]) * FPType(0.5);
        FPType E = (g[2][0] - g[2][1] + g[2][2]) * FPType(0.5);

        FPType G = (g[1][0] + g[1][1] + g[1][2]) * FPType(0.5);
        FPType H = (g[1][0] - g[1][1] + g[1][2]) * FPType(0.5);

        FPType I = (g[0][2] + g[1][2] + g[2][2]) * FPType(0.5);
        FPType J = (g[0][2] - g[1][2] + g[2][2]) * FPType(0.5);

        wrTf(gBaseOut, 0, 0, elem) = g[0][0];
        wrTf(gBaseOut, 0, 1, elem) = A;
        wrTf(gBaseOut, 0, 2, elem) = B;
        wrTf(gBaseOut, 0, 3, elem) = g[0][2];

        wrTf(gBaseOut, 1, 0, elem) = C;
        wrTf(gBaseOut, 1, 1, elem) = (A + G + D) * FPType(0.5);
        wrTf(gBaseOut, 1, 2, elem) = (B + H + E) * FPType(0.5);
        wrTf(gBaseOut, 1, 3, elem) = I;

        wrTf(gBaseOut, 2, 0, elem) = F;
        wrTf(gBaseOut, 2, 1, elem) = (A - G + D) * FPType(0.5);
        wrTf(gBaseOut, 2, 2, elem) = (B - H + E) * FPType(0.5);
        wrTf(gBaseOut, 2, 3, elem) = J;

        wrTf(gBaseOut, 3, 0, elem) = g[2][0];
        wrTf(gBaseOut, 3, 1, elem) = D;
        wrTf(gBaseOut, 3, 2, elem) = E;
        wrTf(gBaseOut, 3, 3, elem) = g[2][2];
      }
    }
    return true;
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
  Vector<Input<Vector<FPType>>, ONE_PTR> dTf;

  /* kernel transform vector. Each vector is of length inpChanDepth*outChanDepth
   * The same input data is used to generate outChanDepth outputs for each input
   * vector
   */
  Vector<Input<Vector<FPType>>> wTf;

  /* Output for each of the nGroups 1D vectors. Each input vector results in a
   * 1xoutChanDepth vector.
   */
  Vector<InOut<Vector<FPType>>> partials;

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
  Vector<Input<Vector<FPType, ONE_PTR>>> inPartial;

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
  Vector<Output<Vector<FPType, ONE_PTR>>, ONE_PTR> dOut;

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
  Vector<Output<Vector<FPType, ONE_PTR>>, ONE_PTR> act;

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
};

template class WgdConvComplete<float>;
template class WgdConvComplete<half>;

template <typename T>
class
[[poplar::constraint("elem(**src) != elem(**dst)")]]
Transpose2d : public Vertex {
public:
  Transpose2d();

  Vector<Input<Vector<T, ONE_PTR,8>>> src;
  Vector<Output<Vector<T, ONE_PTR,8>>, ONE_PTR> dst;
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
class
[[poplar::constraint("elem(*src) != elem(*dst)")]]
Transpose : public Vertex {
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
[[poplar::constraint("elem(*src) != elem(*dst)")]]
TransposeSupervisor : public SupervisorVertex {
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
    unsigned totalTranspositions = workerCount*numTranspositions +
                                   (CTXT_WORKERS - workerCount)*
                                   (numTranspositions-1);

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

  Vector<Input<Vector<MeanType>>> mean;
  Vector<Input<Vector<PowerType, ONE_PTR>>, ONE_PTR> power;
  Vector<Output<Vector<OutType, ONE_PTR>>, ONE_PTR> iStdDev;
  const float scaleVar;
  const float eps;

  bool compute() {
    for (unsigned i = 0; i != mean.size(); ++i) {
      for (unsigned j = 0; j != mean[i].size(); ++j) {
        float varianceEst =
          float(power[i][j]) - float(mean[i][j] * mean[i][j]) + eps;
        varianceEst *= scaleVar;
        float invStdDev = sqrt(1.0f / varianceEst);
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
class
[[poplar::constraint("elem(*weights) != elem(**out)")]]
OuterProduct : public Vertex {
public:
  OuterProduct();

  Input<Vector<T>> in;
  Input<Vector<T, ONE_PTR,8>> weights;
  Vector<Output<Vector<T, ONE_PTR,8>>> out;
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
class
ReduceAdd : public SupervisorVertex {
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
