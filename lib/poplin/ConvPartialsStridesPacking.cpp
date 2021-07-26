// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
// Helper functions to pack/pack and calculate transformed stirdes
// used by AMP codelets

#include "ConvPartialsStridesPacking.hpp"

namespace poplin {

int getTransformedInStride(unsigned convUnitWeightHeight, unsigned inStride,
                           int inRowStride, unsigned convInputLoadElems,
                           unsigned inChansPerGroup) {
  return (static_cast<int>(inStride) - 1 -
          static_cast<int>(convUnitWeightHeight - 1) * inRowStride) *
             static_cast<int>(inChansPerGroup / convInputLoadElems) +
         1;
}

int getTransformedInRowStride(int inRowStride, unsigned convInputLoadElems,
                              unsigned inChansPerGroup) {
  return (inRowStride - 1) *
             static_cast<int>(inChansPerGroup / convInputLoadElems) +
         1;
}

int getTransformedOutStride(int outStride, unsigned outChansPerGroup,
                            unsigned numConvUnits, bool accumTypeIsFloat,
                            bool flipOut) {
  auto outStrideThresh = accumTypeIsFloat ? -6 : -4;
  // For dual AMP codelets need to offset stride threshold by extra 8 elements
  if (numConvUnits > 8) {
    outStrideThresh += -8;
  }

  int scaledOutStride = static_cast<int>(outStride * outChansPerGroup);
  return outStrideThresh + (flipOut ? -scaledOutStride : scaledOutStride);
}

int reverseTransfromedInStride(int transformedInStride,
                               unsigned convInputLoadElems,
                               unsigned inChansPerGroup,
                               unsigned ampKernelHeight, int inRowStride) {
  return (transformedInStride - 1) * convInputLoadElems / inChansPerGroup + 1 +
         (ampKernelHeight - 1) * inRowStride;
}

int reverseTransfromedInRowStride(int transformedInRowStride,
                                  unsigned convInputLoadElems,
                                  unsigned inChansPerGroup) {
  return (transformedInRowStride - 1) * convInputLoadElems / inChansPerGroup +
         1;
}

std::pair<bool, int> reverseTransfromedOutStride(int transformedOutStride,
                                                 bool accumTypeIsFloat,
                                                 unsigned numConvUnits,
                                                 unsigned outChansPerGroup) {
  auto outStrideThresh = accumTypeIsFloat ? -6 : -4;
  // For dual AMP codelets need to offset stride threshold by extra 8 elements
  if (numConvUnits > 8) {
    outStrideThresh += -8;
  }

  const auto flipOut = transformedOutStride < outStrideThresh;
  const int outStride =
      flipOut ? (-transformedOutStride + outStrideThresh) / outChansPerGroup
              : (transformedOutStride - outStrideThresh) / outChansPerGroup;

  return std::make_pair(flipOut, outStride);
}

unsigned packAmpNx1Stride(const unsigned strideBits, int p2, int p1, int p0) {
  const unsigned strideMask = (1 << strideBits) - 1;
  const unsigned s10 = strideBits;
  const unsigned s20 = 2 * strideBits;

  return ((p2 & strideMask) << s20) | ((p1 & strideMask) << s10) |
         (p0 & strideMask);
}

int unpackAmpNx1Stride(const unsigned strideBits, unsigned stride, unsigned pX,
                       bool signExtention) {
  const unsigned strideMask = (1 << strideBits) - 1;
  const unsigned shift = pX * strideBits;
  const int mask = 1 << (strideBits - 1);

  int result = (stride >> shift) & strideMask;
  if (signExtention) {
    result = (result ^ mask) - mask;
  }

  return result;
}

int getSecondPtrOffset(unsigned ampKernelHeight, int transformedInRowStride) {
  if (ampKernelHeight == 4) {
    return 2 * transformedInRowStride;
  } else {
    return transformedInRowStride + 1;
  }
}

int unpackAmpNx1InRowStride(unsigned ampKernelHeight, int secondPtrOffset) {
  if (ampKernelHeight == 4) {
    return secondPtrOffset /= 2;
  } else {
    return secondPtrOffset -= 1;
  }
}

int getTransformedInStrideNx1(unsigned ampKernelHeight, int transformedInStride,
                              int transformedInRowStride) {
  auto inStrideAdjustment =
      getSecondPtrOffset(ampKernelHeight, transformedInRowStride);

  return transformedInStride + inStrideAdjustment;
}

int reverseTransformedInStrideNx1(int transformedInStride,
                                  int secondPtrOffset) {

  return transformedInStride - secondPtrOffset;
}

} // namespace poplin
