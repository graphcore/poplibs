// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
// Helper functions to pack/pack and calculate transformed strides
// used by AMP codelets

#ifndef poplin_ConvPartialsStridesPacking_hpp
#define poplin_ConvPartialsStridesPacking_hpp

#include <utility>

namespace poplin {

// Number of stride bits when unlimited codelet versions are used
constexpr unsigned inline numStrideBitsUnlimited() { return 21; }

// Below 6 functions designed to convert between C++ codelet strides and
// strides used by ASM load/stores instructions
int getTransformedInStride(unsigned convUnitWeightHeight, unsigned inStride,
                           int inRowStride, unsigned convInputLoadElems,
                           unsigned inChansPerGroup);

int getTransformedInRowStride(int inRowStride, unsigned convInputLoadElems,
                              unsigned inChansPerGroup);

int getTransformedOutStride(int outStride, unsigned outChansPerGroup,
                            unsigned numConvUnitsRequired, bool isPartialsFloat,
                            bool flipOut);

int reverseTransformedInStride(int transformedInStride,
                               unsigned convInputLoadElems,
                               unsigned inChansPerGroup,
                               unsigned ampKernelHeight = 0,
                               int inRowStride = 0);

int reverseTransformedInRowStride(int transformedInRowStride,
                                  unsigned convInputLoadElems,
                                  unsigned inChansPerGroup);

std::pair<bool, int> reverseTransformedOutStride(int transformedOutStride,
                                                 bool accumTypeIsFloat,
                                                 unsigned numConvUnits,
                                                 unsigned outChansPerGroup);

// Allows to pack 3 strides into a special register format used by load/store
// instructions. Each stride (p0, p1 and p2) will be masked by stride mask
// and placed accordingly.
// For example for p1: (p1 & strideMask) << (1 * strideBits)
// Always return unsigned long long as the limited and unlimited
// variants of the codelets can result in different stride widths.
unsigned long long packAmpNx1Stride(const unsigned strideBits, int p2, int p1,
                                    int p0);

// Extracts a stride from <stride> register based on position index <pX>.
// Could have an overload with unsigned rather than unsigned long long as this
// function is used by codelets. It is not implemented because we use ASM
// codelets with the packed stride fits within 32 bits.
int unpackAmpNx1Stride(const unsigned strideBits, unsigned long long stride,
                       unsigned pX, bool signExtention = true);

// To overcome HW limitation of 10bits per stride AMP Nx1 vertices for a
// limited version has two independent pointer that move alongside.
// A first pointer points to a start of the data where
// a second pointer has an offset of 2 * inRowStrides. That allows to avoid
// big return strides and only use <X> and <-X+1> strides to read input.
int getSecondPtrOffset(unsigned ampKernelHeight, int transformedInRowStride);

// Reverse actions done by getSecondPtrOffset method. secondPtrOffset is
// calculated by getSecondPtrOffset function
int unpackAmpNx1InRowStride(unsigned ampKernelHeight, int secondPtrOffset);

// AMP Nx1 codelet uses two pointers pattern hence doesn't need -3x stride
// and require smaller InStride
int getTransformedInStrideNx1(unsigned ampKernelHeight, int transformedInStride,
                              int transformedInRowStride);

// Reverse actions done by getTransformedInStrideNx1. secondPtrOffset is
// calculated by getSecondPtrOffset function
int reverseTransformedInStrideNx1(int transformedInStride, int secondPtrOffset);

} // namespace poplin

#endif // poplin_ConvPartialsStridesPacking_hpp
