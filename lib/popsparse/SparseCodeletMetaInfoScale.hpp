// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// In the Sparse meta information the row and column
// references are scaled.  This scaling can be broken
// down into 2 parts.  Firstly scaled by the size of the zSplit
// and then by the type size. This is helpful in addressing memory for the
// mat-mul but not for the multi slice.
// After all scaling is applied we have a value that fits in an
// unsigned short.
//
// The method of dividing by multiplication followed by a right
// shift is used.
// result = quotient / divisor
// Calculated as:
// result = (q * reciprocalMulFactor(divisor))>>reciprocalMulShift
//
// There will be no remainder as each number was generated as a product to start
// with, using:
// d = [1, 65535], q = n * d, q < 65535
// (q for quotient, d for divisor)
//
// Given the range of values a shift of 16
// is sufficient for accuracy and that choice ensures that the intermediate
// value a * reciprocalMulFactor fits into 32 bits

static constexpr unsigned reciprocalMulShift = 16;

// Compute the factor to multiply by
static inline unsigned reciprocalMulFactor(unsigned zFactor) {
  return 1 + (1 << reciprocalMulShift) / zFactor;
}

// Apply the factor and shift to divide
static inline unsigned reciprocalMulDiv(unsigned input,
                                        unsigned reciprocalMulFactor) {
  return (input * reciprocalMulFactor) >> reciprocalMulShift;
}
