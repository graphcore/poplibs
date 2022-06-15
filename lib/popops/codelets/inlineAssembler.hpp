// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <ipu_builtins.h>

// Write 16 bits to memory assuming a 32 bit aligned destination pointer
static __attribute__((always_inline)) void write16Aligned32(half in,
                                                            half2 *outPtr) {
  // Ensure that the operand that is put into a 32 register is 32 bits in size
  half2 result = {in, in};
  half2 toPreserve = *outPtr;
  *outPtr = __builtin_shufflevector(result, toPreserve, 1, 3);
}
// Combine four 8bit values in the 8 lsbs of each input into a single 32
// bit result. bits 8..31 of the inputs are ignored
static __attribute__((always_inline)) unsigned
combine8bit(unsigned in0, unsigned in1, unsigned in2, unsigned in3) {

  auto a = reinterpret_cast<ushort2>(__builtin_ipu_shuf8x8lo(in0, in1));
  auto b = reinterpret_cast<ushort2>(__builtin_ipu_shuf8x8lo(in2, in3));
  return reinterpret_cast<unsigned>(__builtin_shufflevector(a, b, 0, 2));
}
