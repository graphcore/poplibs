// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Write 16 bits to memory assuming a 32 bit aligned destination pointer
static __attribute__((always_inline)) void write16Aligned32(half in,
                                                            half2 *outPtr) {
  // Ensure that the operand that is put into a 32 register is 32 bits in size
  half2 result = {in, in};
  asm volatile("  ldb16 $a0, $mzero, %[h2Out], 1\n"
               "  sort4x16lo $a0, %[result], $a0\n"
               "  st32  $a0, $mzero, %[h2Out],0\n"
               :
               : [result] "r"(result), [h2Out] "r"(outPtr)
               : "$a0", "memory");
}

// Combine four 8bit values in the 8 lsbs of each input into a single 32
// bit result. bits 8..31 of the inputs are ignored
static __attribute__((always_inline)) unsigned
combine8bit(unsigned in0, unsigned in1, unsigned in2, unsigned in3) {
  unsigned out;
  asm volatile(" shuf8x8lo $m1, %[in0], %[in1]\n"
               " shuf8x8lo $m0, %[in2], %[in3]\n"
               " sort4x16lo %[out], $m1, $m0\n"
               : [out] "+r"(out)
               : [in0] "r"(in0), [in1] "r"(in1), [in2] "r"(in2), [in3] "r"(in3)
               : "$m0", "$m1");
  return out;
}
