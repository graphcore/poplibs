// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Common definitions for all types of Cast vertices

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

// Registers used as input for the 'cast_<INTYPE>_<INTYPE>_core' functions
#define IN_PTR                m3
#define OUT_PTR               m2
#define N_ELEMS               m4


// When converting from float or half to signed/unsigned chat, the input value
// could be outside the range of the output (-128..127 or 0..255)
// The definitions and macros below define what we do in this case (clamp the
// input to the range)
#define CLAMP_MIN             a6
#define CLAMP_MAX             a7
#define CLAMP_LIMITS          a6:7

// Setup of clamping limit registers
.macro SETUP_UCHAR_OVERFLOW_CHECK
  setzi       $CLAMP_MIN, 0
  setzi       $CLAMP_MAX, 255.0h
  f16tof32    $CLAMP_MAX, $CLAMP_MAX
.endm

.macro SETUP_SCHAR_OVERFLOW_CHECK
  setzi       $CLAMP_MIN, -128.0h
  f16tof32    $CLAMP_MIN, $CLAMP_MIN
  setzi       $CLAMP_MAX, 127.0h
  f16tof32    $CLAMP_MAX, $CLAMP_MAX
.endm

#endif // __IPU__
