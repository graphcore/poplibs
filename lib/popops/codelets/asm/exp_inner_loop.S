// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Functions to perform inner loop of exponent for half and float types

#ifdef __IPU__
#include "poplar/StackSizeDefs.hpp"

#define HALF_FUNC exponent_half_1d_inner_loop
#define FLOAT_FUNC exponent_float_1d_inner_loop

// function parameters
#define src_ptr m0
#define dst_ptr m1
#define num_samples m2
#define stride m3

.globl HALF_FUNC
.type HALF_FUNC, @function

DEF_STACK_USAGE 0 HALF_FUNC
.section .text.HALF_FUNC
.align 8

#define num_x4 m4
#define num_x2 m4
#define num_x1 m4

HALF_FUNC:
  // exponent can overflow and hence we cannot compute it on anything but data
  // that needs conversion. Otherwise we must guarantee that it is computed on
  // data that cannot overflow.
  // inner loop count
  shr $num_x4, $num_samples, 2

  brz $num_x4, .Lend_half_rem
  ld64step $a0:1, $mzero, $src_ptr+=, $stride
  {
    add $num_x4, $num_x4, -1
    f16v2exp $a2, $a0
  }
  rpt $num_x4, (2f - 1f) / 8 - 1
1:
    {
      ld64step $a0:1, $mzero, $src_ptr+=, $stride
      f16v2exp $a3, $a1
    }
    {
      st64step $a2:3, $mzero, $dst_ptr+=, $stride 
      f16v2exp $a2, $a0
    }
2:
  {
    and $num_samples, $num_samples, 0x3
    f16v2exp $a3, $a1
  }
  st64step $a2:3, $mzero, $dst_ptr+=, $stride
.Lend_half_rem:
  // quick exit as most cases are nice multiple of 4
  brz $num_samples, .Lend_half
  and $num_x2, $num_samples, 0x2
  brz $num_x2, .Lhalf_last_sample
  ld32step $a0, $mzero, $src_ptr+=, 1
  {
    and $num_samples, $num_samples, 0x1
    f16v2exp $a2, $a0
  }
  st32step $a2, $mzero, $dst_ptr+=, 1 
.Lhalf_last_sample:
  brz $num_samples, .Lend_half
  ldb16 $a0, $mzero, $src_ptr, 0
  {
    ld32 $a3, $mzero, $dst_ptr, 0
    f16v2exp $a2, $a0
  }
  sort4x16hi $a0, $a2, $a3
  st32 $a0, $mzero, $dst_ptr, 0 
.Lend_half:  
  br $lr

.size HALF_FUNC, .-HALF_FUNC

#undef num_x4
#undef num_x2
#undef num_x1


//------------------------------------------------------------------------------


.globl FLOAT_FUNC
.type FLOAT_FUNC, @function

DEF_STACK_USAGE 0 FLOAT_FUNC
.section .text.FLOAT_FUNC
.align 8

#define num_x2 m4
#define num_x1 m4

FLOAT_FUNC:
  // exponent can overflow and hence we cannot compute it on anything but data
  // that needs conversion. Otherwise we must guarantee that it is computed on
  // data that cannot overflow.

  // inner loop count
  shr $num_x2, $num_samples, 1

  brz $num_x2, .Lend_float_rem
  ld64step $a0:1, $mzero, $src_ptr+=, $stride
  {
    add $num_x2, $num_x2, -1
    f32exp $a2, $a0
  }
  rpt $num_x2, (2f - 1f) / 8 - 1
1:
    {
      ld64step $a0:1, $mzero, $src_ptr+=, $stride
      f32exp $a3, $a1
    }
    {
      st64step $a2:3, $mzero, $dst_ptr+=, $stride 
      f32exp $a2, $a0
    }
2:
  {
    and $num_samples, $num_samples, 0x1
    f32exp $a3, $a1
  }
  st64step $a2:3, $mzero, $dst_ptr+=, $stride 
.Lend_float_rem:
  brz $num_samples, .Lend_float
  ld32step $a0, $mzero, $src_ptr+=, 1
  f32exp $a2, $a0
  st32 $a2, $mzero, $dst_ptr, 0 
.Lend_float:  
  br $lr
.size FLOAT_FUNC, .-FLOAT_FUNC

#endif
