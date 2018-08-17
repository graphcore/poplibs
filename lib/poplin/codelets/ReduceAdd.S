#ifdef __IPU__
#include "tilearch.h"
#include "tileimplconsts_tommy.h"

#define IN_OFF      0
#define OUT_OFF     1
#define OFFSET_OFF  4
#define NUM_OFF     5

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

#define OUT_i_PTR       m0
#define OUT_j_PTR       m1
#define IN_i_PTR        m2
#define IN_j_PTR        m3
#define OUT_i_SIZE      m4
#define OUT_j_SIZE      m5
#define BASE            m6
#define NUM_PART        m7
#define SCRATCH         m8
#define IN_i_DELTA      m9
#define IN_j_DELTA      m10
#define OUT_j_ALIGN     m11
#define OFFSET_MASK     m11

#define VALUES_0        a4
#define VALUES_1        a5
#define VALUES_2        a6
#define VALUES_3        a7
#define ASCRATCH_0      a2
#define ASCRATCH_1      a3
#define ZAACC           a0

#define ReduceAdd_Half_Float __runCodelet_poplin__ReduceAdd___float_half

.section .text.Reduce_Half_Float, "ax"
// Would be useful to have an antialign directive instead of .align 8; nop;
.globl ReduceAdd_Half_Float
.type ReduceAdd_Half_Float, @function
.align 8

ReduceAdd_Half_Float:
  ld32        $BASE, $mvertex_base, $mzero, OUT_OFF
  ldz16       $OUT_i_PTR, $mvertex_base, $mzero, OFFSET_OFF

  // unpack scaled ptr to DeltaN offsets
  {
    setzi       $SCRATCH, TMEM_REGION0_BASE_ADDR
    setzi       $ZAACC, ZAACC_BITMASK
  }
  shl         $OUT_i_PTR, $OUT_i_PTR, 2
  add         $OUT_i_PTR, $OUT_i_PTR, $SCRATCH

  shr         $OUT_i_SIZE, $BASE, 20
  shl         $BASE, $BASE, 12
  shr         $BASE, $BASE, 12

  ld32        $IN_i_PTR, $mvertex_base, $mzero, IN_OFF
  ldz16       $NUM_PART, $mvertex_base, $mzero, NUM_OFF

  brnzdec     $OUT_i_SIZE, _skip_e
  bri         _exit
  _skip_e:


_loop_over_reductions:
  // Here Out j ptr is an offset:18, size:14. Offset to be added to $BASE
  ld32step    $OUT_j_PTR, $mzero, $OUT_i_PTR+=, 1
  shr         $OUT_j_SIZE, $OUT_j_PTR, 18
  shl         $OUT_j_PTR, $OUT_j_PTR, 14
  shr         $OUT_j_PTR, $OUT_j_PTR, 14
  add         $OUT_j_PTR, $OUT_j_PTR, $BASE

  {
    shr         $SCRATCH, $OUT_j_SIZE, 3
    uput        $FP_CLR, $ZAACC
  }

  setzi       $IN_j_DELTA, 0
  brnzdec     $SCRATCH, _skip_r
  bri         _partials_rpt_4_loop
  _skip_r:

  // and to see if it is 8 bytes aligned, vertex state guarentees it is 4 byte
  and         $OUT_j_ALIGN, $OUT_j_PTR, 4
_loop_over_outer_dim:
  {
    setzi       $IN_i_DELTA, 0
    zero        $VALUES_0:5
  }

_partials_rpt_8_loop:
  // If force alignment of IN_i_PTR to 16 bytes can use ld128, save cycle in rpt
  {rpt         $NUM_PART, 2; zero        $VALUES_2:7}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f16v8acc  $VALUES_0:7
  }
  {
    ld64      $VALUES_0:5, $IN_j_PTR, $IN_j_DELTA, 0
    fnop
  }
  {
    ld64      $VALUES_2:7, $IN_j_PTR, $IN_j_DELTA, 1
    fnop
  }
  // Get values out of accumulators and store
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 16
    f16v8acc    $VALUES_0:7
  }
  {
    brnz      $OUT_j_ALIGN, _misaligned_st_8
    f32v2gina $VALUES_0:5, $azeros, 0
  }
  {
    st64step    $VALUES_0:5, $mzero, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  {
    st64step    $VALUES_0:5, $mzero, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  {
    st64step    $VALUES_0:5, $mzero, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  // as $azeros fed into AACC pipeline loading out the values also zeros it
  st64step    $VALUES_0:5, $mzero, $OUT_j_PTR+=, 1
  brnzdec     $SCRATCH, _loop_over_outer_dim
_partials_rpt_8_loop_end:
// If could do floating point calculations on over read data we wouldn't
// have to treat any of the special cases at the end we could just do a
// last ld128 f16v8acc loop and only store the relevant data
_partials_rpt_4_loop:
  and         $SCRATCH, $OUT_j_SIZE, 4
  brz         $SCRATCH, _partials_rpt_2_loop

  {
    setzi       $IN_i_DELTA, 0
    zero        $VALUES_0:5
  }
  {rpt         $NUM_PART, 1; fnop}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f16v4acc  $VALUES_0:5
  }
  {
    ld64      $VALUES_0:5, $IN_j_PTR, $IN_j_DELTA, 0
    fnop
  }
  // Store values
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 8
    f16v4acc    $VALUES_0:5
  }
  {
    brnz        $OUT_j_ALIGN, _misaligned_st_4
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  {
    st64step    $VALUES_0:5, $mzero, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  st64step    $VALUES_0:5, $mzero, $OUT_j_PTR+=, 1
_partials_rpt_2_loop:
// in this loop going to accumulate values i'm not going to use. Can guarentee
// this won't introduce NANs as the values in the a registers hold the values
// from the last cycle. Advantage is automatically converts to f32
  {
    and         $SCRATCH, $OUT_j_SIZE, 2
    uput        $FP_CLR, $ZAACC
  }
  brz         $SCRATCH, _partials_rpt_1_loop
  {
    setzi       $IN_i_DELTA, 0
    zero        $VALUES_0
  }
  {rpt         $NUM_PART, 1; fnop}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f16v4acc  $VALUES_0:5
  }
  {
    ld32      $VALUES_0, $IN_j_PTR, $IN_j_DELTA, 0
    fnop
  }
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 4
    f16v4acc    $VALUES_0:5
  }
  {
    brnz      $OUT_j_ALIGN, _misaligned_st_2
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  st64step    $VALUES_0:5, $mzero, $OUT_j_PTR+=, 1
_partials_rpt_1_loop:
  and         $SCRATCH, $OUT_j_SIZE, 1
  {brz         $SCRATCH, _reduction_loop_finale; fnop}

  {
    setzi       $IN_i_DELTA, 0
    zero        $VALUES_0
  }
  {rpt         $NUM_PART, 1; zero $VALUES_2}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f16tof32  $VALUES_0, $VALUES_0
  }
  {
    ldb16     $VALUES_0, $IN_j_PTR, $IN_j_DELTA, 0
    f32add    $VALUES_2, $VALUES_2, $VALUES_0
  }
  f16tof32    $VALUES_0, $VALUES_0
  f32add      $VALUES_2, $VALUES_2, $VALUES_0
  st32step    $VALUES_2, $mzero, $OUT_j_PTR+=, 1

_reduction_loop_finale:
  add         $IN_i_PTR, $IN_i_PTR, $IN_i_DELTA
  brnzdec     $OUT_i_SIZE, _loop_over_reductions

_exit:
  exitz       $mzero


_misaligned_st_8:
  st32step      $VALUES_0, $mzero, $OUT_j_PTR+=, 1
  {
    st32step    $VALUES_1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  st32step      $VALUES_0, $mzero, $OUT_j_PTR+=, 1
  {
    st32step    $VALUES_1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  st32step      $VALUES_0, $mzero, $OUT_j_PTR+=, 1
  {
    st32step    $VALUES_1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  st32step      $VALUES_0, $mzero, $OUT_j_PTR+=, 1
  st32step      $VALUES_1, $mzero, $OUT_j_PTR+=, 1

  brnzdec     $SCRATCH, _loop_over_outer_dim
  bri         _partials_rpt_4_loop

_misaligned_st_4:
  st32step      $VALUES_0, $mzero, $OUT_j_PTR+=, 1
  {
    st32step    $VALUES_1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  st32step      $VALUES_0, $mzero, $OUT_j_PTR+=, 1
  st32step      $VALUES_1, $mzero, $OUT_j_PTR+=, 1

  bri         _partials_rpt_2_loop

_misaligned_st_2:
  st32step      $VALUES_0, $mzero, $OUT_j_PTR+=, 1
  st32step      $VALUES_1, $mzero, $OUT_j_PTR+=, 1

  bri         _partials_rpt_1_loop

.size ReduceAdd_Half_Float, .-ReduceAdd_Half_Float


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

#define ReduceAdd_Half_Half __runCodelet_poplin__ReduceAdd___half_half
.section .text.Reduce_Half_Half, "ax"
.globl ReduceAdd_Half_Half
.type ReduceAdd_Half_Half, @function
.align 8

ReduceAdd_Half_Half:
  ld32        $BASE, $mvertex_base, $mzero, OUT_OFF
  ldz16       $OUT_i_PTR, $mvertex_base, $mzero, OFFSET_OFF

  // unpack scaled ptr to DeltaN offsets
  {
    setzi       $SCRATCH, TMEM_REGION0_BASE_ADDR
    setzi       $ZAACC, ZAACC_BITMASK
  }
  shl         $OUT_i_PTR, $OUT_i_PTR, 2
  add         $OUT_i_PTR, $OUT_i_PTR, $SCRATCH

  shr         $OUT_i_SIZE, $BASE, 20
  shl         $BASE, $BASE, 12
  shr         $BASE, $BASE, 12

  ld32        $IN_i_PTR, $mvertex_base, $mzero, IN_OFF
  ldz16       $NUM_PART, $mvertex_base, $mzero, NUM_OFF

  brnzdec     $OUT_i_SIZE, _skip_e_h_h
  bri         _exit_h_h
  _skip_e_h_h:


_loop_over_reductions_h_h:
  // Here Out j ptr is an offset:18, size:14. Offset to be added to $BASE
  ld32step    $OUT_j_PTR, $mzero, $OUT_i_PTR+=, 1
  shr         $OUT_j_SIZE, $OUT_j_PTR, 18
  shl         $OUT_j_PTR, $OUT_j_PTR, 14
  shr         $OUT_j_PTR, $OUT_j_PTR, 14
  add         $OUT_j_PTR, $OUT_j_PTR, $BASE

  {
    shr         $SCRATCH, $OUT_j_SIZE, 3
    uput        $FP_CLR, $ZAACC
  }

  setzi       $IN_j_DELTA, 0
  brnzdec     $SCRATCH, _skip_r_h_h
  bri         _partials_rpt_4_loop_h_h
  _skip_r_h_h:

  // and to see if it is 8 bytes aligned, vertex state guarentees it is 4 byte
  and         $OUT_j_ALIGN, $OUT_j_PTR, 4
_loop_over_outer_dim_h_h:
  {
    setzi       $IN_i_DELTA, 0
    zero        $VALUES_0:5
  }

_partials_rpt_8_loop_h_h:
  // If force alignment of IN_i_PTR to 16 bytes can use ld128, save cycle in rpt
  {rpt         $NUM_PART, 2; zero        $VALUES_2:7}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f16v8acc  $VALUES_0:7
  }
  {
    ld64      $VALUES_0:5, $IN_j_PTR, $IN_j_DELTA, 0
    fnop
  }
  {
    ld64      $VALUES_2:7, $IN_j_PTR, $IN_j_DELTA, 1
    fnop
  }
  // Get values out of accumulators and store
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 16
    f16v8acc    $VALUES_0:7
  }
  {
    nop
    f16v2gina $VALUES_0, $azero, 0
  }
  {
    st32step    $VALUES_0, $mzero, $OUT_j_PTR+=, 1
    f16v2gina   $VALUES_0, $azero, 0
  }
  {
    st32step    $VALUES_0, $mzero, $OUT_j_PTR+=, 1
    f16v2gina   $VALUES_0, $azero, 0
  }
  {
    st32step    $VALUES_0, $mzero, $OUT_j_PTR+=, 1
    f16v2gina   $VALUES_0, $azero, 0
  }
  // as $azeros fed into AACC pipeline loading out the values also zeros it
  st32step    $VALUES_0, $mzero, $OUT_j_PTR+=, 1
  brnzdec     $SCRATCH, _loop_over_outer_dim_h_h
_partials_rpt_8_loop_end_h_h:
// If could do floating point calculations on over read data we wouldn't
// have to treat any of the special cases at the end we could just do a
// last ld128 f16v8acc loop and only store the relevant data
_partials_rpt_4_loop_h_h:
  and         $SCRATCH, $OUT_j_SIZE, 4
  brz         $SCRATCH, _partials_rpt_2_loop_h_h
  {
    setzi       $IN_i_DELTA, 0
    zero        $VALUES_0:5
  }
  {rpt         $NUM_PART, 1; fnop}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f16v4acc  $VALUES_0:5
  }
  {
    ld64      $VALUES_0:5, $IN_j_PTR, $IN_j_DELTA, 0
    fnop
  }
  // Store values
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 8
    f16v4acc    $VALUES_0:5
  }
  {
    // brnz        $OUT_j_ALIGN, _misaligned_st_4_h_h
    nop
    f16v2gina   $VALUES_0, $azero, 0
  }
  {
    st32step    $VALUES_0, $mzero, $OUT_j_PTR+=, 1
    f16v2gina   $VALUES_0, $azero, 0
  }
  st32step    $VALUES_0, $mzero, $OUT_j_PTR+=, 1
_partials_rpt_2_loop_h_h:
// in this loop going to accumulate values i'm not going to use. Can guarentee
// this won't introduce NANs as the values in the a registers hold the values
// from the last cycle. Advantage is automatically converts to f32
  {
    and         $SCRATCH, $OUT_j_SIZE, 2
    uput        $FP_CLR, $ZAACC
  }
  brz         $SCRATCH, _partials_rpt_1_loop_h_h
  {
    setzi       $IN_i_DELTA, 0
    zero        $VALUES_0
  }
  {rpt         $NUM_PART, 1; fnop}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f16v4acc  $VALUES_0:5
  }
  {
    ld32      $VALUES_0, $IN_j_PTR, $IN_j_DELTA, 0
    fnop
  }
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 4
    f16v4acc    $VALUES_0:5
  }
  {
    nop
    f16v2gina   $VALUES_0, $azero, 0
  }
  st32step    $VALUES_0, $mzero, $OUT_j_PTR+=, 1
_partials_rpt_1_loop_h_h:
  and         $SCRATCH, $OUT_j_SIZE, 1
  {brz         $SCRATCH, _reduction_loop_finale_h_h; fnop}

  {
    setzi       $IN_i_DELTA, 0
    zero        $VALUES_0
  }
  {rpt         $NUM_PART, 1; zero $VALUES_2}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f16tof32  $VALUES_0, $VALUES_0
  }
  {
    ldb16     $VALUES_0, $IN_j_PTR, $IN_j_DELTA, 0
    f32add    $VALUES_2, $VALUES_2, $VALUES_0
  }
  f16tof32    $VALUES_0, $VALUES_0
  f32add      $VALUES_2, $VALUES_2, $VALUES_0
  f32tof16     $VALUES_2, $VALUES_2
  ldb16       $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
  sort4x16lo  $VALUES_2, $VALUES_2, $ASCRATCH_0
  st32step    $VALUES_2, $mzero, $OUT_j_PTR+=, 1

_reduction_loop_finale_h_h:
  add         $IN_i_PTR, $IN_i_PTR, $IN_i_DELTA
  brnzdec     $OUT_i_SIZE, _loop_over_reductions_h_h

_exit_h_h:
  exitz       $mzero

.size ReduceAdd_Half_Half, .-ReduceAdd_Half_Half



//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

#define ReduceAdd_Float_Half __runCodelet_poplin__ReduceAdd___half_float
.section .text.Reduce_Float_Half, "ax"
.globl ReduceAdd_Float_Half
.type ReduceAdd_Float_Half, @function
.align 8
ReduceAdd_Float_Half:
  ld32        $BASE, $mvertex_base, $mzero, OUT_OFF
  ldz16       $OUT_i_PTR, $mvertex_base, $mzero, OFFSET_OFF

  // unpack scaled ptr to DeltaN offsets
  {
    setzi       $SCRATCH, TMEM_REGION0_BASE_ADDR
    setzi       $ZAACC, ZAACC_BITMASK
  }
  shl         $OUT_i_PTR, $OUT_i_PTR, 2
  add         $OUT_i_PTR, $OUT_i_PTR, $SCRATCH

  shr         $OUT_i_SIZE, $BASE, 20
  shl         $BASE, $BASE, 12
  shr         $BASE, $BASE, 12

  ld32        $IN_i_PTR, $mvertex_base, $mzero, IN_OFF
  ldz16       $NUM_PART, $mvertex_base, $mzero, NUM_OFF
  setzi       $OFFSET_MASK, 0x3FFFF

  brnzdec     $OUT_i_SIZE, _skip_e_f_h
  bri         _exit_f_h
  _skip_e_f_h:


_loop_over_reductions_f_h:
  // Here Out j ptr is an offset:18, size:14. Offset to be added to $BASE
  ld32step    $OUT_j_PTR, $mzero, $OUT_i_PTR+=, 1
  shr         $OUT_j_SIZE, $OUT_j_PTR, 18
  and         $OUT_j_PTR, $OUT_j_PTR, $OFFSET_MASK
  setzi       $IN_i_DELTA, 0
  {
    shr         $SCRATCH, $OUT_j_SIZE, 2
    uput        $FP_CLR, $ZAACC
  }
  {
    setzi       $IN_j_DELTA, 0
    zero        $VALUES_0:5
  }
  {
    brnzdec     $SCRATCH, _skip_r_f_h
    zero        $VALUES_2:7
  }
  bri         _partials_rpt_2_loop_f_h
  _skip_r_f_h:

_loop_over_outer_dim_f_h:

_partials_rpt_8_loop_f_h:
  // If force alignment of IN_i_PTR to 16 bytes can use ld128, save cycle in rpt
  rpt         $NUM_PART, 2
    {
      ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
      f32v4acc  $VALUES_0:7
    }
    {
      ld64      $VALUES_0:5, $IN_j_PTR, $IN_j_DELTA, 0
      fnop
    }
    {
      ld64      $VALUES_2:7, $IN_j_PTR, $IN_j_DELTA, 1
      fnop
    }
  // Get values out of accumulators and store
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 16
    f32v4acc    $VALUES_0:7
  }
  {
    setzi       $IN_i_DELTA, 0
    f16v2gina   $VALUES_0, $azero, 0
  }
  {
    st32step    $VALUES_0, $BASE, $OUT_j_PTR+=, 1
    f16v2gina   $VALUES_0, $azero, 0
  }
  {
    st32step    $VALUES_0, $BASE, $OUT_j_PTR+=, 1
    zero        $VALUES_0:5
  }
  {
    brnzdec     $SCRATCH, _loop_over_outer_dim_f_h
    zero        $VALUES_2:7
  }

_partials_rpt_8_loop_end_f_h:
_partials_rpt_2_loop_f_h:
// in this loop going to accumulate values i'm not going to use. Can guarentee
// this won't introduce NANs as the values in the a registers hold the values
// from the last cycle. Advantage is automatically converts to f32
  and         $SCRATCH, $OUT_j_SIZE, 2
  brz         $SCRATCH, _partials_rpt_1_loop_f_h
  {rpt         $NUM_PART, 1; fnop}
    {
      ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
      f32v2add  $VALUES_2:7, $VALUES_2:7, $VALUES_0:5
    }
    {
      ld64      $VALUES_0:5, $IN_j_PTR, $IN_j_DELTA, 0
      fnop
    }
  {
    add       $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2add  $VALUES_2:7, $VALUES_2:7, $VALUES_0:5
  }
  {
    setzi       $IN_i_DELTA, 0
    f32v2tof16  $VALUES_2, $VALUES_2:7
  }
  {
    st32step    $VALUES_2, $BASE, $OUT_j_PTR+=, 1
    zero        $VALUES_0:5
  }

_partials_rpt_1_loop_f_h:
  and         $SCRATCH, $OUT_j_SIZE, 1
  brz         $SCRATCH, _reduction_loop_finale_f_h
  {rpt         $NUM_PART, 1; fnop}
    {
      ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
      fnop
    }
    {
      ld32     $VALUES_0, $IN_j_PTR, $IN_j_DELTA, 0
      f32add    $VALUES_1, $VALUES_1, $VALUES_0
    }
  f32add      $VALUES_1, $VALUES_1, $VALUES_0
  {
    ldb16       $ASCRATCH_0, $BASE, $OUT_j_PTR,  1
    f32tof16     $VALUES_1, $VALUES_1
  }
  sort4x16lo  $VALUES_1, $VALUES_1, $ASCRATCH_0
  st32step    $VALUES_1, $BASE, $OUT_j_PTR+=, 1

_reduction_loop_finale_f_h:
  // dummy load to move pointer
  ld32step    $mzero, $mzero, $IN_i_PTR+=, $NUM_PART
  brnzdec     $OUT_i_SIZE, _loop_over_reductions_f_h

_exit_f_h:
  exitz       $mzero

.size ReduceAdd_Float_Half, .-ReduceAdd_Float_Half


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

#define ReduceAdd_Float_Float __runCodelet_poplin__ReduceAdd___float_float
.section .text.Reduce_Float_Float, "ax"
.globl ReduceAdd_Float_Float
.type ReduceAdd_Float_Float, @function
.align 8
ReduceAdd_Float_Float:
  ld32        $BASE, $mvertex_base, $mzero, OUT_OFF
  ldz16       $OUT_i_PTR, $mvertex_base, $mzero, OFFSET_OFF

  // unpack scaled ptr to DeltaN offsets
  {
    setzi       $SCRATCH, TMEM_REGION0_BASE_ADDR
    setzi       $ZAACC, ZAACC_BITMASK
  }
  shl         $OUT_i_PTR, $OUT_i_PTR, 2
  add         $OUT_i_PTR, $OUT_i_PTR, $SCRATCH

  shr         $OUT_i_SIZE, $BASE, 20
  shl         $BASE, $BASE, 12
  shr         $BASE, $BASE, 12

  ld32        $IN_i_PTR, $mvertex_base, $mzero, IN_OFF
  ldz16       $NUM_PART, $mvertex_base, $mzero, NUM_OFF

  setzi       $OFFSET_MASK, 0x3FFFF

  brnzdec     $OUT_i_SIZE, _skip_e_f_f
  bri         _exit_f_f
  _skip_e_f_f:


_loop_over_reductions_f_f:
  // Here Out j ptr is an offset:18, size:14. Offset to be added to $BASE
  ld32step    $OUT_j_PTR, $mzero, $OUT_i_PTR+=, 1
  shr         $OUT_j_SIZE, $OUT_j_PTR, 18
  and         $OUT_j_PTR, $OUT_j_PTR, $OFFSET_MASK
  {
    shr         $SCRATCH, $OUT_j_SIZE, 2
    uput        $FP_CLR, $ZAACC
  }
  {
    setzi       $IN_j_DELTA, 0
    zero        $VALUES_0:5
  }

  {
    brnzdec     $SCRATCH, _skip_r_f_f
    zero        $VALUES_2:7
  }

  bri         _partials_rpt_2_loop_f_f
  _skip_r_f_f:

_loop_over_outer_dim_f_f:
  setzi       $IN_i_DELTA, 0

_partials_rpt_8_loop_f_f:
  // If force alignment of IN_i_PTR to 16 bytes can use ld128, save cycle in rpt
  rpt         $NUM_PART, 2
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f32v4acc  $VALUES_0:7
  }
  {
    ld64      $VALUES_0:5, $IN_j_PTR, $IN_j_DELTA, 0
    fnop
  }
  {
    ld64      $VALUES_2:7, $IN_j_PTR, $IN_j_DELTA, 1
    fnop
  }
  // Get values out of accumulators and store
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 16
    f32v4acc    $VALUES_0:7
  }
  {
    // brnz      $OUT_j_ALIGN, _misaligned_st_8
    nop
    f32v2gina $VALUES_0:5, $azeros, 0
  }
  st32step    $VALUES_0, $BASE, $OUT_j_PTR+=, 1
  {
    st32step    $VALUES_1, $BASE, $OUT_j_PTR+=, 1
    f32v2gina   $VALUES_0:5, $azeros, 0
  }
  // as $azeros fed into AACC pipeline loading out the values also zeros it
  {
    st32step    $VALUES_0, $BASE, $OUT_j_PTR+=, 1;
    zero        $VALUES_2:7
  }
  {
    st32step    $VALUES_1, $BASE, $OUT_j_PTR+=, 1
    zero        $VALUES_0:5
  }
  brnzdec     $SCRATCH, _loop_over_outer_dim_f_f
_partials_rpt_8_loop_end_f_f:
_partials_rpt_2_loop_f_f:
// in this loop going to accumulate values i'm not going to use. Can guarentee
// this won't introduce NANs as the values in the a registers hold the values
// from the last cycle. Advantage is automatically converts to f32
  and         $SCRATCH, $OUT_j_SIZE, 2
  brz         $SCRATCH, _partials_rpt_1_loop_f_f
  setzi       $IN_i_DELTA, 0
  rpt         $NUM_PART, 1
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    f32v2add  $VALUES_2:7, $VALUES_2:7, $VALUES_0:5
  }
  {
    ld64      $VALUES_0:5, $IN_j_PTR, $IN_j_DELTA, 0
    fnop
  }
  {
    add         $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2add    $VALUES_2:7, $VALUES_2:7, $VALUES_0:5
  }
  {
    st32step    $VALUES_2, $BASE, $OUT_j_PTR+=, 1
    zero        $VALUES_0:5
  }
  st32step    $VALUES_3, $BASE, $OUT_j_PTR+=, 1
_partials_rpt_1_loop_f_f:
  and         $SCRATCH, $OUT_j_SIZE, 1
  brz         $SCRATCH, _reduction_loop_finale_f_f
  setzi       $IN_i_DELTA, 0
  {rpt         $NUM_PART, 1; fnop}
  {
    ld32step  $IN_j_PTR, $IN_i_PTR, $IN_i_DELTA+=, 1
    fnop
  }
  {
    ld32     $VALUES_0, $IN_j_PTR, $IN_j_DELTA, 0
    f32add    $VALUES_1, $VALUES_1, $VALUES_0
  }
  f32add      $VALUES_1, $VALUES_1, $VALUES_0
  st32step    $VALUES_1, $BASE, $OUT_j_PTR+=, 1

_reduction_loop_finale_f_f:
  // dummy load to move pointer
  ld32step    $mzero, $mzero, $IN_i_PTR+=, $NUM_PART
  brnzdec     $OUT_i_SIZE, _loop_over_reductions_f_f

_exit_f_f:
  exitz       $mzero

.size ReduceAdd_Float_Float, .-ReduceAdd_Float_Float






#endif // __IPU__
