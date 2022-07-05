// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

#define VERTEX_ADD_HALF_SCALE __runCodelet_popops__ScaledAdd2D___half_float_half_false
#define VERTEX_ADD_FLOAT_SCALE __runCodelet_popops__ScaledAdd2D___half_float_float_false

#define VERTEX_COMMON __ScaledAdd2D___half_float_common


// constants
// Vertex state offsets in bytes
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 4
#define VERTEX_DATA_B_OFFSET 8
#define VERTEX_SCALE_OFFSET 12

// integer variables
#define outData m0
#define outDataSize m1
#define outDataB m2
#define dataA m3
#define dataSize m4
#define dataSizeD4 m5
#define subVCalc m5
#define dataB m6
#define origDataSize m7

// float variables
// aux register aliases
#define aBScale a2    //f32v1

#define aA01f   a0:1
#define aA0123h             a6:7
#define aA01h               a6
#define aA23h                 a7

#define aB01f       a2:3
#define aB0f        a2
#define aB1f          a3
#define aB23f           a4:5

#define aTmpfA          a4:5
#define aTmpf       a2:3

#define aR01f   a0:1
#define aR0123h a0:1
#define aR01h   a0 
#define aR23f       a2:3
#define aR0123f a0:3
#define aR23h     a1

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif


FN_WORKER_ENTRY_POINT VERTEX_ADD_FLOAT_SCALE
  // load vertex state specific to this version of the vertex : Tensor(float): via a pointer
  ld32  $m0, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ld32  $aBScale, $mzero, $m0, 0
  bri   VERTEX_COMMON
FN_SIZE VERTEX_ADD_FLOAT_SCALE

FN_WORKER_ENTRY_POINT VERTEX_ADD_HALF_SCALE
  // load vertex state specific to this version of the vertex : Tensor(half): via a pointer
  ld32  $dataA, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/4
  ldb16  $aBScale, $mzero, $dataA, 0
  {bri   VERTEX_COMMON
   f16tof32 $aBScale, $aBScale}
FN_SIZE VERTEX_ADD_HALF_SCALE

FN_SECTION VERTEX_COMMON 8
VERTEX_COMMON:
  // load vertex state
 {ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
    uput $TAS, $aBScale}
 {ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET/4
    setzi $a0, ZAACC_BITMASK}
 {ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4
    uput $FP_CLR, $a0}
  // minus 1 for the outer loop brnzdec
  add $outDataSize, $outDataSize, -1

.Louter_loop:
#ifdef VECTOR_AVAIL_SHORT_SPAN
  ld32step $dataA, $mzero, $outData+=, 1
  shr $origDataSize, $dataA, SHORT_SPAN_PTR_SIZE
  shl $dataA, $dataA, SHORT_SPAN_LENGTH_SIZE
  shr $dataA, $dataA, SHORT_SPAN_LENGTH_SIZE
#else
  ld32step $dataA, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
#endif

  ld32step $dataB, $mzero, $outDataB+=, 1

  // process 4 at a time first as this is the optimal scenario
  shr $dataSizeD4, $origDataSize, 2
  brz $dataSizeD4, .Lvector4_loop_end

  ld64            $aA0123h, $mzero,   $dataA, 0                                 //                   ^^ ^^ # 
 {ld64step        $aB01f,   $mzero,   $dataB+=, 1                               //       ^^ ^^             # 
    f16v2tof32    $aTmpfA,  $aA01h}                                             //             ^^ ^^ vv    # 
 {ld32            $aA01h,   $mzero,   $dataA, 2                                 //                   ^^    #                      
    f32v2axpy     $azeros,  $aB01f,   $aTmpfA}                                  //       vv vv vv vv       #
 {ld64step        $aB23f,   $mzero,   $dataB+=, 1                               //             ^^ ^^       # 
    f16v2tof32    $aTmpf,   $aA23h}                                             //       ^^ ^^          vv # 
  // Unrolling
  add $dataSizeD4, $dataSizeD4, -1
  {rpt $dataSizeD4, (2f-1f)/8-1
      f32v2axpy     $aR01f,     $aB23f,   $aTmpf}                                 // ^^ ^^ vv vv vv vv       #

1:
  # A registers                                                                 // a0 a1 a2 a3 a4 a5 a6 a7 

 {ld64step        $aB01f,   $mzero,   $dataB+=, 1                               //       ^^ ^^             #
    f16v2tof32    $aTmpfA,  $aA01h}                                             //             ^^ ^^ vv    #
 {ld32            $aA23h,   $mzero,   $dataA, 3                                 //                      ^^ #                                           
    f32v2axpy     $aR23f,   $aB01f,   $aTmpfA}                                  //       ++ ++ vv vv       #
 {ld64step        $aB23f,   $mzero,   $dataB+=, 1                               //             ^^ ^^       #
    f32v4tof16    $aR0123h, $aR0123f}                                           // ++ ++ vv vv             # 
 {st64step        $aR0123h, $mzero,   $dataA+=, 1                               // vv vv                   #  
    f16v2tof32    $aTmpf,   $aA23h}                                             //       ^^ ^^          vv #   
 {ld32            $aA01h,   $mzero,   $dataA, 2                                 //                   ^^    #       
    f32v2axpy     $aR01f,   $aB23f,   $aTmpf}                                   // ^^ ^^ vv vv vv vv       #            
2:
  // flush
    f32v2tof16    $aR01h,   $aR01f                                              // vv vv             ^^                   

    f16v2gina     $aR23h,   $azero,   0                                         //                      ^^             
  st64step        $aR0123h, $mzero,   $dataA+=, 1                               //                   vv vv
  // All full/4 vectors have now been processed and stored.
.Lvector4_loop_end:
  // Any remaining partials must be loaded.
  and $subVCalc, $origDataSize, 0x2
  brz $subVCalc, .LhandleLastElement

  // process next 32bit of result
  ld32            $aA01h,   $mzero,   $dataA, 0                                 //             ^^
 {ld64step        $aB01f,   $mzero,   $dataB+=, 1                               //       ^^ ^^    
    f16v2tof32    $aA01f,   $aA01h}                                             // ^^ ^^       vv
    f32v2axpy     $azeros,  $aB01f,   $aA01f                                    // vv vv       vv vv
    f16v2gina     $aR01h,   $azero,   0                                         // ^^
  st32step        $aR01h,   $mzero,   $dataA+=, 1                               // vv

.LhandleLastElement:
  // how many left do we have? maximum of 1.
  and $subVCalc, $origDataSize, 0x1
  brz $subVCalc, .LouterEnd
 // Note the subword off-end value may be a NaN so we mustn't do fp operations
 // on it.
 {ldb16           $aA01h,   $mzero,   $dataA, 0                                 //             ^^
    setzi         $aB1f,    0}                                                  //          ^^    
 {ld32            $aB0f,    $mzero,   $dataB, 0                                 //       ^^
    f16v2tof32    $aA01f,   $aA01h}                                             // ^^ ^^       vv
    f32v2axpy     $azeros,  $aB01f,   $aA01f                                    // vv vv vv vv
 {ldb16           $aA01h,   $mzero,   $dataA, 1    // [ pastend | pastend ]     //             ^^
    f16v2gina     $aR01h,   $azero,   0}             // [result | result ]      // ^^

  sort4x16lo      $aR01h,    $aR01h,  $aA01h     // [ value   | 0]              // ++          vv         
  st32            $aR01h,    $mzero,  $dataA, 0                                 // vv

.LouterEnd:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

FN_SIZE VERTEX_COMMON

#endif // __IPU__
