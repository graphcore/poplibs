// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
// Codelets to pad and normalise 3channel image tensors

#ifdef __IPU__
#include "poplar/StackSizeDefs.hpp"
#include "poplar/TileConstants.hpp"
#include "CommonPoplibsMacros.h.S"
// -----------------------------------------------------------------------------

#define NORM_IMAGE_UINT8 __runCodelet_popops__NormaliseImage___unsigned_char_half
// Constants: Vertex state
// The first two are used for 2D and all are used for 1D
#define V_IN_OS               0 // bytes
#define V_OUT_OS              4
#define V_SCALES_OS           8
#define V_OFFSETS_OS         12
#define V_INSCALE_OS         16
#define V_PACKED_N_PIXELS_OS 20


#define mIn             m3
#define mOut            m4
#define mNumPixels      m5
#define mWid            m6
#define mTemp           m7

#define aScales         a2:3
#define aScales1        a3
#define aOffsets        a4:5
#define aOffsets1       a5
#define a4096           a6
#define aInScale        a7


// -----------------------------------------------------------------------------
FN_WORKER_ENTRY_POINT NORM_IMAGE_UINT8 8
  ld32 $mIn,  $mvertex_base, $mzero, V_IN_OS/4
  ld32 $mOut, $mvertex_base, $mzero, V_OUT_OS/4
  get  $mWid, $WSR
  and  $mWid, $mWid,         CSR_W_WSR__CTXTID_M1__MASK
  // advance mIn and mOut to this worker's first input/output
  ld64step  $azeros, $m15, $mOut+=, $mWid   // mOut+=8B/worker
  ldz16step $mzero,  $m15, $mIn+=,  $mWid   // mIn+=2B/worker
  ldz8step  $mzero,  $m15, $mIn+=,  $mWid   // mIn+=B/worker

  ld32      $mNumPixels, $mvertex_base, $mzero, V_PACKED_N_PIXELS_OS/4
  and       $mTemp,      $mNumPixels,   0x7
  shr       $mNumPixels, $mNumPixels,   3
  cmpult    $mWid,       $mWid,         $mTemp
  add       $mNumPixels, $mNumPixels,   $mWid
  // Don't process any off-end data as it could cause fp exceptions
  brz $mNumPixels, .Lreturn
  // $mNumPixels now contains the number of pixels this worker will process

  // load scales and offsets for the 3 channels and zero for the 4th/padding
  // channel
  ld32        $mTemp,     $mvertex_base, $mzero, V_SCALES_OS/4
  ld64        $aScales,   $mTemp,        $mzero, 0
  sort4x16lo  $aScales1,  $aScales1,     $azero
  ld32        $mTemp,     $mvertex_base, $mzero, V_OFFSETS_OS/4
  ld64        $aOffsets,  $mTemp,        $mzero, 0
  sort4x16lo  $aOffsets1, $aOffsets1,    $azero
  ldb16       $aInScale,  $mvertex_base, $mzero, V_INSCALE_OS/2
  setzi       $a4096,     0x6c00 // half 4096==2^12

  // Each uint8 is loaded into the lsbs which makes it a denorm, where
  // 1lsb=2^-24. So we multiply by 2^24 to finish the conversion to half.
  ldz8step  $m0, $m15, $mIn+=, 1
  ldz8step  $m1, $m15, $mIn+=, 1
  ldz8step  $m2, $m15, $mIn+=, CTXT_WORKERS*3-2
  sort8x8lo $m0, $m0,  $m1
  st32      $m0, $m15, $mworker_base, 0
  st32      $m2, $m15, $mworker_base, 1

  rpt $mNumPixels, (2f-1f)/8-1
1:
  {ld64      $a0:1, $m15, $mworker_base, 0;       fnop}
  {ldz8step  $m0,   $m15, $mIn+=, 1;              fnop}
  {ldz8step  $m1,   $m15, $mIn+=, 1;              f16v4mul $a0:1, $a4096:BL,    $a0:1}  // * 2^12
  {ldz8step  $m2,   $m15, $mIn+=, 1+5*3;          f16v4mul $a0:1, $a4096:BL,    $a0:1}  // * 2^12
  {sort8x8lo $m0,   $m0,  $m1;                    f16v4mul $a0:1, $aInScale:BL, $a0:1}   // * inScale
  {st32 $m0, $m15,  $mworker_base, 0;             f16v4sub $a0:1, $a0:1,        $aOffsets} // - offsets
  {st32 $m2, $m15,  $mworker_base, 1;             f16v4mul $a0:1, $aScales,     $a0:1}     // * scales
  {st64step  $a0:1, $m15, $mOut+=, CTXT_WORKERS;  fnop}
2:
.Lreturn:
  exitnz $m15
FN_SIZE NORM_IMAGE_UINT8

#endif
