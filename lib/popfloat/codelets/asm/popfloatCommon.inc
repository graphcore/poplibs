#ifdef __IPU__
#ifndef POPFLOAT_COMMON_H
#define POPFLOAT_COMMON_H

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"

// supervisor variables
#define mWorkerEntry              m6

.macro POPFLOAT_SUPERVISOR_CAST_OP wrkEntry
  setzi   $mWorkerEntry  , \wrkEntry
  runall  $mWorkerEntry  , $m0                   , 0
  sync    TEXCH_SYNCZONE_LOCAL
  br      $lr
.endm

.macro POPFLOAT_GET_WORKER_INDEX workerIdx
  get     \workerIdx, $WSR
  and     \workerIdx, \workerIdx, CSR_W_WSR__CTXTID_M1__MASK
.endm

.macro POPFLOAT_MAYBE_LOAD_SCALED_PTR params baseAddr offset
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16   \params, \baseAddr, $mzero, \offset
#else
  ld32    \params, \baseAddr, $mzero, \offset
#endif
.endm

.macro POPFLOAT_CONVERT_SCALED_PTR64_TO_PTR params
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  shl     \params, \params, 3
#endif
.endm

.macro POPFLOAT_CONVERT_SCALED_PTR32_TO_PTR params tmem_base
#if defined(VECTOR_AVAIL_SCALED_PTR32)
  add     \params, \params, \tmem_base
  shl     \params, \params, 2
#endif
.endm


#endif
#endif
