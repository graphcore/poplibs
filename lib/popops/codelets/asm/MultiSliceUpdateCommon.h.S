// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef MultiSliceUpdateCommon_h_S
#define MultiSliceUpdateCommon_h_S
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"


// Update worker offset and sizes
// Updated/new outputs :
//   - $baseOffset, $wkrStart, $numBaseElements
// Given the baseOffset and numBaseElements for this supervisor vertex, computes
// the baseOffset and numBaseElements for this worker. This it does by using
// the "maxElementsPerWorker" vertex field. Workers have atmost "maxElementsPerWorker"
// elements in the sliced dimension.
.macro UPDATE_OFFSET_AND_SIZES NUM_BASE_ELEMENTS, MAX_ELEMENTS_PER_WKR, WKR_ID, WKR_START, WKR_END, BASE_ELEMENTS, S_NUM_BASE_ELEMENTS_OFFSET, S_ELEMENTS_PER_WORKER, S_BASE_OFFSET_OFFSET
  ld32 \NUM_BASE_ELEMENTS, $mzero, $mvertex_base, (\S_NUM_BASE_ELEMENTS_OFFSET)/4
  ld32 \MAX_ELEMENTS_PER_WKR, $mzero, $mvertex_base, (\S_ELEMENTS_PER_WORKER)/4
  get \WKR_ID, $WSR
  and \WKR_ID, \WKR_ID, CSR_W_WSR__CTXTID_M1__MASK
  mul \WKR_START, \WKR_ID, \MAX_ELEMENTS_PER_WKR
  min \WKR_START, \WKR_START, \NUM_BASE_ELEMENTS
  add \WKR_END, \WKR_START, \MAX_ELEMENTS_PER_WKR
  min \WKR_END, \WKR_END, \NUM_BASE_ELEMENTS
  sub \NUM_BASE_ELEMENTS, \WKR_END, \WKR_START
  ld32 \BASE_ELEMENTS, $mzero, $mvertex_base, (\S_BASE_OFFSET_OFFSET)/4
  add  \BASE_ELEMENTS, \BASE_ELEMENTS, \WKR_START
.endm


#endif // __IPU__
#endif // MultiSliceUpdateCommon_h_S
