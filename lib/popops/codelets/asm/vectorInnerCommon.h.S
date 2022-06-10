// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Stack space used to store all context needed for an outer loop
#define SCRATCH_OFFSET_OUTER_FACTOR 0
#define SCRATCH_OFFSET_ACTS 1
#define SCRATCH_OFFSET_OUT 2
#define SCRATCH_OFFSET_BLOCK_COUNT 5
#define SCRATCH_OFFSET_ADDEND 6

#include "workDivision.h.S"

.macro VECTOR_INNER_OUTER_LOOP_WRAPPER INNER_FUNCTION INCREMENT_INSTRUCTION ACTS OUT ADDEND_LEN ADDEND BLOCK_COUNT
  st32 \BLOCK_COUNT, $mworker_base, $mzero, SCRATCH_OFFSET_BLOCK_COUNT
3:
  st32 $remaining_blocks, $mworker_base, $mzero, SCRATCH_OFFSET_OUTER_FACTOR
  // Preserve the pointers for the next loop
  st32 \ACTS, $mworker_base, $mzero, SCRATCH_OFFSET_ACTS
  st32 \OUT, $mworker_base, $mzero, SCRATCH_OFFSET_OUT
  st32 \ADDEND, $mworker_base, $mzero, SCRATCH_OFFSET_ADDEND

  call $lr, \INNER_FUNCTION

  // Load and advance the addend pointer
  ldz16 \ADDEND_LEN, $mvertex_base, $mzero, VERTEX_DATA_BSLICE_LEN_OFFSET
  ld32 \ADDEND, $mworker_base, $mzero, SCRATCH_OFFSET_ADDEND
  \INCREMENT_INSTRUCTION $azero, $mzero, \ADDEND+=, \ADDEND_LEN

  // Load and advance the in,out pointers
  ldz16 $mscratch0, $mvertex_base, $mzero, VERTEX_DATA_BBROADCAST_FACTOR_OFFSET
  mul  $mscratch0, \ADDEND_LEN, $mscratch0
  ld32 \ACTS, $mworker_base, $mzero, SCRATCH_OFFSET_ACTS
  ld32 \OUT, $mworker_base, $mzero, SCRATCH_OFFSET_OUT

  \INCREMENT_INSTRUCTION $azero, $mzero, \ACTS+=, $mscratch0
  \INCREMENT_INSTRUCTION $azero, $mzero, \OUT+=, $mscratch0

  ld32 \BLOCK_COUNT, $mworker_base, $mzero, SCRATCH_OFFSET_BLOCK_COUNT

  // Loop count
  ld32 $remaining_blocks, $mworker_base, $mzero, SCRATCH_OFFSET_OUTER_FACTOR
  brnzdec $remaining_blocks, 3b
.endm

#endif
