// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Name mangling

#define VERTEX_NAME __runCodelet_popops__ForLoopCounter___\TYPE

// Vertex state
#define VERTEX_STATE_OFFSET_COUNT 0
#define VERTEX_STATE_OFFSET_LIMIT 4
#define VERTEX_STATE_OFFSET_COMPARISON_RESULT 8
#define VERTEX_STATE_OFFSET_INCREMENT 12

// Register defines (Supervisor)
#define msupervisor_vertex_base m0
#define mcount m1
#define mlimit m2
#define mcountPtr m3
#define mincrement m4
#define mresultPtr m5

// *****************************************************************************
// Supervisor vertex to increment a tensor by a fixed increment and compare to
// a limit, providing a comparison result output.

.macro INSTANTIATE TYPE
.globl VERTEX_NAME
.type VERTEX_NAME, @function
.section .text.VERTEX_NAME, "ax"
.align 4
VERTEX_NAME:
    ld32 $mcountPtr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_COUNT/4
    ld32 $mlimit, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_LIMIT/4
    ld32 $mincrement, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_INCREMENT/4
    ld32 $mresultPtr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_COMPARISON_RESULT/4
    ld32 $mlimit, $mzero, $mlimit, 0
    nop  // Avoid pipeline bubble for mcountPtr

    ld32 $mcount, $mzero, $mcountPtr, 0
    // We are stuck with these bubbles now, depend on $mcount and so the
    // pipline will stall
    add  $mcount, $mcount, $mincrement
    cmpne $mlimit, $mcount, $mlimit
    // count = count + increment
    st32 $mcount, $mzero, $mcountPtr, 0

    //comparisonResult = count!=limit
    st32  $mlimit,$mzero, $mresultPtr, 0

    br          $lr
.size VERTEX_NAME, .-VERTEX_NAME

DEF_STACK_USAGE 0 VERTEX_NAME

.endm
//******************************************************************************
// Use the macro to instantiate the vertices

INSTANTIATE int
INSTANTIATE unsigned_int


#endif
