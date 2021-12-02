// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Bit to force all accumulators to zero when written in FP_CLR register
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// To create a worker section to begin linking into create a label and declare 
// it as a global function  
.macro FN_WORKER_ENTRY_POINT VERTEX_NAME ALIGNMENT=4 OPTIONAL_NOP="" STACK=0
.ifnc "\STACK", "NONE"
    DEF_STACK_USAGE  \STACK \VERTEX_NAME
.endif
    .section .text.\VERTEX_NAME, FUNCTION_IS_WORKER
    .global \VERTEX_NAME
    .type \VERTEX_NAME, @function
    .align \ALIGNMENT
    .worker
\VERTEX_NAME\()_START:
\OPTIONAL_NOP
\VERTEX_NAME:
.endm

// To create a supervisor section to begin linking into create a label and declare 
// it as a global function  

.macro FN_SUPERVISOR_ENTRY_POINT VERTEX_NAME ALIGNMENT=4 OPTIONAL_NOP="" STACK=0
.ifnc "\STACK", "NONE"
    DEF_STACK_USAGE  \STACK  \VERTEX_NAME
.endif
    .section .text.\VERTEX_NAME, "ax"
    .global \VERTEX_NAME
    .type \VERTEX_NAME, @function
    .align \ALIGNMENT
    .supervisor
\VERTEX_NAME\()_START:
\OPTIONAL_NOP
\VERTEX_NAME:
.endm

// To create a worker section to begin linking into
.macro FN_SECTION VERTEX_NAME ALIGNMENT=4 OPTIONAL_NOP="" STACK=0
.ifnc "\STACK", "NONE"
    DEF_STACK_USAGE  \STACK .text.\VERTEX_NAME 
.endif
    .section .text.\VERTEX_NAME, FUNCTION_IS_WORKER
    .align \ALIGNMENT
    .worker
\VERTEX_NAME\()_START:
\OPTIONAL_NOP
.endm

// To create a supervisor section to begin linking into
.macro FN_SUPERVISOR_SECTION VERTEX_NAME ALIGNMENT=4 OPTIONAL_NOP="" STACK=0
.ifnc "\STACK", "NONE"
    DEF_STACK_USAGE  \STACK .text.\VERTEX_NAME 
.endif
    .section .text.\VERTEX_NAME
    .align \ALIGNMENT
    .supervisor
\VERTEX_NAME\()_START:
\OPTIONAL_NOP
.endm

// To create a size directive when paired with one of the 3 macros above
.macro FN_SIZE LABEL
    .size \LABEL, . - \LABEL\()_START
.endm

// This macro defines a label as global, function
.macro FN_EXPORT label
.globl \label
.type \label, @function
\label:
.endm


// To split a 32 bit immediate value into two parts and load it in 2 instructions
// which can potentially be bundled
#define MASK_LOWER(INPUT) (INPUT & ~((1 << 20) - 1))
#define MASK_UPPER(INPUT) (INPUT & ((1 << 20) - 1))

// Auto align rpt where macro/code before can be of variable size, reference
// a label that is aligned to an 8 byte boundary
.macro RPT_ALIGNED_TO LABEL REGISTER
.if (. - \LABEL) & 0x7
  rpt \REGISTER, (2f-1f)/8 -1
.else
  {rpt \REGISTER, (2f-1f)/8 -1
   fnop}
.endif
.endm

#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_PTR_MASK ((1 << SHORT_SPAN_PTR_SIZE) - 1)
#define SHORT_SPAN_LENGTH_SIZE 12

#define SCALED_PTR16_SHIFTS 1
#define SCALED_PTR32_SHIFTS 2
#define SCALED_PTR64_SHIFTS 3
