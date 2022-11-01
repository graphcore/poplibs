// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Header for Sparse-Dense Block matrix multiplication for asm codelets

#ifndef _StaticBlockSparseDenseMatMul_h_
#define _StaticBlockSparseDenseMatMul_h_

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// =============================================================================

.extern zeroStaticDenseOutput64

// =============================================================================

//// Supervisor vertex state
#define SUP_VBASE_R_BASE             0    // one pointer to NZ values
#define SUP_VBASE_METAINFO           4    // one pointer to meta info
#define SUP_VBASE_S_BASE             8    // one pointer to dense input S
#define SUP_VBASE_Q_BASE             12   // one pointer to dense output Q
#define SUP_VBASE_ZERO_INFO          16   // ushort: number of 64-bits to zero
#define SUP_VBASE_S_SIZE             18   // Size of S in 64-bits


// =============================================================================

// State used by workers for zeroing partials
#define VBASE_ZERO_PTR               0
#define VBASE_ZERO_NUM64             4

// =============================================================================
// Worklist fields for each worker
#define WORKLIST_OFFSET_Z            0
#define WORKLIST_NUM_Z               2
#define SIZEOF_WORKLIST_BYTES        4

// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////
#define W_Z_BASE                     0
#define W_Z_QBASE                    W_Z_BASE + VBASE_ZERO_PTR
#define W_Z_NUM_ZERO                 W_Z_BASE + VBASE_ZERO_NUM64
#define W_XOFFSET                    0
#define W_YOFFSET                    4       // Scaled by 8
#define W_VERTEX_STATE               8
#define STACK_SIZE                   (W_YOFFSET + 8)

// =============================================================================

#define WORKER_NAME_PREFIX staticBlockSparseDenseMultiply

// =============================================================================

// Registers allocation
#define s_vertexBase                 m0
#define s_metaInfo                   m1
#define s_rBase                      m2
#define s_wkrFunction                m3
#define s_zeroWkrFunction            m3
#define s_qBase                      m4
#define s_numZero64                  m5
#define s_XOrYOffset                 m5
#define s_numXm1                     m6
#define s_numYm1                     m7
#define s_clrCWEI                    m6
#define s_sSize                      m8
#define s_sBase                      m0


// =============================================================================


// utilities
.macro SET_WKR_FUNCTION_XRETAINED NAME POSTFIX
setzi                  $s_wkrFunction, \NAME\()_\POSTFIX\()_retained
.endm

.macro SET_WKR_FUNCTION_YRETAINED NAME POSTFIX
setzi                  $s_wkrFunction, \NAME\()_\POSTFIX\()_retained_yOff
.endm

.macro SET_WKR_FUNCTION NAME POSTFIX
setzi                  $s_wkrFunction, \NAME\()_\POSTFIX\()
.endm

.macro WKR_FUNCTION_LABEL_XRETAINED NAME POSTFIX
\NAME\()_\POSTFIX\()_retained:
.endm

.macro WKR_FUNCTION_LABEL_YRETAINED NAME POSTFIX
\NAME\()_\POSTFIX\()_retained_yOff:
.endm

.macro WORKER_FN_HEADER PREFIX, POSTFIX 
DEF_STACK_USAGE 0 \PREFIX\()_\POSTFIX\()
.section .text.\PREFIX\()_\POSTFIX\(), FUNCTION_IS_WORKER
.type  \PREFIX\()_\POSTFIX\(), @function
.global \PREFIX\()_\POSTFIX\()
.global \PREFIX\()_\POSTFIX\()_retained
.global \PREFIX\()_\POSTFIX\()_retained_yOff
.align 8
.worker
// worker code
\PREFIX\()_\POSTFIX\():
.endm

.macro WORKER_FN_FOOTER PREFIX, POSTFIX
.size \PREFIX\()_\POSTFIX\(), . - \PREFIX\()_\POSTFIX\()
.endm

// =============================================================================

#define w_qBaseRetained                    m0
#define w_sBaseRetained                    m1
#define w_id                               m2
#define w_vertexState                      m2
#define w_off                              m4
#define w_workList                         m7
#define w_num_orig                         m8

.macro LOAD_VERTEX_STATE
// the supervisor base is used directly to extract fields required by the workers
ld32                  $w_vertexState, $mvertex_base, W_VERTEX_STATE/4
ld32                  $w_sBaseRetained, $w_vertexState, SUP_VBASE_S_BASE/4
ld32                  $w_qBaseRetained, $w_vertexState, SUP_VBASE_Q_BASE/4
ld32                  $w_workList, $w_vertexState, SUP_VBASE_METAINFO/4

// two entries per worker: offset into batch and number of batch elements to process
get                   $w_id, $WSR
and                   $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
mul                   $w_id, $w_id, SIZEOF_WORKLIST_BYTES
ldz16                 $w_off, $w_id, $w_workList, WORKLIST_OFFSET_Z/2
ldz16                 $w_num_orig, $w_id, $w_workList, WORKLIST_NUM_Z/2
.endm


// =============================================================================


.macro BLOCK_SPARSE_MATMUL CNAME, INTYPE, PTYPE, POSTFIX, BLOCKSIZE

// supervisor base is $m0 - passed to this function
DEF_STACK_USAGE  (STACK_SIZE) \CNAME\()

.section .text.\CNAME\()
.align 4
.globl \CNAME\()
.type \CNAME\(), @function
\CNAME\():
.supervisor
setzi                  $s_clrCWEI, (1 << 2)
add                    $sp, $sp, -STACK_SIZE
ldz16                  $s_sSize, $s_vertexBase, SUP_VBASE_S_SIZE/2

.ifc \PTYPE, float
.ifnc \BLOCKSIZE, 4
ld32                   $s_qBase, $s_vertexBase, SUP_VBASE_Q_BASE/4
ldz16                  $s_numZero64, $s_vertexBase, SUP_VBASE_ZERO_INFO/2
.endif // .ifnc \BLOCKSIZE, 4
.endif // .ifc \PTYPE, float


ld32                   $s_metaInfo, $s_vertexBase, SUP_VBASE_METAINFO/4
st32                   $s_vertexBase, $sp, W_VERTEX_STATE/4
nop
// clear $CWEI
put                    $CR, $s_clrCWEI
ld32                   $s_rBase, $s_vertexBase, SUP_VBASE_R_BASE/4
ld32                   $s_sBase, $s_vertexBase, SUP_VBASE_S_BASE/4

.ifc \PTYPE, float 
.ifnc \BLOCKSIZE, 4
setzi                  $s_zeroWkrFunction, zeroStaticDenseOutput64

// Vertex state for zeroing: ptr and number of 8-bytes
st32                   $s_qBase, $sp, W_Z_QBASE/4
st32                   $s_numZero64, $sp, W_Z_NUM_ZERO/4
.endif // .ifnc \BLOCKSIZE, 4
.endif // .ifc float

// sSize is in 64-bits. Shift up
shl                    $s_sSize, $s_sSize, 3

// move metaInfo pointer over the worklist
add                    $s_metaInfo, $s_metaInfo, CTXT_WORKERS * SIZEOF_WORKLIST_BYTES

// zero output
// TODO: If we had a path in the worker to know when the partials are zero
//       we can call it exactly when needed from the supervisor as we know
//       when the first time is. It will just be a matter of setting the
//       correct worker function.
.ifc \PTYPE, float
.ifnc \BLOCKSIZE, 4
runall                 $s_zeroWkrFunction, $sp, 0
.endif
.endif

SET_WKR_FUNCTION       WORKER_NAME_PREFIX \POSTFIX\()

ldz16step              $s_numXm1, $mzero, $s_metaInfo+=, 1
ldz16step              $s_XOrYOffset, $mzero, $s_metaInfo+=, 1

// We let CCCSLOAD to increment between blocks rather then tracking it using
// a general register
put                    $CCCSLOAD, $s_rBase
sync                   TEXCH_SYNCZONE_LOCAL
// zero out padding: A fixed size is allocated regardless of data type
st32                   $mzero, $s_sSize, $s_sBase, 0
st32                   $mzero, $s_sSize, $s_sBase, 1
st32                   $mzero, $s_sSize, $s_sBase, 2
st32                   $mzero, $s_sSize, $s_sBase, 3

// load entries from output
Ls_XLoop_\INTYPE\()_\PTYPE\():
    ldz16step              $s_numYm1, $mzero, $s_metaInfo+=, 1
    st32                   $s_XOrYOffset, $sp, W_XOFFSET / 4
    ldz16step              $s_XOrYOffset, $mzero, $s_metaInfo+=, 1

Ls_YLoop_\INTYPE\()_\PTYPE\():
.ifc \BLOCKSIZE, 4
      mul                   $s_XOrYOffset, $s_XOrYOffset, 8 * \BLOCKSIZE\()
.ifc \INTYPE, float
      ld64putcs              0x0 
      ld64putcs              0x1 
      ld64putcs              0x4
      ld64putcs              0x5
      ld64putcs              0x8
      st32                   $s_XOrYOffset, $sp, W_YOFFSET/4
      ld64putcs              0x9
      ld64putcs              0xc
      ld64putcs              0xd
.else
      ld64putcs              0x0
      ld64putcs              0x4
      ld64putcs              0x20
      ld64putcs              0x24
      nop
      st32                   $s_XOrYOffset, $sp, W_YOFFSET/4
.endif // \INTYPE float
.endif // \BLOCKSIZE, 4

.ifc \BLOCKSIZE, 8
      mul                   $s_XOrYOffset, $s_XOrYOffset, 8 * \BLOCKSIZE\()
.ifc \INTYPE, float
      ld64putcs              0
      ld64putcs              1
      ld64putcs              2
      ld64putcs              3
      ld64putcs              4
      ld64putcs              5
      st32                   $s_XOrYOffset, $sp, W_YOFFSET/4
      ld64putcs              6
      ld64putcs              7
      ld64putcs              8
      ld64putcs              9
      ld64putcs              10
      ld64putcs              11
      ld64putcs              12
      ld64putcs              13
      ld64putcs              14
      ld64putcs              15
      ld64putcs              16
      ld64putcs              17
      ld64putcs              18
      ld64putcs              19
      ld64putcs              20
      ld64putcs              21
      ld64putcs              22
      ld64putcs              23
      ld64putcs              24
      ld64putcs              25
      ld64putcs              26
      ld64putcs              27
      ld64putcs              28
      ld64putcs              29
      ld64putcs              30
      ld64putcs              31
.else

      ld64putcs              0
      ld64putcs              1
      ld64putcs              4
      ld64putcs              5
      ld64putcs              32
      st32                   $s_XOrYOffset, $sp, W_YOFFSET/4
      ld64putcs              33
      ld64putcs              36
      ld64putcs              37
    
      ld64putcs              8
      ld64putcs              9
      ld64putcs              12
      ld64putcs              13
      ld64putcs              40
      ld64putcs              41
      ld64putcs              44
      ld64putcs              45
.endif // \INTYPE, float
.endif // \BLOCKSIZE, 8


.ifc \BLOCKSIZE, 16
      mul                   $s_XOrYOffset, $s_XOrYOffset, 8 * \BLOCKSIZE\()
      ld64putcs              0
      ld64putcs              1
      ld64putcs              2
      ld64putcs              3
      ld64putcs              4
      st32                   $s_XOrYOffset, $sp, W_YOFFSET/4
      ld64putcs              5
      ld64putcs              6
      ld64putcs              7
      ld64putcs              32
      ld64putcs              33
      ld64putcs              34
      ld64putcs              35
      ld64putcs              36
      ld64putcs              37
      ld64putcs              38
      ld64putcs              39
      ld64putcs              8
      ld64putcs              9
      ld64putcs              10
      ld64putcs              11
      ld64putcs              12
      ld64putcs              13
      ld64putcs              14
      ld64putcs              15
      ld64putcs              40
      ld64putcs              41
      ld64putcs              42
      ld64putcs              43
      ld64putcs              44
      ld64putcs              45
      ld64putcs              46
      ld64putcs              47
      ld64putcs              16
      ld64putcs              17
      ld64putcs              18
      ld64putcs              19
      ld64putcs              20
      ld64putcs              21
      ld64putcs              22
      ld64putcs              23
      ld64putcs              48
      ld64putcs              49
      ld64putcs              50
      ld64putcs              51
      ld64putcs              52
      ld64putcs              53
      ld64putcs              54
      ld64putcs              55
      ld64putcs              24
      ld64putcs              25
      ld64putcs              26
      ld64putcs              27
      ld64putcs              28
      ld64putcs              29
      ld64putcs              30
      ld64putcs              31
      ld64putcs              56
      ld64putcs              57
      ld64putcs              58
      ld64putcs              59
      ld64putcs              60
      ld64putcs              61
      ld64putcs              62
      ld64putcs              63
.endif // \BLOCKSIZE, 16
      runall                 $s_wkrFunction, $sp, 0
      SET_WKR_FUNCTION_YRETAINED WORKER_NAME_PREFIX \POSTFIX\()
      sync                   TEXCH_SYNCZONE_LOCAL
.ifc \BLOCKSIZE, 16
.ifc \PTYPE, float
      add                    $s_XOrYOffset, $s_XOrYOffset, 32
      ld64putcs              0
      ld64putcs              1
      ld64putcs              2
      ld64putcs              3
      ld64putcs              4
      st32                   $s_XOrYOffset, $sp, W_YOFFSET/4
      ld64putcs              5
      ld64putcs              6
      ld64putcs              7
      ld64putcs              32
      ld64putcs              33
      ld64putcs              34
      ld64putcs              35
      ld64putcs              36
      ld64putcs              37
      ld64putcs              38
      ld64putcs              39
      ld64putcs              8
      ld64putcs              9
      ld64putcs              10
      ld64putcs              11
      ld64putcs              12
      ld64putcs              13
      ld64putcs              14
      ld64putcs              15
      ld64putcs              40
      ld64putcs              41
      ld64putcs              42
      ld64putcs              43
      ld64putcs              44
      ld64putcs              45
      ld64putcs              46
      ld64putcs              47
      ld64putcs              16
      ld64putcs              17
      ld64putcs              18
      ld64putcs              19
      ld64putcs              20
      ld64putcs              21
      ld64putcs              22
      ld64putcs              23
      ld64putcs              48
      ld64putcs              49
      ld64putcs              50
      ld64putcs              51
      ld64putcs              52
      ld64putcs              53
      ld64putcs              54
      ld64putcs              55
      ld64putcs              24
      ld64putcs              25
      ld64putcs              26
      ld64putcs              27
      ld64putcs              28
      ld64putcs              29
      ld64putcs              30
      ld64putcs              31
      ld64putcs              56
      ld64putcs              57
      ld64putcs              58
      ld64putcs              59
      ld64putcs              60
      ld64putcs              61
      ld64putcs              62
      ld64putcs              63
      runall                 $s_wkrFunction, $sp, 0
      sync                   TEXCH_SYNCZONE_LOCAL
.endif // \PTYPE, float
.endif // \BLOCKSIZE, 16
      ldz16step              $s_XOrYOffset, $mzero, $s_metaInfo+=, 1
      brnzdec                $s_numYm1, Ls_YLoop_\INTYPE\()_\PTYPE\()
    SET_WKR_FUNCTION_XRETAINED WORKER_NAME_PREFIX \POSTFIX\()
    brnzdec                $s_numXm1, Ls_XLoop_\INTYPE\()_\PTYPE\()
add                    $sp, $sp, STACK_SIZE
sync                   TEXCH_SYNCZONE_LOCAL
br                     $lr

.size \CNAME\(), . - \CNAME\()
.endm

// =============================================================================
#endif // #define _StaticBlockSparseDenseMatMul_h_
// =============================================================================
