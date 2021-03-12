// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// Supervisor registers
#define SUP_VERTEX_BASE m0
#define WORKER_ENTRY    m4

//************************************************
// macro to create mangled names as an entry label into the supervior vertex code,
// and to reference the worker code entry point.
//************************************************
.macro MAKE_ONE_OP OPTYPE FULLNAME OPERATION DATATYPE

DEF_STACK_SIZE_OWN 0 __runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\OPTYPE\()Type__\OPERATION\()_\DATATYPE
.section .text.popops__\FULLNAME\()_\OPERATION\()_\DATATYPE
.align 4
 .global __runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\OPTYPE\()Type__\OPERATION\()_\DATATYPE
 .type __runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\OPTYPE\()Type__\OPERATION\()_\DATATYPE @function
 .supervisor
__runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\OPTYPE\()Type__\OPERATION\()_\DATATYPE:
  setzi        $WORKER_ENTRY, __runCodelet_popops__\FULLNAME\()___popops__expr__\OPTYPE\()Type__\OPERATION\()_\DATATYPE
  runall       $WORKER_ENTRY, $SUP_VERTEX_BASE, 0
  sync         TEXCH_SYNCZONE_LOCAL
  br           $lr

  .size __runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\OPTYPE\()Type__\OPERATION\()_\DATATYPE,\
    .-__runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\OPTYPE\()Type__\OPERATION\()_\DATATYPE
.endm

//************************************************
// Macro to make binary operation supervisor stubs
// each in a  different section so they can be
// linked only when needed.
// The number of parameters (types)can be varied
//************************************************

 .macro INSTANTIATE_OP NAME FULLNAME OPERATION TYPE1="void1" TYPE2="void2" TYPE3="void3" TYPE4="void4" TYPE5="void5" TYPE6="void6" TYPE7="void7" TYPE8="void8" TYPE9="void9"

 // Macro expansion for first type
  .ifnc "\TYPE1","void1"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE1
  .endif

// Macro expansion for 2nd type (optional parameter)
  .ifnc "\TYPE2","void2"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE2
  .endif

// Macro expansion for 3rd type (optional parameter)
  .ifnc "\TYPE3","void3"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE3
  .endif

// Macro expansion for 4th type (optional parameter)
  .ifnc "\TYPE4","void4"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE4
  .endif

// Macro expansion for 5th type (optional parameter)
  .ifnc "\TYPE5","void5"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE5
  .endif

// Macro expansion for 6th type (optional parameter)
  .ifnc "\TYPE6","void6"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE6
  .endif

// Macro expansion for 7th type (optional parameter)
  .ifnc "\TYPE7","void7"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE7
  .endif

// Macro expansion for 8th type (optional parameter)
  .ifnc "\TYPE8","void8"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE8
  .endif

// Macro expansion for 9th type (optional parameter)
  .ifnc "\TYPE9","void9"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE9
  .endif

.endm

//************************************************
// Create a supervisor stub for each operation
// Note _ is mangled to ___
//************************************************

// outputting bool :
INSTANTIATE_OP BinaryOp BinaryOp1D GREATER___THAN, float, half, int, unsigned_int, bool
INSTANTIATE_OP BinaryOp BinaryOp1D GREATER___THAN___EQUAL, float, half, int, unsigned_int, bool
INSTANTIATE_OP BinaryOp BinaryOp1D LESS___THAN, float, half, int, unsigned_int, bool
INSTANTIATE_OP BinaryOp BinaryOp1D LESS___THAN___EQUAL, float, half, int, unsigned_int, bool
INSTANTIATE_OP BinaryOp BinaryOp1D EQUAL, float, half, int, unsigned_int, bool, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1D NOT___EQUAL, float, half, int, unsigned_int, bool, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1D, LOGICAL___AND, bool
INSTANTIATE_OP BinaryOp BinaryOp1D, LOGICAL___OR, bool

INSTANTIATE_OP BinaryOp BinaryOp1D ADD, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D ATAN2, float, half
INSTANTIATE_OP BinaryOp BinaryOp1D BITWISE___AND, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1D BITWISE___OR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1D BITWISE___XOR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1D BITWISE___XNOR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1D DIVIDE, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D MAXIMUM, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D MINIMUM, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D MULTIPLY, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D POWER, float, half
INSTANTIATE_OP BinaryOp BinaryOp1D REMAINDER, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D SHIFT___LEFT, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D SHIFT___RIGHT, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D SHIFT___RIGHT___SIGN___EXTEND, int
INSTANTIATE_OP BinaryOp BinaryOp1D SUBTRACT,float, half, int, unsigned_int

// outputting  bool:
INSTANTIATE_OP UnaryOp UnaryOp1D IS___FINITE, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D IS___INF, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D IS___NAN, float, half

INSTANTIATE_OP UnaryOp UnaryOp1D ABSOLUTE, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1D ASIN, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D BITWISE___NOT, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP UnaryOp UnaryOp1D CEIL, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D COS, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D COUNT___LEADING___ZEROS, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1D EXPONENT, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D EXPONENT___MINUS___ONE, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D FLOOR, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D INVERSE, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D LOGARITHM, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D LOGARITHM___ONE___PLUS, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D NEGATE, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1D POPCOUNT, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1D SIGNUM, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1D SIN, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D TAN, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D TANH, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D RELU, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D ROUND, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D SQRT, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1D SQUARE, float, half, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1D SIGMOID, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D RSQRT, float, half

INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, ABSOLUTE, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, ASIN, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, BITWISE___NOT, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, CEIL, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, COS, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, COUNT___LEADING___ZEROS, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, EXPONENT, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, EXPONENT___MINUS___ONE, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, FLOOR, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, INVERSE, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, LOGARITHM, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, LOGARITHM___ONE___PLUS, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, NEGATE, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, POPCOUNT, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SIGNUM, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SIN, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, TAN, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, ROUND, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SQRT, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SQUARE, float, half, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, RSQRT, float, half

INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, ADD, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, ATAN2, float, half
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, BITWISE___AND, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, BITWISE___OR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, BITWISE___XOR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, BITWISE___XNOR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, DIVIDE, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, MAXIMUM, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, MINIMUM, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, MULTIPLY, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, POWER, float, half
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, REMAINDER, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, SHIFT___LEFT, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, SHIFT___RIGHT, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, SHIFT___RIGHT___SIGN___EXTEND, int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, SUBTRACT, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, GREATER___THAN, bool
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, GREATER___THAN___EQUAL, bool
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, LESS___THAN, bool
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, LESS___THAN___EQUAL, bool
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, EQUAL, bool
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, NOT___EQUAL, bool
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, LOGICAL___AND, bool
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, LOGICAL___OR, bool

// Broadcasting 2nd operand without producing copies to memory

INSTANTIATE_OP BinaryOp, BroadcastScalar1D, ADD, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, ATAN2, half, float
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, BITWISE___AND, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, BITWISE___OR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, BITWISE___XOR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, BITWISE___XNOR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, DIVIDE, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, MAXIMUM, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, MINIMUM, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, MULTIPLY, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, POWER, half, float
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, REMAINDER,  half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, SHIFT___LEFT, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, SHIFT___RIGHT, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, SHIFT___RIGHT___SIGN___EXTEND, int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, SUBTRACT, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, EQUAL, half, float, int, unsigned_int, bool, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, NOT___EQUAL, half, float, int, unsigned_int, bool, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, GREATER___THAN, half, float, int, unsigned_int, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, GREATER___THAN___EQUAL, half, float, int, unsigned_int, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, LESS___THAN, half, float, int, unsigned_int, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, LESS___THAN___EQUAL, half, float, int, unsigned_int, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, LOGICAL___AND, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, LOGICAL___OR, bool

// Note INV___STD___DEV___TO___VARIANCE has an assembly implementation for half
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, INV___STD___DEV___TO___VARIANCE, float
INSTANTIATE_OP BinaryOp, BroadcastScalar1D, VARIANCE___TO___INV___STD___DEV, half, float

INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByColumn, ADD, half_true, float_true, half_false, float_false
INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByColumn, MULTIPLY, half_true, float_true, half_false, float_false
INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByColumn, SUBTRACT, half_true, float_true, half_false, float_false

INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByRow, ADD, half_true, float_true, half_false, float_false
INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByRow, MULTIPLY, half_true, float_true, half_false, float_false
INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByRow, SUBTRACT, half_true, float_true, half_false, float_false

INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, ADD, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, ATAN2, half, float
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, BITWISE___AND, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, BITWISE___OR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, BITWISE___XOR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, BITWISE___XNOR, int, unsigned_int, short, unsigned_short
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, DIVIDE, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, MAXIMUM, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, MINIMUM, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, MULTIPLY, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, POWER, half, float
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, REMAINDER,  half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, SHIFT___LEFT, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, SHIFT___RIGHT, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, SHIFT___RIGHT___SIGN___EXTEND, int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, SUBTRACT, half, float, int, unsigned_int
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, EQUAL, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, NOT___EQUAL, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, GREATER___THAN, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, GREATER___THAN___EQUAL, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, LESS___THAN, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, LESS___THAN___EQUAL, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, LOGICAL___AND, bool
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, LOGICAL___OR, bool

// Note INV___STD___DEV___TO___VARIANCE has an assembly implementation for half
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, INV___STD___DEV___TO___VARIANCE, float
INSTANTIATE_OP BinaryOp, BroadcastScalar1DInPlace, VARIANCE___TO___INV___STD___DEV, half, float

INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByColumnInPlace, ADD, half_true, float_true, half_false, float_false
INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByColumnInPlace, MULTIPLY, half_true, float_true, half_false, float_false
INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByColumnInPlace, SUBTRACT, half_true, float_true, half_false, float_false

INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByRowInPlace, ADD, half_true, float_true, half_false, float_false
INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByRowInPlace, MULTIPLY, half_true, float_true, half_false, float_false
INSTANTIATE_OP BinaryOp, BroadcastVectorOuterByRowInPlace, SUBTRACT, half_true, float_true, half_false, float_false


INSTANTIATE_OP BinaryOp, BroadcastVectorInner, DIVIDE, half, float
INSTANTIATE_OP BinaryOp, BroadcastVectorInnerInPlace, DIVIDE, half, float

#endif
