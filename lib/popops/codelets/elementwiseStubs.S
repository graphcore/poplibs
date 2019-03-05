#ifdef __IPU__
#include "colossus/tilearch.h"
#include "colossus/tileimplconsts.h"


// Supervisor registers
#define SUP_VERTEX_BASE m0
#define WORKER_ENTRY    m4

//************************************************
// macro to create mangled names as an entry label into the supervior vertex code,
// and to reference the worker code entry point.
//************************************************
.macro MAKE_ONE_OP NAME FULLNAME OPERATION TYPE

.section .text.popops__\FULLNAME\()_\OPERATION\()_\TYPE
 .global __runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\NAME\()Type__\OPERATION\()_\TYPE
 .type __runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\NAME\()Type__\OPERATION\()_\TYPE @function

__runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\NAME\()Type__\OPERATION\()_\TYPE:
  setzi        $WORKER_ENTRY, __runCodelet_popops__\FULLNAME\()___popops__expr__\NAME\()Type__\OPERATION\()_\TYPE
  runall       $WORKER_ENTRY, $SUP_VERTEX_BASE, 0
  sync         TEXCH_SYNCZONE_LOCAL
  br           $lr

  .size popops__\FULLNAME\()_\OPERATION\()_\TYPE,\
   .-__runCodelet_popops__\FULLNAME\()Supervisor___popops__expr__\NAME\()Type__\OPERATION\()_\TYPE
.endm
//************************************************
// Macro to make binary operation supervisor stubs
// each in a  different section so they can be
// linked only when needed.
// The number of parameters (types)can be varied
//************************************************
 .macro INSTANTIATE_OP NAME FULLNAME OPERATION TYPE1="void1" TYPE2="void2" TYPE3="void3" TYPE4="void4"

 // Macro expansion for first type
  .ifnc \TYPE1,"void1"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE1
  .endif

// Macro expansion for 2nd type (optional parameter)
  .ifnc \TYPE2,"void2"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE2
 .endif

// Macro expansion for 3rd type (optional parameter)
  .ifnc \TYPE3,"void3"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE3
 .endif

// Macro expansion for 4th type (optional parameter)
  .ifnc \TYPE4,"void4"
  MAKE_ONE_OP \NAME \FULLNAME \OPERATION \TYPE4
 .endif

.endm

//************************************************
// Create a supervisor stub for each operation
// Note _ is mangled to ___
// omit those that list bool as an input/output type
//************************************************

// outputting bool :
INSTANTIATE_OP BinaryOp BinaryOp1D GREATER___THAN, float, half
INSTANTIATE_OP BinaryOp BinaryOp1D GREATER___THAN___EQUAL, float, half
INSTANTIATE_OP BinaryOp BinaryOp1D LESS___THAN, float, half
INSTANTIATE_OP BinaryOp BinaryOp1D LESS___THAN___EQUAL, float, half
INSTANTIATE_OP BinaryOp BinaryOp1D EQUAL, float, half
INSTANTIATE_OP BinaryOp BinaryOp1D NOT___EQUAL, float, half

INSTANTIATE_OP BinaryOp BinaryOp1D ADD, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D ATAN2, float, half
INSTANTIATE_OP BinaryOp BinaryOp1D BITWISE___AND, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1D BITWISE___OR, int, unsigned_int
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

INSTANTIATE_OP UnaryOp UnaryOp1D ABSOLUTE, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1D BITWISE___NOT, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1D CEIL, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D COS, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D COUNT___LEADING___ZEROS, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1D EXPONENT, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D EXPONENT___MINUS___ONE, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D FLOOR, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D LOGARITHM, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D LOGARITHM___ONE___PLUS, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D NEGATE, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1D POPCOUNT, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1D SIGNUM, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1D SIN, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D TANH, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D ROUND, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D SQRT, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1D SQUARE, float, half, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1D SIGMOID, float, half
INSTANTIATE_OP UnaryOp UnaryOp1D RSQRT, float, half

INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, ABSOLUTE, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, BITWISE___NOT, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, CEIL, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, COS, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, COUNT___LEADING___ZEROS, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, EXPONENT, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, FLOOR, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, LOGARITHM, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, NEGATE, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, POPCOUNT, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SIGNUM, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SIN, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, TANH, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, ROUND, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SQRT, float, half, int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SQUARE, float, half, int, unsigned_int
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, SIGMOID, float, half
INSTANTIATE_OP UnaryOp UnaryOp1DInPlace, RSQRT, float, half

INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, ADD, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, ATAN2, float, half
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, BITWISE_AND, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, BITWISE_OR, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, DIVIDE, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, MAXIMUM, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, MINIMUM, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, MULTIPLY, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, POWER, float, half
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, REMAINDER, float, half, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, SHIFT_LEFT, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, SHIFT_RIGHT, int, unsigned_int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, SHIFT_RIGHT_SIGN_EXTEND, int
INSTANTIATE_OP BinaryOp BinaryOp1DInPlace, SUBTRACT, float, half, int, unsigned_int

// Broadcasting 2nd operand without producing copies to memory

INSTANTIATE_OP BroadcastOp ,BroadcastOp1DInPlace, ADD, half, float
INSTANTIATE_OP BroadcastOp, BroadcastOp1DInPlace, MULTIPLY, half, float
INSTANTIATE_OP BroadcastOp, BroadcastOp1DInPlace, SUBTRACT, half, float
INSTANTIATE_OP BroadcastOp, BroadcastOp1DInPlace, INV___STD___DEV___TO___VARIANCE, half, float
INSTANTIATE_OP BroadcastOp, BroadcastOp1DInPlace, VARIANCE___TO___INV___STD___DEV, half, float

#endif
