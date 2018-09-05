#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

#define PTR_OFF       0
#define START_PTR_OFF 4
#define PTR           m7
#define START_PTR     m6
#define VALUE         m1
#define VALUE_2       m2
#define SCRATCH       m3
#define START         m4
#define START_2       m5


#define CYCLE_COUNT_L 0x60
#define CYCLE_COUNT_U 0x61
#define COUNT_VERTEX __runCodelet_popsys__TimeItStart

.section .text.COUNT_VERTEX, "ax"
.align 4
.globl COUNT_VERTEX
.type COUNT_VERTEX, @function
COUNT_VERTEX:
  ld32     $PTR, $m0, $mzero, PTR_OFF/4
  get      $VALUE, $SCOUNT_L
  get      $VALUE_2, $SCOUNT_U

  add      $SCRATCH, $mzero, -(6 + 1)
  cmpult   $SCRATCH, $SCRATCH, $VALUE   // check scount_u hasn't ticked
  sub      $VALUE_2, $VALUE_2, $SCRATCH

  st32     $VALUE, $PTR, $mzero, 0
  st32     $VALUE_2, $PTR, $mzero, 1
  br       $lr



#define TIME_VERTEX __runCodelet_popsys__TimeItEnd

.section .text.TIME_VERTEX, "ax"
.align 4
.globl TIME_VERTEX
.type TIME_VERTEX, @function
TIME_VERTEX:
  ld32     $PTR, $m0, $mzero, PTR_OFF/4
  get      $VALUE, $SCOUNT_L
  get      $VALUE_2, $SCOUNT_U

  add      $SCRATCH, $mzero, -(6 + 1)
  cmpult   $SCRATCH, $SCRATCH, $VALUE   // check scount_u hasn't ticked
  sub      $VALUE_2, $VALUE_2, $SCRATCH

  // substitute values from old. I'm not going to account for the
  // time taken by this code in this substitution
  ld32     $START_PTR, $m0, $mzero, START_PTR_OFF/4

  ld32     $START, $START_PTR, $mzero, 0
  ld32     $START_2, $START_PTR, $mzero, 1

  cmpult   $SCRATCH, $VALUE, $START
  sub      $VALUE_2, $VALUE_2, $SCRATCH

  sub      $VALUE, $VALUE, $START
  sub      $VALUE_2, $VALUE_2, $START_2

  st32     $VALUE, $PTR, $mzero, 0
  st32     $VALUE_2, $PTR, $mzero, 1
  br       $lr

#endif
