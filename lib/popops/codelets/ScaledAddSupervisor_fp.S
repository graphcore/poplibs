#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty

// vertex state

#define VERTEX_DATA_OFFSET 0
#define VERTEX_PACKED_COUNT_OFFSET 1
#define VERTEX_DELTAS_OFFSET 2
#define VERTEX_K_OFFSET 3

// create a new vertex state on the supervisor stack that has the input values
// preprocessed for all of the workers to use.
#define SV_STATE_VPTR_OFFSET 0
#define SV_STATE_COUNT_OFFSET 1
#define SV_STATE_REMM1_OFFSET 2
#define SV_STATE_FINAL_OFFSET 3

#define SV_STATE_SIZE 16

// place on the stack to save the link register.
#define STACK_LR_OFFSET 4

// total space required on the stack
#define STACK_SIZE (SV_STATE_SIZE + 4)

// constants

// to avoid sub-word writes we must make sure that each worker processes
// a number of elements so that we fall exactly into a 64-bit load. for floats
// this is 8/sizeof(float) = 2 and 8/sizeof(half) = 4
#define LOG2_FLOAT_ATOM_SIZE 1
#define LOG2_HALF_ATOM_SIZE 2

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// supervisor variables
#define vertexPtr m0
#define countD2 m1
#define final m2
#define remM1 m3

#define mscratch m4
#define mscratch2 m5

#define log2AtomSize m7
#define atomSizeMask m8

.globl VERTEX(float)
.type VERTEX(float), @function

.section .text.VERTEX(float)
.align 4
VERTEX(float):
  add $sp, $sp, -STACK_SIZE

  // save $lr before we use a call instruction.
  st32 $lr, $sp, $mzero, STACK_LR_OFFSET // 6 cycles

  // call the common supervisor code to set-up the preprocessed vertex state on
  // the supervisor stack. m0 is already set to the vertex base.
  setzi $log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1
  call $lr, VERTEX(supervisor) // 6 cycles

  setzi $m0, VERTEX(float).kernel
  runall $m0, $sp, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  ld32 $lr, $sp, $mzero, STACK_LR_OFFSET

  // restore the stack pointer that was changed in the supervisor common code.
  add $sp, $sp, STACK_SIZE
  br $lr // 6 cycles

.size VERTEX(float), .-VERTEX(float)

.globl VERTEX(half)
.type VERTEX(half), @function

.section .text.VERTEX(half)
.align 4
VERTEX(half):
  add $sp, $sp, -STACK_SIZE

  // save $lr before we use a call instruction.
  st32 $lr, $sp, $mzero, STACK_LR_OFFSET // 6 cycles

  // call the common supervisor code to set-up the preprocessed vertex state on
  // the supervisor stack. m0 is already set to the vertex base.
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1
  call $lr, VERTEX(supervisor) // 6 cycles

  setzi $m0, VERTEX(half).kernel
  runall $m0, $sp, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL

  ld32 $lr, $sp, $mzero, STACK_LR_OFFSET

  // restore the stack pointer that was changed in the supervisor common code.
  add $sp, $sp, STACK_SIZE
  br $lr // 6 cycles

.size VERTEX(half), .-VERTEX(half)

// common supervisor code for both half and float vertices.
.type VERTEX(supervisor), @function

.section .text.VERTEX(supervisor)
.align 4
VERTEX(supervisor):
  // initialise some constants used below.
  setzi $mscratch, 0xAAAB

  // load vertex state
  ld32 $mscratch2, $vertexPtr, $mzero, VERTEX_DATA_OFFSET
  ld32 $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET

  // transform the total count into remM1, final and count/6:
  //  where remM1 is the amount of workers (minus 1) that are required to
  //  process an extra atom size of elements, final is the non atom size
  //  remainder the final worker must process (when N is not divisible by the
  //  atoms size) and count is how many elements every worker processes

  // final = count % atomSize
  and $final, $countD2, $atomSizeMask // 6 cycles

  // for the rest calculate n / 6 and n % 6 by reciprocal multiplcation
  //   n/6 = (n * 0xAAAB) >> 18
  //   n%6 = n - (n/6)*6
  // where n = count/atomSize
  // see recipe #1 for how these constants were derived:
  //   https://embeddedgurus.com/stack-overflow/2009/06/division-of-integers-by-constants/
  shr $countD2, $countD2, $log2AtomSize // 6 cycles

  // mscratch = n/6
  mul $mscratch, $countD2, $mscratch // 6 cycles
  shr $mscratch, $mscratch, 18 // 6 cycles

  // avoid the register bubble from using $final too soon.
  nop ; nop ; nop

  // rem = (count / atomSize) % numWorkers + ceil(final, atomSize)
  //  where ceil(x, y) = x / y + (x % y > 0);
  shr $remM1, $final, $log2AtomSize

  // mscratch2 = n%6
  mul $mscratch2, $mscratch, 6
  sub $mscratch2, $countD2, $mscratch2 // 6 cycles

  // countPerWorker = (count / atomSize) / numWorkers * atomSize
  shl $countD2, $mscratch, $log2AtomSize

  and $mscratch, $final, $atomSizeMask
  cmpne $mscratch, $mscratch, $mzero // 6 cycles

  // avoid register bubble
  nop

  add $remM1, $remM1, $mscratch2
  add $remM1, $remM1, $mscratch // 6 cycles

  // when final is zero that means that the final worker can process an entire
  // block of elements. the easiest way to represent this is to add one to
  // remM1 (or just don't decrement it) in that case.
  //  cycles: 6 if final is zero, 7 if not.
  brz $final, 1f
  add $remM1, $remM1, -1 // 6 cycles
1:

  // allocate space on the stack for our new state and store it.
  st32 $remM1, $sp, $mzero, SV_STATE_REMM1_OFFSET // 6 cycles if final != 0
  st32 $vertexPtr, $sp, $mzero, SV_STATE_VPTR_OFFSET
  st32 $final, $sp, $mzero, SV_STATE_FINAL_OFFSET
  st32 $countD2, $sp, $mzero, SV_STATE_COUNT_OFFSET

  br $lr // 6 cycles

.size VERTEX(supervisor), .-VERTEX(supervisor)

// clear supervisor variables
#undef vertexPtr
#undef dataPtr
#undef countD2
#undef final
#undef remM1
#undef mscratch
#undef mscratch2

// worker variables

// integer variables
#define vertexPtr m0
#define dataPtr m1
#define remM1 m2
#define final m3
#define countD2 m4
#define countD4 m4
#define deltaPtr m5
#define triPtr m6:7
#define triPtri0 m6
#define triPtri1 m7
#define workerIdM1 m8
#define stride m9

// float variables
#define data a0:1
#define datai0 a0
#define datai1 a1
#define delta a2:3
#define deltai0 a2
#define deltai1 a3
#define result a4:5
#define k a6

// scratch variables
#define mscratch m10
#define ascratch a7

.type VERTEX(float).kernel, @function

.section .text.VERTEX(float).kernel
.align 8
  nop
VERTEX(float).kernel:
  // load vertex state
  ld32 $vertexPtr, $mvertex_base, $mzero, SV_STATE_VPTR_OFFSET
  ld32 $countD2, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET
  ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET

  ld32 $dataPtr, $vertexPtr, $mzero, VERTEX_DATA_OFFSET
  ld32 $deltaPtr, $vertexPtr, $mzero, VERTEX_DELTAS_OFFSET
  ld32 $k, $vertexPtr, $mzero, VERTEX_K_OFFSET

  {
    get $workerIdM1, $WSR
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    setzi $ascratch, ZAACC_BITMASK
  }

  // process 2 at a time first as this is the optimal scenario
  shr $countD2, $countD2, 1

  // pack out points (in is never used).
  tapack $triPtr, $dataPtr, $deltaPtr, $mzero

  // if worker id is less than the remainder this worker can process an extra 4.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD2, $countD2, $mscratch

  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  // use $data as a temporary scratch register as we can't write to $azeros
  // twice in the same instruction.
  ld2x64pace $azeros, $data, $triPtr+=, $workerIdM1, 0b0101

  brz $countD2, .Lfloat_loop_epilogue // 16

  // each worker's data is interleaved so set a stride of how many workers
  // we have.
  setzi $stride, CTXT_WORKERS

  // preload 4 values and fill the accumulators.
  ld2x64pace $data, $delta, $triPtr+=, $stride, 0b0101
  {
    // minus 1 because we pipeline the first value.
    add $mscratch, $countD2, -1
    f32v2axpy $azeros, $delta, $data
  }

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {
    ld2x64pace $data, $delta, $triPtr+=, $stride, 0b0101
    f32v2axpy $result, $azeros, $azeros
  }
  {
    st64step $result, $mzero, $dataPtr+=, $stride
    f32v2axpy $azeros, $delta, $data
  }
2:
  // store the final 2 processed values.
  f32v2axpy $result, $azeros, $azeros
  st64step $result, $mzero, $dataPtr+=, $stride

.Lfloat_loop_epilogue:
  // at most one of our workers will have to do the remaining element. this
  // worker id is equal to the $rem value in the vertex state. the amount
  // of elements remaining is the $final value. $final will be 1 at most.
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lfloat_epilogue
  brz $final, .Lfloat_epilogue

  // unpack the data and delta pointers from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  {
    and $dataPtr, $triPtri0, $mscratch
    // zero the top half of data and delta so we can safely accumulate them
    zero $datai1
  }
  {
    and $deltaPtr, $triPtri1, $mscratch
    zero $deltai1
  }

  // scalar.
  ld32 $datai0, $dataPtr, $mzero, 0
  ld32step $deltai0, $mzero, $deltaPtr+=, 1

  f32v2axpy $azeros, $delta, $data
  f32v2axpy $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

.Lfloat_epilogue:
  exitz $mzero

.size VERTEX(float).kernel, .-VERTEX(float).kernel

.type VERTEX(half).kernel, @function

.section .text.VERTEX(half).kernel
.align 8
  nop
VERTEX(half).kernel:
  // load vertex state
  ld32 $vertexPtr, $mvertex_base, $mzero, SV_STATE_VPTR_OFFSET
  ld32 $countD4, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET
  ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET

  ld32 $dataPtr, $vertexPtr, $mzero, VERTEX_DATA_OFFSET
  ld32 $deltaPtr, $vertexPtr, $mzero, VERTEX_DELTAS_OFFSET
  {
    ld32 $k, $vertexPtr, $mzero, VERTEX_K_OFFSET
    f16v2exp $ascratch, $azero
  }
  {
    get $workerIdM1, $WSR
    // $k should have the form of {k, 1}
    sort4x16lo $k, $k, $ascratch
  }
  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  {
    // pack out points (in is never used).
    tapack $triPtr, $dataPtr, $deltaPtr, $mzero
    setzi $ascratch, ZAACC_BITMASK
  }

  // process 4 at a time first as this is the optimal scenario
  shr $countD4, $countD4, 2

  // if worker id is less than the remainder this worker can process an extra 4.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD4, $countD4, $mscratch

  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  // use $data as a temporary scratch register as we can't write to $azeros
  // twice in the same instruction.
  ld2x64pace $azeros, $data, $triPtr+=, $workerIdM1, 0b0101

  brz $countD4, .Lhalf_loop_epilogue

  // each worker's data is interleaved so set a stride of how many workers
  // we have.
  setzi $stride, CTXT_WORKERS

  // preload 4 values and fill the accumulators.
  ld2x64pace $data, $delta, $triPtr+=, $stride, 0b0101
  {
    // minus 1 because we pipeline the first value.
    add $mscratch, $countD4, -1
    f16v4mix $azeros, $delta, $data
  }

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {
    ld2x64pace $data, $delta, $triPtr+=, $stride, 0b0101
    f16v4mix $result, $azeros, $azeros
  }
  {
    st64step $result, $mzero, $dataPtr+=, $stride
    f16v4mix $azeros, $delta, $data
  }
2:
  // store the final 4 processed values.
  f16v4mix $result, $azeros, $azeros
  st64step $result, $mzero, $dataPtr+=, $stride

.Lhalf_loop_epilogue:
  // at most one of our workers will have to do the remaining elements. this
  // worker id is equal to the $rem value in the vertex state. the amount
  // of elements remaining is the $final value. $final will be 3 at most.
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lhalf_epilogue
  brz $final, .Lhalf_epilogue

  // unpack the data and delta pointers from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  and $dataPtr, $triPtri0, $mscratch
  and $deltaPtr, $triPtri1, $mscratch

  {
    // is there at least 2 left?
    cmpult $mscratch, $final, 2
    // zero the top half of data and delta so we can safely accumulate them
    // for the x2 and x1 cases.
    zero $datai1
  }
  {
    brnz $mscratch, .Lhalf_scalar
    zero $deltai1
  }

  // remainder 2
  ld32 $datai0, $dataPtr, $mzero, 0
  ld32step $deltai0, $mzero, $deltaPtr+=, 1

  f16v4mix $azeros, $delta, $data
  f16v4mix $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

  // finish now if that's all.
  cmpeq $mscratch, $final, 2
  brnz $mscratch, .Lhalf_epilogue

.Lhalf_scalar:
  ldb16 $datai0, $dataPtr, $mzero, 0
  ldb16 $deltai0, $deltaPtr, $mzero, 0

  f16v4mix $azeros, $delta, $data

  {
    // load the last word and perform a read/modify/write.
    ld32 $ascratch, $dataPtr, $mzero, 0
    f16v4mix $data, $azeros, $azeros
  }

  sort4x16hi $ascratch, $datai0, $ascratch
  st32 $ascratch, $dataPtr, $mzero, 0

.Lhalf_epilogue:
  exitz $mzero

.size VERTEX(half).kernel, .-VERTEX(half).kernel

#endif // __IPU__
