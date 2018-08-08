#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

#define VERTEX __runCodelet_popops__ScaledAddSupervisor___float

.globl VERTEX
.type VERTEX, @function

// constants
#define VERTEX_DATA_OFFSET 0
#define VERTEX_PACKED_COUNT_OFFSET 1
#define VERTEX_DELTAS_OFFSET 2
#define VERTEX_K_OFFSET 3

#define LOG2_SIZEOF_FLOAT 2
// to avoid sub-word writes we must make sure that each worker processes
// a number of elements so that we fall exactly into a 64-bit load. for floats
// this is 8/sizeof(float) = 2.
#define LOG2_ATOM_SIZE 1

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

.section .text.VERTEX
.align 4

VERTEX:
  setzi $m1, scaled_add_kernel
  runall $m1, $m0, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

// worker variables

// integer variables
#define dataPtr m0
#define count m1
#define remM1 m2
#define final m3
#define countD m4
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
#define k a4

// scratch variables
#define mscratch m10
#define mscratch2 m11
#define ascratch a7

.align 8
scaled_add_kernel:
  // load vertex state
  ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_OFFSET
  ld32 $count, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET
  ld32 $deltaPtr, $mvertex_base, $mzero, VERTEX_DELTAS_OFFSET
  ld32 $k, $mvertex_base, $mzero, VERTEX_K_OFFSET

  // derive count from (end-start)/sizeof(float)
  sub $count, $count, $dataPtr
  shr $count, $count, LOG2_SIZEOF_FLOAT

  // transform the total count into remM1, final and count/6:
  //  where remM1 is the amount of workers (minus 1) that are required to
  //  process an extra atom size of elements, final is the non atom size
  //  remainder the final worker must process (when N is not divisible by the
  //  atoms size) and count is how many elements every worker processes

  // final = count % atomSize
  and $final, $count, 0x1

  // for the rest calculate n / 6 and n % 6 by reciprocal multiplcation
  //   n/6 = (n * 0xAAAB) >> 18
  //   n%6 = n - (n/6)*6
  // where n = count/atomSize
  // see recipe #1 for how these constants were derived:
  //   https://embeddedgurus.com/stack-overflow/2009/06/division-of-integers-by-constants/
  shr $count, $count, LOG2_ATOM_SIZE

  // mscratch = n/6
  setzi $mscratch, 0xAAAB
  mul $mscratch, $count, $mscratch
  shr $mscratch, $mscratch, 18

  // mscratch2 = n%6
  mul $mscratch2, $mscratch, 6
  sub $mscratch2, $count, $mscratch2

  // countPerWorker = (count / atomSize) / numWorkers * atomSize
  shl $count, $mscratch, LOG2_ATOM_SIZE

  // rem = (count / atomSize) % numWorkers + ceil(final, atomSize)
  //  where ceil(x, y) = x / y + (x % y > 0);
  shr $remM1, $final, LOG2_ATOM_SIZE
  add $remM1, $remM1, $mscratch2

  and $mscratch, $final, 0x1
  cmpne $mscratch, $mscratch, $mzero
  add $remM1, $remM1, $mscratch

  // remM1 = rem - 1 (obviously).
  add $remM1, $remM1, -1

  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  {
    // pack out points (in is never used).
    tapack $triPtr, $dataPtr, $deltaPtr, $mzero
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  {
    // process 2 at a time first as this is the optimal scenario
    shr $countD, $count, 1
    setzi $ascratch, ZAACC_BITMASK
  }

  // if worker id is less than the remainder or equal to and final is zero then
  // this worker can process an extra 4.
  cmpslt $mscratch, $workerIdM1, $remM1
  cmpeq $mscratch2, $workerIdM1, $remM1
  movz $mscratch2, $final, $mzero
  or $mscratch, $mscratch, $mscratch2
  add $countD, $countD, $mscratch

  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  // use $data as a temporary scratch register as we can't write to $azeros
  // twice in the same instruction.
  ld2x64pace $azeros, $data, $triPtr+=, $workerIdM1, 0b0101

  {
    brz $countD, .Lvector_epilogue
    // zero accumulators
    uput $FP_CLR, $ascratch
  }

  // each worker's data is interleaved so set a stride of how many workers
  // we have.
  setzi $stride, CTXT_WORKERS

  // preload 4 values and fill the accumulators.
  ld2x64pace $data, $delta, $triPtr+=, $stride, 0b0101
  {
    // minus 1 because we pipeline the first value.
    add $mscratch, $countD, -1
    f32v2axpy $azeros, $delta, $data
  }

  rpt $mscratch, (.Lvector_loop_end - .Lvector_loop_begin) / 8 - 1
.Lvector_loop_begin:
  {
    ld2x64pace $data, $delta, $triPtr+=, $stride, 0b0101
    fnop
  }
  {
    nop
    f32v2axpy $data, $delta, $data
  }
  {
    st64step $data, $mzero, $dataPtr+=, $stride
    fnop
  }
.Lvector_loop_end:
  // store the final 2 processed values.
  f32v2axpy $data, $azeros, $azeros
  st64step $data, $mzero, $dataPtr+=, $stride

.Lvector_epilogue:
  // at most one of our workers will have to do the remaining element. this
  // worker id is equal to the $rem value in the vertex state. the amount
  // of elements remaining is the $final value. $final will be 1 at most.
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lepilogue
  brz $final, .Lepilogue

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

.Lscalar:
  ld32 $datai0, $dataPtr, $mzero, 0
  ld32step $deltai0, $mzero, $deltaPtr+=, 1

  f32v2axpy $azeros, $delta, $data
  f32v2axpy $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

.Lepilogue:
  exitz $mzero

.size VERTEX, .-VERTEX

#endif // __IPU__
