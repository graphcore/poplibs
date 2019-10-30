#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty
#define SUBTRACT_VERTEX(ty) __runCodelet_popops__ScaledSubtractSupervisor___ ## ty
#define AXPLUSBY_VERTEX(ty) __runCodelet_popops__aXPlusbYSupervisor___ ## ty

#define NEGATE_HALF_BY_XOR 0x8000
#define NEGATE_FLOAT_BY_XOR 0x80000000

// vertex state (offsets in bytes)

#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_PACKED_COUNT_OFFSET 2
#define VERTEX_DATA_B_OFFSET 4
// scale variable offset (pointer to tensor or constant half) option.
#define VERTEX_SCALE_OFFSET 6
#define VERTEX_SCALE_B_OFFSET 8
// scale variable offset (float) option.
#define VERTEX_SCALE_OFFSET_FLOAT_CONST 8
#define VERTEX_SCALE_B_OFFSET_FLOAT_CONST 12

// create a new vertex state on the supervisor stack that has the input values
// preprocessed for all of the workers to use.
#define SV_STATE_DATA_OFFSET 0
#define SV_STATE_COUNT_OFFSET 4
#define SV_STATE_REMM1_OFFSET 8
#define SV_STATE_FINAL_OFFSET 12
#define SV_STATE_SCALES_OFFSET 16
#define SV_STATE_DATA_B_OFFSET 20
#define SV_STATE_MEM_CONSTRAINTS 24

#define SV_STATE_SIZE 28

// total space required on the stack
#define STACK_SIZE (SV_STATE_SIZE)

// constants
#define SCALED_PTR32_SHL_BITS 2
#define SCALED_PTR64_SHL_BITS 3

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
#define memConstraints m2
// Flag for memConstraints
#define MEM_CONSTRAINTS_MASK 0x1

#define remM1 m3

#define mscratch m4
#define mscratch2 m5

#define mworkerFunction m6

#define log2AtomSize m7
#define atomSizeMask m8


//******************************************************************************
// Float variant entry points
//******************************************************************************
.globl VERTEX(float_float_float_false_false)
.type VERTEX(float_float_float_false_false), @function
.globl VERTEX(float_float_float_false_true)
.type VERTEX(float_float_float_false_true), @function

.section .text.VERTEX(float_float_float_false)
.align 4
.supervisor
VERTEX(float_float_float_false_false):
  setzi $memConstraints, 0
  bri   1f
VERTEX(float_float_float_false_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $mscratch, $mscratch, SCALED_PTR64_SHL_BITS
 // keeping this before the branch means it doesn't cause a stall later
  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(float).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ld32  $mscratch, $mzero, $mscratch, 0
  bri   VERTEX(supervisor)
.size VERTEX(float_float_float_false), .-VERTEX(float_float_float_false_false)


.globl VERTEX(float_float_float_true_false)
.type VERTEX(float_float_float_true_false), @function
.globl VERTEX(float_float_float_true_true)
.type VERTEX(float_float_float_true_true), @function

.section .text.VERTEX(float_float_float_true)
.align 4
VERTEX(float_float_float_true_false):
  setzi $memConstraints, 0
  bri   1f
VERTEX(float_float_float_true_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ld32  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET_FLOAT_CONST/4
 // keeping this before the branch means it doesn't cause a stall later
  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(float).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  bri   VERTEX(supervisor)
.size VERTEX(float_float_float_true), .-VERTEX(float_float_float_true_false)


.globl SUBTRACT_VERTEX(float_float_false)
.type SUBTRACT_VERTEX(float_float_false), @function
.globl SUBTRACT_VERTEX(float_float_true)
.type SUBTRACT_VERTEX(float_float_true), @function

.section .text.SUBTRACT_VERTEX(float_float)
.align 4
SUBTRACT_VERTEX(float_float_false):
  setzi $memConstraints, 0
  bri   1f
SUBTRACT_VERTEX(float_float_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl $mscratch, $mscratch, SCALED_PTR64_SHL_BITS

  or    $mscratch2, $mzero, NEGATE_FLOAT_BY_XOR
 // keeping this before the branch means it doesn't cause a stall later
  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_FLOAT_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_FLOAT_ATOM_SIZE) - 1

  // pointer to the worker code to run, which needs to do a negate
  setzi $mworkerFunction, VERTEX(float).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ld32  $mscratch, $mzero, $mscratch, 0
  xor   $mscratch, $mscratch, $mscratch2 // 6 cycles

  bri   VERTEX(supervisor)
.size SUBTRACT_VERTEX(float_float), .-SUBTRACT_VERTEX(float_float_false)


//******************************************************************************
// Half variant entry points
//******************************************************************************

.globl VERTEX(half_half_half_false_false)
.type VERTEX(half_half_half_false_false), @function
.globl VERTEX(half_half_half_false_true)
.type VERTEX(half_half_half_false_true), @function

.section .text.VERTEX(half_half_half_false)
.align 4
VERTEX(half_half_half_false_false):
  setzi $memConstraints, 0
  bri   1f

VERTEX(half_half_half_false_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl $mscratch, $mscratch, SCALED_PTR64_SHL_BITS

  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(half).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ldz16 $mscratch, $mzero, $mscratch, 0
  bri   VERTEX(supervisor) // 6 cycles
.size VERTEX(half_half_half_false), .-VERTEX(half_half_half_false_false)


.globl VERTEX(half_half_half_true_false)
.type VERTEX(half_half_half_true_false), @function
.globl VERTEX(half_half_half_true_true)
.type VERTEX(half_half_half_true_true), @function

.section .text.VERTEX(half_half_half_true)
.align 4
VERTEX(half_half_half_true_false):
  setzi $memConstraints, 0
  bri 1f
VERTEX(half_half_half_true_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2

  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(half).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  bri   VERTEX(supervisor) // 6 cycles
.size VERTEX(half_half_half_true), .-VERTEX(half_half_half_true_false)


.globl SUBTRACT_VERTEX(half_half_false)
.type SUBTRACT_VERTEX(half_half_false), @function
.globl SUBTRACT_VERTEX(half_half_true)
.type SUBTRACT_VERTEX(half_half_true), @function

.section .text.SUBTRACT_VERTEX(half_half)
.align 4
SUBTRACT_VERTEX(half_half_false):
  setzi $memConstraints, 0
  bri   1f
SUBTRACT_VERTEX(half_half_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $mscratch, $mscratch, SCALED_PTR64_SHL_BITS

  setzi $mscratch2, NEGATE_HALF_BY_XOR
  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(half).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ldz16 $mscratch, $mzero, $mscratch, 0
  xor   $mscratch, $mscratch, $mscratch2 // 6 cycles
  bri   VERTEX(supervisor) // 6 cycles
.size SUBTRACT_VERTEX(half_half), .-SUBTRACT_VERTEX(half_half_false)


.globl AXPLUSBY_VERTEX(half_true_false)
.type AXPLUSBY_VERTEX(half_true_false), @function
.globl AXPLUSBY_VERTEX(half_true_true)
.type AXPLUSBY_VERTEX(half_true_true), @function

.section .text.AXPBY_VERTEX(half_true)
.align 4
AXPLUSBY_VERTEX(half_true_false):
  setzi $memConstraints, 0
  bri   1f
AXPLUSBY_VERTEX(half_true_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  // 2x16 bit factors
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  ldz16  $mscratch2, $vertexPtr, $mzero, VERTEX_SCALE_B_OFFSET/2

  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(axplusby_half).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2

  nop
  sort4x16lo $mscratch, $mscratch, $mscratch2
  bri   VERTEX(supervisor) // 6 cycles
.size AXPLUSBY_VERTEX(half_true), .-AXPLUSBY_VERTEX(half_true_false)


.globl AXPLUSBY_VERTEX(half_false_false)
.type AXPLUSBY_VERTEX(half_false_false), @function
.globl AXPLUSBY_VERTEX(half_false_true)
.type AXPLUSBY_VERTEX(half_false_true), @function

.section .text.AXPBY_VERTEX(half_false)
.align 4
AXPLUSBY_VERTEX(half_false_false):
  setzi $memConstraints, 0
  bri   1f
AXPLUSBY_VERTEX(half_false_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  ldz16  $mscratch2, $vertexPtr, $mzero, VERTEX_SCALE_B_OFFSET/2
  shl    $mscratch, $mscratch, SCALED_PTR64_SHL_BITS
  shl    $mscratch2, $mscratch2, SCALED_PTR64_SHL_BITS
  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(axplusby_half).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2

  // load factors using pointers - here to avoid pipeline hit
  nop
  ldz16  $mscratch, $mzero, $mscratch, 0
  ldz16  $mscratch2, $mzero, $mscratch2, 0
  sort4x16lo $mscratch, $mscratch, $mscratch2

  bri   VERTEX(supervisor) // 6 cycles
.size AXPLUSBY_VERTEX(half_false), .-AXPLUSBY_VERTEX(half_false_false)


//******************************************************************************
// Mixed precision variant entry points
//******************************************************************************
.globl VERTEX(half_float_half_false_false)
.type VERTEX(half_float_half_false_false), @function
.globl VERTEX(half_float_half_false_true)
.type VERTEX(half_float_half_false_true), @function

.section .text.VERTEX(half_float_half_false)
.align 4
VERTEX(half_float_half_false_false):
  setzi $memConstraints, 0
  bri   1f
VERTEX(half_float_half_false_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl $mscratch, $mscratch, SCALED_PTR64_SHL_BITS

  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(half_float).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ldz16 $mscratch, $mzero, $mscratch, 0
  bri   VERTEX(supervisor) // 6 cycles

.size VERTEX(half_float_half_false), .-VERTEX(half_float_half_false_false)

.globl VERTEX(half_float_half_true_false)
.type VERTEX(half_float_half_true_false), @function
.globl VERTEX(half_float_half_true_true)
.type VERTEX(half_float_half_true_true), @function


.section .text.VERTEX(half_float_half_true)
.align 4
VERTEX(half_float_half_true_false):
  setzi $memConstraints, 0
  bri   1f
VERTEX(half_float_half_true_true):
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(half_float).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  bri   VERTEX(supervisor) // 6 cycles
.size VERTEX(half_float_half_true), .-VERTEX(half_float_half_true_false)


.globl SUBTRACT_VERTEX(half_float_false)
.type SUBTRACT_VERTEX(half_float_false), @function
.globl SUBTRACT_VERTEX(half_float_true)
.type SUBTRACT_VERTEX(half_float_true), @function

.section .text.SUBTRACT_VERTEX(half_float)
.align 4
SUBTRACT_VERTEX(half_float_false):
  setzi $memConstraints, 0
  bri   1f
SUBTRACT_VERTEX(half_float_true):
  setzi $memConstraints, 1
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  shl    $mscratch, $mscratch, SCALED_PTR64_SHL_BITS

  setzi $mscratch2, NEGATE_HALF_BY_XOR
  add   $sp, $sp, -STACK_SIZE
  setzi $log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $mworkerFunction, VERTEX(half_float).kernel
  ldz16  $countD2, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ldz16 $mscratch, $mzero, $mscratch, 0
  xor   $mscratch, $mscratch, $mscratch2 //6 cycles
  bri   VERTEX(supervisor) // 6 cycles

.size SUBTRACT_VERTEX(half_float), .-SUBTRACT_VERTEX(half_float_false)

//******************************************************************************
// common supervisor code for mixed precision, half and float vertices.
//******************************************************************************
.globl VERTEX(supervisor)
.type VERTEX(supervisor), @function

.section .text.VERTEX(supervisor)
.align 4
VERTEX(supervisor):
  ldz16  $mscratch2, $vertexPtr, $mzero, VERTEX_DATA_A_OFFSET/2
  ldz16  $remM1, $vertexPtr, $mzero, VERTEX_DATA_B_OFFSET/2

  st32 $memConstraints, $sp, $mzero, SV_STATE_MEM_CONSTRAINTS/4
  // transform the total count into remM1, final and count/6:
  //  where remM1 is the amount of workers (minus 1) that are required to
  //  process an extra atom size of elements, final is the non atom size
  //  remainder the final worker must process (when N is not divisible by the
  //  atoms size) and count is how many elements every worker processes

  st32  $mscratch, $sp, $mzero, SV_STATE_SCALES_OFFSET/4
  // for the rest calculate n / 6 and n % 6 by reciprocal multiplcation
  //   n/6 = (n * 0xAAAB) >> 18
  //   n%6 = n - (n/6)*6
  // where n = count/atomSize
  // see recipe #1 for how these constants were derived:
  //   https://embeddedgurus.com/stack-overflow/2009/06/division-of-integers-by-constants/
  setzi $mscratch, 0xAAAB
  // final = count % atomSize
  and   $final, $countD2, $atomSizeMask
  shr $countD2, $countD2, $log2AtomSize

  shl    $mscratch2, $mscratch2, SCALED_PTR64_SHL_BITS
  shl    $remM1, $remM1, SCALED_PTR64_SHL_BITS

  // mscratch = n/6
  mul $mscratch, $countD2, $mscratch // 6 cycles
  shr $mscratch, $mscratch, 18 // 6 cycles

  st32  $mscratch2, $sp, $mzero, SV_STATE_DATA_OFFSET
  st32  $remM1, $sp, $mzero, SV_STATE_DATA_B_OFFSET/4

  // rem = (count / atomSize) % numWorkers + ceil(final, atomSize)
  //  where ceil(x, y) = x / y + (x % y > 0);
  shr $remM1, $final, $log2AtomSize

  // Avoid mscratch register bubble
  nop
  nop
  // mscratch2 = n%6
  mul $mscratch2, $mscratch, 6
  sub $mscratch2, $countD2, $mscratch2 // 6 cycles

  // countPerWorker = (count / atomSize) / numWorkers * atomSize
  shl $countD2, $mscratch, $log2AtomSize

  and   $mscratch, $final, $atomSizeMask
  cmpne $mscratch, $mscratch, $mzero // 6 cycles

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
  st32 $remM1, $sp, $mzero, SV_STATE_REMM1_OFFSET/4 // 6 cycles if final != 0
  st32 $final, $sp, $mzero, SV_STATE_FINAL_OFFSET/4
  st32 $countD2, $sp, $mzero, SV_STATE_COUNT_OFFSET/4

  runall $mworkerFunction, $sp, 0 // 6 cycles
  // restore the stack pointer that was changed in the supervisor common code.
  add  $sp, $sp, STACK_SIZE
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

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
#undef memConstraints

// worker variables

// integer variables
#define memConstraints m0
#define dataPtr m1
#define remM1 m2
#define final m3
#define countD2 m4
#define countD4 m4
#define dataBPtr m5
#define triPtr m6:7
#define triPtri0 m6
#define triPtri1 m7
#define workerIdM1 m8
#define stride m9

#define data a0:1
#define datai0 a0
#define datai1 a1
#define dataBHiLo a4:7
#define dataB a4:5
#define dataBHi a6:7
#define dataBi0 a4
#define dataBi1 a5
#define result a2:3
#define k a6

// scratch variables
#define mscratch m10
#define ascratch a7


.macro CHOOSE_FAST_OR_SLOW_PATH SLOW_PATH_LABEL
  // The fast path is always OK if constraints were applied
  brnz $memConstraints, 1f
  // Or if the data start is far enough apart.  It could be ok in some other
  // circumstances but this is time consuming to check correctly.
  sub $memConstraints, $dataPtr, $dataBPtr
  abs $memConstraints, $memConstraints
  // +8 is to account for really wanting a <= instruction
  cmpult $memConstraints, $memConstraints, (2 * TMEM_ELEMSIZE) + 8
  brnz $memConstraints, \SLOW_PATH_LABEL
1:
.endm

.type VERTEX(float).kernel, @function

.section .text.VERTEX(float).kernel
.align 8
.worker
  nop // rpt alignment

VERTEX(float).kernel:
  // load vertex state
  ld32 $countD2, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET/4
  ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET/4
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET/4
  ld32 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
  ld32 $memConstraints, $mvertex_base, $mzero, SV_STATE_MEM_CONSTRAINTS/4
  {
    ld32 $dataPtr, $mvertex_base, $mzero, SV_STATE_DATA_OFFSET
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $dataBPtr, $mvertex_base, $mzero, SV_STATE_DATA_B_OFFSET/4
    uput $FP_CLR, $ascratch
  }

  {
    get $workerIdM1, $WSR
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  // process 2 at a time first as this is the optimal scenario
  shr $countD2, $countD2, 1

  // if worker id is less than the remainder this worker can process an extra 4.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD2, $countD2, $mscratch

  // pack out points (in is never used).
  tapack $triPtr, $dataPtr, $dataBPtr, $mzero
  CHOOSE_FAST_OR_SLOW_PATH .Lfloat_slow_path
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1


  // offset each worker's pointer into the data to interleave them.
  // use $data as a temporary scratch register as we can't write to $azeros
  // twice in the same instruction.
  ld2x64pace $azeros, $data, $triPtr+=, $workerIdM1, 0b0101

  brz $countD2, .Lfloat_loop_epilogue

  // each worker's data is interleaved so set a stride of how many workers
  // we have.
  setzi $stride, CTXT_WORKERS

  // preload 4 values and fill the accumulators.
  ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
  {
    // minus 1 because we pipeline the first value.
    add $mscratch, $countD2, -1
    f32v2axpy $azeros, $dataB, $data
  }
  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {
    ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
    f32v2axpy $result, $azeros, $azeros
  }
  {
    st64step $result, $mzero, $dataPtr+=, $stride
    f32v2axpy $azeros, $dataB, $data
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

  // unpack the data and dataB pointers from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  and $dataPtr, $triPtri0, $mscratch
  and $dataBPtr, $triPtri1, $mscratch
.Lfloat_epilogue_common:
  // scalar.
  {ld32 $datai0, $dataPtr, $mzero, 0
    // zero the top half of data and dataB so we can safely accumulate them
    zero $datai1}
  {ld32step $dataBi0, $mzero, $dataBPtr+=, 1
    zero $dataBi1}

  f32v2axpy $azeros, $dataB, $data
  f32v2axpy $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

.Lfloat_epilogue:
  exitz $mzero

.align 8
  nop // rpt align
// No assumptions made about operands being in different memory segments
.Lfloat_slow_path:
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  brz $countD2, .Lfloat_loop_epilogue_slow

  mov $triPtri0, $dataPtr
  ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
  ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   // minus 1 because we pipeline the first value.
  {add $mscratch, $countD2, -1
   f32v2axpy $azeros, $dataB, $data}

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
   f32v2axpy $result, $azeros, $azeros}
  {ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   fnop}
  {st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS
   f32v2axpy $azeros, $dataB, $data}
2:
  f32v2axpy $result, $azeros, $azeros
  st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS

.Lfloat_loop_epilogue_slow:
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lfloat_epilogue
  brnz $final, .Lfloat_epilogue_common
  exitz $mzero

.size VERTEX(float).kernel, .-VERTEX(float).kernel
//******************************************************************************
.type VERTEX(half).kernel, @function

.section .text.VERTEX(half).kernel

.globl VERTEX(half_half_float_continue)

.align 8
VERTEX(half_half_float_continue):
  // load k with a float scale, cast it and continue as normal
  {ld32  $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
   f16v2exp $ascratch, $azero}
   f32tof16 $k, $k
  {
    bri not_axplusby_continue
    // $k should have the form of {1, k}
    sort4x16lo $k, $ascratch, $k
  }

VERTEX(half).kernel:
  {
    ldb16 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/2
    f16v2exp $ascratch, $azero
  }

  {
    bri not_axplusby_continue
    // $k should have the form of {1, k}
    sort4x16lo $k, $ascratch, $k
  }

VERTEX(axplusby_half).kernel:
  ld32 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
not_axplusby_continue:
  ld32 $memConstraints, $mvertex_base, $mzero, SV_STATE_MEM_CONSTRAINTS/4

  get $workerIdM1, $WSR
   // load vertex state
  {
    ld32 $countD4, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET/4
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET/4
    uput $FP_CLR, $ascratch
  }
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET/4

  ld32 $dataPtr, $mvertex_base, $mzero, SV_STATE_DATA_OFFSET
  ld32 $dataBPtr, $mvertex_base, $mzero, SV_STATE_DATA_B_OFFSET/4

  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
 // pack out points (in is never used).
  tapack $triPtr, $dataPtr, $dataBPtr, $mzero

  // process 4 at a time first as this is the optimal scenario
  shr $countD4, $countD4, 2

  // if worker id is less than the remainder this worker can process an extra 4.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD4, $countD4, $mscratch

  CHOOSE_FAST_OR_SLOW_PATH .Lhalf_slow_path
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1

  // offset each worker's pointer into the data to interleave them.
  // use $data as a temporary scratch register as we can't write to $azeros
  // twice in the same instruction.
  ld2x64pace $azeros, $data, $triPtr+=, $workerIdM1, 0b0101

  brz $countD4, .Lhalf_loop_epilogue

  // each worker's data is interleaved so set a stride of how many workers
  // we have.
  setzi $stride, CTXT_WORKERS

  // preload 4 values and fill the accumulators.
  ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
  {
    // minus 1 because we pipeline the first value.
    add $mscratch, $countD4, -1
    f16v4mix $azeros, $data, $dataB
  }
  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {
    ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
    f16v4mix $result, $azeros, $azeros
  }
  {
    st64step $result, $mzero, $dataPtr+=, $stride
    f16v4mix $azeros, $data, $dataB
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

  // unpack the data and dataB pointers from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  and $dataPtr, $triPtri0, $mscratch
  and $dataBPtr, $triPtri1, $mscratch
.Lhalf_epilogue_common:
  {
    // is there at least 2 left?
    cmpult $mscratch, $final, 2
    // zero the top half of data and dataB so we can safely accumulate them
    // for the x2 and x1 cases.
    zero $datai1
  }
  {
    brnz $mscratch, .Lhalf_scalar
    zero $dataBi1
  }

  // remainder 2
  ld32 $datai0, $dataPtr, $mzero, 0
  ld32step $dataBi0, $mzero, $dataBPtr+=, 1

  f16v4mix $azeros, $data, $dataB
  f16v4mix $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

  // finish now if that's all.
  cmpeq $mscratch, $final, 2
  brnz $mscratch, .Lhalf_epilogue

.Lhalf_scalar:
  ldb16 $datai0, $dataPtr, $mzero, 0
  ldb16 $dataBi0, $dataBPtr, $mzero, 0

  f16v4mix $azeros, $data, $dataB

  {
    // load the last word and perform a read/modify/write.
    ld32 $ascratch, $dataPtr, $mzero, 0
    f16v4mix $data, $azeros, $azeros
  }

  sort4x16hi $ascratch, $datai0, $ascratch
  st32 $ascratch, $dataPtr, $mzero, 0

.Lhalf_epilogue:
  exitz $mzero
.align 8
  nop // rpt align
// No assumptions made about operands being in different memory segments
.Lhalf_slow_path:
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  brz $countD2, .Lhalf_loop_epilogue_slow

  mov $triPtri0, $dataPtr
  ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
  ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   // minus 1 because we pipeline the first value.
  {add $mscratch, $countD2, -1
   f16v4mix $azeros, $data, $dataB}

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
   f16v4mix $result, $azeros, $azeros}
  {ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   fnop}
  {st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS
   f16v4mix $azeros, $data, $dataB}
2:
  f16v4mix $result, $azeros, $azeros
  st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS

.Lhalf_loop_epilogue_slow:
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lhalf_epilogue
  brnz $final, .Lhalf_epilogue_common
  exitz $mzero

.size VERTEX(half).kernel, .-VERTEX(half).kernel

//******************************************************************************
// variant that accepts dataBs, as floats and data, factor as half
//******************************************************************************

// defines for selection of strides to use in addressing
#define DATAp0_DATABp1 0b1100
#define DATAp6_DATABp11 0b0110
#define DATAWp6_DATABp1 0b0100


.type VERTEX(half_float).kernel, @function

.section .text.VERTEX(half_float).kernel

// Alignment to ensure repeat body is 8 byte aligned
.align 8
VERTEX(half_float).kernel:
  // load vertex state
  ld32 $countD4, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET/4
  ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET/4
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET/4

  {
    ld32 $dataPtr, $mvertex_base, $mzero, SV_STATE_DATA_OFFSET
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $dataBPtr, $mvertex_base, $mzero, SV_STATE_DATA_B_OFFSET/4
    uput $FP_CLR, $ascratch
  }
  {
    ldb16 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/2
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

  // process 4 at a time first as this is the optimal scenario
  shr $countD4, $countD4, 2

  // if worker id is less than the remainder this worker can process an extra 4.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD4, $countD4, $mscratch


  // advance dataptr(halves) by workerId * (workers * 8 bytes)
  // advance dataBptr(floats) by workerId * (workers * 16 bytes)
  ld64step $data, $mzero, $dataPtr+=, $workerIdM1
  ld64step $data, $mzero, $dataBPtr+=, $workerIdM1
  ld64step $data, $mzero, $dataBPtr+=, $workerIdM1

  brz $countD4, .Lhalf_loop_epilogue2

  // There are slow/fast loops.  We must use the slow loop if:
  // There are < 3 groups of 4 to process OR the elements that the data is in
  // do not allow ld2x64pace instructions
  ld32 $memConstraints, $mvertex_base, $mzero, SV_STATE_MEM_CONSTRAINTS/4
  CHOOSE_FAST_OR_SLOW_PATH .Lhalf_float_slow_loop

  // minus 3 because we pipeline with code pre/post the loop.
  add $mscratch, $countD4, -3
  brpos $mscratch, .Lmore_than_two_loops

.Lhalf_float_slow_loop:
  // One or two groups of 4 to process - the optimal loop body below is piplelined to the
  // extent that it needs to process 3 or more items.  This one is slower but
  // has minimal pipelining
  add $mscratch, $countD4, -1


  ld64step $dataB, $mzero, $dataBPtr+=, 1
  ld64step $dataBHi, $mzero, $dataBPtr+=, (2*CTXT_WORKERS)-1
  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {
    ld64    $data, $mzero, $dataPtr, 0
    f32v4tof16 $dataBHi, $dataBHiLo
  }
  {
    ld64step $dataB, $mzero, $dataBPtr+=, 1
    f16v4mix $result, $dataBHi, $data
  }
  {
    ld64step $dataBHi, $mzero, $dataBPtr+=, (2*CTXT_WORKERS)-1
    f16v4gacc $result
  }
  {
    st64step $result, $mzero, $dataPtr+=, CTXT_WORKERS
    fnop
  }
2:
  {
    ld64    $data, $mzero, $dataPtr, 0
    f32v4tof16 $dataBHi, $dataBHiLo
  }

  f16v4mix $result, $dataBHi, $data

  f16v4gacc $result
  st64step $result, $mzero, $dataPtr+=, CTXT_WORKERS

  bri      .Lhalf_loop_epilogue2

// Align here, as a repeat will follow this label, which we only ever branch to
.align 8
  nop
.Lmore_than_two_loops:
  tapack $triPtr, $dataBPtr, $dataPtr, $dataPtr
  // each worker's data is interleaved so set a stride of how many workers
  // we have for data addresses and 2*workers -1 for dataB addressing
  setzi $stride, ((2 * CTXT_WORKERS - 1) << 10) | CTXT_WORKERS

  // preload values and fill the accumulators.
  ld2x64pace $dataB, $result, $triPtr+=, $stride, DATAp0_DATABp1
  ld2x64pace $dataBHi, $data, $triPtr+=, $stride, DATAp6_DATABp11
  {
    ld2x64pace $dataB, $result, $triPtr+=, $stride, DATAp0_DATABp1
    f32v4tof16 $dataBHi, $dataBHiLo
  }
  {
    ld2x64pace $dataBHi, $data, $triPtr+=, $stride, DATAp6_DATABp11
    f16v4mix   $result, $dataBHi, $data
  }
  {
    ld2x64pace $dataB, $result, $triPtr+=, $stride, DATAp0_DATABp1
    f32v4tof16 $dataBHi, $dataBHiLo
  }

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {
    ld2x64pace $dataBHi, $data, $triPtr+=, $stride, DATAp6_DATABp11
    f16v4mix   $result, $dataBHi, $data
  }
  {
    ldst64pace $dataB, $result, $triPtr+=, $stride, DATAWp6_DATABp1
    f32v4tof16 $dataBHi, $dataBHiLo
  }
2:
  // store the final results. At the loop end/no loop we have already:
  // Read 64 bits of data that have not yet been processed
  // Read 64 bits of dataBs that have not yet been processed (2nd 64 bits still to read)
  // a result ready to read from the accumulators
  // Got a dataB Hi 4xhalf already cast
  // We need to complete processing this data, otherwise we'd have significant overread
  // due to the per worker stride

  // Read the dataBHi and data words to go with the dataB(Lo) that's already read
  {
    ld2x64pace $dataBHi, $data, $triPtr+=, $stride, DATAp6_DATABp11
    f16v4mix $result, $dataBHi, $data
  }

  // extract packed pointers as need a dataW ptr now. Extracting/modifying the
  // data read ptr is faster, however the data read pointer has already advanced
  // so adjust it backward to generate the write pointer
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  and     $dataBPtr, $triPtri0, $mscratch
  and     $dataPtr, $triPtri1, $mscratch
  {
    add    $dataPtr, $dataPtr, -(8*3*CTXT_WORKERS)
    f32v4tof16  $dataBHi, $dataBHiLo
  }

  {
    st64step    $result, $mzero, $dataPtr+=,CTXT_WORKERS
    f16v4mix    $result, $dataBHi, $data
  }
  {
    st64step    $result, $mzero, $dataPtr+=,CTXT_WORKERS
    f16v4gacc   $result
  }
  st64step    $result, $mzero, $dataPtr+=,CTXT_WORKERS

.Lhalf_loop_epilogue2:
  // at most one of our workers will have to do the remaining elements. this
  // worker id is equal to the $rem value in the vertex state. the amount
  // of elements remaining is the $final value. $final will be 3 at most.
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lhalf_epilogue2
  brz $final, .Lhalf_epilogue2

  {
    // are there at least 2 left?
    cmpult $mscratch, $final, 2
    // zero the top half of data and dataB so we can safely accumulate them
    // for the x2 and x1 cases.
    zero $datai1
  }
  {
    brnz $mscratch, .Lhalf_scalar2
    zero $dataBi1
  }

  // remainder 2
  ld32 $datai0, $dataPtr, $mzero, 0
  ld64step $dataB, $mzero, $dataBPtr+=, 1

  f32v2tof16 $dataBi0, $dataB

  f16v4mix $azeros, $dataB, $data
  {
    cmpeq $mscratch, $final, 2
    f16v4gacc $data
  }

  st32step $datai0, $mzero, $dataPtr+=, 1

  // finish now if that's all.
  brnz $mscratch, .Lhalf_epilogue2

.Lhalf_scalar2:
  ld32 $dataBi0, $dataBPtr, $mzero, 0
  {
    ldb16 $datai0, $dataPtr, $mzero, 0
    f32tof16 $dataBi0, $dataBi0
  }
  f16v4mix $azeros, $dataB, $data
  {
    // load the last word and perform a read/modify/write.
    ld32 $ascratch, $dataPtr, $mzero, 0
    f16v4gacc $data
  }

  sort4x16hi $ascratch, $datai0, $ascratch
  st32 $ascratch, $dataPtr, $mzero, 0

.Lhalf_epilogue2:
  exitz $mzero

.size VERTEX(half_float).kernel, .-VERTEX(half_float).kernel

#endif // __IPU__
