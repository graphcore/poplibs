#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

/* -------------------------------------------------------------------------- */
// Dynamic Slice and Dynamic Update Slice vertex code for int, half and float
// variants
/* -------------------------------------------------------------------------- */
// Register aliases
#define COPY_PTR        m0
#define BASET           m1

#define SUBT            m2
#define BASE_ELEM       m3
#define SUB_ELEM        m4
#define BASET_DELTA_PTR m5
#define SUBT_DELTAN_PTR m6

#define mSCRATCH        m7
#define mSCRATCH2       m8

#define BASE_SLICE      m9
#define BASE_IDX        m10
#define REGION_SIZE     m11

#define VAL12   a0:1
#define VAL1    a0
#define VAL2    a1
#define VAL3    a2
#define OFFSET  a4

//****************************************************************************
// The input structure parameters:
// 32 bit offset
// 32 ptr baseT
// 16 ptr deltaN
// 16 ptr numSubElements
// 32 ptr subT
// 16 ptr deltaN
// 16 bit numregions
// 32 bit numBaseElements
//****************************************************************************
#define VOFF_OFFSET       0
#define VOFF_BASET        1
#define VOFF_BASET_DELTAN (2*2)    // 16 bit - offsets for use with 16 bit loads
#define VOFF_SUB_ELEM     (2*2+1)  // 16 bit - offsets for use with 16 bit loads
#define VOFF_SUBT         3
#define VOFF_SUBT_DELTAN  (4*2)    // 16 bit - offsets for use with 16 bit loads
#define VOFF_REGIONS      (4*2+1)  // 16 bit - offsets for use with 16 bit loads
#define VOFF_BASE_ELEM    5

#define DELTAN_BASE_PTR_BITS 20
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)
#define DELTAN_OFFSET_BITS 18
#define DELTAN_OFFSET_MASK ((1 << DELTAN_OFFSET_BITS) - 1)

//****************************************************************************
// Worker base local variable storage
//****************************************************************************
#define WOFF_REGIONS         0
#define WOFF_BASET_BASE      1
#define WOFF_SUBT_BASE       2

//******************************************************************************
// Labels names for each variant
//******************************************************************************
#define DS_HALF_FUNC __runCodelet_popops__DynamicSlice2d___half
#define DUS_HALF_FUNC __runCodelet_popops__DynamicUpdateSlice2d___half

#define DS_FLOAT_FUNC __runCodelet_popops__DynamicSlice2d___float
#define DUS_FLOAT_FUNC __runCodelet_popops__DynamicUpdateSlice2d___float

#define DS_INT_FUNC __runCodelet_popops__DynamicSlice2d___int
#define DUS_INT_FUNC __runCodelet_popops__DynamicUpdateSlice2d___int

#define DS_UNSIGNED_FUNC __runCodelet_popops__DynamicSlice2d___unsigned_int
#define DUS_UNSIGNED_FUNC __runCodelet_popops__DynamicUpdateSlice2d___unsigned_int

.globl DUS_HALF_FUNC
.type DUS_HALF_FUNC, @function

.globl DS_HALF_FUNC
.type DS_HALF_FUNC, @function

.globl DUS_FLOAT_FUNC
.type DUS_FLOAT_FUNC, @function

.globl DS_FLOAT_FUNC
.type DS_FLOAT_FUNC, @function

.globl DUS_INT_FUNC
.type DUS_INT_FUNC, @function

.globl DS_INT_FUNC
.type DS_INT_FUNC, @function

.globl DUS_UNSIGNED_FUNC
.type DUS_UNSIGNED_FUNC, @function

.globl DS_UNSIGNED_FUNC
.type DS_UNSIGNED_FUNC, @function

//******************************************************************************
// Entry points for each function, each setting up a pointer to the copy
// function and then using common code
//******************************************************************************
.section .text.DynamicSlice2d_half_entry

DynamicSlice2d_half_entry:
.align 4
DUS_HALF_FUNC:
   setzi   $COPY_PTR, copy_loop_DUS_16
   bri     DynamicSlice2d_common

DS_HALF_FUNC:
   setzi   $COPY_PTR, copy_loop_DS_16
    bri    DynamicSlice2d_common

.size DynamicSlice2d_half_entry, .-DynamicSlice2d_half_entry

.section .text.DynamicSlice2d_32_entry

DUS_FLOAT_FUNC:
DUS_INT_FUNC:
DUS_UNSIGNED_FUNC:
   setzi   $COPY_PTR, copy_loop_DUS_32
   bri     DynamicSlice2d_common

DS_FLOAT_FUNC:
DS_INT_FUNC:
DS_UNSIGNED_FUNC:
   setzi   $COPY_PTR, copy_loop_DS_32
   bri     DynamicSlice2d_common

.size DynamicSlice2d_32_entry, .-DynamicSlice2d_32_entry

//******************************************************************************
// Common code - loop for fetching pointers to regions
//******************************************************************************
.section .text.DynamicSlice2d_common
.align 4
DynamicSlice2d_common:
    // Load most of the vertex state
    ld32     $mSCRATCH,$mzero, $mvertex_base, VOFF_OFFSET
    ld32     $OFFSET, $mzero, $mSCRATCH, 0

    // Load, unpack and store BaseT base pointer and DELTAN pointer
    // Unpack BASET pointer. Abandon COUNT for now. Save it to  worker stack
    // BaseT Base pointer
    ldconst  $mSCRATCH2, DELTAN_BASE_PTR_MASK
    ld32     $BASET, $mzero, $mvertex_base, VOFF_BASET
    and      $BASET, $BASET, $mSCRATCH2
    st32     $BASET, $mzero, $mworker_base, WOFF_BASET_BASE

    // SubT Base pointer
    ld32     $SUBT, $mzero, $mvertex_base, VOFF_SUBT
    // $mSCRATCH2 still preloaded with DELTAN_BASE_PTR_MASK
    and      $SUBT, $SUBT, $mSCRATCH2
    st32     $SUBT, $mzero, $mworker_base, WOFF_SUBT_BASE

    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR.
    // Actually added offset to reclaim a register rather than
    // using extra offset parameters for loads
    // BaseT DELTAN
    ldz16    $mSCRATCH, $mzero, $mvertex_base, VOFF_BASET_DELTAN
    setzi    $mSCRATCH2, TMEM_REGION0_BASE_ADDR
    shl      $mSCRATCH, $mSCRATCH, 2
    add      $BASET_DELTA_PTR, $mSCRATCH, $mSCRATCH2

    // SubT DELTAN - uses separate reg to avoid load/store inside inner loop
    ldz16    $mSCRATCH, $mzero, $mvertex_base, VOFF_SUBT_DELTAN
    shl      $mSCRATCH, $mSCRATCH, 2
    // $mSCRATCH2 still preloaded with TMEM_REGION0_BASE_ADDR
    add      $SUBT_DELTAN_PTR, $mSCRATCH, $mSCRATCH2

    // Load rest of the state constants
    ld32     $BASE_ELEM, $mzero, $mvertex_base, VOFF_BASE_ELEM
    ldz16    $REGION_SIZE, $mzero, $mvertex_base, VOFF_REGIONS

    zero    $BASE_IDX
    add     $mSCRATCH, $REGION_SIZE, -1
.Louter_loop:
    st32    $mSCRATCH, $mzero, $mworker_base, WOFF_REGIONS

    // Apply the offset to the base index and check the result is in bounds.
    // Result is base slice
    atom    $mSCRATCH, $OFFSET
    add     $BASE_SLICE, $BASE_IDX, $mSCRATCH
    add     $mSCRATCH, $BASE_IDX, $BASE_ELEM
    cmpult  $mSCRATCH, $BASE_SLICE, $mSCRATCH
    brnz    $mSCRATCH, 3f
    mov     $BASE_SLICE, $BASE_IDX
3:
    // Setup loop over subElements count
    ldz16   $SUB_ELEM, $mzero, $mvertex_base, VOFF_SUB_ELEM
    add     $SUB_ELEM, $SUB_ELEM, -1
.Lsub_elem_loop:
    // ***** Jump to addresses unpacking function *******
    bri     DynamicSlice2d_unpack_addr_and_copy

    // Not a local label as referenced across sections.
DynamicSlice_loop_resume:
    // Increment the base slice and check the result is within bounds
    add     $BASE_SLICE, $BASE_SLICE, 1
    add     $mSCRATCH, $BASE_IDX, $BASE_ELEM
    cmpult  $mSCRATCH, $BASE_SLICE, $mSCRATCH
    brnz    $mSCRATCH, 3f
    sub     $BASE_SLICE, $BASE_SLICE, $BASE_ELEM
3:
    brnzdec $SUB_ELEM, .Lsub_elem_loop

    // add num Base elements to the base index for the next region
    add     $BASE_IDX, $BASE_IDX, $BASE_ELEM
    ld32    $mSCRATCH, $mzero, $mworker_base, WOFF_REGIONS

    brnzdec $mSCRATCH, .Louter_loop
    exitz   $mzero
.size DynamicSlice2d_common, .-DynamicSlice2d_common

//******************************************************************************
// Addresses unpacking step
//******************************************************************************
.section .text.DynamicSlice2d_unpack_addr_and_copy
DynamicSlice2d_unpack_addr_and_copy:
.align 8
    // Get a region size = BaseT[baseIdx]
    ld32     $mSCRATCH, $mzero, $BASET_DELTA_PTR, $BASE_IDX
    shr      $REGION_SIZE, $mSCRATCH, DELTAN_OFFSET_BITS // get size

    // Build src ptr = &BaseT[regionIdx * numBaseElements + baseSlice]
    ld32     $mSCRATCH, $mzero, $BASET_DELTA_PTR, $BASE_SLICE
    ldconst  $mSCRATCH2, DELTAN_OFFSET_MASK
    ld32     $BASET, $mzero, $mworker_base, WOFF_BASET_BASE
    and      $mSCRATCH, $mSCRATCH, $mSCRATCH2
    // Offset and base address will be added below based on required Op
    // Do not re-use $BASET and $mSCRATCH pair until copy function

    // Build dst pointer
    // Load next SubT DeltaN (next subSlice info)
    ld32step $SUBT, $mzero, $SUBT_DELTAN_PTR+=, 1

    // Construct destination region pointer
    and      $SUBT, $SUBT, $mSCRATCH2 // get offset
    ld32     $mSCRATCH2, $mzero, $mworker_base, WOFF_SUBT_BASE
    // Offset and base address will be added below based on required Op
    // Do not re-use $SUBT and $mSCRATCH2 pair until copy function

    // ***** Jump to selected function which loads ptrs and copies *******
    br       $COPY_PTR

.size DynamicSlice2d_unpack_addr_and_copy, .-DynamicSlice2d_unpack_addr_and_copy

//******************************************************************************
// copy loop for DynamicSlice2d and DynamicUpdateSlice2d (float or int)
//******************************************************************************
.section .text.DynamicSlice2d_32_copy
.align 8
    nop // rpt align
DynamicSlice2d_32_copy:
// load the next SUBT pointer to use as a source
// Index to load the next BASET pointer to use as a destination
copy_loop_DUS_32:
    add      $mSCRATCH, $BASET, $mSCRATCH  //output
    add      $BASET, $SUBT, $mSCRATCH2     //input
    mov      $SUBT, $mSCRATCH              //output
    bri      .DynamicSlice2d_copy_DS_common

// load the next SUBT pointer to use as a destination
// Index to load the next BASET input pointer
copy_loop_DS_32:
    add      $SUBT, $SUBT, $mSCRATCH2   //output
    add      $BASET, $BASET, $mSCRATCH  //input

.DynamicSlice2d_copy_DS_common:
    // Aligned output?
    and      $mSCRATCH2, $SUBT, 4
    brz      $mSCRATCH2, 1f

    // Output not aligned -load/store a word to force it to be aligned
    // for the loop
    ld32step     $VAL1, $mzero, $BASET+=,1
    st32step     $VAL1, $mzero, $SUBT+=,1
    add          $REGION_SIZE, $REGION_SIZE, -1
1:
    // input aligned?
    and     $mSCRATCH2, $BASET,4
    brz     $mSCRATCH2, 5f
    shr     $mSCRATCH2, $REGION_SIZE,1

    // Copy pairs of items - output is 64 bit aligned input is not
    rpt $mSCRATCH2, ((2f-1f)/8) -1

1:
    {ld32step $VAL1, $mzero, $BASET+=,1
     fnop}
    {ld32step $VAL2,$mzero,$BASET+=,1
     fnop}
    {st64step  $VAL12, $mzero, $SUBT+=,1
     fnop}
2:
    and     $REGION_SIZE, $REGION_SIZE,1
    brz     $REGION_SIZE, DynamicSlice_loop_resume
    bri     3f

5:
    // Copy pairs of items - both input and output are 64 bit aligned
    shr     $mSCRATCH2,$REGION_SIZE,1
    {rpt $mSCRATCH2, ((2f-1f)/8) -1
        fnop}
1:
    {ld64step    $VAL12, $mzero, $BASET+=,1
        fnop}
    {st64step    $VAL12, $mzero, $SUBT+=,1
        fnop}
2:
    // Is there a last one?
    and     $REGION_SIZE, $REGION_SIZE,1
    brz     $REGION_SIZE, DynamicSlice_loop_resume
3:
    // Copy the last 32 bits
    ld32step $VAL1, $mzero, $BASET+=,1
    st32step $VAL1, $mzero, $SUBT+=,1

    bri     DynamicSlice_loop_resume

.size DynamicSlice2d_32_copy, .-DynamicSlice2d_32_copy
//******************************************************************************
// copy loop for DynamicSlice2d and DynamicUpdateSlice2d (half)
//******************************************************************************
.section .text.DynamicSlice2d_16_copy
.align 8
    nop // rpt align
DynamicSlice2d_16_copy:

// load the next SUBT pointer to use as a source
// Index to load the next BASET pointer to use as a destination
copy_loop_DUS_16:
    add      $mSCRATCH, $BASET, $mSCRATCH  //output
    add      $BASET, $SUBT, $mSCRATCH2     //input
    mov      $SUBT, $mSCRATCH              //output
    bri      .Lcopy_DS_half_common

// load the next SUBT pointer to use as a destination
// Index to load the next BASET input pointer
copy_loop_DS_16:
    add      $SUBT, $SUBT, $mSCRATCH2   //output
    add      $BASET, $BASET, $mSCRATCH  //input

.Lcopy_DS_half_common:
    // 32/16 alignment half copy begins

    // Aligned output?
    and      $mSCRATCH2, $SUBT, 2
    brz      $mSCRATCH2, 1f

    // Output not aligned -load/store to force it to be aligned for the loop
    ldb16step    $VAL1, $mzero, $BASET+=,1
    add          $SUBT, $SUBT, -2
    ldb16        $VAL2, $mzero, $SUBT, 0
    {add          $REGION_SIZE, $REGION_SIZE, -1
     roll16       $VAL1, $VAL2, $VAL1}
    st32step     $VAL1, $mzero, $SUBT+=,1

1:
    // input aligned?
    and     $mSCRATCH2, $BASET,2
    brz     $mSCRATCH2, 5f

    // Case for misaligned input
    shr     $mSCRATCH2, $REGION_SIZE,2
    ldb16step $VAL1, $mzero, $BASET+=,1

    // Copy 2 pairs of items per loop, output is 32 bit aligned, input is not
    rpt $mSCRATCH2, ((2f-1f)/8) -1
1:
    {ld32step $VAL2, $mzero, $BASET+=,1
     fnop}
    {ld32step $VAL3, $mzero, $BASET+=,1
     roll16   $VAL1, $VAL1, $VAL2}
    {st32step $VAL1, $mzero, $SUBT+=,1
     roll16   $VAL2, $VAL2, $VAL3}
    {st32step $VAL2, $mzero, $SUBT+=,1
     mov      $VAL1, $VAL3}
2:
    // Are there 32 bits or more bits left?
    and     $mSCRATCH2, $REGION_SIZE,2
    brz     $mSCRATCH2, 1f
    // Store remaining 32 bits, fetch 16 more incase we have 1 more to store
    ldb16step $VAL2, $mzero, $BASET+=,1
    roll16   $VAL1, $VAL1, $VAL2
    st32step $VAL1, $mzero, $SUBT+=,1
    ldb16step $VAL1, $mzero, $BASET+=,1
1:
    // Is there a last one?  We have already fetched it if so
    and     $REGION_SIZE, $REGION_SIZE,1
    brz     $REGION_SIZE, DynamicSlice_loop_resume
    bri     3f

5:
    // Case for aligned input
    shr     $mSCRATCH2,$REGION_SIZE,1

    // Copy pairs of items, both 32 bit aligned
    {rpt $mSCRATCH2, ((2f-1f)/8) -1
        fnop}
1:
    {ld32step    $VAL1, $mzero, $BASET+=,1
        fnop}
    {st32step    $VAL1, $mzero, $SUBT+=,1
        fnop}
2:
    // Is there a last one?
    and     $REGION_SIZE, $REGION_SIZE,1
    brz     $REGION_SIZE, DynamicSlice_loop_resume
    ldb16step    $VAL1, $mzero, $BASET+=,1
3:
    // Write the 1st 2 bytes into the last 4 byte word - preserve bytes 3,4
    ldb16    $VAL2, $mzero, $SUBT, 1
    roll16   $VAL1, $VAL1, $VAL2
    st32step $VAL1, $mzero, $SUBT+=,1
    bri     DynamicSlice_loop_resume

.size DynamicSlice2d_16_copy, .-DynamicSlice2d_16_copy

#endif
/* -------------------------------------------------------------------------- */
