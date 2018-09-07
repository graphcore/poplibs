#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Dynamic Slice and Dynamic Update Slice vertex code for int, half and float
// variants
/* -------------------------------------------------------------------------- */


// Register aliases

#define COPY_PTR    m0
#define BASET       m1

#define SUBT        m2
#define BASE_ELEM   m3
#define SUB_ELEM    m4
#define REGIONS     m5

#define mSCRATCH    m6
#define mSCRATCH2   m7

#define BASE_SLICE  m8
#define BASE_IDX    m9
#define REGION_SIZE m11

#define VAL12   a0:1
#define VAL1    a0
#define VAL2    a1
#define VAL3    a2

#define OFFSET  a5

//****************************************************************************
// The input structure parameters:
// 32 bit offset
// 32 ptr baseT
// 32 ptr subT
// 16 bit numBaseElements
// 16 bit numSubElements
// 16 bit numregions
//****************************************************************************
#define VOFF_OFFSET     0
#define VOFF_BASET      1
#define VOFF_SUBT       2
// 16 bit - offsets for use with 16 bit loads
#define VOFF_BASE_ELEM  3*2
#define VOFF_SUB_ELEM   3*2 + 1
#define VOFF_REGIONS    3*2 + 2

//****************************************************************************
// Worker base local variable storage
//****************************************************************************
#define WOFF_REGIONS 0

//******************************************************************************
// Labels names for each variant 
//******************************************************************************
#define DS_HALF_FUNC __runCodelet_popops__DynamicSlice2d___half
#define DUS_HALF_FUNC __runCodelet_popops__DynamicUpdateSlice2d___half

#define DS_FLOAT_FUNC __runCodelet_popops__DynamicSlice2d___float
#define DUS_FLOAT_FUNC __runCodelet_popops__DynamicUpdateSlice2d___float

#define DS_INT_FUNC __runCodelet_popops__DynamicSlice2d___int
#define DUS_INT_FUNC __runCodelet_popops__DynamicUpdateSlice2d___int

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
   setzi   $COPY_PTR, copy_loop_DUS_32
   bri     DynamicSlice2d_common

DS_FLOAT_FUNC:
DS_INT_FUNC:
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
    ld32     $SUBT, $mzero, $mvertex_base, VOFF_SUBT
    ldz16    $BASE_ELEM, $mzero, $mvertex_base, VOFF_BASE_ELEM
    ldz16    $REGIONS, $mzero, $mvertex_base, VOFF_REGIONS
    ld32     $BASET, $mzero, $mvertex_base, VOFF_BASET                       

    zero    $BASE_IDX
    add     $REGIONS, $REGIONS, -1
    ld32    $OFFSET, $mzero, $mSCRATCH, 0
.Louter_loop:
    st32    $REGIONS, $mzero, $mworker_base, WOFF_REGIONS
    // Find the size of next region to copy, by indexing into the
    // array of baset pointers (2-pointers) 
    shl     $mSCRATCH, $BASE_IDX,3
    ld32    $REGION_SIZE, $BASET, $mSCRATCH, 0
    ld32    $mSCRATCH, $BASET, $mSCRATCH, 1
    sub     $REGION_SIZE, $mSCRATCH, $REGION_SIZE 

    // Apply the offset to the base index and check the result is in bounds.
    // Result is base slice
    atom    $mSCRATCH, $OFFSET
    add     $BASE_SLICE, $BASE_IDX, $mSCRATCH
    add     $mSCRATCH, $BASE_IDX, $BASE_ELEM
    cmpult  $mSCRATCH, $BASE_SLICE, $mSCRATCH 
    brnz    $mSCRATCH, 3f
    sub     $BASE_SLICE, $BASE_SLICE, $BASE_ELEM
3:
    // Setup loop over subElements count
    ldz16   $SUB_ELEM, $mzero, $mvertex_base, VOFF_SUB_ELEM
    add     $SUB_ELEM, $SUB_ELEM, -1
.Lsub_elem_loop:
    shl     $mSCRATCH, $BASE_SLICE, 3

    // ***** Jump to selected function which loads ptrs and copies *******
    br      $COPY_PTR
    // Not a local label as referenced across sections.
DynamicSlice_loop_resume: 
    // Increment the base slice and check the result is within bounds
    add     $BASE_SLICE, $BASE_SLICE, 1
    add     $mSCRATCH, $BASE_IDX, $BASE_ELEM
    cmpult  $mSCRATCH, $BASE_SLICE, $mSCRATCH
    brnz    $mSCRATCH, 3f                                                       
    sub     $BASE_SLICE, $BASE_SLICE, $BASE_ELEM
3:
    ld32     $BASET, $mzero, $mvertex_base, VOFF_BASET
    brnzdec $SUB_ELEM, .Lsub_elem_loop
   
    // add num Base elements to the base index for the next region
    add     $BASE_IDX, $BASE_IDX, $BASE_ELEM
    ld32    $REGIONS, $mzero, $mworker_base, WOFF_REGIONS

    brnzdec $REGIONS, .Louter_loop
    exitz   $mzero
.size DynamicSlice2d_common, .-DynamicSlice2d_common

//******************************************************************************
// copy loop for DynamicSlice2d and DynamicUpdateSlice2d (float or int)
//******************************************************************************
.section .text.DynamicSlice2d_32_copy
DynamicSlice2d_32_copy:
.align 8
// load the next SUBT pointer to use as a source
// Index to load the next BASET pointer to use as a destination
copy_loop_DUS_32:
    ld32     $mSCRATCH2, $BASET, $mSCRATCH, 0
    ld32step $mSCRATCH, $mzero, $SUBT+=,1
    bri      .Lcopy_DS_common

// load the next SUBT pointer to use as a destination
// Index to load the next BASET input pointer
copy_loop_DS_32:
    ld32step $mSCRATCH2, $mzero, $SUBT+=,1
    ld32     $mSCRATCH, $BASET, $mSCRATCH, 0

.Lcopy_DS_common:
    // 32/64 bit alignment float/int copy begins
    // Note REGIONS, BASET registers are re-used
    // REGIONS = number of 32 bit words to copy
    shr      $REGIONS, $REGION_SIZE,2         
    // Aligned output?
    and      $BASET, $mSCRATCH2, 4
    brz      $BASET, 1f

    // Output not aligned -load/store a word to force it to be aligned 
    // for the loop
    ld32step     $VAL1, $mzero, $mSCRATCH+=,1
    st32step     $VAL1, $mzero, $mSCRATCH2+=,1
    add          $REGIONS, $REGIONS, -1
1:
    // input aligned?
    and     $BASET, $mSCRATCH,4
    brz     $BASET, 5f
    shr     $BASET, $REGIONS,1
    // Copy pairs of items - output is 64 bit aligned input is not
    {rpt $BASET, ((2f-1f)/8) -1
      fnop}
1:
    {ld32step $VAL1, $mzero, $mSCRATCH+=,1
     fnop}
    {ld32step $VAL2,$mzero,$mSCRATCH+=,1
     fnop}
    {st64step  $VAL12, $mzero, $mSCRATCH2+=,1
     fnop}
2:
    and     $REGIONS, $REGIONS,1
    brz     $REGIONS, DynamicSlice_loop_resume
    bri     3f

5:
    // Copy pairs of items - both input and output are 64 bit aligned
    shr     $BASET,$REGIONS,1
    {rpt $BASET, ((2f-1f)/8) -1
        fnop}
1:
    {ld64step    $VAL12, $mzero, $mSCRATCH+=,1
        fnop}
    {st64step    $VAL12, $mzero, $mSCRATCH2+=,1
        fnop}
2:
    // Is there a last one?
    and     $REGIONS, $REGIONS,1
    brz     $REGIONS, DynamicSlice_loop_resume
3:  
    // Copy the last 32 bits
    ld32step $VAL1, $mzero, $mSCRATCH+=,1
    st32step $VAL1, $mzero, $mSCRATCH2+=,1
    bri     DynamicSlice_loop_resume

.size DynamicSlice2d_32_copy, .-DynamicSlice2d_32_copy
//******************************************************************************
// copy loop for DynamicSlice2d and DynamicUpdateSlice2d (half)
//******************************************************************************
.section .text.DynamicSlice2d_16_copy

DynamicSlice2d_16_copy:
.align 8

// load the next SUBT pointer to use as a source
// Index to load the next BASET pointer to use as a destination
copy_loop_DUS_16:
    ld32     $mSCRATCH2, $BASET, $mSCRATCH, 0
    ld32step $mSCRATCH, $mzero, $SUBT+=,1
    bri     .Lcopy_DS_half_common

// load the next SUBT pointer to use as a destination
// Index to load the next BASET input pointer
copy_loop_DS_16:
    ld32step $mSCRATCH2, $mzero, $SUBT+=,1
    ld32     $mSCRATCH, $BASET, $mSCRATCH, 0

.Lcopy_DS_half_common:

    // 32/16 alignment half copy begins
    // Note REGIONS, BASET registers are re-used
    shr      $REGIONS, $REGION_SIZE, 1         
    // Aligned output?
    and      $BASET, $mSCRATCH2, 2
    brz      $BASET, 1f

    // Output not aligned -load/store to force it to be aligned for the loop
    ldb16step    $VAL1, $mzero, $mSCRATCH+=,1
    add          $mSCRATCH2, $mSCRATCH2, -2
    ldb16        $VAL2, $mzero, $mSCRATCH2, 0
    {add          $REGIONS, $REGIONS, -1
     roll16       $VAL1, $VAL2, $VAL1}
    st32step     $VAL1, $mzero, $mSCRATCH2+=,1

1:
    // input aligned?
    and     $BASET, $mSCRATCH,2
    brz     $BASET, 5f
 
    // Case for misaligned input
    shr     $BASET, $REGIONS,2
    ldb16step $VAL1, $mzero, $mSCRATCH+=,1
    
    // Copy 2 pairs of items per loop, output is 32 bit aligned, input is not
    {rpt $BASET, ((2f-1f)/8) -1
        fnop}
1:
    {ld32step $VAL2, $mzero, $mSCRATCH+=,1
     fnop}
    {ld32step $VAL3, $mzero, $mSCRATCH+=,1
     roll16   $VAL1, $VAL1, $VAL2}
    {st32step $VAL1, $mzero, $mSCRATCH2+=,1
     roll16   $VAL2, $VAL2, $VAL3}
    {st32step $VAL2, $mzero, $mSCRATCH2+=,1
     mov      $VAL1, $VAL3}
2:
    // Are there 32 bits or more bits left?
    and     $BASET, $REGIONS,2
    brz     $BASET, 1f
    // Store remaining 32 bits, fetch 16 more incase we have 1 more to store
    ldb16step $VAL2, $mzero, $mSCRATCH+=,1
    roll16   $VAL1, $VAL1, $VAL2
    st32step $VAL1, $mzero, $mSCRATCH2+=,1
    ldb16step $VAL1, $mzero, $mSCRATCH+=,1
1:
    // Is there a last one?  We have already fetched it if so
    and     $REGIONS, $REGIONS,1
    brz     $REGIONS, DynamicSlice_loop_resume
    bri     3f

5:
    // Case for aligned input
    shr     $BASET,$REGIONS,1

    // Copy pairs of items, both 32 bit aligned
    {rpt $BASET, ((2f-1f)/8) -1
        fnop}
1:
    {ld32step    $VAL1, $mzero, $mSCRATCH+=,1
        fnop}
    {st32step    $VAL1, $mzero, $mSCRATCH2+=,1
        fnop}
2:
    // Is there a last one?
    and     $REGIONS, $REGIONS,1
    brz     $REGIONS, DynamicSlice_loop_resume
    ldb16step    $VAL1, $mzero, $mSCRATCH+=,1
3:
    // Write the 1st 2 bytes into the last 4 byte word - preserve bytes 3,4
    ldb16    $VAL2, $mzero, $mSCRATCH2, 1
    roll16   $VAL1, $VAL1, $VAL2
    st32step $VAL1, $mzero, $mSCRATCH2+=,1
    bri     DynamicSlice_loop_resume
 
.size DynamicSlice2d_16_copy, .-DynamicSlice2d_16_copy

#endif
/* -------------------------------------------------------------------------- */
