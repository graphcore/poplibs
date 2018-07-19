//***************************************************************************************************
// Clamp: float and half type versions
//***************************************************************************************************
#ifdef __IPU__
.globl __runCodelet_popops__Clamp___float
.type __runCodelet_popops__Clamp___float, @function


.globl __runCodelet_popops__Clamp___half
.type __runCodelet_popops__Clamp___half, @function


/* Input vertex structure offsets */
#define VOFF_IN1_PTR                0
#define VOFF_IN2_PTR                1
#define VOFF_IN3_PTR                2
#define VOFF_OUT_START_PTR          3
#define VOFF_OUT_END_PTR            4

// Register aliases
// Registers for each of the passed parameters
#define IN1_PTR                    $m1
#define IN2_PTR                    $m2
#define IN3_PTR                    $m3
#define OUT_START_PTR              $m4
#define OUT_END_PTR                $m5

#define mSCRATCH $m0

#define CLAMP_LOW   $a0        //intended register pair for clamp range input
#define CLAMP_HIGH  $a1
#define CLAMP_PAIR  $a0:1      //MUST BE CONSISTENT WITH CLAMP_LOW AND CLAMP_HIGH
#define CLAMP_IN    $a2
#define CLAMP_RESULT $a3
#define CLAMP_RESULT2 $a4      //result of 2nd clamp in a loop which processed 2 items
#define aSCRATCH    $a5

#define IN1 $m7                 //inner loop pointers
#define IN2 $m8
#define IN3 $m9
#define OUT $m6

#define OUTER_LOOP_COUNT $m11
#define INNER_LOOP_COUNT $m10
//***************************************************************************************************
// Float version, using clamp instruction to do all calculation.
// Due to there being 3 input arrays and 1 output array dealt with in the inner optimised loop,
// alignment of data can be complex if we try to optimise further to process 2 items at once.
// In other words, in1,in2,in3 or out may or may not be 64 bit aligned.  An optimal loop could be written
// if they were all aligned, but dealing with the cases where some are, some aren't is complex.
// Maybe an "all aligned" version could be an option?
//
// Benchmark: 8+ vectors*(12+2) +(total_inputs_in_all_vectors-vectors)*4
//***************************************************************************************************

.section .text.__runCodelet_popops__Clamp___float
.align 8

clamp_prepad:                      //Pad with an instruction to ensure the loop start below is aligned to a 64 bit boundary
                                   //doing it here means that the instruction never gets executed and costs no time
  nop
__runCodelet_popops__Clamp___float:
    ld32     IN1_PTR,       $mzero, $mvertex_base, VOFF_IN1_PTR         //fetch a pointer to the in1 array
    ld32     IN2_PTR,       $mzero, $mvertex_base, VOFF_IN2_PTR         //fetch a pointer to the in2 array
    ld32     IN3_PTR,       $mzero, $mvertex_base, VOFF_IN3_PTR         //fetch a pointer to the in3 array
    ld32     OUT_START_PTR, $mzero, $mvertex_base, VOFF_OUT_START_PTR   //fetch ptr to start of the output array of start,end pairs
    ld32     OUT_END_PTR,   $mzero, $mvertex_base, VOFF_OUT_END_PTR     //fetch ptr to end of the output array of start,end pairs

 //calculate outer loop count - number of vectors to process
    sub         OUTER_LOOP_COUNT,OUT_END_PTR,OUT_START_PTR
    shr         OUTER_LOOP_COUNT,OUTER_LOOP_COUNT,3                     //2 words (start,end address) per entry, 4 bytes per word: div by 8
    add         OUTER_LOOP_COUNT,OUTER_LOOP_COUNT,-1                    //decrement, to work with the brnzdec instruction post decrement

clamp_loop_outer:
    ld32step    IN1,$mzero,IN1_PTR+=,1                                  //get an in1 vector pointer
    ld32step    IN2,$mzero,IN2_PTR+=,1                                  //get an in2 vector pointer
    ld32step    IN3,$mzero,IN3_PTR+=,1                                  //get an in3 vector pointer

    ld32step    OUT,$mzero,OUT_START_PTR+=,1                            //fetch address of 1st output (start)
    ld32step    INNER_LOOP_COUNT,$mzero,OUT_START_PTR+=,1               //and the end address of the first output
    sub         INNER_LOOP_COUNT,INNER_LOOP_COUNT,OUT                   //find the vector length in bytes
    shr         INNER_LOOP_COUNT,INNER_LOOP_COUNT,2                     //4 bytes per word: div by 4
    add         INNER_LOOP_COUNT,INNER_LOOP_COUNT,-1                    //one less loop as unrolled for one pass

//unrolled loop portion: fetch and clamp the first entry
    ld32step    CLAMP_IN,$mzero,IN1+=,1
    ld32step    CLAMP_LOW,$mzero,IN2+=,1
    ld32step    CLAMP_HIGH,$mzero,IN3+=,1

//Inner loop - note that there are vectorised clamp instructions, but they apply the same max,min to input[0],input[1] etc,
//which is not what this function is required to do.
    {rpt INNER_LOOP_COUNT,((clamp_loop_inner_end-clamp_loop_inner)/8) -1
     f32clamp    CLAMP_RESULT,CLAMP_IN,CLAMP_PAIR }
clamp_loop_inner:
        {ld32step    CLAMP_IN,$mzero,IN1+=,1                                 //fetch in1, index to the next item
         fnop}
        {ld32step    CLAMP_LOW,$mzero,IN2+=,1                                //and in2, in3
         fnop}
        {ld32step    CLAMP_HIGH,$mzero,IN3+=,1
         fnop}
        {st32step    CLAMP_RESULT,$mzero,OUT+=,1                             //write PREVIOUS output
         f32clamp    CLAMP_RESULT,CLAMP_IN,CLAMP_PAIR }                      //clamp: result = input limited to range of max,min

clamp_loop_inner_end:
//unrolled loop portion: store the last one
    st32step    CLAMP_RESULT,$mzero,OUT+=,1                              //write final output from the inner loop
    brnzdec     OUTER_LOOP_COUNT,clamp_loop_outer                        //dec and outer loop

    exitz    $mzero

.size __runCodelet_popops__Clamp___float, .-__runCodelet_popops__Clamp___float

//***************************************************************************************************
// Half version, using clamp instruction to do all calculation.
// Due to there being 3 input arrays and 1 output array dealt with in the inner optimised loop,
// alignment of data can be complex if we try to optimise further to process 2 items at once.
// In other words, in1,in2,in3 or out may or may not be 32 bit aligned.  An optimal loop could be written
// if they were all aligned, but dealing with the cases where some are, some aren't is complex.
// Maybe an "all aligned" version could be an option?
//
// Benckmark (aligned)= 8 + vectors*(14+2) + total_inputs_in_all_vectors * 7/2
//***************************************************************************************************

.section .text.__runCodelet_popops__Clamp___half
.align 8

clamp16_prepad:                      //Pad with an instruction to ensure the loop start below is aligned to a 64 bit boundary
                                     //doing it here means that the instruction never gets executed and costs no time
  nop
__runCodelet_popops__Clamp___half:
    ld32     IN1_PTR,       $mzero, $mvertex_base, VOFF_IN1_PTR         //fetch a pointer to the in1 array
    ld32     IN2_PTR,       $mzero, $mvertex_base, VOFF_IN2_PTR         //fetch a pointer to the in2 array
    ld32     IN3_PTR,       $mzero, $mvertex_base, VOFF_IN3_PTR         //fetch a pointer to the in3 array
    ld32     OUT_START_PTR, $mzero, $mvertex_base, VOFF_OUT_START_PTR   //fetch ptr to start of the output array of start,end pairs
    ld32     OUT_END_PTR,   $mzero, $mvertex_base, VOFF_OUT_END_PTR     //fetch ptr to end of the output array of start,end pairs

 //calculate outer loop count - number of vectors to process
    sub         OUTER_LOOP_COUNT,OUT_END_PTR,OUT_START_PTR
    shr         OUTER_LOOP_COUNT,OUTER_LOOP_COUNT,3                     //2 words (start,end address) per entry, 4 bytes per word: div by 8
    add         OUTER_LOOP_COUNT,OUTER_LOOP_COUNT,-1                    //decrement, to work with the brnzdec instruction post decrement

clamp16_loop_outer:
    ld32step    IN1,$mzero,IN1_PTR+=,1                                  //get an in1 vector pointer
    ld32step    IN2,$mzero,IN2_PTR+=,1                                  //get an in2 vector pointer
    ld32step    IN3,$mzero,IN3_PTR+=,1                                  //get an in3 vector pointer

    ld32step    OUT,$mzero,OUT_START_PTR+=,1                            //fetch address of 1st output (start)
    ld32step    INNER_LOOP_COUNT,$mzero,OUT_START_PTR+=,1               //and the end address of the first output
    sub         INNER_LOOP_COUNT,INNER_LOOP_COUNT,OUT                   //find the vector length in bytes
    shr         INNER_LOOP_COUNT,INNER_LOOP_COUNT,1                     //2 bytes per word: div by 2

//Note that we have only a 32 bit (no 16 bit) write instruction which we use to write the output.
//Therefore we have to deal with 3 things based on output array alignment and data length:
//Choose to write an inner loop that processes pairs of 16bit words for efficiency.
//If the 1st word is not 32 bit aligned we need to treat that as a special case
//If the last word is not part of a 32bit aligned pair it needs treating as a special case
// We need to use the loop count and start address to determine if we have these special cases,
// not forgetting the case of no loops

    and         mSCRATCH,OUT,0x2                                        //test the bit which will indicate a non alignment in the address
    brz         mSCRATCH,clamp16_start_aligned
//Deal with a single first word which isn't 32 bit aligned
clamp_do_misaligned:
        andc         OUT,OUT,0x3
        ldb16step   CLAMP_LOW,$mzero,IN2+=,1                                //Fetch in2 for 1st clamp
        ldb16step   CLAMP_HIGH,$mzero,IN3+=,1                               //Fetch in3 for 1st clamp
        {ldb16step  CLAMP_IN,$mzero,IN1+=,1                                 //fetch in1
         sort4x16lo CLAMP_HIGH,CLAMP_LOW,CLAMP_HIGH }                       //Combine lower word of in2, in3
        {ld32       aSCRATCH,$mzero,OUT,0                                   //load the output word we are about to overwrite
         f16v2clamp CLAMP_RESULT,CLAMP_IN,CLAMP_HIGH }                      //clamp: result = input limited to range of max,min

        {add         INNER_LOOP_COUNT,INNER_LOOP_COUNT,-1                   //adjust count:one less word to process
         sort4x16lo  CLAMP_RESULT,aSCRATCH,CLAMP_RESULT}
        st32step    CLAMP_RESULT,$mzero,OUT+=,1                             //write output, 1 word
clamp16_start_aligned:

    and         mSCRATCH,INNER_LOOP_COUNT,0x1                           //test the bit which will indicate an odd number of items left to process
    shr         INNER_LOOP_COUNT,INNER_LOOP_COUNT,1                     //2 items per inner loop
    ldb16step   CLAMP_LOW,$mzero,IN2+=,1                                //Fetch in2 for 1st clamp in the loop
    ldb16step   CLAMP_HIGH,$mzero,IN3+=,1                               //Fetch in3 for 1st clamp in the loop

//Inner loop: Processes 2 inputs at once, which are combined in a 32 bit word before being written to avoid memory read modify write 
//(there is no 16 bit write)
//Note that we are using a vectorised f16v2clamp instruction but the instruction applies the same max,min to input[0] and input[1],
//which is not what this function is required to do.  So we only use one of the two results generated. 
    {rpt INNER_LOOP_COUNT,((clamp16_loop_inner_end-clamp16_loop_inner)/8) -1
     fnop }
clamp16_loop_inner:
        {ldb16step    CLAMP_IN,$mzero,IN1+=,1                                  //fetch in1, index to the next item
         sort4x16lo   CLAMP_HIGH,CLAMP_LOW,CLAMP_HIGH }                        //Combine lower word of in2, in3
        {ldb16step    CLAMP_LOW,$mzero,IN2+=,1                                 //Fetch in2 for 2nd clamp in the loop
         f16v2clamp   CLAMP_RESULT,CLAMP_IN,CLAMP_HIGH }                       //clamp: result = input limited to range of max,min

        {ldb16step    CLAMP_HIGH,$mzero,IN3+=,1                                //fetch in3
         fnop}
        {ldb16step    CLAMP_IN,$mzero,IN1+=,1                                  //fetch in1, index to the next item
         sort4x16lo   CLAMP_HIGH,CLAMP_LOW,CLAMP_HIGH }                        //Combine lower word of in2,in3
        {ldb16step    CLAMP_LOW,$mzero,IN2+=,1                                 //Fetch in2 for 1st clamp in the loop
         f16v2clamp   CLAMP_RESULT2,CLAMP_IN,CLAMP_HIGH }                      //clamp: result = input limited to range of max,min

        {ldb16step    CLAMP_HIGH,$mzero,IN3+=,1                                //Fetch in3 for 1st clamp in the loop
         sort4x16lo   CLAMP_RESULT,CLAMP_RESULT,CLAMP_RESULT2}                 //combine output words
        {st32step     CLAMP_RESULT,$mzero,OUT+=,1                              //write output, 2 words at once
         fnop}
clamp16_loop_inner_end:

//Deal with a single last word, noting that low and high bounds are already fetched but input isn't
     brz          mSCRATCH,clamp16_loop_outer_end                        //previously gathered discarded loop count LSB
                                                                         //as the condition for odd number of items left after alignment
        {ldb16step    CLAMP_IN,$mzero,IN1+=,1                                //fetch in1, index to the next item
         sort4x16lo   CLAMP_HIGH,CLAMP_LOW,CLAMP_HIGH }                      //Combine lower word of in2, in3
        {ld32         aSCRATCH,$mzero,OUT,0                                  //load the output word we are about to overwrite
         f16v2clamp    CLAMP_RESULT,CLAMP_IN,CLAMP_HIGH }                    //clamp: result = input limited to range of max,min

        sort4x16hi   CLAMP_RESULT,CLAMP_RESULT,aSCRATCH
        st32step     CLAMP_RESULT,$mzero,OUT+=,1                             //write output, 1 word, preserve next word
clamp16_loop_outer_end:
    brnzdec     OUTER_LOOP_COUNT,clamp16_loop_outer                      //dec and outer loop

    exitz    $mzero

.size __runCodelet_popops__Clamp___half, .-__runCodelet_popops__Clamp___half

#endif
