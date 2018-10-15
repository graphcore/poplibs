#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts.h"

// Assembly implementation of popnn::Loss{SumSquared,Softmax}Transform vertecies
#define lossSumSquaredFloat __runCodelet_popnn__LossSumSquaredTransform___float
#define lossSumSquaredHalf __runCodelet_popnn__LossSumSquaredTransform___half
#define lossSoftmaxFloat __runCodelet_popnn__LossSoftmaxTransform___float
#define lossSoftmaxHalf __runCodelet_popnn__LossSoftmaxTransform___half

#define PROB_PTR_OFFSET         0
#define EXPECTED_PTR_OFFSET     1
#define DELTA_PTR_OFFSET        2
#define TRANSFORMED_PTR_OFFSET  3
#define SIZE_OFFSET             4

.text
/*****************************************************************************/
// popnn::LossSumSquaredTransform<half>
// popnn::LossSumSquaredTransform<float>
/*****************************************************************************/
.section .text.lossSumSquaredFloat
.align 8

.global lossSumSquaredFloat
.type lossSumSquaredFloat, @function

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    m5
#define PROB                    a0
#define EXPECTED                a1
#define DELTA                   a2
#define TRANSFORMED             a3
#define CONST_HALF              a4

lossSumSquaredFloat:
  ldz16 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET
  ldz16 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET
  ldz16 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET
  ldz16 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET
  ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET

  // CONST_HALF = 0x3F000000, where 0x3F000000 represents 0.5 in FP32
  {setzi $BASE, TMEM_REGION0_BASE_ADDR; or $CONST_HALF, $azero, 0x3F000000}
  shl $PROB_PTR, $PROB_PTR, 2
  shl $EXPECTED_PTR, $EXPECTED_PTR, 2
  shl $DELTA_PTR, $DELTA_PTR, 2
  shl $TRANSFORMED_PTR, $TRANSFORMED_PTR, 2;

  // Load ahead
  ld32step $PROB, $BASE, $PROB_PTR+=, 1
  ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1

  rpt $SIZE, (lossSumSquaredFloat_loopEnd - lossSumSquaredFloat_loopStart)/8 - 1
lossSumSquaredFloat_loopStart:
  {ld32step $PROB, $BASE, $PROB_PTR+=, 1; f32sub $DELTA, $PROB, $EXPECTED}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1; f32mul $TRANSFORMED, $CONST_HALF, $DELTA}
  {st32step $DELTA, $BASE, $DELTA_PTR+=, 1; f32mul $TRANSFORMED, $TRANSFORMED, $DELTA}
  {st32step $TRANSFORMED, $BASE, $TRANSFORMED_PTR+=, 1; fnop}
lossSumSquaredFloat_loopEnd:
  exitz $mzero

.size lossSumSquaredFloat, .-lossSumSquaredFloat

#undef PROB_PTR
#undef EXPECTED_PTR
#undef DELTA_PTR
#undef TRANSFORMED_PTR
#undef SIZE
#undef BASE
#undef PROB
#undef EXPECTED
#undef DELTA
#undef TRANSFORMED
#undef CONST_HALF

.section .text.lossSumSquaredHalf

.global lossSumSquaredHalf
.type lossSumSquaredHalf, @function
.align 8

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    m5
#define SIZE_D2                 m6
#define REMAINDER               m7
#define PROB                    a0
#define EXPECTED                a1
#define DELTA                   a2
#define TRANSFORMED             a3
#define CONST_HALF              a4
#define DELTA_RMW               a5
#define TRANSFORMED_RMW         a6
#define ASCRATCH                a7

lossSumSquaredHalf:
  ldz16 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET
  ldz16 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET
  ldz16 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET
  ldz16 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET
  ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET

  // CONST_HALF = [0.5h, 0.5h] in FP16
  {setzi $BASE, TMEM_REGION0_BASE_ADDR; f16v2sigm $CONST_HALF, $azero}
  shl $PROB_PTR, $PROB_PTR, 2
  shl $EXPECTED_PTR, $EXPECTED_PTR, 2
  shl $DELTA_PTR, $DELTA_PTR, 2
  shl $TRANSFORMED_PTR, $TRANSFORMED_PTR, 2;

  // Divide size by two - working on two elements at a time
  shr $SIZE_D2, $SIZE, 1

  // Load ahead
  ld32step $PROB, $BASE, $PROB_PTR+=, 1
  ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1

  {rpt $SIZE_D2, (lossSumSquaredHalf_loopEnd - lossSumSquaredHalf_loopStart)/8 - 1; fnop}
lossSumSquaredHalf_loopStart:
  {ld32step $PROB, $BASE, $PROB_PTR+=, 1; f16v2sub $DELTA, $PROB, $EXPECTED}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1; f16v2mul $TRANSFORMED, $CONST_HALF, $DELTA}
  {st32step $DELTA, $BASE, $DELTA_PTR+=, 1; f16v2mul $TRANSFORMED, $TRANSFORMED, $DELTA}
  {st32step $TRANSFORMED, $BASE, $TRANSFORMED_PTR+=, 1; fnop}
lossSumSquaredHalf_loopEnd:

  // There is potentially 1 more element to work on - need to perfom RMW
  and $REMAINDER, $SIZE, 0x1
  brz $REMAINDER, lossSumSquaredHalf_end

  // Read, Modify, Write
  {ldb16 $DELTA_RMW, $BASE, $DELTA_PTR, 1; f16v2sub $DELTA, $PROB, $EXPECTED}
  {ldb16 $TRANSFORMED_RMW, $BASE, $TRANSFORMED_PTR, 1; f16v2mul $TRANSFORMED, $CONST_HALF, $DELTA}
  sort4x16lo $DELTA_RMW, $DELTA, $DELTA_RMW
  f16v2mul $TRANSFORMED, $TRANSFORMED, $DELTA
  {st32  $DELTA_RMW, $BASE, $DELTA_PTR, 0; sort4x16lo $TRANSFORMED_RMW, $TRANSFORMED, $TRANSFORMED_RMW}
  st32  $TRANSFORMED_RMW, $BASE, $TRANSFORMED_PTR, 0

lossSumSquaredHalf_end:
  exitz $mzero

.size lossSumSquaredHalf, .-lossSumSquaredHalf

#undef PROB_PTR
#undef EXPECTED_PTR
#undef DELTA_PTR
#undef TRANSFORMED_PTR
#undef SIZE
#undef BASE
#undef SIZE_D2
#undef REMAINDER
#undef PROB
#undef EXPECTED
#undef DELTA
#undef TRANSFORMED
#undef CONST_HALF
#undef DELTA_RMW
#undef TRANSFORMED_RMW
#undef ASCRATCH

/*****************************************************************************/
// popnn::LossSoftmaxTransform<half>
// popnn::LossSoftmaxTransform<float>
/*****************************************************************************/
.section .text.lossSoftmaxFloat

.global lossSoftmaxFloat
.type lossSoftmaxFloat, @function
.align 8

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    m5
#define PROB                    a0
#define EXPECTED                a1
#define DELTA                   a2
#define TRANSFORMED             a3
#define LN_PROB                 a4
#define NEG_EXPECTED            a5

lossSoftmaxFloat:
  ldz16 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET
  ldz16 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET
  ldz16 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET
  ldz16 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET
  ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET

  setzi $BASE, TMEM_REGION0_BASE_ADDR
  shl   $PROB_PTR, $PROB_PTR, 2
  shl   $EXPECTED_PTR, $EXPECTED_PTR, 2
  shl   $DELTA_PTR, $DELTA_PTR, 2
  shl   $TRANSFORMED_PTR, $TRANSFORMED_PTR, 2

  ld32step $PROB, $BASE, $PROB_PTR+=, 1
  ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1

  {rpt $SIZE, (lossSoftmaxFloat_loopEnd - lossSoftmaxFloat_loopStart)/8 - 1; fnop}
lossSoftmaxFloat_loopStart:
  {nop; f32sub $DELTA, $PROB, $EXPECTED}
  {ld32step $PROB, $BASE, $PROB_PTR+=, 1; f32ln $LN_PROB, $PROB}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1; f32sub $NEG_EXPECTED, $azero, $EXPECTED}
  {st32step $DELTA, $BASE, $DELTA_PTR+=, 1; f32mul $TRANSFORMED, $NEG_EXPECTED, $LN_PROB}
  {st32step $TRANSFORMED, $BASE, $TRANSFORMED_PTR+=, 1; fnop}
lossSoftmaxFloat_loopEnd:
  exitz $mzero

.size lossSoftmaxFloat, .-lossSoftmaxFloat

#undef PROB_PTR
#undef EXPECTED_PTR
#undef DELTA_PTR
#undef TRANSFORMED_PTR
#undef SIZE
#undef BASE
#undef PROB
#undef EXPECTED
#undef DELTA
#undef TRANSFORMED
#undef LN_PROB
#undef NEG_EXPECTED

.section .text.lossSoftmaxHalf

.global lossSoftmaxHalf
.type lossSoftmaxHalf, @function
.align 8

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    m5
#define SIZE_D2                 m6
#define REMAINDER               m7
#define PROB                    a0
#define EXPECTED                a1
#define DELTA                   a2
#define TRANSFORMED             a3
#define DELTA_RMW               a4
#define TRANSFORMED_RMW         a5
#define LN_PROB                 a6
#define NEG_EXPECTED            a7

lossSoftmaxHalf:
  ldz16 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET
  ldz16 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET
  ldz16 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET
  ldz16 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET
  ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET

  setzi $BASE, TMEM_REGION0_BASE_ADDR
  shl   $PROB_PTR, $PROB_PTR, 2
  shl   $EXPECTED_PTR, $EXPECTED_PTR, 2
  shl   $DELTA_PTR, $DELTA_PTR, 2
  shl   $TRANSFORMED_PTR, $TRANSFORMED_PTR, 2;

  // Divide size by two - working on two elements at a time
  shr $SIZE_D2, $SIZE, 1

  ld32step $PROB, $BASE, $PROB_PTR+=, 1
  ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1

  rpt $SIZE_D2, (lossSoftmaxHalf_loopEnd - lossSoftmaxHalf_loopStart)/8 - 1
lossSoftmaxHalf_loopStart:
  {nop; f16v2sub $DELTA, $PROB, $EXPECTED}
  {ld32step $PROB, $BASE, $PROB_PTR+=, 1; f16v2ln $LN_PROB, $PROB}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1; f16v2sub $NEG_EXPECTED, $azero, $EXPECTED}
  {st32step $DELTA, $BASE, $DELTA_PTR+=, 1; f16v2mul $TRANSFORMED, $NEG_EXPECTED, $LN_PROB}
  {st32step $TRANSFORMED, $BASE, $TRANSFORMED_PTR+=, 1; fnop}
lossSoftmaxHalf_loopEnd:
  // There is potentially 1 more element to work on - need to perfom RMW
  and $REMAINDER, $SIZE, 0x1
  brz $REMAINDER, lossSoftmaxHalf_end

  // Read, Modify, Write
  {ldb16 $DELTA_RMW, $BASE, $DELTA_PTR, 1; f16v2sub $DELTA, $PROB, $EXPECTED}
  {ldb16 $TRANSFORMED_RMW, $BASE, $TRANSFORMED_PTR, 1; f16v2sub $NEG_EXPECTED, $azero, $EXPECTED}
  f16v2ln $LN_PROB, $PROB
  f16v2mul $TRANSFORMED, $NEG_EXPECTED, $LN_PROB
  sort4x16lo $DELTA_RMW, $DELTA, $DELTA_RMW
  {st32  $DELTA_RMW, $BASE, $DELTA_PTR, 0; sort4x16lo $TRANSFORMED_RMW, $TRANSFORMED, $TRANSFORMED_RMW}
  st32  $TRANSFORMED_RMW, $BASE, $TRANSFORMED_PTR, 0

lossSoftmaxHalf_end:
  exitz $mzero

.size lossSoftmaxHalf, .-lossSoftmaxHalf

#undef PROB_PTR
#undef EXPECTED_PTR
#undef DELTA_PTR
#undef TRANSFORMED_PTR
#undef SIZE
#undef BASE
#undef SIZE_D2
#undef REMAINDER
#undef PROB
#undef EXPECTED
#undef DELTA
#undef TRANSFORMED
#undef DELTA_RMW
#undef TRANSFORMED_RMW
#undef LN_PROB
#undef NEG_EXPECTED

#endif // __IPU__
