#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"
#include "popops/EncodingConstants.hpp"

// Assembly implementation of popnn::Loss{SumSquared,CrossEntropy}Transform vertecies
#define lossSumSquaredFloat __runCodelet_popnn__LossSumSquaredTransform___float
#define lossSumSquaredHalf __runCodelet_popnn__LossSumSquaredTransform___half
#define lossCrossEntropyFloat __runCodelet_popnn__LossCrossEntropyTransform___float
#define lossCrossEntropyHalf __runCodelet_popnn__LossCrossEntropyTransform___half

#define PROB_PTR_OFFSET         0
#define EXPECTED_PTR_OFFSET     1
#define DELTA_PTR_OFFSET        2
#define TRANSFORMED_PTR_OFFSET  3
#define SIZE_OFFSET             4
#define DELTA_SCALE_OFFSET      3 // 32 bit offset
#define MODEL_OUT_SCALING_OFFSET    4 // 32 bit offset

// eps
#define EPS_FLOAT   0x00800000   // 1.17549435e-38F, min value which is not a denorm
#define EPS_HALF    0x0001       // 0.000000059605

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
// popnn::LossCrossEntropyTransform<half>
// popnn::LossCrossEntropyTransform<float>
/*****************************************************************************/
.section .text.lossCrossEntropyFloat

.global lossCrossEntropyFloat
.type lossCrossEntropyFloat, @function
.align 8

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    m5
#define SCALE_PTR               m6

#define PROB                    a0
#define EXPECTED                a1
#define DELTAS_SCALE            a2
#define aSCRATCH                a3
#define LN_PROB                 a4
#define MODEL_OUT_SCALING       a5
#define EPS                     a6
#define NEG_LOG_MODEL_OUT_SCALING   a7
  nop           // For repeat body alignment
lossCrossEntropyFloat:
  ldz16 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET
  ldz16 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET
  ldz16 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET
  ldz16 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET
  ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET
  ld32  $SCALE_PTR, $mvertex_base, $mzero, MODEL_OUT_SCALING_OFFSET

  ld32 $MODEL_OUT_SCALING, $mzero, $SCALE_PTR, 0;
  ld32 $SCALE_PTR, $mvertex_base, $mzero, DELTA_SCALE_OFFSET
  ld32 $DELTAS_SCALE, $mzero, $SCALE_PTR, 0;

  {setzi $BASE, TMEM_REGION0_BASE_ADDR
   f32div $DELTAS_SCALE, $DELTAS_SCALE, $MODEL_OUT_SCALING}
  {shl   $PROB_PTR, $PROB_PTR, 2
   f32ln  $NEG_LOG_MODEL_OUT_SCALING, $MODEL_OUT_SCALING}
  {shl   $EXPECTED_PTR, $EXPECTED_PTR, 2
   f32sub $NEG_LOG_MODEL_OUT_SCALING, $azero, $NEG_LOG_MODEL_OUT_SCALING}

  shl   $DELTA_PTR, $DELTA_PTR, 2
  shl   $TRANSFORMED_PTR, $TRANSFORMED_PTR, 2
  ld32step $PROB, $BASE, $PROB_PTR+=, 1
  ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1
  {
    rpt $SIZE, (lossCrossEntropyFloat_loopEnd - lossCrossEntropyFloat_loopStart)/8 - 1
    // assumes EPS_FLOAT can be an immediate to the following instruction
    or  $EPS, $azero, EPS_FLOAT
  }
lossCrossEntropyFloat_loopStart:
  {nop; f32mul $aSCRATCH, $EXPECTED, $MODEL_OUT_SCALING}
  {nop; f32sub $aSCRATCH, $PROB, $aSCRATCH}
  {nop; f32mul $aSCRATCH, $aSCRATCH, $DELTAS_SCALE}
  {st32step $aSCRATCH, $BASE, $DELTA_PTR+=, 1; f32add $PROB, $PROB, $EPS}
  {ld32step $PROB, $BASE, $PROB_PTR+=, 1; f32ln $LN_PROB, $PROB}
  {nop; f32add $LN_PROB, $LN_PROB, $NEG_LOG_MODEL_OUT_SCALING}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1; f32sub $aSCRATCH, $azero, $EXPECTED}
  {nop ; f32mul $aSCRATCH, $aSCRATCH, $LN_PROB}
  {st32step $aSCRATCH, $BASE, $TRANSFORMED_PTR+=, 1; fnop}
lossCrossEntropyFloat_loopEnd:
  exitz $mzero

.size lossCrossEntropyFloat, .-lossCrossEntropyFloat

#undef PROB_PTR
#undef EXPECTED_PTR
#undef DELTA_PTR
#undef TRANSFORMED_PTR
#undef SIZE
#undef BASE
#undef SCALE_PTR

#undef PROB
#undef EXPECTED
#undef DELTA
#undef aSCRATCH
#undef LN_PROB
#undef MODEL_OUT_SCALING
#undef EPS
#undef NEG_LOG_MODEL_OUT_SCALING

.section .text.lossCrossEntropyHalf

.global lossCrossEntropyHalf
.type lossCrossEntropyHalf, @function
.align 8

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    m5
#define SIZE_D2                 m6
#define REMAINDER               m7
#define SCALE_PTR               m8

#define PROB                    a0
#define EXPECTED                a1
#define DELTA                   a2
#define MODEL_OUT_SCALING       a3
#define DELTA_RMW               a4
#define EPS                     a5
#define LN_PROB                 a6
#define aSCRATCH                a7

// To avoid getting div by 0 (-Inf), the smallest denorm value is added to
// the probability.
// i.e. log(prob + 2^-24) is computed instead of log(prob)
// This means that there will be a small error in the estimated loss
// Eg: log(1 + 2^-24) = 0.000000059605 when it should be 0

lossCrossEntropyHalf:
  ldz16 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET
  ldz16 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET
  ldz16 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET
  ldz16 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET
  ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET
  ld32  $SCALE_PTR, $mvertex_base, $mzero, MODEL_OUT_SCALING_OFFSET

  ldb16 $MODEL_OUT_SCALING, $mzero, $SCALE_PTR, 0
  ld32  $SCALE_PTR, $mvertex_base, $mzero, DELTA_SCALE_OFFSET
  // deltasScale / softmaxScale.  Only have a div32 instruction, so cast
  {ldb16 $aSCRATCH, $mzero, $SCALE_PTR, 0
   f16tof32 $EPS, $MODEL_OUT_SCALING}

  {setzi $BASE, TMEM_REGION0_BASE_ADDR
   f16tof32 $aSCRATCH, $aSCRATCH}
  {shl   $PROB_PTR, $PROB_PTR, 2
   f32div   $aSCRATCH, $aSCRATCH, $EPS}
  {shl   $EXPECTED_PTR, $EXPECTED_PTR, 2
   f32tof16 $aSCRATCH, $aSCRATCH}

  // Softmax scale (upper = -log (softmax scale)), (lower = softmax scale)
  {shl   $DELTA_PTR, $DELTA_PTR, 2
   f16v2ln  $EPS, $MODEL_OUT_SCALING}
  {shl   $TRANSFORMED_PTR, $TRANSFORMED_PTR, 2;
   f16v2sub $EPS, $azero, $EPS}
  // Divide size by two - working on two elements at a time
  {shr $SIZE_D2, $SIZE, 1
   sort4x16lo $MODEL_OUT_SCALING, $MODEL_OUT_SCALING, $EPS}

  // EPS: (upper = deltasScale/softmaxScale, lower = EPS)
  {ld32step $PROB, $BASE, $PROB_PTR+=, 1
   setzi $EPS, EPS_HALF}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1
   sort4x16lo $EPS, $EPS, $aSCRATCH}
  {
    rpt $SIZE_D2, (lossCrossEntropyHalf_loopEnd - lossCrossEntropyHalf_loopStart)/8 - 1
    fnop
  }
lossCrossEntropyHalf_loopStart:
  {nop; f16v2mul $aSCRATCH, $MODEL_OUT_SCALING:BL, $EXPECTED}
  {nop; f16v2sub $DELTA, $PROB, $aSCRATCH}
  {nop; f16v2add $PROB,  $EPS:BL, $PROB}
  {nop; f16v2mul $DELTA, $EPS:BU, $DELTA}
  {ld32step $PROB, $BASE, $PROB_PTR+=, 1; f16v2ln $LN_PROB, $PROB}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1; f16v2sub $aSCRATCH, $azero, $EXPECTED}
  {nop; f16v2add $LN_PROB, $MODEL_OUT_SCALING:BU, $LN_PROB}
  {st32step $DELTA, $BASE, $DELTA_PTR+=, 1; f16v2mul $aSCRATCH, $aSCRATCH, $LN_PROB}
  {st32step $aSCRATCH, $BASE, $TRANSFORMED_PTR+=, 1; fnop}
lossCrossEntropyHalf_loopEnd:
  // There is potentially 1 more element to work on - need to perfom RMW
  and $REMAINDER, $SIZE, 0x1
  {
    brz $REMAINDER, lossCrossEntropyHalf_end
    // feed in zero as the unused element. This is further added with EPS before
    // ln is taken
    sort4x16lo $PROB, $PROB, $azero
  }

  // Read, Modify, Write
  f16v2mul $aSCRATCH, $MODEL_OUT_SCALING:BL, $EXPECTED
  {ldb16 $DELTA_RMW, $BASE, $DELTA_PTR, 1; f16v2sub $DELTA, $PROB, $aSCRATCH}
  {nop; f16v2add $PROB, $EPS:BL, $PROB}
  f16v2mul $DELTA, $EPS:BU, $DELTA

  f16v2sub $aSCRATCH, $azero, $EXPECTED

  f16v2ln $LN_PROB, $PROB
  f16v2add $LN_PROB, $MODEL_OUT_SCALING:BU, $LN_PROB
  f16v2mul $aSCRATCH, $aSCRATCH, $LN_PROB

  {ldb16 $EXPECTED, $BASE, $TRANSFORMED_PTR, 1; sort4x16lo $DELTA_RMW, $DELTA, $DELTA_RMW}
  {st32  $DELTA_RMW, $BASE, $DELTA_PTR, 0; sort4x16lo $EXPECTED, $aSCRATCH, $EXPECTED}
  st32  $EXPECTED, $BASE, $TRANSFORMED_PTR, 0

lossCrossEntropyHalf_end:
  exitz $mzero

.size lossCrossEntropyHalf, .-lossCrossEntropyHalf

#undef PROB_PTR
#undef EXPECTED_PTR
#undef DELTA_PTR
#undef TRANSFORMED_PTR
#undef SIZE
#undef BASE
#undef SIZE_D2
#undef REMAINDER
#undef SCALE_PTR

#undef PROB
#undef EXPECTED
#undef DELTA
#undef MODEL_OUT_SCALING
#undef DELTA_RMW
#undef EPS
#undef LN_PROB
#undef aSCRATCH

#endif // __IPU__
