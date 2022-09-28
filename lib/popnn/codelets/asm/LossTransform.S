// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "popops/EncodingConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Assembly implementation of popnn::Loss{SumSquared,CrossEntropy}Transform vertecies
#define lossSumSquaredFloat __runCodelet_popnn__LossSumSquaredTransform___float
#define lossSumSquaredHalf __runCodelet_popnn__LossSumSquaredTransform___half
#define lossCrossEntropyFloat __runCodelet_popnn__LossCrossEntropyTransform___float
#define lossCrossEntropyHalf __runCodelet_popnn__LossCrossEntropyTransform___half

// All offsets are specified in bytes
#define PROB_PTR_OFFSET         0
#define EXPECTED_PTR_OFFSET     4
#define DELTA_PTR_OFFSET        8
#define TRANSFORMED_PTR_OFFSET 12
#define SIZE_OFFSET            16
#define DELTA_SCALE_OFFSET     20
#define MODEL_OUT_SCALING_OFFSET 24

// eps
#define EPS_FLOAT   0x00800000   // 1.17549435e-38F, min value which is not a denorm
#define EPS_HALF    0x0001       // 0.000000059605

/*****************************************************************************/
// popnn::LossSumSquaredTransform<half>
// popnn::LossSumSquaredTransform<float>
/*****************************************************************************/
DEF_STACK_USAGE 0 lossSumSquaredFloat
.section .text.lossSumSquaredFloat
.align 8

.global lossSumSquaredFloat
.type lossSumSquaredFloat, @function

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    mzero
#define PROB                    a0
#define EXPECTED                a1
#define DELTA                   a2
#define TRANSFORMED             a3
#define CONST_HALF              a4

  nop              // for rpt body alignment
lossSumSquaredFloat:
  ld32  $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET/4
  ld32  $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET/4
  ld32  $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET/4
  ld32  $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET/4

  // CONST_HALF = 0x3F000000, where 0x3F000000 represents 0.5 in FP32
  {ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET/2; or $CONST_HALF, $azero, 0x3F000000}

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

DEF_STACK_USAGE 0 lossSumSquaredHalf
.section .text.lossSumSquaredHalf

.global lossSumSquaredHalf
.type lossSumSquaredHalf, @function
.align 8

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    mzero
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
  ld32 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET/4
  ld32 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET/4
  ld32 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET/4
  ld32 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET/4

  // CONST_HALF = [0.5h, 0.5h] in FP16
  {ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET/2; f16v2sigm $CONST_HALF, $azero}

  // Divide size by two - working on two elements at a time
  shr $SIZE_D2, $SIZE, 1

  // Load ahead
  ld32step $PROB, $BASE, $PROB_PTR+=, 1
  ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1

  rpt $SIZE_D2, (lossSumSquaredHalf_loopEnd - lossSumSquaredHalf_loopStart)/8 - 1
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
DEF_STACK_USAGE 0 lossCrossEntropyFloat
.section .text.lossCrossEntropyFloat

.global lossCrossEntropyFloat
.type lossCrossEntropyFloat, @function
.align 8

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    mzero
#define SCALE_PTR               m6

#define PROB                    a0
#define EXPECTED                a1
#define DELTAS_SCALE            a2
#define aSCRATCH                a3
#define LN_PROB                 a4
#define MODEL_OUT_SCALING       a5
#define EPS                     a6
#define NEG_LOG_MODEL_OUT_SCALING   a7

lossCrossEntropyFloat:
  ld32 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET/4
  ld32 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET/4
  ld32 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET/4
  ld32 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET/4
  ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET/2
  ld32  $SCALE_PTR, $mvertex_base, $mzero, MODEL_OUT_SCALING_OFFSET/4

  ld32 $MODEL_OUT_SCALING, $mzero, $SCALE_PTR, 0;
  ld32 $SCALE_PTR, $mvertex_base, $mzero, DELTA_SCALE_OFFSET/4
  ld32 $DELTAS_SCALE, $mzero, $SCALE_PTR, 0;
  f32div $DELTAS_SCALE, $DELTAS_SCALE, $MODEL_OUT_SCALING


  {ld32step $PROB, $BASE, $PROB_PTR+=, 1
   f32ln  $NEG_LOG_MODEL_OUT_SCALING, $MODEL_OUT_SCALING}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1
   f32sub $NEG_LOG_MODEL_OUT_SCALING, $azero, $NEG_LOG_MODEL_OUT_SCALING}

  {
    rpt $SIZE, (lossCrossEntropyFloat_loopEnd - lossCrossEntropyFloat_loopStart)/8 - 1;
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


DEF_STACK_USAGE 0 lossCrossEntropyHalf
.section .text.lossCrossEntropyHalf

.global lossCrossEntropyHalf
.type lossCrossEntropyHalf, @function
.align 8

#define PROB_PTR                m0
#define EXPECTED_PTR            m1
#define DELTA_PTR               m2
#define TRANSFORMED_PTR         m3
#define SIZE                    m4
#define BASE                    mzero
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
  ld32 $DELTA_PTR, $mvertex_base, $mzero, DELTA_PTR_OFFSET/4
  ld32 $TRANSFORMED_PTR, $mvertex_base, $mzero, TRANSFORMED_PTR_OFFSET/4
  ldz16 $SIZE, $mvertex_base, $mzero, SIZE_OFFSET/2
  ld32  $SCALE_PTR, $mvertex_base, $mzero, MODEL_OUT_SCALING_OFFSET/4

  ldb16 $MODEL_OUT_SCALING, $mzero, $SCALE_PTR, 0
  ld32  $SCALE_PTR, $mvertex_base, $mzero, DELTA_SCALE_OFFSET/4
  // deltasScale / softmaxScale.  Only have a div32 instruction, so cast
  {ldb16 $aSCRATCH, $mzero, $SCALE_PTR, 0
   f16tof32 $EPS, $MODEL_OUT_SCALING}

  {ld32 $PROB_PTR, $mvertex_base, $mzero, PROB_PTR_OFFSET/4
   f16tof32 $aSCRATCH, $aSCRATCH}
  {ld32 $EXPECTED_PTR, $mvertex_base, $mzero, EXPECTED_PTR_OFFSET/4
   f32div   $aSCRATCH, $aSCRATCH, $EPS}
  f32tof16 $aSCRATCH, $aSCRATCH

  // Softmax scale (upper = -log (softmax scale)), (lower = softmax scale)
  f16v2ln  $EPS, $MODEL_OUT_SCALING
  f16v2sub $EPS, $azero, $EPS

  // Divide size by two - working on two elements at a time
  {shr $SIZE_D2, $SIZE, 1
   sort4x16lo $MODEL_OUT_SCALING, $MODEL_OUT_SCALING, $EPS}

  // EPS: (upper = deltasScale/softmaxScale, lower = EPS)
  {ld32step $PROB, $BASE, $PROB_PTR+=, 1
   setzi $EPS, EPS_HALF}
  {ld32step $EXPECTED, $BASE, $EXPECTED_PTR+=, 1
   sort4x16lo $EPS, $EPS, $aSCRATCH}
  rpt $SIZE_D2, (lossCrossEntropyHalf_loopEnd - lossCrossEntropyHalf_loopStart)/8 - 1

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
  {
    and $REMAINDER, $SIZE, 0x1
    // feed in zero as unused expected element. This guarantees that the
    // f16v2mul below can never exceed max
    sort4x16lo $EXPECTED, $EXPECTED, $azero
  }
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
