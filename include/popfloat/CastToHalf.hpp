// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef POPFLOAT_HALF_UTILS_H
#define POPFLOAT_HALF_UTILS_H

#include <array>
#include <cmath>

/*
 * MACROs for manipulating the raw bit format of half-precision values
 */
#define HALF_MAN_SIZE   (10)
#define HALF_MANT_SHIFT (0)
#define HALF_MANT_MASK ((1 << HALF_MAN_SIZE) - 1)
#define HALF_EXP_SIZE   (5)
#define HALF_EXP_SHIFT HALF_MAN_SIZE
#define HALF_EXP_MASK ((1 << HALF_EXP_SIZE) - 1)
#define HALF_MAX_EXP HALF_EXP_MASK
#define HALF_EXP_BIAS (15)
#define HALF_SIGN_SHIFT (HALF_EXP_SHIFT + HALF_EXP_SIZE)
#define HALF_Q_SHIFT (HALF_EXP_SHIFT - 1)
#define HALF_EXP(v) (((v) >> HALF_EXP_SHIFT) & HALF_EXP_MASK)
#define HALF_MANT(v) (((v) >> HALF_MANT_SHIFT) & HALF_MANT_MASK)
#define HALF_SIGN(v) (((v) >> HALF_SIGN_SHIFT) & 1)
#define HALF_IS_NEG(v) (HALF_SIGN(v) != 0)
#define HALF_IS_ZERO(v) ((HALF_EXP(v) == 0) && (HALF_MANT(v) == 0))
#define HALF_IS_SUBNORM(v) ((HALF_EXP(v) == 0) && (HALF_MANT(v) != 0))
#define HALF_IS_INFINITY(v) ((HALF_EXP(v) == HALF_MAX_EXP) && \
                             (HALF_MANT(v) == 0))
#define HALF_IS_NAN(v) ((HALF_EXP(v) == HALF_MAX_EXP) && \
                         (HALF_MANT(v) != 0))
#define HALF_IS_QNAN(v) (HALF_IS_NAN(v) && (((v >> HALF_Q_SHIFT) & 1) == 1))
#define HALF_IS_SNAN(v) (HALF_IS_NAN(v) && (((v >> HALF_Q_SHIFT) & 1) == 0))
#define HALF_INFINITY (HALF_MAX_EXP << HALF_EXP_SHIFT)

/*
 * MACROs for manipulating the raw bit format of single-precision values
 */
#define SINGLE_MAN_SIZE (23)
#define SINGLE_MANT_SHIFT (0)
#define SINGLE_MANT_MASK ((1 << SINGLE_MAN_SIZE) - 1)
#define SINGLE_EXP_SHIFT SINGLE_MAN_SIZE
#define SINGLE_EXP_SIZE (8)
#define SINGLE_EXP_MASK ((1 << SINGLE_EXP_SIZE) - 1)
#define SINGLE_MAX_EXP SINGLE_EXP_MASK
#define SINGLE_SIGN_SHIFT (SINGLE_EXP_SHIFT + SINGLE_EXP_SIZE)
#define SINGLE_Q_SHIFT (SINGLE_EXP_SHIFT - 1)
#define SINGLE_EXP_BIAS (127)
#define SINGLE_EXP(v) (((v) >> SINGLE_EXP_SHIFT) & SINGLE_EXP_MASK)
#define SINGLE_MANT(v) (((v) >> SINGLE_MANT_SHIFT) & SINGLE_MANT_MASK)
#define SINGLE_SIGN(v) (((v) >> SINGLE_SIGN_SHIFT) & 1)
#define SINGLE_IS_NEG(v) (SINGLE_SIGN(v) != 0)
#define SINGLE_IS_ZERO(v) ((SINGLE_EXP(v) == 0) && (SINGLE_MANT(v) == 0))
#define SINGLE_IS_SUBNORM(v) ((SINGLE_EXP(v) == 0) && (SINGLE_MANT(v) != 0))
#define SINGLE_IS_INFINITY(v) ((SINGLE_EXP(v) == SINGLE_MAX_EXP) && \
                                (SINGLE_MANT(v) == 0))
#define SINGLE_IS_NAN(v) ((SINGLE_EXP(v) == SINGLE_MAX_EXP) && \
                           (SINGLE_MANT(v) != 0))
#define SINGLE_IS_QNAN(v) (SINGLE_IS_NAN(v) && \
                           ((((v) >> SINGLE_Q_SHIFT) & 1) == 1))
#define SINGLE_IS_SNAN(v) (SINGLE_IS_NAN(v) && \
                            ((((v) >> SINGLE_Q_SHIFT) & 1) == 0))
#define SINGLE_INFINITY (SINGLE_MAX_EXP << SINGLE_EXP_SHIFT)

namespace popfloat {

/** Cast a single precision input to half precision
 *
 * \param value          Single precision input
 * \param enNanoo        Enable Nan on overflow
 * \return               The 16-bit representation of the half precision output
 */
uint16_t singleToHalf(float value, bool enNanoo = false);

/** Cast a half precision input to single precision
 *
 * \param ihalf          The 16-bit representation of the half precision input
 * \param enNanoo        Enable Nan on overflow
 * \return               Single precision output
 */
float halfToSingle(uint16_t ihalf);

} //end namespace popfloat
#endif
