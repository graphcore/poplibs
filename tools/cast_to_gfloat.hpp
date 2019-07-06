#ifndef TOOLS_CAST_TO_GFLOAT_H_
#define TOOLS_CAST_TO_GFLOAT_H_

#include <array>
#include <cmath>
#include <popfloat/GfloatExpr.hpp>
#include <popfloat/CastToGfloat.hpp>

using namespace poplar;

#if 0
#include <iomanip>
#define DEBUG_CAST_OPS
#endif

/*
 * MACROs for manipulating the raw bit format of half-precision values
 */
#define HALF_MANT_SHIFT (0)
#define HALF_MANT_SIZE (10)
#define HALF_MANT_MASK ((1 << HALF_MANT_SIZE) - 1)
#define HALF_EXP_SHIFT HALF_MANT_SIZE
#define HALF_EXP_SIZE (5)
#define HALF_EXP_MASK ((1 << HALF_EXP_SIZE) - 1)
#define HALF_MAX_EXP HALF_EXP_MASK
#define HALF_SIGN_SHIFT (HALF_EXP_SHIFT + HALF_EXP_SIZE)
#define HALF_Q_SHIFT (HALF_EXP_SHIFT - 1)
#define HALF_BIAS (15)
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
#define SINGLE_MANT_SHIFT (0)
#define SINGLE_MANT_SIZE (23)
#define SINGLE_MANT_MASK ((1 << SINGLE_MANT_SIZE) - 1)
#define SINGLE_EXP_SHIFT SINGLE_MANT_SIZE
#define SINGLE_EXP_SIZE (8)
#define SINGLE_EXP_MASK ((1 << SINGLE_EXP_SIZE) - 1)
#define SINGLE_MAX_EXP SINGLE_EXP_MASK
#define SINGLE_SIGN_SHIFT (SINGLE_EXP_SHIFT + SINGLE_EXP_SIZE)
#define SINGLE_Q_SHIFT (SINGLE_EXP_SHIFT - 1)
#define SINGLE_BIAS (127)
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

using namespace popfloat;
using namespace popfloat::gfexpr;

const int manSizeFp32 = 23;
const int manMaskFp32 = (1 << manSizeFp32) - 1;
const int expSizeFp32 = 8;
const int expMaskFp32 = ((1 << expSizeFp32) - 1) << manSizeFp32;
const int expBiasFp32 = 127;
const int sgnMaskFp32 = 1 << (manSizeFp32 + expSizeFp32);
const int qnanFp32    = 0x7FD9C07E;

const int manSizeFp16 = 10;
const int manMaskFp16 = (1 << manSizeFp16) - 1;
const int expSizeFp16 = 5;
const int expMaskFp16 = ((1 << expSizeFp16) - 1) << manSizeFp16;
const int expBiasFp16 = 15;
const int sgnMaskFp16 = 1 << (manSizeFp16 + expSizeFp16);
const int qnanFp16    = 0x7ece;

uint32_t roundMantissaRN(uint32_t m_single,
                         uint32_t masklen) {
  bool msfBitVal = (m_single >> (masklen - 1)) & 1;
  bool lsbs = (m_single & ((1 << (masklen - 1)) - 1)) != 0;
  bool lsBitVal = (m_single >> masklen) & 1;
  uint32_t newMant = (m_single >> masklen);
  if (msfBitVal && (lsBitVal || lsbs)) {
    newMant += 1;
  }
  return newMant;
}

uint16_t floatToHalf(float value, bool enNanoo = false) {

  uint16_t result;
  uint32_t ivalue;
  int exp;

  std::memcpy(&ivalue, &value, sizeof(ivalue));

  result = SINGLE_SIGN(ivalue) << HALF_SIGN_SHIFT;
  exp = SINGLE_EXP(ivalue) - SINGLE_BIAS;

  if (exp < -25) {
    /* Very small values map to +-0
       - nothing more to do.
     */
  } else if (exp < -24) {
    /* Half the smallest denorm could round up to the smallest denorm.
     */
    uint32_t mant = SINGLE_MANT(ivalue) | (1 << (SINGLE_MANT_SIZE + 1));
    mant = roundMantissaRN(mant,
                           (SINGLE_MANT_SIZE + 1));
    result |= (mant << HALF_MANT_SHIFT);
  } else if (exp < -14) {
    /* Small numbers map to denorms - will lose precision
     */

    /* Shift the exponent into the mantissa
     */
    int shift = -exp - (HALF_BIAS);

    /* Combine with the original mantissa shifted into place
     */
    uint16_t mant = 1 << ((HALF_MANT_SIZE - 1) - shift);
    mant += roundMantissaRN(SINGLE_MANT(ivalue),
                            ((SINGLE_MANT_SIZE - HALF_MANT_SIZE) +
                             shift + 1));
    result |= (mant << HALF_MANT_SHIFT);

  } else if (exp <= 15) {
    /* Normal numbers - will lose precision
     */

    uint32_t mant = SINGLE_MANT(ivalue) | (1 << (SINGLE_MANT_SIZE));
    mant = roundMantissaRN(mant,
                           (SINGLE_MANT_SIZE - HALF_MANT_SIZE));
    if (mant >> (HALF_MANT_SIZE + 1)) {
      mant >>= 1;
      ++exp;
    }
    if (exp > 15) {
      if (enNanoo) {
        result |= 0x7BFF;
      } else {
        result |= 65504;
      }
    } else {
      result |= (mant & HALF_MANT_MASK) << HALF_MANT_SHIFT;
      result |= ((exp + HALF_BIAS) << HALF_EXP_SHIFT);
    }
  } else if (exp < 128) {
    /* Large numbers map to infinity if NANOO is OFF, otherwise clip to F16 Max
     */
    if (enNanoo) {
      result |= 0x7BFF;
    } else {
      result |= 65504;
    }

  } else if (isnan(value)) {
    /* NaNs map to NaNs
     */
    uint16_t mant = roundMantissaRN(SINGLE_MANT(ivalue),
                                    (SINGLE_MANT_SIZE - HALF_MANT_SIZE));

    if (SINGLE_IS_QNAN(ivalue)) {
      mant |= (1 << HALF_Q_SHIFT);
    } else {
      mant &= ~(1 << HALF_Q_SHIFT);

      if (mant == 0) {
        /* Ensure NaNs stay as NaNs (non-zero mantissa)
         */
        mant |= 1;
      }

    }

    result |= HALF_INFINITY;
    result |= mant << HALF_MANT_SHIFT;

  } else {
    /* Infinity maps to infinity
     */
    result |= HALF_INFINITY;

  }

  return result;
}

template<typename T, bool deviceHalf>
static void readAndConvertTensor(const Target &target, Engine &eng,
                                 const std::string &handle,
                                 T *out, std::size_t N,
                                 typename std::enable_if<!deviceHalf,
                                 int>::type = 0) {
  eng.readTensor(handle, out);
}

template<typename T, bool deviceHalf = false>
static void readAndConvertTensor(const Target &target, Engine &eng,
                                 const std::string &handle,
                                 T *out, std::size_t N,
                                 typename std::enable_if<std::is_same<T,
                                 float>::value &&deviceHalf,
                                 int>::type = 0) {
  std::vector<char> buf(target.getTypeSize(HALF) * N);
  eng.readTensor(handle, buf.data());
  copyDeviceHalfToFloat(target, buf.data(), out, N);
}

template<typename T, bool deviceHalf>
static void convertAndWriteTensor(const Target &target, Engine &eng,
                                  const std::string &handle,
                                  T *in, std::size_t N,
                                  typename std::enable_if<!deviceHalf,
                                  int>::type = 0) {
  eng.writeTensor(handle, in);
}

template<typename T, bool deviceHalf = false>
static void convertAndWriteTensor(const Target &target, Engine &eng,
                                  const std::string &handle,
                                  T *in, std::size_t N,
                                  typename std::enable_if<std::is_same<T,
                                  float>::value &&deviceHalf,
                                  int>::type = 0) {
  std::vector<char> buf(target.getTypeSize(HALF) * N);
  copyFloatToDeviceHalf(target, in, buf.data(), N);
  eng.writeTensor(handle, buf.data());
}

#ifdef DEBUG_CAST_OPS
void PrintHalf(float      *inVec,
               int         expBias,
               unsigned    sizeVec){
  for (int idx = 0; idx < sizeVec; ++idx) {
    uint16_t bits = floatToHalf(inVec[idx]);

    int sgnBit = (bits >> 15) & 1;
    int expBits = ((bits & expMaskFp16) >> manSizeFp16);
    int manBits = (bits & manMaskFp16);
    int expVal = (expBits - expBias);
    int manVal = manBits;
    manVal |= ((expBits == 0) ? 0 : (1 << manSizeFp16));
    int baseExp = (expBits == 0) ?
      (1 - expBias - manSizeFp16) : (expVal - manSizeFp16);
    float fpVal = (float)manVal * std::pow(2.0, (float)baseExp);
    fpVal       *= (sgnBit ? -1.0 : 1.0);

    std::cout << bits << ": 0x" << std::hex <<
      bits << ", (" << sgnBit << " , " <<
      std::setw(2) << expBits << " , " <<
      std::setw(3) << manBits << ") => (" << std::dec <<
      std::setw(4) << std::setfill(' ') << expVal << "/" <<
      std::setw(4) << std::setfill(' ') << baseExp << " , " <<
      std::setw(4) << std::setfill('0') << manVal << ") => FP = " <<
      std::setw(8) << std::setfill(' ') << fpVal << "\n";

  }
}

void PrintFloat(float       *inVec,
                unsigned    sizeVec) {
  uint32_t bits;
  for (int idx = 0; idx < sizeVec; ++idx) {
    std::memcpy(&bits, &inVec[idx], sizeof(bits));
    int expBits = ((bits & expMaskFp32) >> manSizeFp32);
    int manBits = (bits & manMaskFp32);
    std::cout << std::setfill(' ') <<
      std::setw(8) << inVec[idx] << ": 0x" << std::hex <<
      std::setw(8) << std::setfill('0') << bits << ", (" <<
      std::setw(2) << std::setfill(' ') << expBits << ", " <<
      std::setw(6) << std::setfill('0') << manBits << ")\n" << std::dec;
  }
}

void PrintGfloat8(char       *inVec,
                  unsigned    man,
                  unsigned    exp,
                  int         expBias,
                  unsigned    sizeVec) {
  int manSize = std::log10(std::ceil(std::pow(2.0,man)));
  for (int idx = 0; idx < sizeVec; ++idx) {
    int bits = (int)inVec[idx] & 0xFF;
    int sgnBit = (bits >> 7) & 1;
    int longMan = 7 - exp;
    int expBits = ((bits >> longMan) & ((1 << exp) - 1));
    int manBits = (bits & ((1 << longMan) - 1));
    int expVal = (expBits - expBias);
    int manVal = manBits | ((expBits == 0) ? 0 : (1 << longMan));
    int baseExp = (expBits == 0) ? (1 - expBias - longMan) : expVal - longMan;
    float fpVal = (float)manVal * std::pow(2.0, (float)baseExp);
    fpVal       *= (sgnBit ? -1.0 : 1.0);

    std::cout <<
      std::setw(3) << std::setfill(' ') << bits << ": 0x" <<
      std::hex << std::setw(2) << std::setfill('0') << bits << ", (" <<
      sgnBit << "," <<
      std::setw((exp + 3) / 4) << expBits << " , " <<
      std::setw((man + 3) / 4) << std::setfill('0') << manBits << ") => (" <<
      std::dec <<
      std::setw(4)       << std::setfill(' ') << expVal << "/" <<
      std::setw(4)       << std::setfill(' ') << baseExp << " , " <<
      std::setw(manSize) << std::setfill('0') << manVal << ") => FP = " <<
      std::setw(8)       << std::setfill(' ') << fpVal << "\n";

  }
}

void PrintGfloat16(short      *inVec,
                   unsigned    man,
                   unsigned    exp,
                   int         expBias,
                   bool        maxAligned,
                   unsigned    sizeVec) {
  int manDec = 1 + std::log10(std::pow(2.0, man + 1));

  for (int idx = 0; idx < sizeVec; ++idx) {
    int bits = (int)inVec[idx] & 0xFFFF;
    int sgnBit = (bits >> 15) & 1;
    int longMan = 15 - exp;
    int expBits = ((bits >> longMan) & ((1 << exp) - 1));
    int manBits = (bits & ((1 << longMan) - 1));
    int expVal = (expBits - expBias + maxAligned);
    int manVal = manBits | ((expBits == 0) ? 0 : (1 << longMan));
    int baseExp = (expBits == 0) ? (1 - expBias - longMan) : expVal - longMan;
    float fpVal = (float)manVal * std::pow(2.0, (float)baseExp);
    fpVal       *= (sgnBit ? -1.0 : 1.0);
    unsigned fpBits;
    std::memcpy(&fpBits, &fpVal, sizeof(fpVal));

    std::cout <<
      std::setw(5) << std::setfill(' ') << bits << ": 0x" << std::hex <<
      std::setw(4) << std::setfill('0') << bits << ", (" << sgnBit << " , " <<
      std::setw((exp + 3) / 4) << std::setfill('0') << expBits << " , " <<
      std::setw((man + 3) / 4) << std::setfill('0') << manBits << ") => (" <<
      std::dec <<
      std::setw(4) << std::setfill(' ') << expVal << "/" <<
      std::setw(4) << std::setfill(' ') << baseExp << " , " <<
      std::setw(manDec) << std::setfill(' ') << manVal << ") => FP = " <<
      std::setw(8) << std::setfill(' ') << fpVal << "(0x" <<
      std::hex << std::setw(4) << fpBits << std::dec << ")\n";
  }
}

#endif
#endif
