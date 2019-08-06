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
