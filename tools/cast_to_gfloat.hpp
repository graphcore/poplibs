#ifndef TOOLS_CAST_TO_GFLOAT_H_
#define TOOLS_CAST_TO_GFLOAT_H_

#include <array>
#include <cmath>
#include <experimental/popfloat/GfloatExpr.hpp>
#include <experimental/popfloat/CastToGfloat.hpp>

using namespace poplar;
using namespace experimental::popfloat;

#if 0
#include <iomanip>
#define DEBUG_CAST_OPS
#endif

SpecType convertToStorageType(poplar::Type dType) {
  if (dType == FLOAT) {
    return SpecType::FP32;
  } else if (dType == HALF) {
    return SpecType::FP16;
  } else {
    return SpecType::AUTO;
  }
}

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
#endif
