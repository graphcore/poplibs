// Copyright (c) Graphcore Ltd, All rights reserved.
#ifndef TOOLS_CAST_TO_GFLOAT_H_
#define TOOLS_CAST_TO_GFLOAT_H_

#include <array>
#include <cmath>
#include <experimental/popfloat/CastToGfloat.hpp>
#include <experimental/popfloat/GfloatExpr.hpp>

const int manSizeFp32 = 23;
const int manMaskFp32 = (1 << manSizeFp32) - 1;
const int expSizeFp32 = 8;
const int expMaskFp32 = ((1 << expSizeFp32) - 1) << manSizeFp32;
const int expBiasFp32 = 127;
const int infAndNanExpFp32 = 255;
const int sgnMaskFp32 = 1 << (manSizeFp32 + expSizeFp32);
const int qnanFp32 = 0x7FD9C07E;

const int manSizeFp16 = 10;
const int manMaskFp16 = (1 << manSizeFp16) - 1;
const int expSizeFp16 = 5;
const int expMaskFp16 = ((1 << expSizeFp16) - 1) << manSizeFp16;
const int expBiasFp16 = 15;
const int sgnMaskFp16 = 1 << (manSizeFp16 + expSizeFp16);
const int qnanFp16 = 0x7ece;

experimental::popfloat::SpecType
convertTypeToGfloatSpecType(poplar::Type dType) {
  if (dType == poplar::FLOAT) {
    return experimental::popfloat::SpecType::FP32;
  } else if (dType == poplar::HALF) {
    return experimental::popfloat::SpecType::FP16;
  } else {
    return experimental::popfloat::SpecType::AUTO;
  }
}

experimental::popfloat::SpecType
convertStringToSpecType(const std::string &specType) {
  if (specType == "AUTO") {
    return experimental::popfloat::SpecType::AUTO;
  } else if (specType == "FP32") {
    return experimental::popfloat::SpecType::FP32;
  } else if (specType == "FP16") {
    return experimental::popfloat::SpecType::FP16;
  } else if (specType == "INT8") {
    return experimental::popfloat::SpecType::INT8;
  } else if (specType == "INT16") {
    return experimental::popfloat::SpecType::INT16;
  }
  throw poputil::poplibs_error("Type not supported");
}

experimental::popfloat::RoundType
convertStringToRoundType(const std::string &roundMode, poplar::Type inType,
                         unsigned srBits) {
  if (roundMode == "RZ") {
    return experimental::popfloat::RoundType::RZ;
  } else if (roundMode == "RN") {
    return experimental::popfloat::RoundType::RN;
  } else if (roundMode == "RA") {
    return experimental::popfloat::RoundType::RA;
  } else if (roundMode == "RU") {
    return experimental::popfloat::RoundType::RU;
  } else if (roundMode == "RD") {
    return experimental::popfloat::RoundType::RD;
  } else if (roundMode == "SR") {
    bool isExtendedSr =
        srBits <
        unsigned((inType == poplar::FLOAT) ? manSizeFp32 : manSizeFp16);
    if (isExtendedSr) {
      return experimental::popfloat::RoundType::SX;
    } else {
      return experimental::popfloat::RoundType::SR;
    }
  }
  throw poputil::poplibs_error("Round Mode not supported");
}

template <typename T, bool deviceHalf>
static void
readAndConvertTensor(const poplar::Target &target, poplar::Engine &eng,
                     const std::string &handle, T *out, std::size_t N,
                     typename std::enable_if<!deviceHalf, int>::type = 0) {
  eng.readTensor(handle, out);
}

template <typename T, bool deviceHalf = false>
static void readAndConvertTensor(
    const poplar::Target &target, poplar::Engine &eng,
    const std::string &handle, T *out, std::size_t N,
    typename std::enable_if<std::is_same<T, float>::value && deviceHalf,
                            int>::type = 0) {
  std::vector<char> buf(target.getTypeSize(poplar::HALF) * N);
  eng.readTensor(handle, buf.data());
  copyDeviceHalfToFloat(target, buf.data(), out, N);
}

template <typename T, bool deviceHalf>
static void
convertAndWriteTensor(const poplar::Target &target, poplar::Engine &eng,
                      const std::string &handle, T *in, std::size_t N,
                      typename std::enable_if<!deviceHalf, int>::type = 0) {
  eng.writeTensor(handle, in);
}

template <typename T, bool deviceHalf = false>
static void convertAndWriteTensor(
    const poplar::Target &target, poplar::Engine &eng,
    const std::string &handle, T *in, std::size_t N,
    typename std::enable_if<std::is_same<T, float>::value && deviceHalf,
                            int>::type = 0) {
  std::vector<char> buf(target.getTypeSize(poplar::HALF) * N);
  copyFloatToDeviceHalf(target, in, buf.data(), N);
  eng.writeTensor(handle, buf.data());
}
#endif
