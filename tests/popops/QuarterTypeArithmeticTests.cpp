// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE QuarterTypeArithmeticTests

#include <cmath>
#include <poplar/Engine.hpp>
#include <poplar/MetadataCreation.hpp>
#include <poplar/TypeConversion.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/Cast.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <vector>

#include <gccs/Algorithm.hpp>

#define QUARTER_METADATA_SCALE_BIAS_MIN -32
#define QUARTER_METADATA_SCALE_BIAS_MAX 31

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_support;

static void testCastFromQuarter(const Type &outType,
                                const QuarterMetadata &metadata,
                                const std::vector<float> &hIn) {
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);

  unsigned numElements = hIn.size();

  auto in = graph.addVariable(QUARTER, {numElements}, "in");
  mapTensorLinearly(graph, in);
  graph.createHostWrite("in", in);

  auto prog = Sequence();
  poplar::Tensor out =
      cast(graph, in, outType, prog, "castTo" + outType.toString());
  graph.createHostRead("out", out);
  std::vector<char> rawIn(target.getTypeSize(QUARTER) * numElements);
  std::vector<char> rawOut(target.getTypeSize(outType) * numElements);

  poplar::convertToDeviceType(QUARTER, metadata, gccs::ArrayRef(hIn),
                              rawIn.data());

  Engine eng(graph, Sequence{prog});
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", metadata, rawIn.data(), rawIn.data() + rawIn.size());
    eng.run();
    eng.readTensor("out", rawOut.data(), rawOut.data() + rawOut.size());
  });
  std::vector<float> hOut(numElements);
  poplar::convertFromDeviceType(outType, rawOut.data(), gccs::ArrayRef(hOut));

  /* Check result */
  for (auto i = 0U; i < numElements; ++i) {
    if (std::isnan(hIn[i])) {
      BOOST_CHECK(std::isnan(hOut[i]));
    } else {
      BOOST_TEST(hOut[i] == hIn[i]);
    }
  }
}

// Sweep across a range of values that can be represented in quarter with the
// given metadata.
static void quarterSweepTest(const Type &otherType,
                             const QuarterMetadata &metadata,
                             unsigned char quartBegin,
                             unsigned numElements = 1) {
  std::vector<unsigned char> quartData(numElements);
  std::iota(quartData.begin(), quartData.end(), quartBegin);
  std::vector<float> testInput(numElements);
  poplar::convertFromDeviceType(QUARTER, metadata, quartData.data(),
                                gccs::ArrayRef(testInput));
  testCastFromQuarter(otherType, metadata, testInput);
}

BOOST_AUTO_TEST_CASE(CastHalfFromQuarterF143Sweep) {
  // Half range is [2^-24, 2^15]
  // Quarter F143 range without scaling is [2^-10, 2^7]
  // The scale range that keeps Quarter representable in Half is [2^-14, 2^8]
  quarterSweepTest(HALF, QuarterMetadata(QuarterMetadata::Format::F143, -14), 0,
                   256);
  quarterSweepTest(HALF, QuarterMetadata(QuarterMetadata::Format::F143, 8), 0,
                   256);
}

BOOST_AUTO_TEST_CASE(CastHalfFromQuarterF152Sweep) {
  // Half range is [2^-24, 2^15]
  // Quarter F152 range without scaling is [2^-17, 2^15]
  // The scale range that keeps Quarter representable in Half is [2^-7, 2^0]
  quarterSweepTest(HALF, QuarterMetadata(QuarterMetadata::Format::F152, -7), 0,
                   256);
  quarterSweepTest(HALF, QuarterMetadata(QuarterMetadata::Format::F152, 0), 0,
                   256);
}

BOOST_AUTO_TEST_CASE(CastFloatFromQuarterF143Sweep) {
  quarterSweepTest(FLOAT,
                   QuarterMetadata(QuarterMetadata::Format::F143,
                                   QUARTER_METADATA_SCALE_BIAS_MIN),
                   0, 256);
  quarterSweepTest(FLOAT,
                   QuarterMetadata(QuarterMetadata::Format::F143,
                                   QUARTER_METADATA_SCALE_BIAS_MAX),
                   0, 256);
}

BOOST_AUTO_TEST_CASE(CastFloatFromQuarterF152Sweep) {
  quarterSweepTest(FLOAT,
                   QuarterMetadata(QuarterMetadata::Format::F152,
                                   QUARTER_METADATA_SCALE_BIAS_MIN),
                   0, 256);
  quarterSweepTest(FLOAT,
                   QuarterMetadata(QuarterMetadata::Format::F152,
                                   QUARTER_METADATA_SCALE_BIAS_MAX),
                   0, 256);
}

static std::vector<float> testCastToQuarter(const Type &inType,
                                            const QuarterMetadata &metadata,
                                            const std::vector<float> &hIn,
                                            bool checkResults = true) {
  auto device = createTestDevice(TEST_TARGET);
  auto target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);

  unsigned numElements = hIn.size();
  auto in = graph.addVariable(inType, {numElements}, "in");
  mapTensorLinearly(graph, in);
  graph.createHostWrite("in", in);

  auto prog = Sequence();
  auto metadataTensor = poplar::createVariableMetadataTensor(
      graph, metadata.getFormat(), metadata.getScale());
  auto out = cast(graph, in, QUARTER, metadataTensor, prog);

  graph.createHostRead("out", out);
  QuarterMetadata resultMetadata;

  // Quantize to Quarter
  auto quantizeToQuarter = [&](const QuarterMetadata &md,
                               const std::vector<float> &in) {
    std::vector<char> quartData(target.getTypeSize(QUARTER) * in.size());
    poplar::convertToDeviceType(QUARTER, md, gccs::ArrayRef(in),
                                quartData.data());
    std::vector<float> quantized(in.size());
    poplar::convertFromDeviceType(QUARTER, md, quartData.data(),
                                  gccs::ArrayRef(quantized));
    return quantized;
  };
  auto hInQuantized = quantizeToQuarter(metadata, hIn);

  std::vector<char> rawIn(target.getTypeSize(inType) * numElements);
  std::vector<char> rawOut(target.getTypeSize(QUARTER) * numElements);

  poplar::convertToDeviceType(inType, hIn, rawIn.data());

  Engine eng(graph, Sequence{prog});
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", rawIn.data(), rawIn.data() + rawIn.size());
    eng.run();
    eng.readTensor("out", resultMetadata, rawOut.data(),
                   rawOut.data() + rawOut.size());
  });

  std::vector<float> hOut(numElements);
  poplar::convertFromDeviceType(QUARTER, resultMetadata, rawOut.data(),
                                gccs::ArrayRef(hOut));

  /* Check result */
  if (checkResults) {
    for (auto i = 0U; i < numElements; ++i) {
      if (std::isnan(hInQuantized[i]) || std::isinf(hIn[i])) {
        BOOST_CHECK(std::isnan(hOut[i]));
      } else {
        BOOST_TEST(hOut[i] == hInQuantized[i]);
      }
    }
    BOOST_TEST(resultMetadata == metadata);
  }
  return hOut;
}

BOOST_AUTO_TEST_CASE(CastFloatNaNToQuarterF143) {
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F143, 1),
                    {std::nanf("0")});
}

BOOST_AUTO_TEST_CASE(CastFloatNaNToQuarterF152) {
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F152, 1),
                    {std::nanf("0")});
}

BOOST_AUTO_TEST_CASE(CastFloatInfToQuarterF143) {
  auto inf = std::numeric_limits<float>::infinity();
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F143, 1),
                    {inf, -inf});
}

BOOST_AUTO_TEST_CASE(CastFloatInfToQuarterF152) {
  auto inf = std::numeric_limits<float>::infinity();
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F152, 1),
                    {inf, -inf});
}

// Test all combinations of a given subset of exponent and mantissa bits.
// The metadata scale is used to center the floating point scale.
static void floatSweepTest(const Type &otherType,
                           const QuarterMetadata &metadata,
                           const std::vector<unsigned> &exponentBits,
                           const std::vector<unsigned> &mantissaBits) {
  unsigned floatExponentBitPos = 23;
  unsigned floatBias = 127;
  unsigned quarterExponentBit =
      metadata.getFormat() == QuarterMetadata::Format::F143 ? 3u : 4u;
  int quarterBias =
      metadata.getFormat() == QuarterMetadata::Format::F143 ? 8 : 16;

  auto numBits = exponentBits.size() + mantissaBits.size();
  auto numCombinations = std::exp2(numBits);
  std::vector<float> testInput(2 * numCombinations, 0);
  for (unsigned i = 0; i < numCombinations; ++i) {
    union {
      float f32;
      uint32_t u32;
    } value;
    value.u32 = 0;
    unsigned bit = 0;

    // Flip mantissa bits
    for (auto it = mantissaBits.rbegin(); it != mantissaBits.rend(); ++it) {
      value.u32 |= ((i >> bit) & 1) << *it;
      bit++;
    }

    int exponent = 0;
    for (auto it = exponentBits.rbegin(); it != exponentBits.rend(); ++it) {
      if (*it > quarterExponentBit) {
        std::cerr << "Exponent bit " << *it << " exceeds the " << metadata
                  << " exponent range." << std::endl;
      }
      exponent |= ((i >> bit) & 1) << *it;
      bit++;
    }
    exponent -= quarterBias;

    // Add metadata cale and exponent bias
    value.u32 += (exponent + metadata.getScale() + floatBias)
                 << floatExponentBitPos;

    // Test for both sign positions
    testInput[i] = value.f32;
    testInput[i + numCombinations] = -value.f32;
  }
  testCastToQuarter(otherType, metadata, testInput);
}

BOOST_AUTO_TEST_CASE(CastHalfToQuarterF143Sweep,
                     *boost::unit_test::precondition(enableIfIpu21())) {
  // Half range is [2^-24, 2^15]
  // Quarter F143 range without scaling is [2^-10, 2^7]
  // The scale range that keeps Quarter representable in Half is [2^-14, 2^8]
  floatSweepTest(HALF, QuarterMetadata(QuarterMetadata::Format::F143, -14),
                 {3, 1, 0}, {9, 4, 1, 0});
  floatSweepTest(HALF, QuarterMetadata(QuarterMetadata::Format::F143, 8),
                 {3, 2, 0}, {9, 4, 1, 0});
}

BOOST_AUTO_TEST_CASE(CastHalfToQuarterF152Sweep) {
  // Half range is [2^-24, 2^15]
  // Quarter F152 range without scaling is [2^-10, 2^7]
  // The scale range that keeps Quarter representable in Half is [2^-14, 2^8]
  floatSweepTest(HALF, QuarterMetadata(QuarterMetadata::Format::F152, -7),
                 {4, 3, 0}, {9, 4, 1, 0});
  floatSweepTest(HALF, QuarterMetadata(QuarterMetadata::Format::F152, 0),
                 {4, 2, 0}, {9, 4, 1, 0});
}

BOOST_AUTO_TEST_CASE(CastFloatToQuarterF143Sweep,
                     *boost::unit_test::precondition(enableIfIpu21())) {
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F143,
                                 QUARTER_METADATA_SCALE_BIAS_MIN),
                 {3, 1, 0}, {22, 21, 20, 19, 12, 0});
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F143,
                                 QUARTER_METADATA_SCALE_BIAS_MAX),
                 {3, 2, 0}, {22, 21, 20, 19, 12, 0});
}

BOOST_AUTO_TEST_CASE(CastFloatToQuarterF152Sweep,
                     *boost::unit_test::precondition(enableIfIpu21())) {
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F152,
                                 QUARTER_METADATA_SCALE_BIAS_MIN),
                 {4, 2, 0}, {22, 21, 20, 19, 12, 0});
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F152,
                                 QUARTER_METADATA_SCALE_BIAS_MAX),
                 {4, 3, 0}, {22, 21, 20, 19, 12, 0});
}

static void testFloatToQuarterRounding(const QuarterMetadata &metadata) {
  unsigned numMantissa =
      metadata.getFormat() == QuarterMetadata::Format::F143 ? 3u : 2u;
  float quarterDenormStep =
      metadata.getFormat() == QuarterMetadata::Format::F143 ? std::exp2(-10)
                                                            : std::exp2(-17);
  float quarterDenormHalfStep = quarterDenormStep / 2;
  float halfDenormStep = std::exp2(-23);
  float quarterScaling = std::exp2(metadata.getScale());
  unsigned numMantissaLevels = 1 << numMantissa;
  unsigned testRange = 4 * numMantissaLevels;
  std::vector<float> testInput(testRange);
  for (unsigned i = 0; i < numMantissaLevels; ++i) {
    float value = quarterDenormStep * i + quarterDenormHalfStep;
    testInput[2 * i] = value * quarterScaling;
    testInput[2 * i + 1] = (value + halfDenormStep) * quarterScaling;
    testInput[2 * i + (testRange / 2)] = -testInput[2 * i];
    testInput[2 * i + 1 + (testRange / 2)] = -testInput[2 * i + 1];
  };
  auto result = testCastToQuarter(FLOAT, metadata, testInput, false);
  for (unsigned i = 0; i < numMantissaLevels; ++i) {
    float nextValue = quarterDenormStep * (i + 1) * quarterScaling;
    float nearestEven =
        quarterDenormStep * gccs::ceildiv(i, 2u) * 2 * quarterScaling;
    BOOST_TEST(result[2 * i] == nearestEven);
    BOOST_TEST(result[2 * i + 1] == nextValue);
    BOOST_TEST(result[2 * i + (testRange / 2)] == -nearestEven);
    BOOST_TEST(result[2 * i + 1 + (testRange / 2)] == -nextValue);
  }
}

BOOST_AUTO_TEST_CASE(RoundingFloatToQuarterF143,
                     *boost::unit_test::precondition(enableIfIpu21())) {
  testFloatToQuarterRounding(QuarterMetadata(QuarterMetadata::Format::F143,
                                             QUARTER_METADATA_SCALE_BIAS_MIN));
  testFloatToQuarterRounding(QuarterMetadata(QuarterMetadata::Format::F143,
                                             QUARTER_METADATA_SCALE_BIAS_MAX));
}

BOOST_AUTO_TEST_CASE(RoundingFloatToQuarterF152,
                     *boost::unit_test::precondition(enableIfIpu21())) {
  testFloatToQuarterRounding(QuarterMetadata(QuarterMetadata::Format::F152,
                                             QUARTER_METADATA_SCALE_BIAS_MIN));
  testFloatToQuarterRounding(QuarterMetadata(QuarterMetadata::Format::F152,
                                             QUARTER_METADATA_SCALE_BIAS_MAX));
}
