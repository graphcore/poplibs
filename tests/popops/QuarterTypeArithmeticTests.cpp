// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE QuarterTypeArithmeticTests

#include <cmath>
#include <poplar/CSRFunctions.hpp>
#include <poplar/Engine.hpp>
#include <poplar/MetadataCreation.hpp>
#include <poplar/TypeConversion.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/Cast.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <vector>

#include <gccs/Algorithm.hpp>

#include <iomanip>

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
  auto device = createTestDevice(TEST_TARGET, 1, 4);
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

// Function to cast to quarter:
// `convertToFloatWithZeroScale` will suppress host side scaling of the
// data when reading back from the device so that the effect of saturation
// is clear and can be checked for
static std::vector<float>
testCastToQuarter(const Type &inType, const QuarterMetadata &metadata,
                  const std::vector<float> &hIn, bool checkResults = true,
                  bool nanoo = true, bool convertToFloatWithZeroScale = false) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);

  unsigned numElements = hIn.size();
  auto in = graph.addVariable(inType, {numElements}, "in");
  mapTensorLinearly(graph, in);
  graph.createHostWrite("in", in);

  auto prog = Sequence();

  poplar::setFloatingPointBehaviour(graph, prog,
                                    {/*inv*/ false,
                                     /*div0*/ true, /*oflo*/ false,
                                     /*esr*/ false, nanoo});

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
                                quartData.data(), nanoo);
    std::vector<float> quantized(in.size());
    auto toFloatMetadata = convertToFloatWithZeroScale
                               ? QuarterMetadata(metadata.getFormat(), 0)
                               : md;
    poplar::convertFromDeviceType(QUARTER, toFloatMetadata, quartData.data(),
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
  auto toFloatMetadata = convertToFloatWithZeroScale
                             ? QuarterMetadata(metadata.getFormat(), 0)
                             : resultMetadata;
  poplar::convertFromDeviceType(QUARTER, toFloatMetadata, rawOut.data(),
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

// Function to cast to quarter:
// `convertToFloatWithZeroScale` will suppress host side scaling of the
// data when reading back from the device so that the effect of saturation
// is clear and can be checked for
static std::vector<float>
testCastToQuarter(const Type &inType, const QuarterMetadata &metadata,
                  const std::vector<float> &hIn,
                  const std::vector<unsigned short> &halfIn,
                  bool checkResults = true, bool nanoo = true,
                  bool convertToFloatWithZeroScale = false) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);

  unsigned numElements = hIn.size();
  auto in = graph.addVariable(inType, {numElements}, "in");
  mapTensorLinearly(graph, in);
  graph.createHostWrite("in", in);

  auto prog = Sequence();

  poplar::setFloatingPointBehaviour(graph, prog,
                                    {/*inv*/ false,
                                     /*div0*/ true, /*oflo*/ false,
                                     /*esr*/ false, nanoo});

  auto metadataTensor = poplar::createVariableMetadataTensor(
      graph, metadata.getFormat(), metadata.getScale());
  auto out = cast(graph, in, QUARTER, metadataTensor, prog);

  graph.createHostRead("out", out);
  QuarterMetadata resultMetadata;

  // Create reference quarter data
  auto rawQuarter = [&](const QuarterMetadata &md,
                        const std::vector<float> &in) {
    std::vector<char> quartData(target.getTypeSize(QUARTER) * in.size());
    poplar::convertToDeviceType(QUARTER, md, gccs::ArrayRef(in),
                                quartData.data(), nanoo);
    return quartData;
  };
  auto quartData = rawQuarter(metadata, hIn);

  std::vector<char> rawIn(target.getTypeSize(inType) * numElements);
  std::vector<char> rawOut(target.getTypeSize(QUARTER) * numElements);

  std::memcpy(&rawIn[0], &halfIn[0], target.getTypeSize(inType) * numElements);
  Engine eng(graph, Sequence{prog});
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", rawIn.data(), rawIn.data() + rawIn.size());
    eng.run();
    eng.readTensor("out", resultMetadata, rawOut.data(),
                   rawOut.data() + rawOut.size());
  });
  auto toFloatMetadata = convertToFloatWithZeroScale
                             ? QuarterMetadata(metadata.getFormat(), 0)
                             : resultMetadata;
  std::vector<float> hOut(numElements);
  poplar::convertFromDeviceType(QUARTER, toFloatMetadata, rawOut.data(),
                                gccs::ArrayRef(hOut));
  std::vector<float> hOutExp(numElements);
  poplar::convertFromDeviceType(QUARTER, toFloatMetadata, quartData.data(),
                                gccs::ArrayRef(hOutExp));
  /* Check result */
  if (checkResults) {
    for (auto i = 0U; i < numElements; ++i) {
      auto result = 0xff & unsigned(rawOut[i]);
      auto expected = 0xff & unsigned(quartData[i]);
      if (result != expected) {
        std::cerr << "Test scale:" << int(metadata.getScale())
                  << " expected:" << expected << "(" << hOutExp[i]
                  << ") result:" << result << "(" << hOut[i] << ")\n";
      }
      BOOST_TEST(result == expected);
    }
    BOOST_TEST(resultMetadata == metadata);
  }
  return hOut;
}

BOOST_AUTO_TEST_CASE(CastFloatNaNToQuarterF143) {
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F143, 1),
                    {std::nanf("0")}, true, true);
}

BOOST_AUTO_TEST_CASE(CastFloatNaNToQuarterF152) {
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F152, 1),
                    {std::nanf("0")}, true, true);
}

BOOST_AUTO_TEST_CASE(CastFloatInfToQuarterF143) {
  auto inf = std::numeric_limits<float>::infinity();
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F143, 1),
                    {inf, -inf}, true, true);
}

BOOST_AUTO_TEST_CASE(CastFloatInfToQuarterF152) {
  auto inf = std::numeric_limits<float>::infinity();
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F152, 1),
                    {inf, -inf}, true, true);
}

void testFloatWithExpectedResult(
    const std::vector<float> &input,
    const std::vector<std::vector<float>> &expected,
    QuarterMetadata::Format quarterType, const std::vector<int> &scales,
    bool nanoo) {

  for (unsigned s = 0; s < scales.size(); s++) {
    auto result =
        testCastToQuarter(FLOAT, QuarterMetadata(quarterType, scales[s]), input,
                          true, nanoo, true);
    std::cerr << "Scale:" << scales[s] << "\n";
    for (unsigned i = 0; i < input.size(); i++) {
      std::cerr << "result[" << i << "]:" << result[i] << "\n";
      if (std::isnan(expected[s][i])) {
        BOOST_TEST(std::isnan(result[i]));
      } else {
        BOOST_TEST(result[i] == expected[s][i]);
      }
    }
  }
}

// Even when using Sim21 to cast from float to float8 types we have to be
// careful about the overflow behaviour as we use half as an intermediate type.
// So check some values against a specific expected result.
// The intent is to produce values that will saturate / give nan as the result
// with specific scales
// For example:
// 2^33 is greater than maxPosF143(240)
// But 2^33 * 2^-32 = 2. which is less than maxPosF143 and exactly representible
//
// So a test case with scale = 0 will result in saturate or nan, and a test case
// with scale = -32 will not
BOOST_AUTO_TEST_CASE(CastFloatOverflowToQuarterF143Nanoo) {
  int testScale = -10;
  float largeIn = std::pow(2, 33.0f);
  float largeOut = largeIn * std::pow(2, -QUARTER_METADATA_SCALE_BIAS_MAX);
  float smallIn = std::pow(2, -12.0f);
  float smallOut = smallIn * std::pow(2, -testScale);

  const auto nan = std::nanf("0");
  auto inf = std::numeric_limits<float>::infinity();
  std::vector<float> input = {smallIn, -smallIn, largeIn, -largeIn, 8.0f,
                              -8.0f,   nan,      -inf,    inf};
  std::vector<std::vector<float>> expected = {
      {0.0f, 0.0f, nan, nan, 8.0f, -8.0f, nan, nan, nan},
      {smallOut, -smallOut, nan, nan, nan, nan, nan, nan, nan},
      {0.0f, 0.0f, largeOut, -largeOut, 0, 0, nan, nan, nan}};

  std::vector<int> scales = {0, testScale, QUARTER_METADATA_SCALE_BIAS_MAX};
  testFloatWithExpectedResult(input, expected, QuarterMetadata::Format::F143,
                              scales, true);
}

// Disable this test for IpuModel as it has no FP_CTL register and so
// behaves as if Nanoo is always set
BOOST_AUTO_TEST_CASE(CastFloatOverflowToQuarterF143Saturate,
                     *boost::unit_test::precondition(enableIfSimOrHw())) {
  float largeIn = std::pow(2, 33.0f);
  float largeOut = largeIn * std::pow(2, -QUARTER_METADATA_SCALE_BIAS_MAX);

  const float maxPosF143 = 240;
  const auto nan = std::nanf("0");
  auto inf = std::numeric_limits<float>::infinity();

  std::vector<float> input = {largeIn, -largeIn, 8.0f, -8.0f, nan, -inf, inf};
  std::vector<std::vector<float>> expected = {
      {maxPosF143, -maxPosF143, 8.0f, -8.0f, nan, nan, nan},
      {maxPosF143, -maxPosF143, maxPosF143, -maxPosF143, nan, nan, nan},
      {largeOut, -largeOut, 0, 0, nan, nan, nan}};

  std::vector<int> scales = {0, -10, QUARTER_METADATA_SCALE_BIAS_MAX};
  testFloatWithExpectedResult(input, expected, QuarterMetadata::Format::F143,
                              scales, false);
}

BOOST_AUTO_TEST_CASE(CastFloatOverflowToQuarterF152Nanoo) {
  int testScale = -13;
  float largeIn = std::pow(2, 46.0f);
  float largeOut = largeIn * std::pow(2, -QUARTER_METADATA_SCALE_BIAS_MAX);
  float smallIn = std::pow(2, -18.0f);
  float smallOut = smallIn * std::pow(2, -testScale);

  const auto nan = std::nanf("0");
  auto inf = std::numeric_limits<float>::infinity();
  std::vector<float> input = {smallIn, -smallIn, largeIn, -largeIn, 8.0f,
                              -8.0f,   nan,      -inf,    inf};
  std::vector<std::vector<float>> expected = {
      {0.0f, 0.0f, nan, nan, 8.0f, -8.0f, nan, nan, nan},
      {smallOut, -smallOut, nan, nan, nan, nan, nan, nan, nan},
      {0.0f, 0.0f, largeOut, -largeOut, 0, 0, nan, nan, nan}};

  std::vector<int> scales = {0, testScale, QUARTER_METADATA_SCALE_BIAS_MAX};
  testFloatWithExpectedResult(input, expected, QuarterMetadata::Format::F152,
                              scales, true);
}

// Disable this test for IpuModel as it has no FP_CTL register and so
// behaves as if Nanoo is always set
BOOST_AUTO_TEST_CASE(CastFloatOverflowToQuarterF152Saturate,
                     *boost::unit_test::precondition(enableIfSimOrHw())) {
  float largeIn = std::pow(2, 46.0f);
  float largeOut = largeIn * std::pow(2, -QUARTER_METADATA_SCALE_BIAS_MAX);

  const float maxPosF152 = 57344;
  const auto nan = std::nanf("0");
  auto inf = std::numeric_limits<float>::infinity();

  std::vector<float> input = {largeIn, -largeIn, 8.0f, -8.0f, nan, -inf, inf};
  std::vector<std::vector<float>> expected = {
      {maxPosF152, -maxPosF152, 8.0f, -8.0f, nan, nan, nan},
      {maxPosF152, -maxPosF152, maxPosF152, -maxPosF152, nan, nan, nan},
      {largeOut, -largeOut, 0, 0, nan, nan, nan}};

  std::vector<int> scales = {0, -13, QUARTER_METADATA_SCALE_BIAS_MAX};
  testFloatWithExpectedResult(input, expected, QuarterMetadata::Format::F152,
                              scales, false);
}

// Test all combinations of a given subset of exponent and mantissa bits.
// The metadata scale is used to center the floating point scale.
static void floatSweepTest(const Type &otherType,
                           const QuarterMetadata &metadata,
                           const std::vector<unsigned> &exponentBits,
                           const std::vector<unsigned> &mantissaBits,
                           bool nanoo) {
  unsigned floatExponentBitPos = 23;
  unsigned floatBias = 127;
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
  testCastToQuarter(otherType, metadata, testInput, true, nanoo);
}

// Sweep across a range of binary values
static void halfQuarterSweepTest(const Type &otherType,
                                 const QuarterMetadata &metadata,
                                 unsigned halfBegin, unsigned numElements = 1,
                                 bool nanoo = false) {
  std::vector<unsigned short> halfData(numElements);
  std::iota(halfData.begin(), halfData.end(), halfBegin);
  std::vector<float> testInput(numElements);
  poplar::convertFromDeviceType(HALF, halfData.data(),
                                gccs::ArrayRef(testInput));
  testCastToQuarter(otherType, metadata, testInput, halfData, true, nanoo);
}

// Note - min scale will be negated and then is an invalid positive 6 bit value
// So use -32 + 1 = -31, which then gives +31
const std::vector<int> scales = {0, QUARTER_METADATA_SCALE_BIAS_MIN + 1,
                                 QUARTER_METADATA_SCALE_BIAS_MAX};

// Disable this test for IpuModel as it has no FP_CTL register and so
// behaves as if Nanoo is always set
BOOST_AUTO_TEST_CASE(CastHalfToQuarterF143SweepSaturate,
                     *boost::unit_test::precondition(enableIfSimOrHw())) {
  // Half range is [2^-24, 2^15]
  // Quarter F143 range without scaling is [2^-10, 2^7]
  // The scale range that keeps Quarter representable in Half is [2^-14, 2^8]
  for (auto scale : scales) {
    halfQuarterSweepTest(HALF,
                         QuarterMetadata(QuarterMetadata::Format::F143, scale),
                         0, 65536, false);
  }
}

BOOST_AUTO_TEST_CASE(CastHalfToQuarterF143SweepNanoo) {
  // Half range is [2^-24, 2^15]
  // Quarter F143 range without scaling is [2^-10, 2^7]
  // The scale range that keeps Quarter representable in Half is [2^-14, 2^8]
  for (auto scale : scales) {
    halfQuarterSweepTest(HALF,
                         QuarterMetadata(QuarterMetadata::Format::F143, scale),
                         0, 65536, true);
  }
}

// Disable this test for IpuModel as it has no FP_CTL register and so
// behaves as if Nanoo is always set
BOOST_AUTO_TEST_CASE(CastHalfToQuarterF152SweepSaturate,
                     *boost::unit_test::precondition(enableIfSimOrHw())) {
  // Half range is [2^-24, 2^15]
  // Quarter F152 range without scaling is [2^-10, 2^7]
  // The scale range that keeps Quarter representable in Half is [2^-14, 2^8]
  for (auto scale : scales) {
    halfQuarterSweepTest(HALF,
                         QuarterMetadata(QuarterMetadata::Format::F152, scale),
                         0, 65536, false);
  }
}

BOOST_AUTO_TEST_CASE(CastHalfToQuarterF152SweepNanoo) {
  // Half range is [2^-24, 2^15]
  // Quarter F152 range without scaling is [2^-10, 2^7]
  // The scale range that keeps Quarter representable in Half is [2^-14, 2^8]
  for (auto scale : scales) {
    halfQuarterSweepTest(HALF,
                         QuarterMetadata(QuarterMetadata::Format::F152, scale),
                         0, 65536, true);
  }
}

BOOST_AUTO_TEST_CASE(CastFloatToQuarterF143SweepNanoo) {
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F143,
                                 QUARTER_METADATA_SCALE_BIAS_MIN),
                 {3, 1, 0}, {22, 21, 20, 19, 12, 0}, true);
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F143,
                                 QUARTER_METADATA_SCALE_BIAS_MAX),
                 {3, 2, 0}, {22, 21, 20, 19, 12, 0}, true);
}

BOOST_AUTO_TEST_CASE(CastFloatToQuarterF152SweepNanoo) {
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F152,
                                 QUARTER_METADATA_SCALE_BIAS_MIN),
                 {4, 2, 0}, {22, 21, 20, 19, 12, 0}, true);
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F152,
                                 QUARTER_METADATA_SCALE_BIAS_MAX),
                 {4, 3, 0}, {22, 21, 20, 19, 12, 0}, true);
}

BOOST_AUTO_TEST_CASE(CastFloatToQuarterF143SweepSaturate,
                     *boost::unit_test::precondition(enableIfSimOrHw())) {
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F143,
                                 QUARTER_METADATA_SCALE_BIAS_MIN),
                 {3, 1, 0}, {22, 21, 20, 19, 12, 0}, false);
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F143,
                                 QUARTER_METADATA_SCALE_BIAS_MAX),
                 {3, 2, 0}, {22, 21, 20, 19, 12, 0}, false);
}

BOOST_AUTO_TEST_CASE(CastFloatToQuarterF152SweepSaturate,
                     *boost::unit_test::precondition(enableIfSimOrHw())) {
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F152,
                                 QUARTER_METADATA_SCALE_BIAS_MIN),
                 {4, 2, 0}, {22, 21, 20, 19, 12, 0}, false);
  floatSweepTest(FLOAT,
                 QuarterMetadata(QuarterMetadata::Format::F152,
                                 QUARTER_METADATA_SCALE_BIAS_MAX),
                 {4, 3, 0}, {22, 21, 20, 19, 12, 0}, false);
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

BOOST_AUTO_TEST_CASE(RoundingFloatToQuarterF143) {
  testFloatToQuarterRounding(QuarterMetadata(QuarterMetadata::Format::F143,
                                             QUARTER_METADATA_SCALE_BIAS_MIN));
  testFloatToQuarterRounding(QuarterMetadata(QuarterMetadata::Format::F143,
                                             QUARTER_METADATA_SCALE_BIAS_MAX));
}

BOOST_AUTO_TEST_CASE(RoundingFloatToQuarterF152) {
  testFloatToQuarterRounding(QuarterMetadata(QuarterMetadata::Format::F152,
                                             QUARTER_METADATA_SCALE_BIAS_MIN));
  testFloatToQuarterRounding(QuarterMetadata(QuarterMetadata::Format::F152,
                                             QUARTER_METADATA_SCALE_BIAS_MAX));
}

// Test Random float values that are prone to rounding errors. The values are
// chosen such that a cast from float to half would cause rounding in such a
// way that would affect the 3 most significant mantissa bits if the half is
// then cast to quarter-F143.
BOOST_AUTO_TEST_CASE(CastFloatPrecisionToQuarterF143) {
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F143, 0),
                    {0.968608438969, 0.593695938587, 0.718697071075439453125000,
                     0.148397162556648254394531, 0.484317153692245483398437,
                     0.148416265845, 0.00927624944597,
                     0.359309315681457519531250, 0.8435215950012207031250000});
}

// Test Random float values that are prone to rounding errors. The values are
// chosen such that a cast from float to half would cause rounding in such a
// way that would affect the 2 most significant mantissa bits if the half is
// then cast to quarter-F152.
BOOST_AUTO_TEST_CASE(CastFloatPrecisionToQuarterF152) {
  testCastToQuarter(FLOAT, QuarterMetadata(QuarterMetadata::Format::F152, 0),
                    {0.687358438969, 0.937445938587, 0.687447071075,
                     0.171834662557, 0.468692153692, 0.687415063381,
                     0.010741093196, 0.343684315681, 0.937271595001});
}
