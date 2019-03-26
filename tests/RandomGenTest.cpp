#define BOOST_TEST_MODULE RandomGenTests
#include "TestDevice.hpp"
#include <poprand/RandomGen.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>
#include <poprand/codelets.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include <poplibs_test/Util.hpp>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <set>

using namespace poplar;
using namespace poplar::program;
using namespace poprand;
using namespace poputil;
using namespace poplibs_test::util;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define ALLONES_SEED static_cast<uint64_t>(~0)
#define DIM_SIZE  200

// Tolerances used in tests
#define FLOAT_REL_TOL  1e-5
#define HALF_REL_TOL   1e-3

const poplar::OptionFlags options {
  {"target.workerStackSizeInBytes", "0x1000"}
};


template <typename T, bool deviceHalf>
static void readAndConvertTensor(
    const Target &target, Engine &eng,
    const std::string &handle,
    T *out, std::size_t N,
    typename std::enable_if<!deviceHalf, int>::type = 0) {
  eng.readTensor(handle, out);
}

template <typename T, bool deviceHalf = false>
static void readAndConvertTensor(
    const Target &target, Engine &eng,
    const std::string &handle,
    T *out, std::size_t N,
    typename std::enable_if<std::is_same<T, float>::value && deviceHalf,
                            int>::type = 0) {
  std::vector<char> buf(target.getTypeSize(HALF) * N);
  eng.readTensor(handle, buf.data());
  copyDeviceHalfToFloat(target, buf.data(), out, N);
}

template <typename T, bool deviceHalf>
static void convertAndWriteTensor(
    const Target &target, Engine &eng,
    const std::string &handle,
    T *in, std::size_t N,
    typename std::enable_if<!deviceHalf, int>::type = 0) {
  eng.writeTensor(handle, in);
}

template <typename T, bool deviceHalf = false>
static void convertAndWriteTensor(
    const Target &target, Engine &eng,
    const std::string &handle,
    T *in, std::size_t N,
    typename std::enable_if<std::is_same<T, float>::value && deviceHalf,
                            int>::type = 0) {
  std::vector<char> buf(target.getTypeSize(HALF) * N);
  copyFloatToDeviceHalf(target, in, buf.data(), N);
  eng.writeTensor(handle, buf.data());
}

template <typename T>
static bool compareMatrices(T a[DIM_SIZE][DIM_SIZE],
                            T b[DIM_SIZE][DIM_SIZE]){
  bool pass = true;
  for (auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      if (a[r][c] != b[r][c]) {
        pass = false;
        std::cerr << "mismatch in matrix elements at [" << r << "][" << c;
        std::cerr << "]" << " : " << a[r][c] << " != " << b[r][c] << "\n";
      }
    }
  }
  return pass;
}


template <typename T>
static bool validateUniform(T mat[DIM_SIZE][DIM_SIZE], double minVal,
                            double maxVal,
                            double percentError) {
  bool boundsMet = true;
  double mean = 0;
  // compute mean and variance and check bounds
  for(auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      if (mat[r][c] < minVal || mat[r][c] > maxVal) {
        boundsMet = false;
        std::cerr << "bounds not met at [" << r << "][" << c << "] ";
        std::cerr << mat[r][c] << "\n";
      }
      mean += mat[r][c];
    }
  }

  mean /= DIM_SIZE * DIM_SIZE;

  double variance = 0;
  for(auto r = 0U; r != DIM_SIZE; ++r){
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      double err = mat[r][c] - mean;
      variance += err * err;
    }
  }

  double stdDev = std::sqrt(variance / (DIM_SIZE * DIM_SIZE - 1));

  const double dist = (maxVal - minVal) / 2;
  const double actualMean = (minVal + maxVal) / 2.0;


  const bool meanTest = mean >= (actualMean - dist * percentError / 100)
                     && mean <= (actualMean + dist * percentError / 100);

  const double rStdDev = stdDev /  dist;
  bool stdDevTest = rStdDev <= ((1 + percentError / 100) / std::sqrt(3.0))
                    && rStdDev >= ((1 - percentError / 100) / std::sqrt(3.0));
  // ignore stddev test for int. It is easy to derive if needed
  if ( std::is_same<T, int>::value) {
    stdDevTest = true;
  }

  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean << " estimated ";
    std::cerr << mean << "\n";
  }
  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }

  // Add further tests if needed
  return boundsMet && meanTest && stdDevTest;
}

template <typename T, bool deviceHalf = false>
static bool uniformTest(T hOut[DIM_SIZE][DIM_SIZE],
                        const double minVal, const double maxVal,
                        double percentError, RandomGenMode mode,
                        uint64_t seed = ~0, unsigned numIPUs = 1) {
  IPUModel ipuModel;
  ipuModel.numIPUs = numIPUs;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  poprand::addCodelets(graph);

  auto dType = deviceHalf ? poplar::HALF : equivalent_device_type<T>().value;

  auto out = graph.addVariable(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);
  graph.createHostRead("out", out);
  auto prog = Sequence();
  Random r(mode, seed);
  r.uniform(graph, out, minVal, maxVal, prog);

  Engine eng(graph, prog);
  eng.load(device);
  eng.run();
  readAndConvertTensor<T, deviceHalf>(graph.getTarget(), eng, "out",
                                      &hOut[0][0],
                                      DIM_SIZE * DIM_SIZE);

  return validateUniform<T>(hOut, minVal, maxVal, percentError);
}

template <typename T>
static bool validateBernoulli(T mat[DIM_SIZE][DIM_SIZE], float prob,
                             double percentError) {
  bool validEvents = true;
  double probEst = 0;
  // compute mean and variance and check bounds
  for(auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      if (mat[r][c] != 1 && mat[r][c] != 0)  {
        validEvents = false;
        std::cerr << "invalid event at [" << r << "][" << c << "] ";
        std::cerr << mat[r][c] << "\n";

      }
      probEst += static_cast<double>(mat[r][c]);
    }
  }
  probEst /= DIM_SIZE * DIM_SIZE;
  const bool probTest = probEst >= (prob - percentError / 100)
                     && probEst <= (prob + percentError / 100);

  if (!probTest) {
    std::cerr << "probability test failed : actual " << prob << " estimated ";
    std::cerr << probEst << "\n";
  }
  // Add further tests if needed
  return validEvents && probTest;
}

template <typename T, bool deviceHalf = false>
static bool bernoulliTest(T hOut[DIM_SIZE][DIM_SIZE], float prob,
                          double percentError, RandomGenMode mode,
                          uint64_t seed = ~0, unsigned numIPUs = 1) {
  IPUModel ipuModel;
  ipuModel.numIPUs = numIPUs;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  poprand::addCodelets(graph);

  auto dType = deviceHalf ? poplar::HALF : equivalent_device_type<T>().value;

  auto out = graph.addVariable(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);
  graph.createHostRead("out", out);

  auto prog = Sequence();
  Random r(mode, seed);
  r.bernoulli(graph, out, prob, prog);

  Engine eng(graph, prog);
  eng.load(device);
  eng.run();
  readAndConvertTensor<T, deviceHalf>(graph.getTarget(), eng, "out",
                                      &hOut[0][0],
                                      DIM_SIZE * DIM_SIZE);
  return validateBernoulli<T>(hOut, prob, percentError);
}




template <typename T>
static bool validateNormal(T mat[DIM_SIZE][DIM_SIZE], float actualMean,
                           float actualStdDev, double percentError) {
  double mean = 0;
  // compute mean and variance and check bounds
  for(auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      mean += mat[r][c];
    }
  }
  mean /= DIM_SIZE * DIM_SIZE;

  double variance = 0;
  for(auto r = 0U; r != DIM_SIZE; ++r){
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      double err = mat[r][c] - mean;
      variance += err * err ;
    }
  }

  const double stdDev = std::sqrt(variance / (DIM_SIZE * DIM_SIZE - 1));
  const bool meanTest = mean >= (actualMean - actualStdDev * percentError / 100)
                    && mean <= (actualMean + actualStdDev * percentError / 100);

  const double rStdDev = stdDev / actualStdDev;
  const bool stdDevTest = rStdDev <= (1 + percentError / 100)
                       && rStdDev >= (1 - percentError / 100);

  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean << " estimated ";
    std::cerr << mean << "\n";
  }
  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }

  // Add further tests if needed
  return meanTest && stdDevTest;
}

template <typename T, bool deviceHalf = false>
static bool normalTest(T hOut[DIM_SIZE][DIM_SIZE], float mean, float stdDev,
                       double percentError, RandomGenMode mode,
                       uint64_t seed = ~0, unsigned numIPUs = 1) {
  IPUModel ipuModel;
  ipuModel.numIPUs = numIPUs;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  poprand::addCodelets(graph);

  auto dType = deviceHalf ? poplar::HALF : equivalent_device_type<T>().value;

  auto out = graph.addVariable(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);
  graph.createHostRead("out", out);

  auto prog = Sequence();
  Random r(mode, seed);
  r.normal(graph, out, mean, stdDev, prog);

  Engine eng(graph, prog);
  eng.load(device);
  eng.run();
  readAndConvertTensor<T, deviceHalf>(graph.getTarget(), eng, "out",
                                      &hOut[0][0],
                                      DIM_SIZE * DIM_SIZE);

  return validateNormal<T>(hOut, mean, stdDev, percentError);
}

template <typename T>
static bool validateTruncNormal(T mat[DIM_SIZE][DIM_SIZE], float actualMean,
                                float actualStdDev, float alpha,
                                double percentError) {
  bool boundsMet = true;
  double mean = 0;
  // compute mean and variance and check bounds
  for(auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      if ((mat[r][c] < (actualMean - alpha * actualStdDev)) ||
          (mat[r][c] > (actualMean + alpha * actualStdDev))) {
        boundsMet = false;
        std::cerr << "bounds not met at [" << r << "][" << c << "] ";
        std::cerr << mat[r][c] << "\n";
      }
      mean += mat[r][c];
    }
  }
  mean /= DIM_SIZE * DIM_SIZE;

  double variance = 0;
  for(auto r = 0U; r != DIM_SIZE; ++r){
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      double err = mat[r][c] - mean;
      variance += err * err ;
    }
  }

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif
  const double phi = 1/std::sqrt(2 * M_PI) * std::exp(-alpha * alpha / 2);
  const double alphaN = alpha / std::sqrt(2.0);
  const double actualVariance =
    1 - 2 * alpha * phi / (0.5 * (std::erf(alphaN) - std::erf(-alphaN)));

  const double stdDev = std::sqrt(variance / (DIM_SIZE * DIM_SIZE - 1));
  const double actualTruncStdDev = std::sqrt(actualVariance) * actualStdDev;

  const bool meanTest =
    mean >= (actualMean - actualTruncStdDev * percentError / 100)
    && mean <= (actualMean + actualTruncStdDev * percentError / 100);

  const double rStdDev = stdDev / actualTruncStdDev;
  const bool stdDevTest = rStdDev <= (1 + percentError / 100)
                       && rStdDev >= (1 - percentError / 100);
  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean << " estimated ";
    std::cerr << mean << "\n";
  }

  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }
  // Add further tests if needed
  return boundsMet && meanTest && stdDevTest;
}

template <typename T, bool deviceHalf = false>
static bool truncatedNormalTest(T hOut[DIM_SIZE][DIM_SIZE], float mean,
                                float stdDev, float alpha,
                                double percentError, RandomGenMode mode,
                                uint64_t seed = ~0, unsigned numIPUs = 1) {
  IPUModel ipuModel;
  ipuModel.numIPUs = numIPUs;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  poprand::addCodelets(graph);

  auto dType = deviceHalf ? poplar::HALF : equivalent_device_type<T>().value;

  auto out = graph.addVariable(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);
  graph.createHostRead("out", out);

  auto prog = Sequence();
  Random r(mode, seed);
  r.truncatedNormal(graph, out, mean, stdDev, alpha, prog);

  Engine eng(graph, prog);
  eng.load(device);
  eng.run();
  readAndConvertTensor<T, deviceHalf>(graph.getTarget(), eng, "out",
                                      &hOut[0][0],
                                      DIM_SIZE * DIM_SIZE);

  return validateTruncNormal<T>(hOut, mean, stdDev, alpha, percentError);
}

BOOST_AUTO_TEST_CASE(RandomGenUniformHalf) {
  float hOut[DIM_SIZE][DIM_SIZE];
  bool result = uniformTest<float, true>(hOut, -2.0, 0, 5.0, SYSTEM_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenUniformInt) {
  using T = int;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = uniformTest<T>(hOut, -20, 2, 5.0, SYSTEM_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenUniformIntMaxRange) {
  using T = int;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = uniformTest<T>(hOut, -2147483648, 2147483647, 5.0,
                               SYSTEM_REPEATABLE);
  BOOST_TEST(result == true);
}


BOOST_AUTO_TEST_CASE(RandomGenUniformFloat) {
  using T = float;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = uniformTest<T>(hOut, -1.0, 1.0, 5.0, NOT_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliHalf) {
  float hOut[DIM_SIZE][DIM_SIZE];
  bool result = bernoulliTest<float, true>(hOut, 0.75, 5.0, SYSTEM_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliFloat) {
  using T = float;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = bernoulliTest<T>(hOut, 0.25, 5.0, SYSTEM_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliInt) {
  using T = int;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = bernoulliTest<T>(hOut, 0.5, 5.0, NOT_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliFloatProb0) {
  using T = float;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = bernoulliTest<T>(hOut, 0, 0, NOT_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliIntProb1) {
  using T = int;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = bernoulliTest<T>(hOut, 1, 0, NOT_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenNormalHalf) {
  float hOut[DIM_SIZE][DIM_SIZE];
  bool result = normalTest<float, true>(hOut, 0.5, 2.5, 5.0, ALWAYS_REPEATABLE);
  BOOST_TEST(result == true);
}
BOOST_AUTO_TEST_CASE(RandomGenNormalFloat) {
  using T = float;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = normalTest<T>(hOut, -0.5, 2.5, 5.0, NOT_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenTruncatedNormalHalf) {
  float hOut[DIM_SIZE][DIM_SIZE];
  bool result = truncatedNormalTest<float, true>(hOut, 1, 1, 2, 5,
                                                 NOT_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenTruncatedNormalFloatRepeat) {
  using T = float;
  T hOut1[DIM_SIZE][DIM_SIZE];
  bool result1 =
      truncatedNormalTest<T>(hOut1, -1, 1, 2, 5, ALWAYS_REPEATABLE, 0x1234, 1);
  T hOut2[DIM_SIZE][DIM_SIZE];
  bool result2 =
      truncatedNormalTest<T>(hOut2, -1, 1, 2, 5, ALWAYS_REPEATABLE, 0x1234, 2);
  bool compareResult = compareMatrices<T>(hOut1, hOut2);
  const auto result = result1 && result2 && compareResult;
  BOOST_TEST(result == true);
}

template <typename T, bool deviceHalf = false>
static bool dropOutTest(unsigned numIPUs,
                        unsigned tilesPerIPU,
                        double dropoutProb,
                        uint32_t seedModifier,
                        DeviceType deviceType) {
  T hOut[DIM_SIZE][DIM_SIZE];
  T hIn[DIM_SIZE][DIM_SIZE];
  uint32_t hSeed[2] = {0xDEADBEEF, 0xBEEFDEAD};

  for (std::size_t i = 0; i != DIM_SIZE; ++i) {
    for (std::size_t j = 0; j != DIM_SIZE; ++j) {
      hIn[i][j] = 1.0;
    }
  }

  auto device = createTestDevice(deviceType, numIPUs, tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  poprand::addCodelets(graph);

  auto dType = deviceHalf ? poplar::HALF : equivalent_device_type<T>().value;
  auto in = graph.addVariable(dType, {DIM_SIZE, DIM_SIZE}, "in");
  auto reference = graph.addVariable(dType, {DIM_SIZE, DIM_SIZE}, "ref");

  mapTensorLinearly(graph, in);
  mapTensorLinearly(graph, reference);

  auto seed = graph.addVariable(poplar::UNSIGNED_INT, {2}, "seed");
  graph.setTileMapping(seed, 0);
  auto prog = Sequence();

  auto out = poprand::dropout(graph, in.transpose(), seed,
                              reference.transpose(),
                              dropoutProb, seedModifier, 1 / dropoutProb, prog);

  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);
  graph.createHostWrite("seed", seed);

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    convertAndWriteTensor<T, deviceHalf>(graph.getTarget(), eng, "in",
                                         &hIn[0][0],
                                         DIM_SIZE * DIM_SIZE);
    eng.writeTensor("seed", hSeed);
    eng.run();
    readAndConvertTensor<T, deviceHalf>(graph.getTarget(), eng, "out",
                                        &hOut[0][0],
                                        DIM_SIZE * DIM_SIZE);
  });

  // Check number of zeros
  double tolerance = deviceHalf ? HALF_REL_TOL : FLOAT_REL_TOL;
  std::size_t numDropout = 0;
  for (std::size_t i = 0; i != DIM_SIZE; ++i) {
    for (std::size_t j = 0; j != DIM_SIZE; ++j) {
      if (hOut[i][j] == 0)
        numDropout++;
      else
        BOOST_CHECK_CLOSE(hOut[i][j], 1 / dropoutProb, tolerance);
    }
  }

  unsigned expectedDrop =
      DIM_SIZE * DIM_SIZE - dropoutProb * DIM_SIZE * DIM_SIZE;
  unsigned allowedError = 0.02 * DIM_SIZE * DIM_SIZE;

  if ((numDropout > expectedDrop + allowedError) ||
      (numDropout + allowedError < expectedDrop)) {
    std::cerr << "\n measured dropout probability doesn't match expected range";
    BOOST_TEST(false);
  }
}

BOOST_AUTO_TEST_CASE(DropoutFloatTest) {
  dropOutTest<float>(1, 16, 0.3, 0x55555555, DeviceType::IpuModel);
}

BOOST_AUTO_TEST_CASE(DropoutHalfTest) {
  dropOutTest<float, true>(1, 16, 0.25, 0x55555555, DeviceType::IpuModel);
}


// This test is just to use the assembler vertex

BOOST_AUTO_TEST_CASE(SetSeedTest) {
  auto device =
      createTestDevice(TEST_TARGET, 1, 16);
  const auto &target = device.getTarget();
  Graph graph(target);
  poprand::addCodelets(graph);
  uint32_t hSeed[2] = {0xDEADBEEF, 0xBEEFDEAD};

  auto seed = graph.addVariable(poplar::UNSIGNED_INT, {2}, "seed");
  graph.setTileMapping(seed, 0);
  auto prog = Sequence();

  poprand::setSeed(graph, seed, 0x12345, prog, "setSeed");
  auto seedsRead = poprand::getHwSeeds(graph, prog);
  std::vector<uint32_t> hostSeedsRead(seedsRead.numElements());

  graph.createHostWrite("seed", seed);
  graph.createHostRead("seedsRead", seedsRead);

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("seed", hSeed);
    eng.run();
    eng.readTensor("seedsRead", hostSeedsRead.data());
  });

  std::set<std::vector<unsigned>> unique_seeds;
  assert(seedsRead.rank() == 3);
  assert(seedsRead.numElements() == hostSeedsRead.size());
  for (unsigned t = 0; t != seedsRead.dim(0); ++t) {
    for (unsigned w = 0; w != seedsRead.dim(1); ++w) {
      std::vector<unsigned> workerSeed(seedsRead.dim(2));
      for (unsigned s = 0; s != seedsRead.dim(2); ++s) {
        auto wSeed = hostSeedsRead[t * seedsRead.dim(1) * seedsRead.dim(2)
                                   + w * seedsRead.dim(2)
                                   + seedsRead.dim(2) - 1 - s];
        workerSeed[s] = wSeed;
      }
      unique_seeds.insert(workerSeed);
    }
  }
  if (TEST_TARGET == DeviceType::Sim || TEST_TARGET == DeviceType::Hw) {
    BOOST_CHECK_EQUAL(unique_seeds.size(), seedsRead.dim(0) * seedsRead.dim(1));
  }
}
