#define BOOST_TEST_MODULE RandomGenTests
#include <poprand/RandomGen.hpp>
#include <popstd/TileMapping.hpp>
#include <popstd/Util.hpp>
#include <poplar/Engine.hpp>
#include <poplar/HalfFloat.hpp>
#include <poprand/codelets.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poprand;
using namespace popstd;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define ALLONES_SEED static_cast<uint64_t>(~0)
#define DIM_SIZE  200

template <typename T>
std::string toTypeStr() {
  if (std::is_same<T, float>::value) {
    return "float";
  } else if (std::is_same<T, half>::value) {
    return "half";
  } else if (std::is_same<T, int>::value) {
    return "int";
  } else {
    assert(0 && "Invalid type");
  }
  return "";
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

template <typename T>
static bool uniformTest(T hOut[DIM_SIZE][DIM_SIZE],
                        const double minVal, const double maxVal,
                        double percentError, RandomGenMode mode,
                        uint64_t seed = ~0, unsigned numIPUs = 1) {
  DeviceInfo info;
  info.numIPUs = numIPUs;
  Graph graph(createIPUModelDevice(info));
  poprand::addCodelets(graph);

  std::string dType = toTypeStr<T>();
  if (dType.empty()) {
    return false;
  }
  auto out = graph.addTensor(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);
  graph.createHostRead("out", out);
  auto prog = Sequence();
  Random r(mode, seed);
  r.uniform(graph, out, minVal, maxVal, prog);

  Engine eng(graph, prog);
  eng.run();
  eng.readTensor("out", hOut);

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


template <typename T>
static bool bernoulliTest(T hOut[DIM_SIZE][DIM_SIZE], float prob,
                          double percentError, RandomGenMode mode,
                          uint64_t seed = ~0, unsigned numIPUs = 1) {
  DeviceInfo info;
  info.numIPUs = numIPUs;
  Graph graph(createIPUModelDevice(info));
  poprand::addCodelets(graph);

  std::string dType = toTypeStr<T>();
  if (dType.empty()) {
    return false;
  }

  auto out = graph.addTensor(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);
  graph.createHostRead("out", out);

  auto prog = Sequence();
  Random r(mode, seed);
  r.bernoulli(graph, out, prob, prog);

  Engine eng(graph, prog);
  eng.run();
  eng.readTensor("out", hOut);

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

template <typename T>
static bool normalTest(T hOut[DIM_SIZE][DIM_SIZE], float mean, float stdDev,
                       double percentError, RandomGenMode mode,
                       uint64_t seed = ~0, unsigned numIPUs = 1) {
  DeviceInfo info;
  info.numIPUs = numIPUs;
  Graph graph(createIPUModelDevice(info));
  poprand::addCodelets(graph);

  std::string dType = toTypeStr<T>();
  if (dType.empty()) {
    return false;
  }

  auto out = graph.addTensor(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);
  graph.createHostRead("out", out);

  auto prog = Sequence();
  Random r(mode, seed);
  r.normal(graph, out, mean, stdDev, prog);

  Engine eng(graph, prog);
  eng.run();
  eng.readTensor("out", hOut);

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

template <typename T>
static bool truncatedNormalTest(T hOut[DIM_SIZE][DIM_SIZE], float mean,
                                float stdDev, float alpha,
                                double percentError, RandomGenMode mode,
                                uint64_t seed = ~0, unsigned numIPUs = 1) {
  DeviceInfo info;
  info.numIPUs = numIPUs;
  Graph graph(createIPUModelDevice(info));
  poprand::addCodelets(graph);

  std::string dType = toTypeStr<T>();
  if (dType.empty()) {
    return false;
  }

  auto out = graph.addTensor(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);
  graph.createHostRead("out", out);

  auto prog = Sequence();
  Random r(mode, seed);
  r.truncatedNormal(graph, out, mean, stdDev, alpha, prog);

  Engine eng(graph, prog);
  eng.run();
  eng.readTensor("out", hOut);

  return validateTruncNormal<T>(hOut, mean, stdDev, alpha, percentError);
}

BOOST_AUTO_TEST_CASE(RandomGenUniformHalf) {
  using T = half;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = uniformTest<T>(hOut, -2.0, 0, 5.0, SYSTEM_REPEATABLE);
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
  using T = half;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = bernoulliTest<T>(hOut, 0.75, 5.0, SYSTEM_REPEATABLE);
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
  using T = half;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = normalTest<T>(hOut, 0.5, 2.5, 5.0, ALWAYS_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenNormalFloat) {
  using T = float;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = normalTest<T>(hOut, -0.5, 2.5, 5.0, NOT_REPEATABLE);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenTruncatedNormalHalf) {
  using T = half;
  T hOut[DIM_SIZE][DIM_SIZE];
  bool result = truncatedNormalTest<T>(hOut, 1, 1, 2, 5, NOT_REPEATABLE);
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
