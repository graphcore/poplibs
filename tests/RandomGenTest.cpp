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


using namespace poplar;
using namespace poplar::program;
using namespace poprand;
using namespace popstd;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#define DIM_SIZE  1000


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
static bool validateUniform(T mat[DIM_SIZE][DIM_SIZE], T minVal, T maxVal,
                            double percentError) {
  bool boundsMet = true;
  double mean = 0;
  // compute mean and variance and check bounds
  for(auto r = 0U; r != DIM_SIZE; ++r) {
    for (auto c = 0U; c != DIM_SIZE; ++c) {
      if (mat[r][c] < minVal || mat[r][c] > maxVal)  {
        boundsMet = false;
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

  const double stdDev = std::sqrt(variance / (DIM_SIZE * DIM_SIZE - 1));
  const double dist = (maxVal - minVal) / 2;
  const double actualMean = (maxVal + minVal) / 2;

  const bool meanTest = mean >= (actualMean - dist * percentError / 100)
                     && mean <= (actualMean + dist * percentError / 100);

  const double rStdDev = stdDev /  dist;
  const bool stdDevTest = rStdDev <= ((1 + percentError / 100) / std::sqrt(3.0))
                      && rStdDev >= ((1 - percentError / 100) / std::sqrt(3.0));

  // Add further tests if needed
  return boundsMet && meanTest && stdDevTest;
}

template <typename T>
static bool uniformTest(const T minVal, const T maxVal, double percentError) {
  Graph graph(createIPUModelDevice());
  poprand::addCodelets(graph);

  std::string dType = toTypeStr<T>();
  if (dType.empty()) {
    return false;
  }

  auto out = graph.addTensor(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);

  auto prog = Sequence();
  uniform(graph, out, minVal, maxVal, prog);

  T hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

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
      }
      probEst += static_cast<double>(mat[r][c]);
    }
  }
  probEst /= DIM_SIZE * DIM_SIZE;
  const bool probTest = probEst >= (prob - percentError / 100)
                     && probEst <= (prob + percentError / 100);
  // Add further tests if needed
  return validEvents && probTest;
}


template <typename T>
static bool bernoulliTest(float prob, double percentError) {
  Graph graph(createIPUModelDevice());
  poprand::addCodelets(graph);

  std::string dType = toTypeStr<T>();
  if (dType.empty()) {
    return false;
  }

  auto out = graph.addTensor(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);

  auto prog = Sequence();
  bernoulli(graph, out, prob, prog);

  T hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

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

  // Add further tests if needed
  return meanTest && stdDevTest;
}

template <typename T>
static bool normalTest(float mean, float stdDev, double percentError) {
  Graph graph(createIPUModelDevice());
  poprand::addCodelets(graph);

  std::string dType = toTypeStr<T>();
  if (dType.empty()) {
    return false;
  }

  auto out = graph.addTensor(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);

  auto prog = Sequence();
  normal(graph, out, mean, stdDev, prog);

  T hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  return validateNormal<T>(hOut, mean, stdDev, percentError);
}


template <typename T>
static bool validateNormal(T mat[DIM_SIZE][DIM_SIZE], float actualMean,
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

  // Add further tests if needed
  return boundsMet && meanTest && stdDevTest;
}

template <typename T>
static bool truncatedNormalTest(float mean, float stdDev, float alpha,
                                double percentError) {
  Graph graph(createIPUModelDevice());
  poprand::addCodelets(graph);

  std::string dType = toTypeStr<T>();
  if (dType.empty()) {
    return false;
  }

  auto out = graph.addTensor(dType, {DIM_SIZE, DIM_SIZE}, "out");
  mapTensorLinearly(graph, out);

  auto prog = Sequence();
  truncatedNormal(graph, out, mean, stdDev, alpha, prog);

  T hOut[DIM_SIZE][DIM_SIZE];
  prog.add(Copy(out, hOut));

  Engine eng(graph, prog);
  eng.run();

  return validateNormal<T>(hOut, mean, stdDev, alpha, percentError);
}

BOOST_AUTO_TEST_CASE(RandomGenUniformHalf) {
  bool result = uniformTest<half>(-2.0, 0, 5.0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenUniformFloat) {
  bool result = uniformTest<float>(-1.0, 1.0, 5.0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliHalf) {
  bool result = bernoulliTest<half>(0.75, 5.0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliFloat) {
  bool result = bernoulliTest<float>(0.25, 5.0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliInt) {
  bool result = bernoulliTest<int>(0.5, 5.0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliIntProb0) {
  bool result = bernoulliTest<float>(0, 0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenBernoulliIntProb1) {
  bool result = bernoulliTest<float>(1, 0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenNormalHalf) {
  bool result = normalTest<half>(0.5, 2.5, 5.0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenNormalFloat) {
  bool result = normalTest<float>(-0.5, 2.5, 5.0);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenTruncatedNormalHalf) {
  bool result = truncatedNormalTest<half>(1, 1, 2, 5);
  BOOST_TEST(result == true);
}

BOOST_AUTO_TEST_CASE(RandomGenTruncatedNormalFloat) {
  bool result = truncatedNormalTest<float>(-1, 1, 2, 5);
  BOOST_TEST(result == true);
}
