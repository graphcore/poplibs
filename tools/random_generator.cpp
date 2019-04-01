// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#include "TestDevice.hpp"
#include <poprand/RandomGen.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>
#include <poprand/codelets.hpp>
#include <poprand/RandomGen.hpp>
//#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include <poplibs_test/Util.hpp>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdint>
#include <algorithm>

#include <boost/program_options.hpp>
#include <popops/ElementWise.hpp>
#include "poputil/VertexTemplates.hpp"
#include <memory>
#include <iostream>
#include <random>

#include <poplibs_test/Pass.hpp>
#include <poplibs_support/Compiler.hpp>

#define FLOAT_REL_TOL  1e-5
#define HALF_REL_TOL   1e-3

const poplar::OptionFlags options {
  {"target.workerStackSizeInBytes", "0x1000"}
};

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using poplibs_test::Pass;

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

template<typename T>
static bool validateUniform(T *mat,
                            unsigned int inSize,
                            double minVal,
                            double maxVal,
                            double percentError) {
  bool boundsMet = true;
  double mean = 0;
  // compute mean and variance and check bounds
  for (auto r = 0U; r != inSize; ++r) {
    if (mat[r] < minVal || mat[r] > maxVal) {
      boundsMet = false;
      std::cerr << "bounds not met at [" << r << "] ";
      std::cerr << mat[r] << "\n";
    }
    mean += mat[r];
  }
  mean /= inSize;

  double variance = 0;
  for (auto r = 0U; r != inSize; ++r) {
    double err = mat[r] - mean;
    variance += err * err;
  }

  double stdDev = std::sqrt(variance / (inSize - 1));

  const double dist = (maxVal - minVal) / 2;
  const double actualMean = (minVal + maxVal) / 2.0;

  const bool meanTest = mean >= (actualMean - dist * percentError / 100)
    && mean <= (actualMean + dist * percentError / 100);

  const double rStdDev = stdDev /  dist;
  bool stdDevTest = rStdDev <= ((1 + percentError / 100) / std::sqrt(3.0))
    && rStdDev >= ((1 - percentError / 100) / std::sqrt(3.0));
  // ignore stddev test for int. It is easy to derive if needed
  if (std::is_same<T, int>::value) {
    stdDevTest = true;
  }

  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean
      << " estimated ";
    std::cerr << mean << "\n";
  }
  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }

  // Add further tests if needed
  return !(boundsMet && meanTest && stdDevTest);
}

template<typename T>
static bool validateBernoulli(T *mat, unsigned int inSize, float prob,
                              double percentError) {
  bool validEvents = true;
  double probEst = 0;
  // compute mean and variance and check bounds
  for (auto r = 0U; r != inSize; ++r) {
    if (mat[r] != 1 && mat[r] != 0)  {
      validEvents = false;
      std::cerr << "invalid event at [" << r << "] ";
      std::cerr << mat[r] << "\n";
    }
    probEst += static_cast<double>(mat[r]);
  }
  probEst /= inSize;
  const bool probTest = probEst >= (prob - percentError / 100)
    && probEst <= (prob + percentError / 100);

  if (!probTest) {
    std::cerr << "probability test failed : actual " <<
      prob << " estimated ";
    std::cerr << probEst << "\n";
  }
  // Add further tests if needed
  return !(validEvents && probTest);
}

template<typename T>
static bool validateNormal(T *mat, unsigned int inSize, float actualMean,
                           float actualStdDev, double percentError) {
  double mean = 0;
  // compute mean and variance and check bounds
  for (auto r = 0U; r != inSize; ++r) {
    mean += mat[r];
  }
  mean /= inSize;

  double variance = 0;
  for (auto r = 0U; r != inSize; ++r) {
    double err = mat[r] - mean;
    variance += err * err;
  }

  const double stdDev = std::sqrt(variance / (inSize - 1));
  const bool meanTest =
    (mean >= (actualMean - actualStdDev * percentError / 100))
    && mean <= (actualMean + actualStdDev * percentError / 100);

  const double rStdDev = stdDev / actualStdDev;
  const bool stdDevTest = rStdDev <= (1 + percentError / 100)
    && rStdDev >= (1 - percentError / 100);

  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean
      << " estimated ";
    std::cerr << mean << "\n";
  }
  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }

  // Add further tests if needed
  return !(meanTest && stdDevTest);
}

template<typename T>
static bool validateTruncNormal(T *mat, unsigned int inSize, float actualMean,
                                float actualStdDev, float alpha,
                                double percentError) {
  bool boundsMet = true;
  double mean = 0;
  // compute mean and variance and check bounds
  for (auto r = 0U; r != inSize; ++r) {
    if ((mat[r] < (actualMean - alpha * actualStdDev)) ||
        (mat[r] > (actualMean + alpha * actualStdDev))) {
      boundsMet = false;
      std::cerr << "bounds not met at [" << r << "] ";
      std::cerr << mat[r] << "\n";
    }
    mean += mat[r];
  }
  mean /= inSize;

  double variance = 0;
  for (auto r = 0U; r != inSize; ++r) {
    double err = mat[r] - mean;
    variance += err * err;
  }

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
  const double phi = 1 / std::sqrt(2 * M_PI) * std::exp(-alpha * alpha / 2);
  const double alphaN = alpha / std::sqrt(2.0);
  const double actualVariance =
    1 - 2 * alpha * phi / (0.5 * (std::erf(alphaN) - std::erf(-alphaN)));

  const double stdDev = std::sqrt(variance / (inSize - 1));
  const double actualTruncStdDev = std::sqrt(actualVariance) * actualStdDev;

  const bool meanTest =
    mean >= (actualMean - actualTruncStdDev * percentError / 100)
    && mean <= (actualMean + actualTruncStdDev * percentError / 100);

  const double rStdDev = stdDev / actualTruncStdDev;
  const bool stdDevTest = rStdDev <= (1 + percentError / 100)
    && rStdDev >= (1 - percentError / 100);
  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean
      << " estimated ";
    std::cerr << mean << "\n";
  }

  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }
  // Add further tests if needed
  return !(boundsMet && meanTest && stdDevTest);
}

template<typename T, bool deviceHalf = false>
static bool validateDropout(T *hOut, unsigned inSize, float dropoutProb,
                            float tolerance, double percentError) {
  // Check number of zeros
  std::size_t numDropout = 0;
  for (std::size_t i = 0; i != inSize; ++i) {
    if (hOut[i] == 0) numDropout++;
  }

  unsigned expectedDrop = inSize - dropoutProb * inSize;
  unsigned allowedError = percentError * inSize / 100;

  return (((numDropout > expectedDrop + allowedError) ||
           (numDropout + allowedError < expectedDrop)));
}

const OptionFlags simDebugOptions {
  {"debug.trace", "false"}
};

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  float       mean;
  float       stdDev;
  double      minVal;
  double      maxVal;
  float       alpha;
  float       prob;
  float       percentError;
  bool        deviceHalf;
  unsigned    inSize;
  std::string randTest;
  DeviceType  deviceType = DeviceType::Cpu; //IpuModel;
  IPUModel    ipuModel;
  unsigned    seed;
  unsigned    seedModifier;
  unsigned    nBins;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type: Cpu | Sim | Hw | IpuModel")
    ("tiles-per-ipu",
     po::value<unsigned>(&ipuModel.tilesPerIPU)->
     default_value(ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
    ("seed", po::value<unsigned>(&seed)->default_value(12352345), "prng seed")
    ("seedModifier", po::value<unsigned>(&seedModifier)->default_value(785439),
     "prng seedModifier")
    ("profile", "Output profiling report")
    ("mean", po::value<float>(&mean)->default_value(0.0),
     "Mean value. Used by Gaussian and Truncated Gaussian")
    ("stdDev", po::value<float>(&stdDev)->default_value(1.0),
     "stdDev value. Used by Gaussian and Truncated Gaussian Tests")
    ("minVal", po::value<double>(&minVal)->default_value(0.0),
     "Min Values used for uniform distribution test")
    ("maxVal", po::value<double>(&maxVal)->default_value(1.0),
     "Max Values used for uniform distribution test")
    ("deviceHalf", po::value<bool>(&deviceHalf)->default_value(false),
     "Half precision input/output")
    ("alpha", po::value<float>(&alpha)->default_value(2.0),
     "Alpha used by the truncated normal test")
    ("prob", po::value<float>(&prob)->default_value(1.0),
     "Probability used by Bernoulli and Dropout tests")
    ("percentError", po::value<float>(&percentError)->default_value(2.0),
     "Tolerance level")
    ("randTest",
     po::value<std::string>(&randTest)->default_value("None"),
     "Random Test: Uniform | UniformInt | Bernoulli| BernoulliInt | Normal "
     "| TruncatedNormal | Dropout")
    ("inSize",
     po::value<unsigned>(&inSize)->default_value(12),
     "Vector size");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  auto dev = [&]() -> TestDevice{
    if (deviceType == DeviceType::IpuModel) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      return createTestDevice(deviceType,
                              ipuModel.numIPUs,
                              ipuModel.tilesPerIPU);
    }
  }
    ();


  const auto &target = dev.getTarget();
  Graph graph(target);
  poprand::addCodelets(graph);

  auto randProg = Sequence();

  uint32_t hSeed[2];
  hSeed[0] =  seed;
  hSeed[1] = ~seed;

  Type dType = (randTest == "UniformInt") ? poplar::INT :
               (deviceHalf ? poplar::HALF : poplar::FLOAT);

  auto tSeed = graph.addVariable(poplar::UNSIGNED_INT, { 2 }, "tSeed");
  graph.setTileMapping(tSeed, 0);
  poprand::setSeed(graph, tSeed, seedModifier, randProg, "setSeed");

  graph.createHostWrite("tSeed", tSeed);

  auto reference = graph.addVariable(dType, { inSize }, "ref");
  mapTensorLinearly(graph, reference);

  //Create stream for cast output
  auto randOutStream = graph.addDeviceToHostFIFO("RandOutputStream",
                                                 dType,
                                                 inSize);

  auto flpRandOut = std::unique_ptr<float[]>(new float[inSize]);
  auto intRandOut = std::unique_ptr<int[]>(new int[inSize]);

  Tensor out;

  if (randTest == "Dropout") {
    auto flpInput = std::unique_ptr<float[]>(new float[inSize]);

    for (int idx = 0; idx != inSize; ++idx) {
      flpInput.get()[idx] = 1.0;
    }

    auto in = graph.addVariable(dType, { inSize }, "in");
    mapTensorLinearly(graph, in);

    graph.createHostWrite("in", in);

    out = poprand::dropout(graph, in, reference, prob, 1.0/prob, randProg);
    graph.createHostRead("out", out);

    Engine engine(graph, randProg, OptionFlags{
      { "target.workerStackSizeInBytes", "0x800" }
    });
    if (deviceHalf) {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.writeTensor("tSeed", hSeed);
        convertAndWriteTensor<float, true>(target, engine, "in",
                                           flpInput.get(), inSize);
        engine.run();
        readAndConvertTensor<float, true>(graph.getTarget(), engine, "out",
                                          flpRandOut.get(), inSize);
      });
    } else {

      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.writeTensor("tSeed", hSeed);
        convertAndWriteTensor<float, false>(target, engine, "in",
                                            flpInput.get(), inSize);
        engine.run();
        readAndConvertTensor<float, false>(graph.getTarget(), engine, "out",
                                           flpRandOut.get(), inSize);
      });
    }
  } else {
    if (randTest == "Normal") {
      out = poprand::normal(graph, reference, dType, mean, stdDev, randProg);
    } else if (randTest == "TruncatedNormal") {
      out = poprand::truncatedNormal(graph, reference, dType, mean, stdDev,
                                     alpha, randProg);
    } else if ((randTest == "Uniform") || (randTest == "UniformInt")) {
      out = poprand::uniform(graph, reference, dType, minVal, maxVal, randProg);
    } else if ((randTest == "Bernoulli")or(randTest == "BernoulliInt")) {
      out = poprand::bernoulli(graph, reference, dType, prob, randProg);
    } else {
      std::cerr << "Test " << randTest << " not supported\n";
    }
    graph.createHostRead("out", out);

    Engine engine(graph, randProg, OptionFlags{
      { "target.workerStackSizeInBytes", "0x800" }
    });

    if (dType == poplar::INT) {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.writeTensor("tSeed", hSeed);
        engine.run();
        readAndConvertTensor<int, false>(graph.getTarget(), engine, "out",
                                         intRandOut.get(),
                                         inSize);
      });
    } else if (deviceHalf) {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.writeTensor("tSeed", hSeed);
        engine.run();
        readAndConvertTensor<float, true>(graph.getTarget(), engine, "out",
                                          flpRandOut.get(), inSize);
      });
    } else {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.writeTensor("tSeed", hSeed);
        engine.run();
        readAndConvertTensor<float, false>(graph.getTarget(), engine, "out",
                                           flpRandOut.get(), inSize);
      });
    }
  }

  if (randTest == "Normal") {
    return validateNormal<float>(flpRandOut.get(), inSize, mean, stdDev,
                                 percentError);
  } else if (randTest == "TruncatedNormal") {
    return validateTruncNormal<float>(flpRandOut.get(), inSize, mean, stdDev,
                                      alpha, percentError);
  } else if ((randTest == "Uniform") || (randTest == "UniformInt")) {
    if (dType == poplar::INT) {
      return validateUniform<int>(intRandOut.get(), inSize, minVal, maxVal,
                                  percentError);
    } else {
      return validateUniform<float>(flpRandOut.get(), inSize, minVal, maxVal,
                                    percentError);
    }
  } else if ((randTest == "Bernoulli")or(randTest == "BernoulliInt")) {
    if (dType == poplar::INT) {
      return validateBernoulli<int>(intRandOut.get(), inSize, prob,
                                    percentError);
    } else {
      return validateBernoulli<float>(flpRandOut.get(), inSize, prob,
                                      percentError);
    }
  } else if (randTest == "Dropout") {
    return validateDropout<float>(flpRandOut.get(), inSize, prob,
                                  deviceHalf ? HALF_REL_TOL : FLOAT_REL_TOL,
                                  percentError);
  }
}
