// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "TestDevice.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <poplar/CSRFunctions.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/RandomSeed.hpp>
#include <poplar/Target.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_test/exceptions.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include "poputil/VertexTemplates.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <memory>
#include <popops/ElementWise.hpp>
#include <random>

#include <poplibs_support/Compiler.hpp>
#include <poplibs_test/Pass.hpp>

#define FLOAT_REL_TOL 1e-5
#define HALF_REL_TOL 1e-3

const poplar::OptionFlags options;

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using poplibs_test::Pass;

template <typename T, bool deviceHalf>
static void
readAndConvertTensor(const Target &target, Engine &eng,
                     const std::string &handle, std::vector<T> &out,
                     std::size_t N,
                     typename std::enable_if<!deviceHalf, int>::type = 0) {
  eng.readTensor(handle, out.data(), &out[N]);
}

template <typename T, bool deviceHalf = false>
static void readAndConvertTensor(
    const Target &target, Engine &eng, const std::string &handle,
    std::vector<T> &out, std::size_t N,
    typename std::enable_if<std::is_same<T, float>::value && deviceHalf,
                            int>::type = 0) {
  std::vector<char> buf(target.getTypeSize(HALF) * N);
  eng.readTensor(handle, buf.data(), buf.data() + buf.size());
  copyDeviceHalfToFloat(target, buf.data(), out.data(), N);
}

template <typename T, bool deviceHalf>
static void
convertAndWriteTensor(const Target &target, Engine &eng,
                      const std::string &handle, const std::vector<T> &in,
                      std::size_t N,
                      typename std::enable_if<!deviceHalf, int>::type = 0) {
  eng.writeTensor(handle, in.data());
}

template <typename T, bool deviceHalf = false>
static void convertAndWriteTensor(
    const Target &target, Engine &eng, const std::string &handle,
    const std::vector<T> &in, std::size_t N,
    typename std::enable_if<std::is_same<T, float>::value && deviceHalf,
                            int>::type = 0) {
  std::vector<char> buf(target.getTypeSize(HALF) * N);
  copyFloatToDeviceHalf(target, in.data(), buf.data(), N);
  eng.writeTensor(handle, buf.data());
}

template <typename T>
static bool validateUniform(const std::vector<T> &mat, unsigned int inSize,
                            double minVal, double maxVal, double percentError) {
  bool boundsMet = true;
  double mean = 0;
  double maxSeen = minVal, minSeen = maxVal;
  // compute mean and variance and check bounds
  for (auto r = 0U; r != inSize; ++r) {
    if (mat[r] < minVal || mat[r] > maxVal) {
      boundsMet = false;
      std::cerr << "bounds not met at [" << r << "] ";
      std::cerr << mat[r] << "\n";
    }
    if (maxSeen < mat[r])
      maxSeen = mat[r];
    if (minSeen > mat[r])
      minSeen = mat[r];
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

  const bool meanTest = mean >= (actualMean - dist * percentError / 100) &&
                        mean <= (actualMean + dist * percentError / 100);

  const double rStdDev = stdDev / dist;
  bool stdDevTest = rStdDev <= ((1 + percentError / 100) / std::sqrt(3.0)) &&
                    rStdDev >= ((1 - percentError / 100) / std::sqrt(3.0));
  // ignore stddev test for int. It is easy to derive if needed
  if (std::is_same<T, int>::value) {
    stdDevTest = true;
  }

  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean << " estimated ";
    std::cerr << mean << "\n";
  }
  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }
  if (false) {
    std::cerr << "limit [" << minVal << ":" << maxVal << "]; range [" << minSeen
              << ":" << maxSeen << "]; mean " << mean << "; sd  " << stdDev
              << "\n";
  }
  // Add further tests if needed
  auto failed = !(boundsMet && meanTest && stdDevTest);
  if (failed) {
    std::cerr << "Validation of Uniform failed\n";
  }
  return failed;
}

template <typename T>
static bool validateBernoulli(const std::vector<T> &mat, unsigned int inSize,
                              float prob, double percentError) {
  bool validEvents = true;
  double probEst = 0;
  // compute mean and variance and check bounds
  for (auto r = 0U; r != inSize; ++r) {
    if (mat[r] != 1 && mat[r] != 0) {
      validEvents = false;
      std::cerr << "invalid event at [" << r << "] ";
      std::cerr << mat[r] << "\n";
    }
    probEst += static_cast<double>(mat[r]);
  }
  probEst /= inSize;
  const bool probTest = probEst >= (prob - percentError / 100) &&
                        probEst <= (prob + percentError / 100);

  if (!probTest) {
    std::cerr << "probability test failed : actual " << prob << " estimated ";
    std::cerr << probEst << "\n";
  }
  // Add further tests if needed
  auto failed = !(validEvents && probTest);
  if (failed) {
    std::cerr << "Validation of Bernoulli failed\n";
  }
  return failed;
}

template <typename T>
static bool validateNormal(const std::vector<T> &mat, unsigned int inSize,
                           float actualMean, float actualStdDev,
                           double percentError) {
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
      (mean >= (actualMean - actualStdDev * percentError / 100)) &&
      mean <= (actualMean + actualStdDev * percentError / 100);

  const double rStdDev = stdDev / actualStdDev;
  const bool stdDevTest = rStdDev <= (1 + percentError / 100) &&
                          rStdDev >= (1 - percentError / 100);

  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean << " estimated ";
    std::cerr << mean << "\n";
  }
  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }

  // Add further tests if needed
  bool failed = !(meanTest && stdDevTest);
  if (failed) {
    std::cerr << "Validation of Normal failed\n";
  }
  return failed;
}

template <typename T>
static bool validateTruncNormal(const std::vector<T> &mat, unsigned int inSize,
                                float actualMean, float actualStdDev,
                                float alpha, double percentError) {
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
      mean >= (actualMean - actualTruncStdDev * percentError / 100) &&
      mean <= (actualMean + actualTruncStdDev * percentError / 100);

  const double rStdDev = stdDev / actualTruncStdDev;
  const bool stdDevTest = rStdDev <= (1 + percentError / 100) &&
                          rStdDev >= (1 - percentError / 100);
  if (!meanTest) {
    std::cerr << "mean test failed : actual " << actualMean << " estimated ";
    std::cerr << mean << "\n";
  }

  if (!stdDevTest) {
    std::cerr << "std dev test failed : ratio " << rStdDev << "\n";
  }
  // Add further tests if needed
  bool failed = !(boundsMet && meanTest && stdDevTest);
  if (failed) {
    std::cerr << "Validation of Truncated Normal failed\n";
  }
  return failed;
}

template <typename T, bool deviceHalf = false>
static bool validateDropout(const std::vector<T> &hOut, unsigned inSize,
                            float dropoutProb, float tolerance,
                            double percentError) {
  // Check number of zeros
  std::size_t numDropout = 0;
  for (std::size_t i = 0; i != inSize; ++i) {
    if (hOut[i] == 0)
      numDropout++;
  }

  unsigned expectedDrop = inSize - dropoutProb * inSize;
  unsigned allowedError = percentError * inSize / 100;

  bool failed = (((numDropout > expectedDrop + allowedError) ||
                  (numDropout + allowedError < expectedDrop)));
  if (failed) {
    std::cerr << "Validation of Dropout failed\n";
  }
  return failed;
}

enum class TestType {
  SetSeeds,
  SetHwSeeds,
  Bernoulli,
  BernoulliInt,
  Uniform,
  UniformInt,
  Normal,
  TruncatedNormal,
  Dropout
};

static TestType getTestType(const std::string &testType) {
  if (testType == "SetSeeds") {
    return TestType::SetSeeds;
  } else if (testType == "SetHwSeeds") {
    return TestType::SetHwSeeds;
  } else if (testType == "Bernoulli") {
    return TestType::Bernoulli;
  } else if (testType == "BernoulliInt") {
    return TestType::BernoulliInt;
  } else if (testType == "Uniform") {
    return TestType::Uniform;
  } else if (testType == "UniformInt") {
    return TestType::UniformInt;
  } else if (testType == "Normal") {
    return TestType::Normal;
  } else if (testType == "TruncatedNormal") {
    return TestType::TruncatedNormal;
  } else if (testType == "Dropout") {
    return TestType::Dropout;
  } else {
    throw poplibs_test::poplibs_test_error("Invalid random test");
  }
}

const OptionFlags simDebugOptions{{"debug.trace", "false"}};

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  float mean;
  float stdDev;
  double minVal;
  double maxVal;
  float alpha;
  float prob;
  float percentError;
  bool deviceHalf;
  bool fpChecking;
  unsigned inSize;
  std::string randTest;
  DeviceType deviceType = DeviceType::Cpu; // IpuModel;
  boost::optional<unsigned> tilesPerIPU;
  unsigned seed;
  unsigned seedModifier;
  unsigned numLoops;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type: Cpu | Sim | Sim2 | Hw | IpuModel | IpuModel2")
    ("tiles-per-ipu",
     po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("seed", po::value<unsigned>(&seed)->default_value(12352345), "prng seed")
    ("seed-modifier", po::value<unsigned>(&seedModifier)->default_value(785439),
     "prng seedModifier")
    ("profile", "Output profiling report")
    ("mean", po::value<float>(&mean)->default_value(0.0),
     "Mean value. Used by Gaussian and Truncated Gaussian")
    ("std-dev", po::value<float>(&stdDev)->default_value(1.0),
     "stdDev value. Used by Gaussian and Truncated Gaussian Tests")
    ("min-val", po::value<double>(&minVal)->default_value(0.0),
     "Min Values used for uniform distribution test")
    ("max-val", po::value<double>(&maxVal)->default_value(1.0),
     "Max Values used for uniform distribution test")
    ("half-data-type", po::value<bool>(&deviceHalf)->default_value(false),
     "Half precision input/output, else float (ignored for UniformInt test)")
    ("alpha", po::value<float>(&alpha)->default_value(2.0),
     "Alpha used by the truncated normal test")
    ("prob", po::value<float>(&prob)->default_value(1.0),
     "Probability used by Bernoulli and Dropout tests")
    ("percent-error", po::value<float>(&percentError)->default_value(2.0),
     "Tolerance level")
    ("repeat", po::value<unsigned>(&numLoops)->default_value(1u),
     "Number of times to repeat test")
    ("rand-test",
     po::value<std::string>(&randTest)->default_value("None"),
     "Random Test: Uniform | UniformInt | Bernoulli| BernoulliInt | Normal "
     "| TruncatedNormal | Dropout | SetSeeds | SetHwSeeds")
    ("in-size",
     po::value<unsigned>(&inSize)->default_value(12),
     "Vector size")
    ("fp-checking",
     po::value(&fpChecking)->default_value(true),
     "Enable hardware floating-point checks"
  );
  // clang-format on

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

  auto testType = getTestType(randTest);

  auto dev = [&]() -> TestDevice {
    if (isIpuModel(deviceType)) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      IPUModel ipuModel(deviceTypeToIPUName(deviceType));
      if (tilesPerIPU)
        ipuModel.tilesPerIPU = *tilesPerIPU;
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      if (tilesPerIPU)
        return createTestDevice(deviceType, 1, *tilesPerIPU);
      else
        return createTestDeviceFullSize(deviceType);
    }
  }();

  const auto &target = dev.getTarget();
  Graph graph(target);
  poprand::addCodelets(graph);

  auto randProg = Sequence();
  auto checkSeedSeq = Sequence();

  const bool defaultSeed = vm["seed"].defaulted();

  uint32_t hSeed[2];
  hSeed[0] = seed;
  hSeed[1] = ~seed;

  Type dType = (randTest == "UniformInt")
                   ? poplar::INT
                   : (deviceHalf ? poplar::HALF : poplar::FLOAT);

  auto tSeed = graph.addVariable(poplar::UNSIGNED_INT, {2}, "tSeed");
  graph.setTileMapping(tSeed, 0);

  // We pad the dropout input with trailing NaNs. Add extra padding if there is
  // an odd half, so that the host copy is a multiple of 4 bytes
  unsigned padSize = 0;
  if (deviceHalf)
    padSize += inSize % 2;
  auto paddedInSize = inSize + padSize;
  auto paddedReference = graph.addVariable(dType, {paddedInSize}, "ref");
  auto reference = paddedReference.slice({0, inSize});
  mapTensorLinearly(graph, paddedReference);

  poprand::setSeed(graph, tSeed, seedModifier, randProg, "setSeed");

  // handle setseed, SetHwSeed tests as they are different from the others
  if (testType == TestType::SetSeeds || testType == TestType::SetHwSeeds) {
    auto seedsWrite = graph.addVariable(
        poplar::UNSIGNED_INT,
        {target.getNumTiles(), target.getNumWorkerContexts(), 4}, "hwSeeds");
    mapTensorLinearly(graph, seedsWrite);
    std::vector<uint32_t> hSeedsWrite(seedsWrite.numElements());
    for (unsigned i = 0; i < hSeedsWrite.size(); i++) {
      hSeedsWrite[i] = 200 * i + 1000;
    }
    if (testType == TestType::SetHwSeeds) {
      poplar::setHwSeeds(graph, seedsWrite, randProg, "setHwSeeds");
    }

    auto seedsRead = poplar::getHwSeeds(graph, randProg);
    std::vector<uint32_t> hostSeedsRead(seedsRead.numElements());
    graph.createHostWrite("seed", tSeed);
    graph.createHostWrite("seedsWrite", seedsWrite);
    graph.createHostRead("seedsRead", seedsRead);
    Engine eng(graph, randProg, options);
    dev.bind([&](const Device &d) {
      eng.load(d);
      eng.writeTensor("seed", hSeed);
      eng.writeTensor("seedsWrite", &hSeedsWrite[0]);
      eng.run();
      eng.readTensor("seedsRead", hostSeedsRead.data(),
                     hostSeedsRead.data() + hostSeedsRead.size());
    });
    std::set<std::vector<unsigned>> unique_seeds;
    assert(seedsRead.rank() == 3);
    assert(seedsRead.numElements() == hostSeedsRead.size());
    for (unsigned t = 0; t != seedsRead.dim(0); ++t) {
      for (unsigned w = 0; w != seedsRead.dim(1); ++w) {
        std::vector<unsigned> workerSeed(seedsRead.dim(2));
        for (unsigned s = 0; s != seedsRead.dim(2); ++s) {
          auto wSeed =
              hostSeedsRead[t * seedsRead.dim(1) * seedsRead.dim(2) +
                            w * seedsRead.dim(2) + seedsRead.dim(2) - 1 - s];
          workerSeed[s] = wSeed;
        }
        unique_seeds.insert(workerSeed);
      }
    }
    if (isSimulator(deviceType) || deviceType == DeviceType::Hw) {
      if (testType == TestType::SetHwSeeds) {
        return !(std::equal(hostSeedsRead.begin(), hostSeedsRead.end(),
                            hSeedsWrite.begin()));
      } else {
        return !(unique_seeds.size() == seedsRead.dim(0) * seedsRead.dim(1));
      }
    }
    return 1;
  }

  if (fpChecking) {
    if (deviceType == DeviceType::Cpu)
      std::cerr << "Warning: fp-checking not supported on CPU target\n";
    poplar::FloatingPointBehaviour checkAll(true, true, true, false, true);
    poplar::setFloatingPointBehaviour(graph, randProg, checkAll,
                                      "/initialValue");
  }
  graph.createHostWrite("tSeed", tSeed);

  auto *seedToUseInTest = defaultSeed ? nullptr : &tSeed;

  std::vector<float> flpRandOut(paddedInSize);
  std::vector<int> intRandOut(paddedInSize);

  // validate returns false on success
  auto validate = [&]() {
    if (randTest == "Normal") {
      return validateNormal<float>(flpRandOut, inSize, mean, stdDev,
                                   percentError);
    } else if (randTest == "TruncatedNormal") {
      return validateTruncNormal<float>(flpRandOut, inSize, mean, stdDev, alpha,
                                        percentError);
    } else if ((randTest == "Uniform") || (randTest == "UniformInt")) {
      if (dType == poplar::INT) {
        return validateUniform(intRandOut, inSize, minVal, maxVal,
                               percentError);
      } else {
        return validateUniform(flpRandOut, inSize, minVal, maxVal,
                               percentError);
      }
    } else if ((randTest == "Bernoulli") or (randTest == "BernoulliInt")) {
      if (dType == poplar::INT) {
        return validateBernoulli<int>(intRandOut, inSize, prob, percentError);
      } else {
        return validateBernoulli<float>(flpRandOut, inSize, prob, percentError);
      }
    } else if (randTest == "Dropout") {
      return validateDropout<float>(flpRandOut, inSize, prob,
                                    deviceHalf ? HALF_REL_TOL : FLOAT_REL_TOL,
                                    percentError);
    }
    return false;
  };
  Tensor out;
  if (testType == TestType::Dropout) {
    std::vector<float> flpInput(paddedInSize);

    for (unsigned idx = 0; idx != inSize; ++idx) {
      flpInput[idx] = 1.0;
    }
    for (unsigned idx = inSize; idx != paddedInSize; ++idx) {
      flpInput[idx] = std::numeric_limits<float>::signaling_NaN();
    }

    auto paddedIn = graph.addVariable(dType, {paddedInSize}, "paddedIn");
    auto in = paddedIn.slice({0, inSize});
    mapTensorLinearly(graph, paddedIn);

    graph.createHostWrite("paddedIn", paddedIn, true);

    auto seedsReadBefore = poplar::getHwSeeds(graph, randProg);

    out = poprand::dropout(graph, seedToUseInTest, seedModifier, in, reference,
                           prob, 1.0 / prob, randProg);
    auto seedsReadAfter = poplar::getHwSeeds(graph, randProg);

    graph.createHostRead("out", out);
    graph.createHostRead("seedsReadBefore", seedsReadBefore);
    graph.createHostRead("seedsReadAfter", seedsReadAfter);

    std::vector<uint32_t> hostSeedsReadBefore(seedsReadBefore.numElements());
    std::vector<uint32_t> hostSeedsReadAfter(seedsReadAfter.numElements());

    Engine engine(graph, randProg);
    if (deviceHalf) {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.writeTensor("tSeed", hSeed);
        convertAndWriteTensor<float, true>(target, engine, "paddedIn", flpInput,
                                           paddedInSize);
        engine.run();
        readAndConvertTensor<float, true>(graph.getTarget(), engine, "out",
                                          flpRandOut, inSize);
        engine.readTensor("seedsReadBefore", hostSeedsReadBefore.data(),
                          hostSeedsReadBefore.data() +
                              hostSeedsReadBefore.size());
        engine.readTensor("seedsReadAfter", hostSeedsReadAfter.data(),
                          hostSeedsReadAfter.data() +
                              hostSeedsReadAfter.size());
      });
    } else {

      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.writeTensor("tSeed", hSeed);
        convertAndWriteTensor<float, false>(target, engine, "paddedIn",
                                            flpInput, paddedInSize);
        engine.run();
        readAndConvertTensor<float, false>(graph.getTarget(), engine, "out",
                                           flpRandOut, inSize);
        engine.readTensor("seedsReadBefore", hostSeedsReadBefore.data(),
                          hostSeedsReadBefore.data() +
                              hostSeedsReadBefore.size());
        engine.readTensor("seedsReadAfter", hostSeedsReadAfter.data(),
                          hostSeedsReadAfter.data() +
                              hostSeedsReadAfter.size());
      });
    }

    if (seedToUseInTest) {
      if (!std::equal(hostSeedsReadBefore.begin(), hostSeedsReadBefore.end(),
                      hostSeedsReadAfter.begin())) {
        std::cerr << "hw seeds read before and after do not match\n";
        return 1;
      }
    } else if (deviceType != DeviceType::Cpu && !isIpuModel(deviceType)) {
      if (std::equal(hostSeedsReadBefore.begin(), hostSeedsReadBefore.end(),
                     hostSeedsReadAfter.begin())) {
        // there is always some work done and hw seeds should change
        std::cerr << "hw seeds read before and after match when they "
                     "should not\n";
        return 1;
      }
    }
  } else {
    auto seedsReadBefore = poplar::getHwSeeds(graph, randProg);

    if (testType == TestType::Normal) {
      out = poprand::normal(graph, seedToUseInTest, seedModifier, reference,
                            dType, mean, stdDev, randProg);
    } else if (testType == TestType::TruncatedNormal) {
      out = poprand::truncatedNormal(graph, seedToUseInTest, seedModifier,
                                     reference, dType, mean, stdDev, alpha,
                                     randProg);
    } else if (testType == TestType::Uniform ||
               testType == TestType::UniformInt) {
      out = poprand::uniform(graph, seedToUseInTest, seedModifier, reference,
                             dType, minVal, maxVal, randProg);
    } else if (testType == TestType::Bernoulli ||
               testType == TestType::BernoulliInt) {
      out = poprand::bernoulli(graph, seedToUseInTest, seedModifier, reference,
                               dType, prob, randProg);
    }
    auto seedsReadAfter = poplar::getHwSeeds(graph, randProg);
    std::vector<uint32_t> hostSeedsReadBefore(seedsReadBefore.numElements());
    std::vector<uint32_t> hostSeedsReadAfter(seedsReadAfter.numElements());

    graph.createHostRead("out", out);
    graph.createHostRead("seedsReadBefore", seedsReadBefore);
    graph.createHostRead("seedsReadAfter", seedsReadAfter);

    Engine engine(graph, randProg);

    dev.bind([&](const Device &d) {
      engine.load(d);
      for (unsigned i = 0; i != numLoops; ++i) {
        engine.writeTensor("tSeed", hSeed);
        engine.run();
        if (dType == poplar::INT) {
          readAndConvertTensor<int, false>(graph.getTarget(), engine, "out",
                                           intRandOut, inSize);
        } else if (deviceHalf) {
          readAndConvertTensor<float, true>(graph.getTarget(), engine, "out",
                                            flpRandOut, inSize);
        } else {
          readAndConvertTensor<float, false>(graph.getTarget(), engine, "out",
                                             flpRandOut, inSize);
        }
        engine.readTensor("seedsReadBefore", hostSeedsReadBefore.data(),
                          hostSeedsReadBefore.data() +
                              hostSeedsReadBefore.size());
        engine.readTensor("seedsReadAfter", hostSeedsReadAfter.data(),
                          hostSeedsReadAfter.data() +
                              hostSeedsReadAfter.size());
        if (seedToUseInTest) {
          if (!std::equal(hostSeedsReadBefore.begin(),
                          hostSeedsReadBefore.end(),
                          hostSeedsReadAfter.begin())) {
            std::cerr << " hw seeds read before and after do not match \n";
            std::exit(1);
          }
        } else if (deviceType != DeviceType::Cpu && !isIpuModel(deviceType)) {
          if (std::equal(hostSeedsReadBefore.begin(), hostSeedsReadBefore.end(),
                         hostSeedsReadAfter.begin())) {
            // there is always some work done and hw seeds should change
            std::cerr << "hw seeds read before and after match when they "
                         "should not\n";
            std::exit(1);
          }
        }
        auto invalid = validate();
        if (numLoops > 1) {
          std::cerr << "Test " << i << "/" << numLoops << ": "
                    << (invalid ? "Fail" : "Pass") << "\n";
        }
        if (invalid)
          std::exit(1);
        hSeed[0]++;
        if (hSeed[0] == 0)
          hSeed[1]++;
      }
    });
  }
}
