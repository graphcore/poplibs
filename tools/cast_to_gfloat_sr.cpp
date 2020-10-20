// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include "poputil/VertexTemplates.hpp"
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <popfloat/experimental/CastToGfloat.hpp>
#include <popfloat/experimental/codelets.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

#include "cast_to_gfloat.hpp"
#include "poputil/TileMapping.hpp"

#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#undef ENABLE_GF16_VERTEX

const float sr_tolerance = 0.03;
#include <poplar/CSRFunctions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using namespace popfloat::experimental;
using namespace poplibs_support;
using poplibs_test::Pass;

const OptionFlags simDebugOptions{{"debug.trace", "false"}};

SRDensityType convertStringToSRDensity(const std::string &dist) {
  if (dist == "Uniform") {
    return SRDensityType::UNIFORM;
  } else if (dist == "Normal") {
    return SRDensityType::NORMAL;
  } else if (dist == "TruncatedNormal") {
    return SRDensityType::TRUNCATED_NORMAL;
  } else if (dist == "Bernoulli") {
    return SRDensityType::BERNOULLI;
  } else if (dist == "TruncatedLogistic") {
    return SRDensityType::TRUNCATED_LOGISTIC;
  } else if (dist == "Logistic") {
    return SRDensityType::LOGISTIC;
  } else if (dist == "Laplace") {
    return SRDensityType::LAPLACE;
  } else if (dist == "TruncatedLaplace") {
    return SRDensityType::TRUNCATED_LAPLACE;
  } else if (dist == "LogitNormal") {
    return SRDensityType::LOGIT_NORMAL;
  } else if (dist == "TruncatedLogitNormal") {
    return SRDensityType::TRUNCATED_LOGIT_NORMAL;
  }
  throw poputil::poplibs_error("Gfloat Sr Distribution not supported");
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned man;
  unsigned exp;
  int bias;
  bool enableDenorm;
  bool enableInfsAndNans;
  bool enableNanoo;
  std::string srDensity;
  std::string calcType;
  int numberSRBits;
  float srNoiseMin;
  float srNoiseMax;
  float srNoiseOffset;
  float srNoiseScale;
  float probDown;
  Type inType = FLOAT;
  unsigned inSize;
  DeviceType deviceType = DeviceType::Cpu;
  boost::optional<unsigned> tilesPerIPU;
  unsigned seed;
  unsigned seedModifier;
  float inValue;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     deviceTypeHelp)
    ("tiles-per-ipu",
      po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("seed", po::value<unsigned>(&seed)->default_value(12352345), "prng seed")
    ("seed-modifier", po::value<unsigned>(&seedModifier)->default_value(785439),
     "prng seedModifier")("profile", "Output profiling report")
    ("man", po::value<unsigned>(&man)->default_value(10),
     "Mantissa size of the Gfloat format")
    ("exp", po::value<unsigned>(&exp)->default_value(5),
     "Exponent size of the Gfloat format")
    ("bias", po::value<int>(&bias)->default_value(15),
     "Exponent bias of the Gfloat format")
    ("enable-denorms",
     po::value<bool>(&enableDenorm)->default_value(true),
     "Enable Denorms")
    ("enable-infs-and-nans",
     po::value<bool>(&enableInfsAndNans)->default_value(true),
     "Enable Infs and Nans")
     ("enable-nanoo-mode",
      po::value<bool>(&enableNanoo)->default_value(true),
     "Enable Nans on overflow ")
     ("sr-noise-density",
     po::value<std::string>(&srDensity)->default_value("Uniform"),
     "SR noise density function: Uniform | Normal | TruncatedNormal | Laplace |"
     " Logistic | LogitNormal | LogitTruncatedNormal")
     ("number-sr-bits",
     po::value<int>(&numberSRBits)->default_value(23),
     "Number of bits for stochastic rounding")
    ("sr-noise-min",
     po::value<float>(&srNoiseMin)->default_value(-0.5),
     "Min value for prng values used for stochastic rounding")
    ("sr-noise-max",
     po::value<float>(&srNoiseMax)->default_value(0.5),
     "Max value for prng values used for stochastic rounding")
    ("sr-noise-offset",
     po::value<float>(&srNoiseOffset)->default_value(0.0),
     "Mean of prng values used for stochastic rounding")
    ("sr-noise-scale",
     po::value<float>(&srNoiseScale)->default_value(1.0),
     "Standard deviation of prng values used for stochastic rounding")
    ("sr-prob-truncate",
     po::value<float>(&probDown)->default_value(0.5),
     "Probability of rounding down (Bernoulli)")
    ("in-type", po::value<Type>(&inType)->default_value(inType),
     "Type of the data")
    ("calc-type",
     po::value<std::string>(&calcType)->default_value("AUTO"),
     "Native type used for gfloat calculation: AUTO | FP32 | FP16")
    ("input-value", po::value<float>(&inValue)->default_value(0.1),
     "Input value")
    ("input-size",
     po::value<unsigned>(&inSize)->default_value(1024),
     "Vector size");
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

  poplar::Device::createCPUDevice();
  const auto &target = dev.getTarget();
  Graph graph(target);
  poprand::addCodelets(graph);
  popops::addCodelets(graph);
  popfloat::experimental::addCodelets(graph);

  auto gfCastProg = Sequence();

  bool enableNanooMode = enableNanoo && enableInfsAndNans && (exp > 0);
  poplar::FloatingPointBehaviour fpBehaviour(false, false, false, false,
                                             enableNanooMode);

  poplar::setFloatingPointBehaviour(graph, gfCastProg, fpBehaviour,
                                    "setFpBehaviour");
  poplar::setStochasticRounding(graph, gfCastProg, false, "setSR");

#if 0 // def ENABLE_PARAM_PRINT
  std::cout
  << "Cast to Gfloat Params:\n"
  "man       :" << man << "\n"
  "exp       :" << exp << "\n"
  "bias      :" << bias << "\n"
  "enDenorm  :" << enableDenorm << "\n"
  "enInf     :" << enableInfsAndNans << "\n"
  "enNanoo   :" << enableNanooMode << "\n"
  "srDensity :" << srDensity << "\n"
  "srBits    :" << numberSRBits << "\n"
  "randMin   :" << srNoiseMin << "\n"
  "randMax   :" << srNoiseMax << "\n"
  "stdDev    :" << srNoiseScale << "\n"
  "inType    :" << inType << "\n"
  "inSize    :" << inSize << "\n";
#endif
  auto noiseDensity = convertStringToSRDensity(srDensity);

  auto calculationType = convertStringToSpecType(calcType);
  auto gfFormatCfg = GfloatCast::FormatConfig(
      man, exp, bias, enableDenorm, enableInfsAndNans, calculationType);

  Tensor input = graph.addVariable(inType, {inSize}, "input");
  mapTensorLinearly(graph, input);

  auto hInput = std::unique_ptr<float[]>(new float[inSize]);
  for (std::size_t i = 0; i != inSize; ++i) {
    hInput.get()[i] = inValue;
  }

  boost::multi_array<double, 1> hostInput(boost::extents[inSize]);

  std::mt19937 randomEngine;
  writeRandomValues(target, inType, hostInput, -5.0, +5.0, randomEngine);

  auto flpCastOut = std::unique_ptr<float[]>(new float[inSize]);

  // Create stream for input data
  auto inStreamV = graph.addHostToDeviceFIFO("InputVector", inType, inSize);

  gfCastProg = Sequence(Copy(inStreamV, input));

  uint32_t hSeed[2];
  hSeed[0] = seed;
  hSeed[1] = ~seed;

  auto tSeed = graph.addVariable(poplar::UNSIGNED_INT, {2}, "tSeed");
  graph.setTileMapping(tSeed, 0);

  // Create stream for input data
  auto seedStreamV = graph.addHostToDeviceFIFO("seed", UNSIGNED_INT, 2);
  gfCastProg.add(Copy(seedStreamV, tSeed));

  poprand::setSeed(graph, tSeed, seedModifier, gfCastProg, "setSeed");

  auto roundCfg = GfloatCast::RoundConfig(
      RoundType::SX, numberSRBits, gfFormatCfg.getCalculationType(),
      noiseDensity, srNoiseOffset, srNoiseScale, srNoiseMax, srNoiseMin,
      probDown);

  auto gfCast =
      GfloatCast(gfFormatCfg, roundCfg, enableNanooMode,
                 convertTypeToGfloatSpecType(gfFormatCfg.getCalculationType()));

  gfCast.createCastOpParamsTensor(graph, gfCastProg);

  auto castOutput = gfCast.castNativeToGfloat(graph, input, gfCastProg);

  graph.createHostRead("castOutput", castOutput);

  Engine engine(graph, gfCastProg,
                OptionFlags{{"debug.allowOutOfMemory", "true"},
                            {"debug.outputAllSymbols", "true"},
                            {"debug.instrumentCompute", "true"},
                            {"debug.floatPointOpException", "false"},
                            {"debug.nanOverflowException", "false"},
                            {"prng.enableStochasticRounding", "true"},
                            {"prng.seed", std::to_string(seed)}});

  engine.connectStream("InputVector", hInput.get());
  engine.connectStream("seed", hSeed);

  // Run the forward pass.
  if (gfFormatCfg.getCalculationType() == poplar::FLOAT) {
    dev.bind([&](const Device &d) {
      engine.load(d);
      engine.run();
      readAndConvertTensor<float, false>(
          graph.getTarget(), engine, "castOutput", flpCastOut.get(), inSize);
    });
  } else if (gfFormatCfg.getCalculationType() == poplar::HALF) {
    dev.bind([&](const Device &d) {
      engine.load(d);
      engine.run();
      readAndConvertTensor<float, true>(graph.getTarget(), engine, "castOutput",
                                        flpCastOut.get(), inSize);
    });
  }

  unsigned inValueMem;
  std::memcpy(&inValueMem, &inValue, sizeof(inValueMem));
  int e_single;
  float inMan = std::frexp(inValue, &e_single);
  e_single -= 1;
  inMan *= 2.0;

  int m_single = (std::abs(inMan) * (1 << manSizeFp32));

  int maskLen;
  maskLen = ((1 - bias) - e_single);
  maskLen = (maskLen < 0) ? 0 : maskLen;
  maskLen += manSizeFp32 - man;
  maskLen = std::min<uint32_t>(maskLen, manSizeFp32 + 1);

  int castAtomNumMantissaBits =
      (gfFormatCfg.getCalculationType() == poplar::FLOAT) ? manSizeFp32
                                                          : manSizeFp16;
  numberSRBits = std::min<int>(castAtomNumMantissaBits, numberSRBits);

  float normMan = 0.0;
  numberSRBits = std::min<int>(maskLen, numberSRBits);

  int srMask;
  srMask = (1 << numberSRBits) - 1;
  m_single >>= (maskLen - numberSRBits);
  m_single &= srMask;
  normMan = (float)m_single / ((float)(1 << numberSRBits));

  float floorRate = 0;
  for (unsigned idx = 0; idx < inSize; ++idx) {
    if (std::abs(flpCastOut.get()[idx]) <= std::abs(inValue)) {
      ++floorRate;
    }
  }
  floorRate = floorRate / inSize;
  float probFloor = probDown;

  if (noiseDensity != SRDensityType::BERNOULLI) {
    probFloor = 1 - normMan;
    if ((noiseDensity == SRDensityType::NORMAL) ||
        (noiseDensity == SRDensityType::TRUNCATED_NORMAL) ||
        (noiseDensity == SRDensityType::LOGISTIC) ||
        (noiseDensity == SRDensityType::TRUNCATED_LOGISTIC) ||
        (noiseDensity == SRDensityType::LAPLACE) ||
        (noiseDensity == SRDensityType::TRUNCATED_LAPLACE)) {
      probFloor -= 0.5;
    }

    if (probFloor < srNoiseMin) {
      probFloor = 0.0;
    } else if (probFloor > srNoiseMax) {
      probFloor = 1.0;
    } else {
      float probMin = 0.0;
      float probMax = 1.0;
      if (noiseDensity == SRDensityType::UNIFORM) {
        probFloor = (probFloor - srNoiseMin) / (srNoiseMax - srNoiseMin);
      } else if (noiseDensity == SRDensityType::NORMAL) {
        probFloor = 0.5 * (1.0 + std::erf((probFloor - srNoiseOffset) /
                                          (std::sqrt(2.0) * srNoiseScale)));
      } else if (noiseDensity == SRDensityType::TRUNCATED_NORMAL) {
        probFloor = (1.0 + std::erf((probFloor - srNoiseOffset) /
                                    (std::sqrt(2.0) * srNoiseScale)));
        probMin = (1.0 + std::erf((srNoiseMin - srNoiseOffset) /
                                  (std::sqrt(2.0) * srNoiseScale)));
        probMax = (1.0 + std::erf((srNoiseMax - srNoiseOffset) /
                                  (std::sqrt(2.0) * srNoiseScale)));
      } else if (noiseDensity == SRDensityType::LAPLACE) {
        probFloor -= srNoiseOffset;
        if (probFloor != 0.0) {
          probFloor =
              0.5 + (std::abs(probFloor) / probFloor) *
                        (1.0 - std::exp(-std::abs(probFloor) / srNoiseScale)) /
                        2.0;
        } else {
          probFloor = 0.5;
        }
      } else if (noiseDensity == SRDensityType::TRUNCATED_LAPLACE) {
        probFloor -= srNoiseOffset;
        if (probFloor != 0) {
          probFloor =
              0.5 + (std::abs(probFloor) / probFloor) *
                        (1.0 - std::exp(-std::abs(probFloor) / srNoiseScale)) /
                        2.0;
        } else {
          probFloor = 0.5;
        }

        probMin = 0.5;
        if (srNoiseMin != 0.0) {
          probMin += (std::abs(srNoiseMin - srNoiseOffset) /
                      (srNoiseMin - srNoiseOffset)) *
                     (1.0 - std::exp(-std::abs(srNoiseMin - srNoiseOffset) /
                                     srNoiseScale)) /
                     2.0;
        }

        probMax = 0.5;
        if (srNoiseMax != 0.0) {
          probMax += (std::abs(srNoiseMax - srNoiseOffset) /
                      (srNoiseMax - srNoiseOffset)) *
                     (1.0 - std::exp(-std::abs(srNoiseMax - srNoiseOffset) /
                                     srNoiseScale)) /
                     2.0;
        }
      } else if (noiseDensity == SRDensityType::LOGISTIC) {
        probFloor =
            1.0 / (1.0 + std::exp(-(probFloor - srNoiseOffset) / srNoiseScale));
      } else if (noiseDensity == SRDensityType::TRUNCATED_LOGISTIC) {
        probFloor =
            1.0 / (1.0 + std::exp(-(probFloor - srNoiseOffset) / srNoiseScale));
        probMin =
            1.0 /
            (1.0 + std::exp(-(srNoiseMin - srNoiseOffset) / srNoiseScale));
        probMax =
            1.0 /
            (1.0 + std::exp(-(srNoiseMax - srNoiseOffset) / srNoiseScale));
      } else if (noiseDensity == SRDensityType::LOGIT_NORMAL) {
        probFloor = std::log(probFloor / (1.0 - probFloor));
        probFloor = 0.5 * (1.0 + std::erf((probFloor - srNoiseOffset) /
                                          (std::sqrt(2.0) * srNoiseScale)));
      } else if (noiseDensity == SRDensityType::TRUNCATED_LOGIT_NORMAL) {
        probFloor = std::log(probFloor / (1.0 - probFloor));
        probFloor = (1.0 + std::erf((probFloor - srNoiseOffset) /
                                    (std::sqrt(2.0) * srNoiseScale)));

        probMin = std::log(srNoiseMin / (1.0 - srNoiseMin));
        probMin = (1.0 + std::erf((probMin - srNoiseOffset) /
                                  (std::sqrt(2.0) * srNoiseScale)));

        probMax = std::log(srNoiseMax / (1.0 - srNoiseMax));
        probMax = (1.0 + std::erf((probMax - srNoiseOffset) /
                                  (std::sqrt(2.0) * srNoiseScale)));
      }
      probFloor = (probFloor - probMin) / (probMax - probMin);
    }
  }
  probFloor = ((int)(probFloor * ((float)(1 << numberSRBits)))) /
              ((float)(1 << numberSRBits));

  bool pass = (std::abs(probFloor - floorRate) <= sr_tolerance);
  if (pass) {
    std::cout << "Test Passed\n";
  }
  return !pass;
}
