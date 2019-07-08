// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <poplar/Engine.hpp>
#include "poputil/VertexTemplates.hpp"
#include <memory>
#include <iostream>
#include <random>
#include <poplar/IeeeHalf.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <popfloat/CastToGfloat.hpp>
#include <popfloat/codelets.hpp>
#include <iomanip>

#include "cast_to_gfloat.hpp"
#include "poputil/TileMapping.hpp"

#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_support/Compiler.hpp>
#include "TestDevice.hpp"
#include "../lib/popfloat/codelets/popfloatParamUtils.hpp"
#undef ENABLE_GF16_VERTEX

const float sr_tolerance = 0.03;
#include <popsys/codelets.hpp>
#include <popsys/CSRFunctions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using namespace popfloat;
using namespace popfloat::gfexpr;
using poplibs_test::Pass;

const OptionFlags simDebugOptions {
  {"debug.trace", "false"}
};

popfloat::gfexpr::GfloatSRDensityType
convertStringToGfloatSRDensity(const std::string &dist) {
  if (dist == "Uniform") {
    return GfloatSRDensityType::UNIFORM;
  } else if (dist == "Normal") {
    return GfloatSRDensityType::NORMAL;
  } else if (dist == "TruncatedNormal") {
    return GfloatSRDensityType::TRUNCATED_NORMAL;
  } else if (dist == "Bernoulli") {
    return GfloatSRDensityType::BERNOULLI;
  } else if (dist == "Logistic") {
    return GfloatSRDensityType::LOGISTIC;
  } else if (dist == "Laplace") {
    return GfloatSRDensityType::LAPLACE;
  } else if (dist == "LogitNormal") {
    return GfloatSRDensityType::LOGIT_NORMAL;
  } else if (dist == "TruncatedLogitNormal") {
    return GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL;
  }
  throw poputil::poplibs_error("Gfloat Sr Distribution not supported");
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned    man;
  unsigned    exp;
  int         bias;
  bool        enDenorm;
  bool        enInf;
  bool        enNanoo;
  std::string srDensity;
  int         srBits;
  float       randMin;
  float       randMax;
  float       stdMean;
  float       stdDev;
  float       probDown;
  Type        inType = FLOAT;
  unsigned    inSize;
  DeviceType  deviceType = DeviceType::Cpu;
  IPUModel    ipuModel;
  unsigned    seed;
  unsigned    seedModifier;
  float       inValue;

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
    ("seed-modifier", po::value<unsigned>(&seedModifier)->default_value(785439),
     "prng seedModifier")("profile", "Output profiling report")
    ("man", po::value<unsigned>(&man)->default_value(10),
     "Mantissa size of the Gfloat format")
    ("exp", po::value<unsigned>(&exp)->default_value(5),
     "Exponent size of the Gfloat format")
    ("bias", po::value<int>(&bias)->default_value(15),
     "Exponent bias of the Gfloat format")
    ("enDnrm",
     po::value<bool>(&enDenorm)->default_value(true),
     "Enable Denorms")
    ("enInf",
     po::value<bool>(&enInf)->default_value(true),
     "Enable Infs and Nans")
     ("enNanoo",
      po::value<bool>(&enNanoo)->default_value(true),
     "Enable Nans on overflow ")
     ("srDensity",
     po::value<std::string>(&srDensity)->default_value("Uniform"),
     "SR noise density function: Uniform | Normal | TruncatedNormal | Laplace |"
     " Logistic | LogitNormal | LogitTruncatedNormal")
     ("srBits",
     po::value<int>(&srBits)->default_value(23),
     "Number of bits for stochastic rounding")
    ("randMin",
     po::value<float>(&randMin)->default_value(-0.5),
     "Min value for prng values used for stochastic rounding")
    ("randMax",
     po::value<float>(&randMax)->default_value(0.5),
     "Max value for prng values used for stochastic rounding")
    ("stdMean",
     po::value<float>(&stdMean)->default_value(0.0),
     "Mean of prng values used for stochastic rounding")
    ("stdDev",
     po::value<float>(&stdDev)->default_value(1.0),
     "Standard deviation of prng values used for stochastic rounding")
    ("probDown",
     po::value<float>(&probDown)->default_value(0.5),
     "Probability of rounding down (Bernoulli)")
    ("inType", po::value<Type>(&inType)->default_value(inType),
     "Type of the data")
    ("inValue", po::value<float>(&inValue)->default_value(0.1), "Input value")
    ("inSize",
     po::value<unsigned>(&inSize)->default_value(1024),
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
  }();

  poplar::Device::createCPUDevice();
  const auto &target = dev.getTarget();
  Graph graph(target);
  poprand::addCodelets(graph);
  popfloat::addCodelets(graph);
  popsys::addCodelets(graph);

  auto gfCastProg = Sequence();

  popsys::FloatingPointBehaviour fpBehaviour(
     true, true, true, false, enNanoo &&enInf &&(exp > 0));
  popsys::setFloatingPointBehaviour(graph, gfCastProg, fpBehaviour,
                                    "setFpBehaviour");

#if def ENABLE_PARAM_PRINT
  std::cout
  << "Cast to Gfloat Params:\n"
  "man            :" << man       << "\n"
  "exp            :" << exp       << "\n"
  "bias           :" << bias      << "\n"
  "enDenorm       :" << enDenorm  << "\n"
  "enInf          :" << enInf     << "\n"
  "enNanoo        :" << enNanoo   << "\n"
  "srDistribution :" << srDensity << "\n"
  "srBits         :" << srBits    << "\n"
  "randMin        :" << randMin   << "\n"
  "randMax        :" << randMax   << "\n"
  "stdDev         :" << stdDev    << "\n"
  "inType         :" << inType    << "\n"
  "inSize         :" << inSize    << "\n";
#endif
  auto noiseDensity = convertStringToGfloatSRDensity(srDensity);

  bool hfMaxAlign = (exp == POPFLOAT_NUM_FP16_EXPONENT_BITS) && !enInf;

  GfloatFormatConfig gfFormatCfg =
    GfloatFormatConfig(man, exp, bias, enDenorm, enInf);
  auto quantisedOpType = gfFormatCfg.getQuantisedOpType();
  if ((quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF32) &&
      (inType != poplar::FLOAT)) {
    throw poplibs_error(
       "popfloat::lookupGfQuantiseParamOp: Ops expects float input");
  }

  Tensor input = graph.addVariable(inType, { inSize }, "input");
  mapTensorLinearly(graph, input);

  auto hInput = std::unique_ptr<float[]>(new float[inSize]);
  for (std::size_t i = 0; i != inSize; ++i) {
    hInput.get()[i] = inValue;
  }

  boost::multi_array<double, 1>
  hostInput(boost::extents[inSize]);

  std::mt19937 randomEngine;
  writeRandomValues(target, inType, hostInput, -5.0, +5.0, randomEngine);

  auto flpCastOut = std::unique_ptr<float[]>(new float[inSize]);

  //Create stream for input data
  auto inStreamV = graph.addHostToDeviceFIFO("InputVector",
                                             inType,
                                             inSize);

  gfCastProg = Sequence(Copy(inStreamV  , input));

  uint32_t hSeed[2];
  hSeed[0] =  seed;
  hSeed[1] = ~seed;


  auto tSeed = graph.addVariable(poplar::UNSIGNED_INT, { 2 }, "tSeed");
  graph.setTileMapping(tSeed, 0);

  //Create stream for input data
  auto seedStreamV = graph.addHostToDeviceFIFO("seed",
                                               UNSIGNED_INT,
                                               2);
  gfCastProg.add(Copy(seedStreamV, tSeed));

  poprand::setSeed(graph, tSeed, seedModifier, gfCastProg, "setSeed");

  // Create Gfloat params for forwad cast
  auto gfCompressed = setPackedGfloatParams(graph, gfCastProg, gfFormatCfg);

  Tensor quantiseParams =
    createCastOpParamsTensor(graph, gfCastProg,
                             gfFormatCfg.getQuantisedOpType(),
                             gfCompressed);

  bool enableNanooMode = enNanoo && enInf && (exp > 0);

  auto gfQuantiseCfg =
    GfloatCastConfig(inType, gfFormatCfg.getQuantisedOutputType(),
                     gfFormatCfg.getQuantisedOpType(),
                     noiseDensity, srBits, enableNanooMode,
                     stdMean, stdDev, randMax, randMin, probDown);

  auto quantiseOutput = castToGfloat(graph, input, quantiseParams,
                                     gfCastProg, gfQuantiseCfg);
  graph.createHostRead("quantiseOutput", quantiseOutput);

  Engine engine(graph, gfCastProg, OptionFlags{
                  { "target.workerStackSizeInBytes", "0x8000" },
                  { "prng.enable", "true" },
                  { "prng.seed", std::to_string(seed) }
                });

  engine.connectStream(inStreamV, hInput.get());
  engine.connectStream(seedStreamV, hSeed);

  // Run the forward pass.
  if (gfFormatCfg.getQuantisedOutputType() == poplar::FLOAT) {
    dev.bind([&](const Device &d) {
      engine.load(d);
      engine.run();
      readAndConvertTensor<float, false>(
         graph.getTarget(),
         engine,
         "quantiseOutput",
         flpCastOut.get(),
         inSize);
                                  });
  } else if (gfFormatCfg.getQuantisedOutputType() == poplar::HALF) {
    dev.bind([&](const Device &d) {
      engine.load(d);
      engine.run();
      readAndConvertTensor<float, true>(
         graph.getTarget(),
         engine,
         "quantiseOutput",
         flpCastOut.get(),
         inSize);
                                  });
  }

  unsigned inValueMem;
  std::memcpy(&inValueMem, &inValue, sizeof(inValueMem));
  int e_single;
  float inMan = std::frexp(inValue, &e_single);
  e_single -= 1;
  inMan *= 2.0;

  int m_single = (std::abs(inMan) * (1 << 23));

  int maskLen;
  maskLen  = ((1 - bias) - e_single);
  maskLen  = (maskLen < 0) ? 0 : maskLen;
  maskLen += 23 - man;
  maskLen  = std::min<uint32_t>(maskLen, manSizeFp32 + 1);

  srBits =
    (gfFormatCfg.getQuantisedOpType() ==
     GfloatCastOpType::CAST_TO_QUANTISED_GF16) ?
    std::min<int>(10, srBits) : std::min<int>(23, srBits);
  int srMask;
  float normMan = 0.0;
  srBits = std::min<int>(maskLen, srBits);
  m_single >>= (maskLen - srBits);
  srMask = (1 << srBits) - 1;
  m_single &= srMask;
  normMan = (float)m_single / ((float)(1 << srBits));

  float floorRate = 0;
  for (unsigned idx = 0; idx < inSize; ++idx) {
    if (std::abs(flpCastOut.get()[idx]) > std::abs(inValue)) {
      ++floorRate;
    }
  }
  floorRate = floorRate / inSize;
  float probFloor = 0.0;
  float probMin = 0.0;
  float probMax = 1.0;

  if (noiseDensity == GfloatSRDensityType::UNIFORM) {
    normMan -= 0.5;
    probFloor = (normMan - randMin) / (randMax - randMin);
  } else if (noiseDensity == GfloatSRDensityType::NORMAL) {
    normMan -= 0.5;
    probFloor = 0.5 * (1.0 + std::erf((normMan - stdMean) /
                                      (std::sqrt(2.0) * stdDev)));
  } else if (noiseDensity == GfloatSRDensityType::TRUNCATED_NORMAL) {
    normMan -= 0.5;
    probFloor = 0.5 * (1 + std::erf((normMan - stdMean) /
                                    (std::sqrt(2.0) * stdDev)));
    probMin = 0.5 * (1.0 + std::erf((randMin - stdMean) /
                                    (std::sqrt(2.0) * stdDev)));
    probMax = 0.5 * (1.0 + std::erf((randMax - stdMean) /
                                    (std::sqrt(2.0) * stdDev)));
  } else if (noiseDensity == GfloatSRDensityType::LAPLACE) {
    float b = std::sqrt(2.0) * stdDev;
    normMan -= 0.5 + stdMean;
    probFloor = 0.5;
    if (normMan != 0) {
      probFloor += (std::abs(normMan) / normMan) *
                   (1.0 - std::exp(-std::abs(normMan) / b));
    }

    probMin = 0.5;
    if (randMin != stdMean) {
      probMin += (std::abs(randMin - stdMean) / (randMin - stdMean)) *
                 (1.0 - std::exp(-std::abs(randMin - stdMean) / b));
    }

    probMax = 0.5;
    if (randMax != stdMean) {
      probMax += (std::abs(randMax - stdMean) / (randMax - stdMean)) *
                 (1.0 - std::exp(-std::abs(randMax - stdMean) / b));
    }
  } else if (noiseDensity == GfloatSRDensityType::LOGISTIC) {
    float s = stdDev * std::sqrt(3.0) / (std::atan(1.0) * 4);

    probFloor = 1.0 / (1.0 + std::exp(-std::abs(normMan - stdMean) / s));
    probMin = 1.0 / (1.0 + std::exp(-std::abs(randMin - stdMean) / s));
    probMax = 1.0 / (1.0 + std::exp(-std::abs(randMax - stdMean) / s));
  } else if (noiseDensity == GfloatSRDensityType::LOGIT_NORMAL) {
    probFloor = std::log(normMan / (1.0 - normMan));
    probFloor = 0.5 * (1.0 + std::erf((normMan - stdMean) /
                                      (std::sqrt(2.0) * stdDev)));
  } else if (noiseDensity == GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL) {
    probFloor = std::log(normMan / (1.0 - normMan));
    probFloor = 0.5 * (1.0 + std::erf((probFloor - stdMean) /
                                      (std::sqrt(2.0) * stdDev)));

    probMin = std::log(randMin / (1.0 - randMin));
    probMin = 0.5 * (1.0 + std::erf((probMin - stdMean) /
                                    (std::sqrt(2.0) * stdDev)));
    probMax = std::log(randMax / (1.0 - randMax));
    probMax = 0.5 * (1.0 + std::erf((probMax - stdMean) /
                                    (std::sqrt(2.0) * stdDev)));
  }

  probFloor = (probFloor - probMin) / (probMax - probMin);
  probFloor =
    ((int)(probFloor * ((float)(1 << srBits)))) / ((float)(1 << srBits));

  bool pass = (std::abs(probFloor - floorRate) <= sr_tolerance);
  if (pass) {
    std::cout << "Test Passed\n";
  }
  return !pass;
}
