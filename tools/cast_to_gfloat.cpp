// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include "cast_to_gfloat.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/VertexTemplates.hpp"
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <experimental/popfloat/CastToGfloat.hpp>
#include <experimental/popfloat/CastToHalf.hpp>
#include <experimental/popfloat/codelets.hpp>
#include <iomanip>
#include <iostream>
#include <memory>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include <random>

#include "TestDevice.hpp"
#include <poplibs_support/Compiler.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>

#include <poplar/CSRFunctions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace experimental::popfloat;
using namespace poputil;
using poplibs_test::Pass;

const OptionFlags simDebugOptions{{"debug.trace", "false"}};

template <typename T, bool pack>
bool castNativeToGfloatCheck(float *inVec, T *outVec, unsigned sizeVec,
                             const GfloatCast::FormatConfig &gfFormatCfg,
                             const GfloatCast::CastConfig &gfCastCfg) {
  int32_t minExp = 1 - gfFormatCfg.getExponentBias();
  if (gfFormatCfg.isDenormEnabled()) {
    minExp -= gfFormatCfg.getNumMantissaBits();
  }
  auto formatType = gfFormatCfg.getFormatType();

  if (gfFormatCfg.getCalculationType() == HALF) {
    minExp = gfFormatCfg.isDenormEnabled()
                 ? -(14 + gfFormatCfg.getNumMantissaBits())
                 : -14;
    minExp -= (formatType == FormatType::MAX_NORM_ALIGN_GF8);
  }

  float minValue = std::pow(2.0, minExp);

  int32_t maxExp = (1 << gfFormatCfg.getNumExponentBits());
  maxExp -= 1 + gfFormatCfg.getExponentBias();
  if (gfFormatCfg.getCalculationType() == HALF) {
    maxExp = (1 << gfFormatCfg.getNumExponentBits()) - 1 - expBiasFp16;
  }

  if (gfFormatCfg.infAndNansEnabled() && !gfFormatCfg.isBlockFloat()) {
    --maxExp;
  }
  float maxValue = (float)((1 << (gfFormatCfg.getNumMantissaBits() + 1)) - 1);
  if (gfFormatCfg.isBlockFloat()) {
    maxValue = (float)((1 << gfFormatCfg.getNumMantissaBits()) - 1);
    maxValue *= std::pow(2.0, minExp);
  } else {
    maxExp -= gfFormatCfg.getNumMantissaBits();
    maxValue *= std::pow(2.0, maxExp);
  }

  float scale = 1.0;

  if (gfFormatCfg.getCalculationType() == HALF) {
    if ((formatType == FormatType::MAX_NORM_ALIGN_GF8) ||
        (!gfFormatCfg.infAndNansEnabled() &&
         (gfFormatCfg.getNumExponentBits() == expSizeFp16))) {
      scale = std::pow(2.0, gfFormatCfg.getExponentBias() - 16);
    } else {
      scale = std::pow(2.0, gfFormatCfg.getExponentBias() - 15);
    }
  }

  unsigned fpSize = 31;
  if ((gfCastCfg.getStorageType() == HALF) ||
      (gfCastCfg.getStorageType() == SHORT)) {
    fpSize = 15;
  } else if (gfCastCfg.getStorageType() == CHAR) {
    fpSize = 7;
  }

  unsigned manSize = fpSize - gfFormatCfg.getNumExponentBits();

  unsigned manExpMask = (1 << fpSize) - 1;

  unsigned fpMask = (1 << (fpSize + 1)) - 1;
  unsigned alignShr = manSizeFp32 - manSize;

  int32_t expBias = gfFormatCfg.getExponentBias();
  if (gfFormatCfg.getCalculationType() == HALF) {
    expBias = (formatType == FormatType::MAX_NORM_ALIGN_GF8) ? (expBiasFp16 + 1)
                                                             : expBiasFp16;
  }
  int32_t minNormExp0 = 1 - expBias;

  bool pass = true;

  int32_t minNormExp1 = 1 - gfFormatCfg.getExponentBias();
  if (gfFormatCfg.getCalculationType() == HALF) {
    minNormExp1 = 1 - expBiasFp16;
  }

  int32_t qNan = qnanFp32;
  if ((gfFormatCfg.getCalculationType() == HALF) ||
      (gfCastCfg.getStorageType() == HALF)) {
    float _qnan = halfToSingle(qnanFp16);
    std::memcpy(&qNan, &_qnan, sizeof(qNan));
  }

  // Quantised FP16 clip input before scaling if NaNOO mode is disable
  uint16_t maxBits = 0x7BFF;
  if (gfFormatCfg.getCalculationType() == HALF) {
    maxBits >>= (manSizeFp16 - gfFormatCfg.getNumMantissaBits());
    maxBits <<= (manSizeFp16 - gfFormatCfg.getNumMantissaBits());
  }

  float maxAbs = halfToSingle(maxBits);
  for (unsigned j = 0; j != sizeVec; ++j) {
    float input = inVec[j] * scale;
    int32_t inBits;

    if (gfFormatCfg.getCalculationType() == HALF) {
      if (!gfCastCfg.isNanooModeEnabled()) {
        if (std::abs(input) > maxAbs) {
          input = (input > 0) ? maxAbs : (-1.0 * maxAbs);
        }
      }
      auto inBits16 = singleToHalf(input, gfCastCfg.isNanooModeEnabled());
      input = halfToSingle(inBits16);
    }
    std::memcpy(&inBits, &input, sizeof(inBits));

    int32_t m_single = inBits & manMaskFp32;
    int32_t e_single = ((inBits & expMaskFp32) >> manSizeFp32);
    if (e_single != 0) {
      m_single |= (1 << manSizeFp32);
    }
    e_single -= expBiasFp32;
    int32_t s_single = (inBits & sgnMaskFp32);

    int32_t masklen;
    masklen = (minNormExp0 - e_single);
    masklen = (masklen < 0) ? 0 : masklen;
    masklen += manSizeFp32 - gfFormatCfg.getNumMantissaBits();
    masklen = std::min<uint32_t>(masklen, manSizeFp32 + 1);

    if ((gfFormatCfg.getCalculationType() == HALF) &&
        (std::abs(input) > maxAbs)) {
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == RoundType::RZ) {
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == RoundType::RN) {
      bool msfBitVal = (m_single >> (masklen - 1)) & 1;
      bool lsbs = (m_single & ((1 << (masklen - 1)) - 1)) != 0;
      bool lsBitVal = (m_single >> masklen) & 1;
      m_single = (m_single >> masklen);
      if (msfBitVal && (lsBitVal || lsbs)) {
        m_single += 1;
      }
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == RoundType::RA) {
      m_single = m_single + ((1 << masklen) >> 1);
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == RoundType::RU) {
      uint32_t corr = (s_single == 0) ? ((1 << masklen) - 1) : 0;
      m_single = m_single + corr;
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == RoundType::RD) {
      uint32_t corr = (s_single == 0) ? 0 : ((1 << masklen) - 1);
      m_single = m_single + corr;
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else {
      std::cout << "Rounding mode not supported" << std::endl;
    }

    float fpOut = (float)m_single;
    fpOut *= std::pow(2.0, e_single - manSizeFp32);
    if (fpOut < minValue) {
      fpOut = 0.0;
    }
    std::memcpy(&inBits, &fpOut, sizeof(inBits));

    if (fpOut > maxValue) {
      if (gfFormatCfg.infAndNansEnabled() && gfCastCfg.isNanooModeEnabled()) {
        inBits = qNan;
        if ((pack || ((gfFormatCfg.getCalculationType() == FLOAT) &&
                      (gfCastCfg.getStorageType() != HALF))) &&
            (inVec[j] < 0)) {
          inBits |= s_single;
        }
      } else {
        fpOut = maxValue / (pack ? 1.0 : scale);
        std::memcpy(&inBits, &fpOut, sizeof(inBits));
        inBits |= s_single;
      }
    } else {
      if (!pack) {
        fpOut /= scale;
        std::memcpy(&inBits, &fpOut, sizeof(inBits));
      }
      inBits |= s_single;
    }

    int32_t outBits;
    if (pack) {
      int32_t m_single = (inBits & manMaskFp32) >> alignShr;
      int32_t e_single = ((inBits & expMaskFp32) >> manSizeFp32);
      int32_t s_single = (inBits & sgnMaskFp32) >> (31 - fpSize);

      outBits = 0;

      if (e_single == infAndNanExpFp32) {
        if (gfFormatCfg.getCalculationType() == FLOAT) {
          outBits = ((qnanFp32 >> alignShr) & manExpMask) | s_single;
        } else {
          outBits = ((qnanFp16 >> (manSizeFp16 - manSize)) & manExpMask);
        }
      } else if ((m_single != 0) || (e_single != 0)) {
        e_single -= expBiasFp32;
        if (e_single < minNormExp1) {
          if ((formatType == FormatType::MAX_NORM_ALIGN_GF8) &&
              (e_single == (minNormExp1 - 1))) {
            e_single = 1;
          } else {
            m_single |= (1 << manSize);
            m_single >>= (minNormExp1 - e_single);
            e_single = 0;
          }
        } else {
          e_single += expBias;
        }
        e_single = e_single << manSize;

        outBits = m_single | e_single | s_single;
      } else {
        if (gfFormatCfg.getCalculationType() == FLOAT) {
          outBits = s_single;
        }
      }

      pass &= ((outBits & fpMask) == ((int32_t)outVec[j] & fpMask));
    } else {
      std::memcpy(&outBits, &outVec[j], sizeof(outBits));
      pass &= (outBits == inBits);
    }
  }
  return pass;
}

template <typename T>
bool castGfloatToNativeCheck(T *inVec, float *outVec, unsigned sizeVec,
                             const GfloatCast::FormatConfig &gfFormatCfg,
                             Type storageType) {
  unsigned fpSize = 31;
  if ((storageType == HALF) || (storageType == SHORT)) {
    fpSize = 15;
  } else if (storageType == CHAR) {
    fpSize = 7;
  }
  unsigned manSize = fpSize - gfFormatCfg.getNumExponentBits();

  unsigned alignShr = manSizeFp32 - manSize;

  int32_t manMask = (1 << manSize) - 1;
  int32_t maxExp =
      (1 << gfFormatCfg.getNumExponentBits()) - gfFormatCfg.infAndNansEnabled();
  int32_t expMask = ((1 << gfFormatCfg.getNumExponentBits()) - 1) << manSize;
  int32_t sgnMask = 1 << (manSize + gfFormatCfg.getNumExponentBits());

  int32_t expBias = gfFormatCfg.getExponentBias();
  int32_t qNan = qnanFp32;

  if (gfFormatCfg.getCalculationType() == HALF) {
    float _qnan = halfToSingle(qnanFp16);
    std::memcpy(&qNan, &_qnan, sizeof(qNan));
  }

  int32_t minNormExp = 1 - expBias;

  bool pass = true;
  for (unsigned j = 0; j != sizeVec; ++j) {
    int32_t inBits = inVec[j];

    int32_t m_single = inBits & manMask;
    int32_t e_single = (inBits & expMask) >> manSize;
    int32_t s_single = (inBits & sgnMask) << (31 - fpSize);

    m_single <<= alignShr;

    inBits = 0;
    if ((e_single < maxExp) || gfFormatCfg.isBlockFloat()) {
      if (e_single == 0) {
        if (m_single != 0) {
          e_single = minNormExp;
          while (m_single < (1 << manSizeFp32)) {
            m_single <<= 1;
            --e_single;
          }
          e_single += expBiasFp32;
          e_single <<= manSizeFp32;
          m_single &= manMaskFp32;
          inBits = m_single | e_single | s_single;
        } else {
          auto gfFormatType = gfFormatCfg.getFormatType();
          if (gfFormatType == FormatType::ENABLE_DENORM_GF16) {
            inBits = s_single;
          }
        }
      } else {
        e_single -= expBias;
        e_single += expBiasFp32;
        e_single <<= manSizeFp32;
        inBits = m_single | e_single | s_single;
      }

      int32_t outBits;
      std::memcpy(&outBits, &outVec[j], sizeof(outBits));
      pass &= (outBits == inBits);
    }
  }
  return pass;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned man;
  unsigned exp;
  int bias;
  bool enableDenorms;
  bool enableInfsAndNans;
  bool enableNanoo;
  std::string roundMode;
  std::string calcType;
  std::string storeType;
  unsigned numberSRBits;
  Type inType = FLOAT;
  unsigned inSize;
  DeviceType deviceType = DeviceType::Cpu;
  IPUModel ipuModel;
  bool prng;
  unsigned seed;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
      ("help", "Produce help message")
      ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Hw | IpuModel")
      ("tiles-per-ipu",
       po::value<unsigned>(&ipuModel.tilesPerIPU)->
           default_value(ipuModel.tilesPerIPU),
       "Number of tiles per IPU")
      ("prng", po::value<bool>(&prng)->default_value(false), "prng enable")
      ("seed", po::value<unsigned>(&seed)->default_value(12352345), "prng seed")
      ("profile", "Output profiling report")
      ("man", po::value<unsigned>(&man)->default_value(10),
       "Mantissa size of the Gfloat format")
      ("exp", po::value<unsigned>(&exp)->default_value(5),
       "Exponent size of the Gfloat format")
      ("bias", po::value<int>(&bias)->default_value(15),
       "Exponent bias of the Gfloat format")
      ("enable-denorms",
       po::value<bool>(&enableDenorms)->default_value(true),
       "Enable Denorms")
      ("enable-infs-and-nans",
       po::value<bool>(&enableInfsAndNans)->default_value(true),
       "Enable Infs and Nans")
      ("calc-type",
       po::value<std::string>(&calcType)->default_value("AUTO"),
       "Native type used for gfloat calculation: AUTO | FP32 | FP16")
      ("storage-type",
       po::value<std::string>(&storeType)->default_value("AUTO"),
       "Storage type used for gfloat format: AUTO | FP32 | FP16")
      ("enable-nanoo-mode",
       po::value<bool>(&enableNanoo)->default_value(true),
       "Propagate Nans, and generate qNan on overflow ")
      ("round-mode",
       po::value<std::string>(&roundMode)->default_value("RZ"),
       "Round mode")
      ("num-sr-bits",
       po::value<unsigned>(&numberSRBits)->default_value(23),
       "Maximum number of prng bits used for stochastic rounding")
      ("in-type", po::value<Type>(&inType)->default_value(poplar::FLOAT),
       "Type of the data")
      ("input-size",
       po::value<unsigned>(&inSize)->default_value(12),
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
    if (deviceType == DeviceType::IpuModel) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      return createTestDevice(deviceType, ipuModel.numIPUs,
                              ipuModel.tilesPerIPU);
    }
  }();

  poplar::Device::createCPUDevice();
  const auto &target = dev.getTarget();
  Graph graph(target);
  experimental::popfloat::addCodelets(graph);
  popops::addCodelets(graph);
  auto gfCastProg = Sequence();

  auto calculationType = convertStringToSpecType(calcType);
  auto rMode = convertStringToRoundType(roundMode, inType, numberSRBits);

  auto gfFormatCfg = GfloatCast::FormatConfig(
      man, exp, bias, enableDenorms, enableInfsAndNans, calculationType);

  // Create input tensor.
  Tensor input = graph.addVariable(inType, {inSize}, "input");
  mapTensorLinearly(graph, input);

  auto hInput = std::unique_ptr<float[]>(new float[inSize]);

  boost::multi_array<double, 1> hostInput(boost::extents[inSize]);

  std::mt19937 randomEngine;
  writeRandomValues(target, inType, hostInput, -5.0, +5.0, randomEngine);
  copy(target, hostInput, inType, hInput.get());

  auto flpCastOut = std::unique_ptr<float[]>(new float[inSize]);

  auto chrCastOut = std::unique_ptr<char[]>(new char[inSize]);
  auto shrCastOut = std::unique_ptr<short[]>(new short[inSize]);

  auto flpUnpackOut = std::unique_ptr<float[]>(new float[inSize]);

  // Create stream for input data
  auto inStreamV = graph.addHostToDeviceFIFO("InputVector", inType, inSize);

  gfCastProg = Sequence(Copy(inStreamV, input));

  SpecType gfStorageType = convertStringToSpecType(storeType);

  auto roundCfg = GfloatCast::RoundConfig(rMode, numberSRBits,
                                          gfFormatCfg.getCalculationType());
  bool enableNanooMode = enableNanoo && enableInfsAndNans && (exp > 0);
  auto gfCast = GfloatCast(gfFormatCfg, roundCfg, enableNanooMode,
                           gfStorageType, calculationType);

  gfCast.createCastOpParamsTensor(graph, gfCastProg);

  auto gfCastOutput = gfCast.castNativeToGfloat(graph, input, gfCastProg);

  // Create stream for pack output
  auto castOutStream = graph.addDeviceToHostFIFO(
      "CastOutputStream", gfCast.getGFStorageType(), inSize);

  graph.createHostRead("castOutput", gfCastOutput);

  if (!gfCast.getStoreAsNative()) {
    auto unpackOutput =
        gfCast.castGfloatToNative(graph, gfCastOutput, gfCastProg);
    graph.createHostRead("unpackOut", unpackOutput);
  }

  Engine engine(graph, gfCastProg,
                OptionFlags{{"target.workerStackSizeInBytes", "0x8000"},
                            {"debug.allowOutOfMemory", "true"},
                            {"debug.outputAllSymbols", "true"},
                            {"debug.instrumentCompute", "true"},
                            {"prng.enable", prng ? "true" : "false"},
                            {"prng.seed", std::to_string(seed)}});

  engine.connectStream(inStreamV, hInput.get());

  if (!gfCast.getStoreAsNative()) {
    if (gfFormatCfg.getStorageType() == poplar::CHAR) {
      engine.connectStream(castOutStream, chrCastOut.get());
    } else if (gfFormatCfg.getStorageType() == poplar::SHORT) {
      engine.connectStream(castOutStream, shrCastOut.get());
    } else {
      std::cout << "packOutType not valid" << std::endl;
    }
  }

  if (gfCast.getGFStorageType() == poplar::FLOAT) {
    dev.bind([&](const Device &d) {
      engine.load(d);
      engine.run();
      readAndConvertTensor<float, false>(
          graph.getTarget(), engine, "castOutput", flpCastOut.get(), inSize);
    });
  } else if (gfCast.getGFStorageType() == poplar::HALF) {
    dev.bind([&](const Device &d) {
      engine.load(d);
      engine.run();
      readAndConvertTensor<float, true>(graph.getTarget(), engine, "castOutput",
                                        flpCastOut.get(), inSize);
    });
  } else if (gfCast.getGFStorageType() == poplar::SHORT) {
    dev.bind([&](const Device &d) {
      engine.load(d);
      engine.run();
      readAndConvertTensor<short, false>(
          graph.getTarget(), engine, "castOutput", shrCastOut.get(), inSize);
      readAndConvertTensor<float, false>(graph.getTarget(), engine, "unpackOut",
                                         flpUnpackOut.get(), inSize);
    });
  } else if (gfCast.getGFStorageType() == poplar::CHAR) {
    if (gfCast.getCalculationType() == poplar::FLOAT) {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.run();
        readAndConvertTensor<char, false>(
            graph.getTarget(), engine, "castOutput", chrCastOut.get(), inSize);
        readAndConvertTensor<float, false>(
            graph.getTarget(), engine, "unpackOut", flpUnpackOut.get(), inSize);
      });
    } else if (gfCast.getCalculationType() == poplar::HALF) {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.run();
        readAndConvertTensor<char, false>(
            graph.getTarget(), engine, "castOutput", chrCastOut.get(), inSize);
        readAndConvertTensor<float, true>(
            graph.getTarget(), engine, "unpackOut", flpUnpackOut.get(), inSize);
      });
    }
  }

  if (vm.count("profile")) {
    auto reportOptions =
        OptionFlags{{"showExecutionSteps", "true"}, {"showVarStorage", "true"}};

    engine.printProfileSummary(std::cout, reportOptions);
  }

  bool pass = true;

  if (!gfCast.getStoreAsNative()) {
    if (gfCast.getGFStorageType() == poplar::SHORT) {
      auto nativeToGFConfig = gfCast.getNativeToGFConfig();
      pass = castNativeToGfloatCheck<short, true>(
          hInput.get(), shrCastOut.get(), inSize, gfFormatCfg,
          nativeToGFConfig);

      if (!pass) {
        std::cout << "castToGfloatCheck failed" << std::endl;
      }

      bool unpackCheck = castGfloatToNativeCheck<short>(
          shrCastOut.get(), flpUnpackOut.get(), inSize, gfFormatCfg,
          gfCast.getGFStorageType());

      pass &= unpackCheck;
      if (!unpackCheck) {
        std::cout << "castFromGfloatCheck failed" << std::endl;
      }
    } else if (gfCast.getGFStorageType() == poplar::CHAR) {
      auto nativeToGFConfig = gfCast.getNativeToGFConfig();
      pass = castNativeToGfloatCheck<char, true>(hInput.get(), chrCastOut.get(),
                                                 inSize, gfFormatCfg,
                                                 nativeToGFConfig);

      if (!pass) {
        std::cout << "castToGfloatCheck failed" << std::endl;
      }

      bool unpackCheck = castGfloatToNativeCheck<char>(
          chrCastOut.get(), flpUnpackOut.get(), inSize, gfFormatCfg,
          gfCast.getGFStorageType());

      pass &= unpackCheck;
      if (!unpackCheck) {
        std::cout << "castFromGfloatCheck failed" << std::endl;
      }
    } else {
      std::cout << "Cast output type (" << gfCast.getGFStorageType()
                << ") not valid" << std::endl;
    }
  } else {
    auto nativeToGFConfig = gfCast.getNativeToGFConfig();
    pass = castNativeToGfloatCheck<float, false>(
        hInput.get(), flpCastOut.get(), inSize, gfFormatCfg, nativeToGFConfig);

    if (!pass) {
      std::cout << "castToGfloatCheck failed" << std::endl;
    }
  }
  if (pass) {
    std::cout << "Test Passed\n";
  }
  return !pass;
}
