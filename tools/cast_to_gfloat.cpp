// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <poplar/Engine.hpp>
#include "poputil/VertexTemplates.hpp"
#include <memory>
#include <iostream>
#include <random>
#include <iomanip>
#include <poplar/IeeeHalf.hpp>
#include <popfloat/CastToGfloat.hpp>
#include <popfloat/codelets.hpp>
#include "cast_to_gfloat.hpp"
#include "poputil/TileMapping.hpp"

#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_support/Compiler.hpp>
#include "TestDevice.hpp"
#include "../lib/popfloat/codelets/popfloatParamUtils.hpp"

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

popfloat::gfexpr::GfloatRoundType
convertStringToGfloatRoundType(const std::string &roundMode,
                               Type inType, unsigned srBits) {
  if (roundMode == "RZ") {
    return GfloatRoundType::RZ;
  } else if (roundMode == "RN") {
    return GfloatRoundType::RN;
  } else if (roundMode == "RA") {
    return GfloatRoundType::RA;
  } else if (roundMode == "RU") {
    return GfloatRoundType::RU;
  } else if (roundMode == "RD") {
    return GfloatRoundType::RD;
  } else if (roundMode == "SR") {
    bool isExtendedSr = srBits < ((inType == FLOAT) ?
                                  POPFLOAT_NUM_FP32_MANTISSA_BITS :
                                  POPFLOAT_NUM_FP16_MANTISSA_BITS);
    if (isExtendedSr) {
      return GfloatRoundType::SX;
    } else {
      return GfloatRoundType::SR;
    }
  }
  throw poputil::poplibs_error("Round Mode not supported");
}

bool quantiseGfloatCheck(float *inVec, float *outVec, unsigned sizeVec,
                         GfloatFormatConfig &gfFormatCfg,
                         GfloatCastConfig &gfCastCfg) {
  int32_t minExp = 1 - gfFormatCfg.getExponentBias();
  if (gfFormatCfg.isDenormEnabled()) {
    minExp -= gfFormatCfg.getNumMantissaBits();
  }
  auto quantisedOpType = gfFormatCfg.getQuantisedOpType();
  auto formatType = gfFormatCfg.getFormatType();

  if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    minExp =
      gfFormatCfg.isDenormEnabled() ?
      -(14 + gfFormatCfg.getNumMantissaBits()) : -14;
  }

  float minValue = std::pow(2.0, minExp);

  int32_t maxExp =
    (1 << gfFormatCfg.getNumExponentBits()) - 1 - gfFormatCfg.getExponentBias();
  if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
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
    maxValue *= std::pow(2.0, maxExp - gfFormatCfg.getNumMantissaBits());
  }

  float scale = 1.0;

  if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    if ((formatType == GfloatFormatType::MAX_NORM_ALIGN_GF8) ||
        (!gfFormatCfg.infAndNansEnabled() &&
         (gfFormatCfg.getNumExponentBits() == expSizeFp16))) {
      scale = std::pow(2.0, gfFormatCfg.getExponentBias() - 16);
    } else {
      scale = std::pow(2.0, gfFormatCfg.getExponentBias() - 15);
    }
  }

  bool pass = true;

  int32_t minNormExp = 1 - gfFormatCfg.getExponentBias();
  if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    minNormExp = 1 - expBiasFp16;
  }

  int32_t qNan = qnanFp32;
  if (gfFormatCfg.getQuantisedOutputType() == HALF) {
    float _qnan = float(poplar::IeeeHalf::fromBits(qnanFp16));
    std::memcpy(&qNan, &_qnan, sizeof(qNan));
  }

  //Quantised FP16 clip input before scaling if enNanoo is set to false
  uint16_t maxBits = 0x7BFF;
  if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    maxBits >>= (manSizeFp16 - gfFormatCfg.getNumMantissaBits());
    maxBits <<= (manSizeFp16 - gfFormatCfg.getNumMantissaBits());
  }
  float maxAbs = float(poplar::IeeeHalf::fromBits(maxBits));
  for (unsigned j = 0; j != sizeVec; ++j) {
    float input = inVec[j] * scale;
    int32_t inBits;
    if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
      if (!gfCastCfg.isNanooModeEnabled()) {
        if (std::abs(input) > maxAbs) {
          input = (input > 0) ? maxAbs : (-1.0 * maxAbs);
        }
      }
      auto inBits16 = floatToHalf(input, gfCastCfg.isNanooModeEnabled());
      input = float(poplar::IeeeHalf::fromBits(inBits16));
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
    masklen  = (minNormExp - e_single);
    masklen  = (masklen < 0) ? 0 : masklen;
    masklen += manSizeFp32 - gfFormatCfg.getNumMantissaBits();
    masklen  = std::min<uint32_t>(masklen, manSizeFp32+1);

    if ((quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) &&
        (std::abs(input) > maxAbs)) {
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == GfloatRoundType::RZ) {
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == GfloatRoundType::RN) {
      bool msfBitVal = (m_single >> (masklen - 1)) & 1;
      bool lsbs = (m_single & ((1 << (masklen - 1)) - 1)) != 0;
      bool lsBitVal = (m_single >> masklen) & 1;
      m_single = (m_single >> masklen);
      if (msfBitVal && (lsBitVal || lsbs)) {
        m_single += 1;
      }
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == GfloatRoundType::RA) {
      m_single = m_single + ((1 << masklen) >> 1);
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == GfloatRoundType::RU) {
      uint32_t corr = (s_single == 0) ? ((1 << masklen) - 1) : 0;
      m_single = m_single + corr;
      m_single = m_single >> masklen;
      m_single = m_single << masklen;
    } else if (gfCastCfg.getRoundMode() == GfloatRoundType::RD) {
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
        if ((gfFormatCfg.getQuantisedOutputType() == FLOAT)
            && (inVec[j] < 0)) {
          inBits |= (1 << 31);
        }
      } else {
        std::memcpy(&inBits, &maxValue, sizeof(inBits));
        inBits |= s_single;
      }
    } else {
      inBits |= s_single;
    }

    int32_t outBits;
    std::memcpy(&outBits, &outVec[j], sizeof(outBits));
    pass &= (outBits == inBits);
  }
  return pass;
}

template<typename T>
bool packGfloatCheck(float *inVec, float *qntVec, T *outVec, unsigned sizeVec,
                     GfloatFormatConfig  &gfFormatCfg) {
  unsigned fpSize = gfFormatCfg.getPackedFloatBits() - 1;
  unsigned manSize = fpSize - gfFormatCfg.getNumExponentBits();

  unsigned manExpMask = (1 << fpSize) - 1;

  unsigned fpMask = (1 << (fpSize + 1)) - 1;
  unsigned alignShr = manSizeFp32 - manSize;

  auto quantisedOpType = gfFormatCfg.getQuantisedOpType();
  auto formatType = gfFormatCfg.getFormatType();
  int32_t expBias = gfFormatCfg.getExponentBias();
  if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    expBias = (formatType == GfloatFormatType::MAX_NORM_ALIGN_GF8) ?
      (expBiasFp16 + 1) : expBiasFp16;
  }
  int32_t minNormExp = 1 - expBias;

  bool pass = true;
  int32_t inBits;
  for (unsigned j = 0; j != sizeVec; ++j) {
    std::memcpy(&inBits, &qntVec[j], sizeof(inBits));

    int32_t m_single = (inBits & manMaskFp32) >> alignShr;
    int32_t e_single = ((inBits & expMaskFp32) >> manSizeFp32) - expBiasFp32;
    int32_t s_single = (inBits & sgnMaskFp32) >> (31 - fpSize);

    int32_t outBits = 0;

    if (e_single > expBiasFp32) {
      if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF32) {
        outBits = ((qnanFp32 >> alignShr) & manExpMask) | s_single;
      } else {
        outBits = ((qnanFp16 >> (manSizeFp16 - manSize)) & manExpMask);
      }
    } else if (qntVec[j] != 0.0) {
      if (e_single < minNormExp) {
        m_single |= (1 << manSize);
        m_single >>= (minNormExp - e_single);
        e_single = 0;
      } else {
        e_single += expBias;
      }
      e_single = e_single << manSize;

      outBits = m_single | e_single | s_single;
    } else {
      if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF32) {
        outBits = s_single;
      }
    }

    pass &= ((outBits & fpMask) == ((int32_t)outVec[j] & fpMask));
  }
  return pass;
}

template<typename T>
bool unpackGfloatCheck(T *inVec, float *outVec, unsigned sizeVec,
                       GfloatFormatConfig  &gfFormatCfg) {
  unsigned fpSize = gfFormatCfg.getPackedFloatBits() - 1;
  unsigned manSize = fpSize - gfFormatCfg.getNumExponentBits();

  unsigned alignShr = manSizeFp32 - manSize;

  int32_t manMask = (1 << manSize) - 1;
  int32_t maxExp =
    (1 << gfFormatCfg.getNumExponentBits()) - gfFormatCfg.infAndNansEnabled();
  int32_t expMask = ((1 << gfFormatCfg.getNumExponentBits()) - 1) << manSize;
  int32_t sgnMask = 1 << (manSize + gfFormatCfg.getNumExponentBits());

  int32_t expBias = gfFormatCfg.getExponentBias();
  int32_t qNan = qnanFp32;

  auto quantisedOpType = gfFormatCfg.getQuantisedOpType();
  auto formatType = gfFormatCfg.getFormatType();
  if (quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    expBias = (formatType == GfloatFormatType::MAX_NORM_ALIGN_GF8) ?
              (expBiasFp16 + 1) : expBiasFp16;

    float _qnan = float(poplar::IeeeHalf::fromBits(qnanFp16));
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
          if (gfFormatType == GfloatFormatType::ENABLE_DENORM_GF16) {
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

  unsigned    man;
  unsigned    exp;
  int         bias;
  bool        enDenorm;
  bool        enInf;
  bool        enNanoo;
  std::string roundMode;
  unsigned    srBits;
  bool        miniFloat;
  Type        inType = FLOAT;
  unsigned    inSize;
  DeviceType  deviceType = DeviceType::Cpu; //IpuModel;
  IPUModel    ipuModel;
  bool        prng;
  unsigned    seed;

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
      ("prng", po::value<bool>(&prng)->default_value(false), "prng enable")
      ("seed", po::value<unsigned>(&seed)->default_value(12352345), "prng seed")
      ("profile", "Output profiling report")
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
       "Propagate Nans, and generate qNan on overflow ")
      ("roundMode",
       po::value<std::string>(&roundMode)->default_value("RZ"),
       "Round mode")
      ("srBits",
       po::value<unsigned>(&srBits)->default_value(23),
       "Maximum number of prng bits used for stochastic rounding")
      ("inType", po::value<Type>(&inType)->default_value(inType),
       "Type of the data")
      ("miniFloat",
       po::value<bool>(&miniFloat)->default_value(true),
       "Enable saving to smaller floar format is possible")
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
  }();

  poplar::Device::createCPUDevice();
  const auto &target = dev.getTarget();
  Graph graph(target);
  popfloat::addCodelets(graph);
  popsys::addCodelets(graph);
  auto gfCastProg = Sequence();

  auto rMode = convertStringToGfloatRoundType(roundMode, inType, srBits);

  GfloatFormatConfig gfFormatCfg =
    GfloatFormatConfig(man, exp, bias, enDenorm, enInf);
  auto quantisedOpType = gfFormatCfg.getQuantisedOpType();
  if ((quantisedOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF32) &&
      (inType != poplar::FLOAT)) {
    throw poplibs_error(
        "popfloat::lookupGfQuantiseParamOp: Ops expects float input");
  }

  // Create input tensor.
  Tensor input = graph.addVariable(inType, { inSize }, "input");
  mapTensorLinearly(graph, input);

  auto hInput = std::unique_ptr<float[]>(new float[inSize]);

  boost::multi_array<double, 1>
      hostInput(boost::extents[inSize]);

  std::mt19937 randomEngine;
  writeRandomValues(target, inType, hostInput, -5.0, +5.0, randomEngine);
  copy(target, hostInput, inType, hInput.get());

  auto flpCastOut = std::unique_ptr<float[]>(new float[inSize]);

  auto chrPackOut = std::unique_ptr<char[]>(new char[inSize]);
  auto shrPackOut = std::unique_ptr<short[]>(new short[inSize]);

  auto flpUnpackOut = std::unique_ptr<float[]>(new float[inSize]);

  //Create stream for input data
  auto inStreamV = graph.addHostToDeviceFIFO("InputVector",
                                             inType,
                                             inSize);

  //Create stream for pack output
  auto packOutStream =
    graph.addDeviceToHostFIFO("PackOutputStream",
                              gfFormatCfg.getPackedOutputType(),
                              inSize);

  gfCastProg = Sequence(Copy(inStreamV, input));

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
                     rMode,enableNanooMode, srBits);

  auto quantiseOutput = castToGfloat(graph, input, quantiseParams,
                                     gfCastProg, gfQuantiseCfg);
  graph.createHostRead("quantiseOutput", quantiseOutput);

  if (miniFloat && gfFormatCfg.isPackedFloatFormat()) {
    Tensor packParams =
      createCastOpParamsTensor(graph, gfCastProg,
                               gfFormatCfg.getPackOpType(),
                               gfCompressed);

    auto gfPackArgs =
      GfloatCastConfig(gfFormatCfg.getQuantisedOutputType(),
                       gfFormatCfg.getPackedOutputType(),
                       gfFormatCfg.getFormatType());
    Tensor packOutput = castToGfloat(graph, quantiseOutput, packParams,
                                     gfCastProg, gfPackArgs);

    Tensor unpackParams =
      createCastOpParamsTensor(graph, gfCastProg,
                               gfFormatCfg.getUnpackOpType(),
                               gfCompressed);

    auto gfUnpackArgs =
      GfloatCastConfig(gfFormatCfg.getPackedOutputType(),
                       gfFormatCfg.getUnpackedOutputType(),
                       gfFormatCfg.getFormatType());

    Tensor unpackOutput = castToGfloat(graph, packOutput, unpackParams,
                                       gfCastProg, gfUnpackArgs);
    graph.createHostRead("unpackOut", unpackOutput);

    gfCastProg.add(Copy(packOutput, packOutStream));
  }

  Engine engine(graph, gfCastProg, OptionFlags{
      { "target.workerStackSizeInBytes", "0x8000" },
      { "debug.allowOutOfMemory" , "true" },
      { "debug.executionProfile", "compute_sets"},
      { "prng.enable", prng ? "true" : "false" },
      { "prng.seed", std::to_string(seed) }
  });

  engine.connectStream(inStreamV, hInput.get());

  if (miniFloat && gfFormatCfg.isPackedFloatFormat()) {
    if (gfFormatCfg.getPackedOutputType() == poplar::CHAR) {
      engine.connectStream(packOutStream, chrPackOut.get());
    } else if (gfFormatCfg.getPackedOutputType() == poplar::SHORT) {
      engine.connectStream(packOutStream, shrPackOut.get());
    } else {
      std::cout << "packOutType not valid" << std::endl;
    }
  }

  // Run the forward pass.
  if (gfFormatCfg.getQuantisedOutputType() == poplar::FLOAT) {
    if (miniFloat && gfFormatCfg.isPackedFloatFormat()) {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.run();
        readAndConvertTensor<float, false>(
            graph.getTarget(),
            engine,
            "quantiseOutput",
            flpCastOut.get(),
            inSize);
        readAndConvertTensor<float, false>(
            graph.getTarget(),
            engine,
            "unpackOut",
            flpUnpackOut.get(),
            inSize);
      });
    } else {
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
    }
  } else if (gfFormatCfg.getQuantisedOutputType() == poplar::HALF) {
    if (miniFloat && gfFormatCfg.isPackedFloatFormat()) {
      dev.bind([&](const Device &d) {
        engine.load(d);
        engine.run();
        readAndConvertTensor<float, true>(
            graph.getTarget(),
            engine,
            "quantiseOutput",
            flpCastOut.get(),
            inSize);
        readAndConvertTensor<float, true>(
            graph.getTarget(),
            engine,
            "unpackOut",
            flpUnpackOut.get(),
            inSize);
      });
    } else {
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
  }

  if (vm.count("profile")) {
    auto reportOptions = OptionFlags{
      { "showExecutionSteps", "true" },
      { "showVarStorage", "true" } };

    engine.printProfileSummary(std::cout, reportOptions);
  }

  bool pass = true;

  bool quantiseCheck = quantiseGfloatCheck(hInput.get(),
                                           flpCastOut.get(),
                                           inSize,
                                           gfFormatCfg,
                                           gfQuantiseCfg);

  pass &= quantiseCheck;
  if (!quantiseCheck) {
    std::cout << "quantiseGfloatCheck failed" << std::endl;
  }

  if (miniFloat && gfFormatCfg.isPackedFloatFormat()) {
    if ((gfFormatCfg.getQuantisedOutputType() == poplar::FLOAT) &&
        (gfFormatCfg.getPackedOutputType() == poplar::SHORT)) {
      bool packCheck = packGfloatCheck<short>(hInput.get(),
                                              flpCastOut.get(),
                                              shrPackOut.get(),
                                              inSize,
                                              gfFormatCfg);
      pass &= packCheck;
      if (!packCheck) {
        std::cout << "packGfloatCheck failed" << std::endl;
      }

      bool unpackCheck = unpackGfloatCheck<short>(shrPackOut.get(),
                                                  flpUnpackOut.get(),
                                                  inSize,
                                                  gfFormatCfg);
      pass &= unpackCheck;
      if (!unpackCheck) {
        std::cout << "unpackGfloatCheck failed" << std::endl;
      }
    } else if (gfFormatCfg.getPackedOutputType() == poplar::CHAR) {
      bool packCheck = packGfloatCheck<char>(hInput.get(),
                                             flpCastOut.get(),
                                             chrPackOut.get(),
                                             inSize,
                                             gfFormatCfg);
      pass &= packCheck;
      if (!packCheck) {
        std::cout << "packGfloatCheck failed" << std::endl;
      }

      bool unpackCheck = unpackGfloatCheck<char>(chrPackOut.get(),
                                                 flpUnpackOut.get(),
                                                 inSize,
                                                 gfFormatCfg);
      pass &= unpackCheck;
      if (!unpackCheck) {
        std::cout << "unpackGfloatCheck failed" << std::endl;
      }
    } else {
      std::cout << "Pack/Unpack output not valid" << std::endl;
    }
  }
  if (pass) {
    std::cout << "Test Passed\n";
  }
  return !pass;
}
