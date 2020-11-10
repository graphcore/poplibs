// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvOptionsTest

#include "poputil/TileMapping.hpp"

#include <boost/test/unit_test.hpp>
#include <popfloat/experimental/CastToGfloat.hpp>
#include <popfloat/experimental/CastToHalf.hpp>
#include <popfloat/experimental/codelets.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>

#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>

#include <stdio.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <poplar/DebugContext.hpp>

#if defined(__clang__)
#define SUPPORTS_FUNCTION_BUILTINS __has_builtin(__builtin_FUNCTION)
#elif __GNUC__ >= 7
#define SUPPORTS_FUNCTION_BUILTINS 1
#else
#define SUPPORTS_FUNCTION_BUILTINS 0
#endif

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace popfloat::experimental;
using namespace poputil;
using namespace poplibs_support;

const int manSizeFp32 = 23;
const int manSizeFp16 = 10;

popfloat::experimental::SpecType
convertStringToSpecType(const std::string &specType) {
  if (specType == "AUTO") {
    return popfloat::experimental::SpecType::AUTO;
  } else if (specType == "FP32") {
    return popfloat::experimental::SpecType::FP32;
  } else if (specType == "FP16") {
    return popfloat::experimental::SpecType::FP16;
  } else if (specType == "INT8") {
    return popfloat::experimental::SpecType::INT8;
  } else if (specType == "INT16") {
    return popfloat::experimental::SpecType::INT16;
  }
  throw poputil::poplibs_error("Type not supported");
}

popfloat::experimental::RoundType
convertStringToRoundType(const std::string &roundMode, poplar::Type inType,
                         unsigned srBits) {
  if (roundMode == "RZ") {
    return popfloat::experimental::RoundType::RZ;
  } else if (roundMode == "RN") {
    return popfloat::experimental::RoundType::RN;
  } else if (roundMode == "RA") {
    return popfloat::experimental::RoundType::RA;
  } else if (roundMode == "RU") {
    return popfloat::experimental::RoundType::RU;
  } else if (roundMode == "RD") {
    return popfloat::experimental::RoundType::RD;
  } else if (roundMode == "SR") {
    bool isExtendedSr =
        srBits <
        unsigned((inType == poplar::FLOAT) ? manSizeFp32 : manSizeFp16);
    if (isExtendedSr) {
      return popfloat::experimental::RoundType::SX;
    } else {
      return popfloat::experimental::RoundType::SR;
    }
  }
  throw poputil::poplibs_error("Round Mode not supported");
}

BOOST_AUTO_TEST_CASE(DebugInfoTest) {

  unsigned man = 10;
  unsigned exp = 15;
  int bias = 15;
  bool enableDenorms = true;
  bool enableInfsAndNans = true;
  std::string roundMode = "RZ";
  std::string calcType = "AUTO";
  std::string storeType = "AUTO";
  unsigned numberSRBits = 23;
  Type inType = FLOAT;
  unsigned inSize = 12;
  DeviceType deviceType = DeviceType::Cpu;
  boost::optional<unsigned> tilesPerIPU;

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

  std::string filename = "popfloat_debuginfotest.json";

  poplar::DebugInfo::initializeStreamer(filename,
                                        poplar::DebugSerializationFormat::JSON);

  poplar::Device::createCPUDevice();
  const auto &target = dev.getTarget();
  Graph graph(target);
  popfloat::experimental::addCodelets(graph);
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
  bool enableNanooMode = true;
  auto gfCast = GfloatCast(gfFormatCfg, roundCfg, enableNanooMode,
                           gfStorageType, calculationType);

  gfCast.createCastOpParamsTensor(graph, gfCastProg);

  auto gfCastOutput = gfCast.castNativeToGfloat(graph, input, gfCastProg);

  poplar::DebugInfo::closeStreamer();

  using namespace boost::property_tree;
  ptree pt;
  json_parser::read_json(filename, pt);

  bool createCastOpParamsTensorDiExist = false;
  for (auto &v : pt.get_child("contexts")) {
    auto api = v.second.get_optional<std::string>("api");
    if (api) {
      if (*api == "createCastOpParamsTensor") {
        createCastOpParamsTensorDiExist = true;
      }
    }
  }

// Can only check the function name is correct if the os supports
// __builtin_Function() This is not supported on macos
#if SUPPORTS_FUNCTION_BUILTINS
  BOOST_CHECK(createCastOpParamsTensorDiExist == true);
#endif

  remove(filename.c_str());
}
