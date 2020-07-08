// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NonLinearitySweepTest
// Test for the Non-Linearity Forward Operations
// Used to verify the accuracy of Non-Linearity forward processing functions
// over the full Half-Precision range.

#include <poplar/CSRFunctions.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include "popops/ElementWise.hpp"
#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDefUtil.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/program_options.hpp>

#include <exception>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;
using namespace popnn;
using namespace poplibs_support;

const poplar::OptionFlags options{{"debug.executionProfile", "compute_sets"}};

#define TOL 0.1 // tolerance of 0.1%
#define FLOAT_ATOL 1e-20
#define HALF_ATOL 1e-7

// The largest number that can be repreesented in half-precision
#define HALF_MAX_CODE 0x7bff

//*****************************************************************************
// Execute the Non-Linearity function over every possible valid Half-Precision
// value and optionally stream the function output to stdout.
//
// The Non-Linearity function is a function of single tensor of activations.
//
// Exclude the following values:
//   o     NaN: Exponent=0b11111 && Fraction=0
//   o  +/-Inf: Exponent=0b11111 && Fraction!=0
//
// The set of valid half-precision codes with the exponent in the range
// [0b00000 - 0b11110]. The full range can be defined by sweeping across the
// following ranges:
//   o codes from 0 to 0b0111101111111111 (0x7bff) and
//   o codes from -0 to 0b1111101111111111 (-0x7bff).
//
// The constant HALF_MAX_CODE is defined as 0x7bff. The full range of inputs is
// stored in the tensor in the following order:
//
//    Index into Tensor       Half-Precision Value
//    -----------------       --------------------
//          0                  -HALF_MAX_CODE
//          1                  -(HALF_MAX_CODE-1)
//          .                        .
//          .                        .
//          .                        .
//    HALF_MAX_CODE                 -0
//    HALF_MAX_CODE+1               +0
//          .                        .
//          .                        .
//          .                        .
//   (HALF_MAX_CODE+1)*2-1      HALF_MAX_CODE
//
//
// Format of the CSV output:
//   - A single row in increasing order of activation-input code.
//
//*****************************************************************************
bool doNonLinearitySweep(const DeviceType &deviceType, NonLinearityType nlType,
                         bool doPrintTensors, bool ignoreData) {
  auto device = createTestDevice(deviceType, 1, 1);
  auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);

  // Layer parameters
  const std::size_t xSize = (HALF_MAX_CODE + 1) * 2;
  auto actH = graph.addVariable(HALF, {xSize}, "actH");

  // Arbitrary mappings
  mapTensorLinearly(graph, actH);

  graph.createHostWrite("inH", actH);
  graph.createHostRead("outH", actH);

  // Sweep the full range of valid half values
  std::vector<uint16_t> rawHActInH(xSize);
  std::vector<uint16_t> rawHActOutH(xSize);
  for (std::size_t i = 0; i < xSize / 2; i++) {
    rawHActInH[(xSize / 2) + i] = i;
    rawHActInH[(xSize / 2) - i - 1] = (i + (1 << 15));
  }

  // reference results calculated in harness
  const auto batchSize = 1;
  boost::multi_array<double, 2> hRefActOut(boost::extents[batchSize][xSize]),
      hRefActIn(boost::extents[batchSize][xSize]),
      hActOutH(boost::extents[batchSize][xSize]);

  // Check forward activation calculation
  poplar::copyDeviceHalfToDouble(target, rawHActInH.data(), &hRefActOut[0][0],
                                 xSize);
  poplibs_test::nonLinearity(nlType, hRefActOut);

  std::ostringstream buffer;

  // Build and run the target code
  auto fwdProg = Sequence();

  //
  // Exceptions:
  //
  //    - Invalid Instruction    Enabled
  //    - Divide by 0            Enabled
  //    - Overflow               Enabled
  //    - Stochastic Rounding    Enabled
  //    - NANO                   Enabled
  //
  setFloatingPointBehaviour(graph, fwdProg, {true, true, true, true, true},
                            "Set");

  nonLinearityInPlace(graph, nlType, actH, fwdProg);
  Engine fwdEng(graph, fwdProg);
  device.bind([&](const Device &d) {
    fwdEng.load(d);
    fwdEng.writeTensor("inH", rawHActInH.data(),
                       rawHActInH.data() + rawHActInH.size());
    fwdEng.run();
    fwdEng.readTensor("outH", rawHActOutH.data(),
                      rawHActOutH.data() + rawHActOutH.size());
  });

  poplar::copyDeviceHalfToDouble(target, rawHActOutH.data(), &hActOutH[0][0],
                                 xSize);

  /* Print output tensor */
  if (doPrintTensors) {
    for (auto i = 0U; i < xSize - 1; ++i) {
      std::cout << hActOutH[0][i] << ", ";
    }
    std::cout << hActOutH[0][xSize - 1] << "\n";
  }

  /* Check result */
  if (!ignoreData) {
    return checkIsClose("outH", hActOutH, hRefActOut, TOL, HALF_ATOL);
  }

  return true;
}

//******************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type dataType;
  NonLinearityType nlType;
  bool doPrintTensors = false;
  bool ignoreData = false;

  po::options_description desc("Options");

  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("print",
     po::value<bool>(&doPrintTensors)->default_value(doPrintTensors),
     "Print the tensors")
    ("ignoreCheck",
     po::value<bool>(&ignoreData)->default_value(ignoreData),
     "Ignore mismatches from the expected results")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("nl-type",
     po::value<NonLinearityType>(&nlType)->required(),
     "Non-linearity type");
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return !doNonLinearitySweep(deviceType, nlType, doPrintTensors, ignoreData);
}
