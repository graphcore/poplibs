// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the Non-Linearity Gradient Operations
// Used to verify the accuracy of Non-Linearity backward processing functions
// over any part of the Half-Precision range for both inputs.

#include <TestDevice.hpp>
#include <poplar/CSRFunctions.hpp>
#include <poplar/Engine.hpp>
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

const poplar::OptionFlags options{{"debug.executionProfile", "compute_sets"}};

#define TOL 0.1 // tolerance of 0.1%
#define FLOAT_ATOL 1e-20
#define HALF_ATOL 1e-7

// The number of activation sweeps executed per tile
#define MAX_ACT_SWEEP_PER_TILE 512

// The largest number that can be repreesented in half-precision
#define HALF_0_TO_6 (0x45ff)

//*****************************************************************************
// Execute the Non-Linearity Gradient function over every possible range of
// valid Half-Precision value, and optionally stream the function output to
// stdout.
//
// The Non-Linearity Gradient function is a function of two tensors:
//    1. Tensor of activation inputs
//    2. Tensor of Gradient Outputs
//
// One sweep of the activation inputs is performed for every Gradient output
// value. Alternatively it is also possible to decimate the range of Gradient
// outputs used. In order to speed up execution, multiple activation sweeps
// can be packed into a single Tensor, with the following caveats/restrictions:
//   - The number of gradient output points in a single tensor must be a
//     power of 2.
//   - Every execution can only process the same multiple of activation sweeps.
//     Therefore if there are not sufficient activation sweeps left over for
//     the final execution, the final execution will be abandoned.
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
//   - The elements in each row are in increasing order with unit decimation.
//   - Each row corresponds to one particular Gradient Output code.
//   - The rows are arranged in decreasing order of Gradient Outputs, beginning
//     at the upper limit and proceeding to as close to the lower limit as
//     possible.
//
//*****************************************************************************
bool doNonLinearityGradSweep(const DeviceType &deviceType,
                             const unsigned ntiles, NonLinearityType nlType,
                             const unsigned activation_max_range,
                             const unsigned gradout_lower_code,
                             const unsigned gradout_upper_code,
                             const unsigned stride_vertex,
                             const unsigned stride_delta, bool doPrintTensors,
                             bool ignoreData) {

  unsigned sweep_activation = (activation_max_range + 1) * 2;
  unsigned num_elements = sweep_activation;

  auto invertHalfCodeSign = [](const unsigned code) { return code ^ 0x8000; };

  auto halfPrecisionToIndex = [invertHalfCodeSign](const unsigned code) {
    int index = ((code & (1 << 15)) ? (-invertHalfCodeSign(code) - 1) : code);
    return index;
  };

  auto indexToHalfPrecision = [invertHalfCodeSign](const int index) {
    unsigned code = ((index >= 0) ? index : invertHalfCodeSign((-index) - 1));
    return code;
  };

  int sweep_upper = halfPrecisionToIndex(gradout_upper_code);
  int sweep_lower = halfPrecisionToIndex(gradout_lower_code);
  if (sweep_lower >= sweep_upper) {
    throw std::invalid_argument("range-hi must be greater than range-lo!");
  }

  const std::size_t xSize = num_elements * stride_vertex;

  auto device = createTestDevice(deviceType, 1, ntiles);
  auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);

  // layer parameters
  auto actH = graph.addVariable(HALF, {xSize}, "actH");
  auto deltaH = graph.addVariable(HALF, {xSize}, "deltaH");

  // arbitrary mappings
  mapTensorLinearly(graph, actH);
  mapTensorLinearly(graph, deltaH);

  graph.createHostWrite("inH", actH);
  graph.createHostWrite("inDeltaH", deltaH);
  graph.createHostRead("outDeltaH", deltaH);

  //
  // Sweep_Full the full range of valid half values
  //
  // Exclude the following values:
  //   o     NaN: Exponent=0b11111 && Fraction=0
  //   o  +/-Inf: Exponent=0b11111 && Fraction!=0
  //
  // The set of valid half codes is the set of all values with the
  // exponent in the range [0b00000 - 0b11110]
  //
  std::vector<uint16_t> rawHActInH(xSize);
  std::vector<uint16_t> rawHDeltaInH(xSize);
  std::vector<uint16_t> rawHDeltaOutH(xSize);

  for (unsigned block = 0; block < stride_vertex; block++) {
    for (unsigned col = 0; (col < sweep_activation / 2) && (col < num_elements);
         col++) {
      rawHActInH[(block * sweep_activation) + col] =
          (((sweep_activation / 2) - 1 - col) | (1 << 15));
    }

    for (unsigned col = 0; (col < sweep_activation / 2) &&
                           ((col + (sweep_activation / 2)) < num_elements);
         col++) {
      rawHActInH[(block * sweep_activation) + (sweep_activation / 2) + col] =
          col;
    }
  }

  // reference results calculated in harness
  const auto batchSize = 1;
  boost::multi_array<double, 2> hRefActIn(boost::extents[batchSize][xSize]),
      hRefDeltaOut(boost::extents[batchSize][xSize]),
      hDeltaOutH(boost::extents[batchSize][xSize]);

  // Save the inputs
  poplar::copyDeviceHalfToDouble(target, rawHActInH.data(), &hRefActIn[0][0],
                                 xSize);

  int stride = stride_vertex * stride_delta;
  unsigned row_code = 0;
  unsigned total_rows = [](const unsigned a, const unsigned b) {
    return (a / b) * b;
  }((sweep_upper - sweep_lower), stride);
  for (int tile_row = sweep_upper; tile_row >= (sweep_lower + stride);
       tile_row -= stride) {

    std::cerr << std::hex << "GradOut:" << row_code << std::dec << "; Progress "
              << (sweep_upper - tile_row) << "/" << total_rows << "\n";

    // Construct activations within limited ranged, one stride_vertex of column
    // sweeps at a time
    for (unsigned block = 0; block < stride_vertex; block++) {
      for (unsigned col = 0; col < num_elements; col++) {
        int row = tile_row - (block * stride_delta);
        row_code = indexToHalfPrecision(row);
        rawHDeltaInH[(block * sweep_activation) + col] = row_code;
      }
    }

    // Obtain backward gradient calculation for reference
    poplar::copyDeviceHalfToDouble(target, rawHDeltaInH.data(),
                                   &hRefDeltaOut[0][0], xSize);
    poplibs_test::bwdNonLinearity(nlType, hRefActIn, hRefDeltaOut);

    // build and run the target code
    auto bwdProg = Sequence();

    //
    // Exceptions:
    //
    //    - Invalid Instruction    Enabled
    //    - Divide by 0            Enabled
    //    - Overflow               Disabled  // Overflow Expected
    //    - Stochastic Rounding    Disabled  // For deterministic results
    //    - NANOO                  Disabled  // Overflow Expected
    //
    setFloatingPointBehaviour(graph, bwdProg, {true, true, false, false, false},
                              "Set");

    auto deltaHH =
        nonLinearityInputGradient(graph, nlType, actH, deltaH, bwdProg);
    bwdProg.add(Copy(deltaHH, deltaH));
    Engine bwdEng(graph, bwdProg);
    device.bind([&](const Device &d) {
      bwdEng.load(d);
      bwdEng.writeTensor("inH", rawHActInH.data(),
                         rawHActInH.data() + rawHActInH.size());
      bwdEng.writeTensor("inDeltaH", rawHDeltaInH.data(),
                         rawHDeltaInH.data() + rawHDeltaInH.size());
      bwdEng.run();
      bwdEng.readTensor("outDeltaH", rawHDeltaOutH.data(),
                        rawHDeltaOutH.data() + rawHDeltaOutH.size());
    });

    poplar::copyDeviceHalfToDouble(target, rawHDeltaOutH.data(),
                                   &hDeltaOutH[0][0], xSize);

    /* Print output as a matrix */
    if (doPrintTensors) {
      for (auto block = 0U; block < xSize; block += sweep_activation) {
        unsigned i;
        for (i = 0U; i < num_elements - 1; ++i) {
          std::cout << hDeltaOutH[0][block + i] << ",";
        }
        std::cout << hDeltaOutH[0][block + i] << "\n";
      }
    }

    /* Check result */
    if (!ignoreData &&
        !checkIsClose("outDeltaH", hDeltaOutH, hRefDeltaOut, TOL, HALF_ATOL)) {
      return false;
    }
  }

  return true;
}

//******************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type dataType;

  bool doPrintTensors = false;
  bool ignoreData = false;
  unsigned tiles = 1;
  unsigned stride = 1;
  std::string act_max_code_string;
  unsigned act_max_code = HALF_0_TO_6;
  NonLinearityType nlType;
  unsigned gradout_upper_code = 0x3C00;
  unsigned gradout_lower_code = 0x3BFF;
  std::string range_lo;
  std::string range_hi;

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
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use")
    ("nl-type",
     po::value<NonLinearityType>(&nlType)->required(),
     "Non-linearity type")
    ("act-max-code",
     po::value<std::string>(&act_max_code_string)
         ->value_name(act_max_code_string),
     "The limit of the sweep of activations, as the Half-precision limit")
    ("stride",
     po::value<unsigned>(&stride)->default_value(stride),
     "Sweep Interval")
    ("range-lo",
     po::value<std::string>(&range_lo)->value_name("range-lo"),
     "Lower range of the sweep")
    ("range-hi",
     po::value<std::string>(&range_hi)->value_name("range-hi"),
     "Upper range of the sweep");
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

  if (stride & (stride - 1)) {
    throw std::invalid_argument("stride must be a power of 2!");
  }

  if (stride > MAX_ACT_SWEEP_PER_TILE) {
    throw std::invalid_argument("stride exceeds maximum!");
  }

  if (vm.count("act-max-code")) {
    act_max_code = std::stoul(act_max_code_string, nullptr, 0);
  }

  if (vm.count("range-lo")) {
    gradout_lower_code = std::stoul(range_lo, nullptr, 0);
  }

  if (vm.count("range-hi")) {
    gradout_upper_code = std::stoul(range_hi, nullptr, 0);
  }

  unsigned stride_vertex = (gradout_upper_code - gradout_lower_code) / stride;
  if (stride_vertex > MAX_ACT_SWEEP_PER_TILE) {
    stride_vertex = MAX_ACT_SWEEP_PER_TILE;
  }

  std::cerr << "Input arguments:"
            << "\n";
  std::cerr << "  Activation range-lo = 0x" << std::hex
            << [](const unsigned code) { return code ^ 0x8000; }(act_max_code)
            << std::dec << "\n";

  std::cerr << "  Activation range-hi = 0x" << std::hex << act_max_code
            << std::dec << "\n";
  std::cerr << "  GradOut range-lo = 0x" << std::hex << gradout_lower_code
            << std::dec << "\n";
  std::cerr << "  GradOut range-hi = 0x" << std::hex << gradout_upper_code
            << std::dec << "\n";
  std::cerr << "  GradOut decimation factor = " << stride << "\n";
  std::cerr << "  Activation Sweeps per Vertex = " << stride_vertex << "\n\n";

  return !doNonLinearityGradSweep(
      deviceType, tiles, nlType, act_max_code, gradout_lower_code,
      gradout_upper_code, stride_vertex, stride, doPrintTensors, ignoreData);
}
