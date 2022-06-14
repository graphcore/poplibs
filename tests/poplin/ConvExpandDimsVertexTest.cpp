// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvExpandDimsVertexTest

#include "poplin/Convolution.hpp"
#include "poplin/codelets.hpp"
#include "popops/Cast.hpp"
#include "popops/codelets.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"

#include "poplibs_support/PlanConstraints.hpp"
#include "poplibs_support/TestDevice.hpp"
#include "poplibs_support/VectorUtils.hpp"

#include <poplar/Device.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Quarter.hpp>
#include <poplar/Target.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/TypeConversion.hpp>

#include <pva/pva.hpp>

#include <boost/filesystem.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/test/unit_test.hpp>

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <thread>

using namespace poplar;
using namespace poplin;
using namespace poplibs_support;

template <typename T>
std::ostream &operator<<(std::ostream &os, ArrayRef<T> ar) {
  os << '[';
  const char *sep = "";
  for (auto const &item : ar) {
    os << sep << item;
    sep = ", ";
  }
  return os << ']';
}

// RAII wrapper around a temporary directory.
// This will not be cleaned up on signal or if exit is called.
struct TemporaryDirectory {
  TemporaryDirectory() = default;
  TemporaryDirectory(TemporaryDirectory &&) = default;
  TemporaryDirectory(TemporaryDirectory const &) = delete;
  void open() {
    constexpr unsigned maxTries = 10;
    auto tmp = boost::filesystem::temp_directory_path();
    boost::filesystem::path pathAttempt;
    for (unsigned i = 0; i < maxTries; ++i) {
      pathAttempt = tmp / boost::filesystem::unique_path();
      boost::system::error_code ec;
      boost::filesystem::create_directory(pathAttempt, ec);
      if (!ec) {
        path = std::move(pathAttempt);
        return; // success
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    throw poputil::poplibs_error("Could not get temporary directory");
  }
  void close() {
    if (!path.empty()) {
      boost::system::error_code ec;
      boost::filesystem::remove_all(path, ec);
      if (ec) {
        std::cerr << "Failed to remove temporary directory: " << path << "\n";
      }
    }
  }
  ~TemporaryDirectory() { close(); }
  boost::filesystem::path path;
};

template <typename T = float> struct HostTensor {
  std::vector<size_t> shape;
  std::vector<T> data;
  Type ipuDataType;
  HostTensor(Tensor out)
      : shape(out.shape()), data(product(out.shape())),
        ipuDataType(out.elementType()) {}
  HostTensor(decltype(shape) shape_, decltype(data) data_)
      : shape(std::move(shape_)), data(std::move(data_)),
        ipuDataType(equivalent_device_type<T>().value) {
    if (!shape.empty() || !data.empty())
      assert(product(shape) == data.size());
  }
  bool operator==(HostTensor const &other) const noexcept {
    if (shape.size() != other.shape.size() ||
        data.size() != other.data.size() ||
        !std::equal(shape.begin(), shape.end(), other.shape.begin()))
      return false;
    // This is a bit arbitrary but seems good enough to
    // ignore precision errors and catch logic errors.
    const float tolerance = (ipuDataType == HALF) ? 1e-4 : 1e-5;
    // Use almost-equals logic for floats and halves. With perfect precision
    // we should get the same answer but the optimisation changes the order in
    // which the floating-point numbers will be used and thus can be different
    // due to different precision losses.
    return std::equal(data.begin(), data.end(), other.data.begin(),
                      [tolerance](T a, T b) {
                        auto rd = boost::math::relative_difference(a, b);
                        return rd <= tolerance;
                      });
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, HostTensor<T> const &ht) {
  using poplar::ArrayRef;
  return os << "Tensor: shape=" << ArrayRef(ht.shape)
            << ", data=" << ArrayRef(ht.data);
}

const auto quarterMetadata = QuarterMetadata(QuarterMetadata::Format::F143, 0);

enum class FillPattern {
  // Fill the data with the range [1, N]
  IOTA,
  // Set all the data to 0.
  ALL_ZEROES,
  // Set all the data to 1.
  ALL_ONES,
  // Set the data to [1, 16].
  MAX_16,
  // Set the data to 1 or -1 in an alternating pattern.
  MIX_OF_MINUS_ONE_AND_ONE,
};

template <typename Iter>
void patternFill(FillPattern pattern, Iter first, Iter last) {
  switch (pattern) {
  case FillPattern::IOTA:
    std::iota(first, last, 1);
    return;
  case FillPattern::ALL_ZEROES:
    std::fill(first, last, 0);
    return;
  case FillPattern::ALL_ONES:
    std::fill(first, last, 1);
    return;
  case FillPattern::MAX_16:
    for (size_t i = 0; first != last; ++i, ++first)
      *first = (i + 1) % 16;
    return;
  case FillPattern::MIX_OF_MINUS_ONE_AND_ONE:
    for (size_t i = 0; first != last; ++i, ++first)
      *first = (i % 2) ? 1 : -1;
    return;
  }
  throw poplar::poplar_error("Unknown pattern type");
}

template <typename HostValueType>
static void fillValues(Graph &graph, program::Sequence &progs,
                       const ConvParams &params, Tensor &input, Tensor &weights,
                       FillPattern initialValuesStrategy) {
  // Set the initial values of the inputs.
  std::vector<HostValueType> inputData(input.numElements());
  std::vector<HostValueType> weightData(weights.numElements());
  patternFill(initialValuesStrategy, inputData.begin(), inputData.end());
  patternFill(initialValuesStrategy, weightData.begin(), weightData.end());
  if (params.inputType == QUARTER) {
    // Write the values in halves as some functions don't support quarter types.
    Tensor halfInput = graph.clone(HALF, input);
    Tensor halfWeights = graph.clone(HALF, weights);
    graph.setInitialValue<HostValueType>(halfInput, inputData);
    graph.setInitialValue<HostValueType>(halfWeights, weightData);
    // Quarter types require metadata.
    Tensor inputMetadata = poputil::createConstantMetadataTensor(
        graph, quarterMetadata.getFormat(), quarterMetadata.getScale());
    progs.add(program::Copy(inputMetadata, input.getMetadata()));
    Tensor weightsMetadata = poputil::createConstantMetadataTensor(
        graph, quarterMetadata.getFormat(), quarterMetadata.getScale());
    progs.add(program::Copy(weightsMetadata, weights.getMetadata()));
    // Convert the halves (with values) to quarter types.
    auto cs = graph.addComputeSet({"CastToQuarter"});
    popops::cast(graph, halfInput, input, cs);
    popops::cast(graph, halfWeights, weights, cs);
    progs.add(program::Execute(cs));
  } else {
    graph.setInitialValue<HostValueType>(input, inputData);
    graph.setInitialValue<HostValueType>(weights, weightData);
  }
}

struct TestOptions {
  // How to generate the initial values of inputs and weights.
  FillPattern fillStrategy = FillPattern::IOTA;
  // True if the plan should contain an expand dims transform.
  bool shouldHaveExpandDims = true;
  // Whether the input rearrangement should have been optimised away or not.
  bool allowInputRearrangement = false;
  // Run the program if true and false otherwise.
  bool runProgram = true;
  // Print the input and weights tensors.
  bool printTensors = true;
  // Print the plan.
  bool printPlan = true;
  // The directory to store the auto-report.
  // If empty a temporary directory will be used.
  std::string autoReportDirectory = "";
  // Instrument the program. Only used when runProgram is true.
  bool profileExecution = false;
};

// Check if expandDims is in the plan and dump the plan for inspection.
static void checkPlan(const Graph &graph, const ConvParams &params,
                      const OptionFlags &options, PlanningCache *cache,
                      const TestOptions &testOptions) {
  std::stringstream planOutputStream;
  reportPlanInfo(planOutputStream, graph, params, options, cache);
  std::string planOutput = planOutputStream.str();

  if (testOptions.printPlan)
    std::cout << planOutput;

  std::regex re("\\s+expandDims.*\\{(.+)\\}");
  auto m = std::regex_search(planOutput, re);
  if (!m) {
    BOOST_TEST_MESSAGE("No expandDims present :(\n");
    BOOST_TEST(!testOptions.shouldHaveExpandDims);
  } else {
    BOOST_TEST_MESSAGE("expandDims is present :)\n");
    BOOST_TEST(testOptions.shouldHaveExpandDims);
  }
}

static OptionFlags
makePlanConstraints(unsigned inChansPerGroup,
                    ArrayRef<unsigned> tileLevelExpandDims,
                    ArrayRef<unsigned> systemLevelExpandDims = {}) {
  // clang-format off
  return OptionFlags{
    {"planConstraints", fmt::format(R"({{
        "method": {{"type": "AMP"}},
        "inChansPerGroup": "{}",
        "0": {{
          "transform": {{
            "expandDims": {},
            "outChanFlattenDims": [],
            "swapOperands": false
          }}
        }},
        "1": {{
          "transform": {{
            "expandDims": {},
            "outChanFlattenDims": [],
            "swapOperands": false
          }}
        }}
      }})", inChansPerGroup, systemLevelExpandDims, tileLevelExpandDims)}};
  // clang-format on
}

template <typename HostValueType>
static HostTensor<HostValueType> convolve(const ConvParams &params,
                                          const OptionFlags &convOptions,
                                          const TestOptions &testOptions = {}) {
  // Construct an IPU device and target.
  constexpr unsigned numIpus = 1;
  constexpr unsigned numTiles = 1;
  TestDevice testDevice = createTestDevice(TEST_TARGET, numIpus, numTiles);
  const Target &target = testDevice.getTarget();

  // Create a graph to store the variables.
  Graph graph{target};
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  // Avoid planning multiple times.
  PlanningCache cache;

  // Create input tensors to convolution.
  Tensor input =
      createInput(graph, params, "MyInputTensor", convOptions, &cache);
  Tensor weights =
      createWeights(graph, params, "MyWeights", convOptions, &cache);

  checkPlan(graph, params, convOptions, &cache, testOptions);

  program::Sequence progs;
  fillValues<HostValueType>(graph, progs, params, input, weights,
                            testOptions.fillStrategy);

  if (testOptions.printTensors) {
    progs.add(program::PrintTensor("Weights", weights));
    progs.add(program::PrintTensor("Input", input));
  }

  constexpr bool transposeAndFlipWeights = false;
  Tensor output = poplin::convolution(graph, input, weights, params,
                                      transposeAndFlipWeights, progs,
                                      "MyConvolution", convOptions, &cache);

  if (testOptions.printTensors) {
    progs.add(program::PrintTensor("Output", output));
  }

  // Read out the output after the convolution.
  graph.createHostRead("getOutput", output);

  // The profile is used to check that no rearrangement of inputs occurs,
  // so if a directory to store the profile hasn't been given
  TemporaryDirectory td; // cleaned up on scope exit
  std::string autoReportDirectory = testOptions.autoReportDirectory;
  if (autoReportDirectory.empty()) {
    td.open();
    autoReportDirectory = td.path.string();
  }

  // Compile the program with profiling enabled.
  OptionFlags compilationOptions;
  compilationOptions.set("autoReport.directory", autoReportDirectory);
  if (testOptions.profileExecution) {
    compilationOptions.set("autoReport.all", "true");
  } else {
    compilationOptions.set("autoReport.outputLoweredVars", "true");
    compilationOptions.set("autoReport.outputGraphProfile", "true");
  }
  Executable exe = compileGraph(graph, {progs}, compilationOptions);

  // Create somewhere to store the output tensor for post-run validation.
  HostTensor hostOutput(output);

  // Return before running the program if requested.
  if (!testOptions.runProgram)
    return hostOutput;

  // Construct an engine, run the program and read the output back.
  {
    OptionFlags runOptions = compilationOptions;
    Engine engine(std::move(exe), runOptions);

    testDevice.bind([&](auto &device) {
      engine.loadAndRun(device);

      // Read the output back so we can validate the values.
      const auto outputType = output.elementType();
      auto &data = hostOutput.data;
      if (outputType == QUARTER || outputType == HALF) {
        // These types are represented by floats on the host. Copy the IPU
        // memory into a temporary buffer, then convert the quarters/halves to
        // floats.
        size_t sizeOnTarget = target.getTypeSize(outputType);
        std::vector<uint8_t> buffer(data.size() * sizeOnTarget);
        engine.readTensor("getOutput", buffer.data(),
                          buffer.data() + buffer.size());
        if (outputType.requiresMetadata()) {
          convertFromDeviceType(outputType, quarterMetadata, buffer.data(),
                                data);
        } else {
          convertFromDeviceType(outputType, buffer.data(), data);
        }
      } else {
        engine.readTensor("getOutput", data.data(), data.data() + data.size());
      }
    });

    // Engine is destroyed on scope exit which writes the profile.pop file.
  }

  // Check that inputs were not rearranged if expand dims were set,
  // which should be the case if the deferment to vertex-level was
  // successful and the optimisation worked as expected.
  if (!testOptions.allowInputRearrangement) {
    struct Visitor : public pva::ProgramVisitor {
      void visitOnTileExecute(
          const pva::OnTileExecuteProgram &onTileExecute) override {
        if (onTileExecute.name().find("PreArrange") != std::string::npos ||
            onTileExecute.name().find("OnTileCopy") != std::string::npos ||
            onTileExecute.name().find("actsRearranged") != std::string::npos) {
          for (const auto &var : onTileExecute.vars()) {
            BOOST_TEST_MESSAGE(var.name());
            BOOST_TEST(var.name().find("MyInputTensor") == std::string::npos);
          }
        }
      }
    };
    Visitor visitor;
    const auto &report = pva::openReport(autoReportDirectory + "/profile.pop");
    for (const auto &p : report.compilation().programs()) {
      p->accept(visitor);
    }
  }

  return hostOutput;
}

BOOST_AUTO_TEST_CASE(Minimal1DExpansion) {
  // Note that this doesn't use the vertex-level expand dims
  // implementation because it achieves no rearrangement anyway.
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {8},
                    /*kernelShape     = */ {4},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 5},
      /* data  = */ {21520, 22048, 22576, 23104, 23632});
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {0}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(MinimalFloatConv) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {2, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 1, 5},
      /* data  = */ {173600, 175680, 177760, 179840, 181920});
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(MinimalHalfConv) {
  ConvParams params{/*dataType        = */ HALF,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {2, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 16,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  params.outputType = FLOAT; // Half is too small to represent output
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 1, 5},
      /* data  = */ {1.39373e+06, 1.40198e+06, 1.41024e+06, 1.4185e+06,
                     1.42675e+06});
  HostTensor output = convolve<float>(params, makePlanConstraints(16, {1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(MinimalQuarterConv,
                     *boost::unit_test::precondition(enableIfIpu21())) {
  ConvParams params{/*dataType        = */ QUARTER,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {2, 4},
                    /*kernelShape     = */ {1, 2},
                    /*inputChannels   = */ 32,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  params.outputType = FLOAT; // Quarter is too small to represent output
  TestOptions testOptions;
  testOptions.fillStrategy = FillPattern::ALL_ONES;
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 2, 3},
      /* data  = */ {64, 64, 64, 64, 64, 64});
  HostTensor output =
      convolve<float>(params, makePlanConstraints(32, {1}), testOptions);
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingLargerDimensions) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {4, 8},
                    /*kernelShape     = */ {4, 4},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 1, 5},
      /* data  = */ {1.39373e+06, 1.40198e+06, 1.41024e+06, 1.4185e+06,
                     1.42675e+06});
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingOuterDimension) {
  // Expanding the outer-most (0th) field dimension should require no
  // worklist adjustment.
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {4, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  // clang-format off
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 3, 5},
      /* data  = */ {333088, 335168, 337248, 339328, 341408,
                     349728, 351808, 353888, 355968, 358048,
                     366368, 368448, 370528, 372608, 374688});
  // clang-format on
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {0}));
  BOOST_TEST(output == expect);
}

// Expanding a middle field dimension should require a coordinated
// worklist adjustment to produce the correct answer.
BOOST_AUTO_TEST_CASE(ExpandingMiddleDimension) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {2, 4, 8},
                    /*kernelShape     = */ {2, 3, 4},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  // clang-format off
  HostTensor expect = HostTensor(
      /* shape = */{1, 1, 1, 2, 5},
      /* data  = */{
          6.21232e+06, 6.23085e+06, 6.24938e+06, 6.26790e+06, 6.28643e+06,
          6.36054e+06, 6.37907e+06, 6.39760e+06, 6.41613e+06, 6.43466e+06});
  // clang-format on
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {1}));
  BOOST_TEST(output == expect);
}

// View-only expand dims followed by regular expand dims.
// This also tests > 1 IC1 dimension.
BOOST_AUTO_TEST_CASE(ExpandingViewAndVertexLevel) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {2, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 1, 5},
      /* data  = */ {173600, 175680, 177760, 179840, 181920});
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {0, 1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingMultipleDimensions) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {3, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  // clang-format off
  HostTensor expect = HostTensor(
      /* shape = */{1, 1, 2, 5},
      /* data  = */{253344, 255424, 257504, 259584, 261664,
                    269984, 272064, 274144, 276224, 278304});
  // clang-format on
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {0, 1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingMultipleDimensionsWithMultipleConvGroups) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {3, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 2};
  // clang-format off
  HostTensor expect = HostTensor(
      /* shape = */{1, 2, 2, 5},
      /* data  = */{
          253344, 255424, 257504, 259584, 261664,
          269984, 272064, 274144, 276224, 278304,
          1.80982e+06, 1.81600e+06, 1.82218e+06, 1.82835e+06, 1.83453e+06,
          1.85923e+06, 1.86541e+06, 1.87158e+06, 1.87776e+06, 1.88394e+06});
  // clang-format on
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {0, 1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingWithInputPadding) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {3, 3},
                    /*kernelShape     = */ {2, 2},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  // Padding a non-expanded outer dimension shouldn't cause rearrangement.
  params.inputTransform.paddingLower[0] = 2;
  params.inputTransform.paddingUpper[0] = 1;
  // clang-format off
  HostTensor expect({1, 1, 5, 2}, {    0,     0,
                                   12268, 12548,
                                   24320, 24848,
                                   25904, 26432,
                                   12700, 12948});
  // clang-format on
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingWithInputPaddingAndRearrangement) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 2,
                    /*inputFieldShape = */ {3, 3},
                    /*kernelShape     = */ {1, 2},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  params.inputTransform.paddingLower[1] = 2;
  params.inputTransform.paddingUpper[1] = 1;
  // Padding the expanded dimension causes some rearrangement, but the input
  // elements that wouldn't have been aliased shouldn't be broadcast.
  TestOptions options;
  options.allowInputRearrangement = true;
  // clang-format off
  HostTensor expect({2, 1, 3, 5}, {0, 3096, 6004, 6140, 2964,
                                   0, 3312, 6412, 6548, 3156,
                                   0, 3528, 6820, 6956, 3348,
                                   // batch 2
                                   0, 8280, 15796, 15932, 7572,
                                   0, 8496, 16204, 16340, 7764,
                                   0, 8712, 16612, 16748, 7956});
  // clang-format on
  HostTensor output =
      convolve<float>(params, makePlanConstraints(8, {1}), options);
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingWithKernelDilation) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {3, 4},
                    /*kernelShape     = */ {1, 2},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  params.kernelTransform.dilation[1] = 2;
  // clang-format off
  HostTensor expect({1, 1, 3, 2}, {8008, 8144,
                                   8552, 8688,
                                   9096, 9232});
  // clang-format on
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(FullyExpandingWithBatchSize) {
  // Fully expanding the field shape means the batch can't be flattened,
  // because there is no field dimension to flatten it in to, so it has
  // to be accounted for by the expand dims.
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 2,
                    /*inputFieldShape = */ {3, 3, 3},
                    /*kernelShape     = */ {2, 2, 2},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  // Generate the expected result by not expanding any dimensions.
  TestOptions options;
  options.shouldHaveExpandDims = false;
  HostTensor expect =
      convolve<float>(params, makePlanConstraints(4, {}), options);
  // Define the test cases as every combination of expand dims.
  std::vector<std::vector<unsigned>> cases = {{0},    {1},    {2},      {0, 1},
                                              {0, 2}, {1, 2}, {0, 1, 2}};
  // Run the test cases.
  for (const auto &dims : cases) {
    for (unsigned inChans : {2, 4, 8}) {
      // Not supported with vertex level expansion.
      if (dims[0] == 0 && inChans < 8)
        continue;
      TestOptions options;
      options.allowInputRearrangement =
          inChans != params.inputChannelsPerConvGroup;
      HostTensor output =
          convolve<float>(params, makePlanConstraints(inChans, dims), options);
      BOOST_TEST(output == expect);
    }
  }
}

BOOST_AUTO_TEST_CASE(ExpandingWithManyTransforms) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 2,
                    /*inputFieldShape = */ {3, 4},
                    /*kernelShape     = */ {1, 3},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  params.inputTransform.truncationLower[1] = 1;
  params.inputTransform.truncationUpper[1] = 1;
  params.inputTransform.paddingLower[1] = 2;
  params.inputTransform.paddingUpper[1] = 1;
  params.kernelTransform.truncationLower[1] = 1;
  params.kernelTransform.truncationUpper[1] = 1;
  params.kernelTransform.paddingLower[1] = 2;
  params.kernelTransform.paddingUpper[1] = 1;
  TestOptions options;
  options.allowInputRearrangement = true;
  // clang-format off
  HostTensor expect({2, 1, 3, 2}, { 5912,  6012,
                                    6312,  6412,
                                    6712,  6812,
                                   // batch 2
                                   15512, 15612,
                                   15912, 16012,
                                   16312, 16412});
  // clang-format on
  HostTensor output =
      convolve<float>(params, makePlanConstraints(8, {1}), options);
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingWithSomeTransformsAndMoreOutputChannels) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 2,
                    /*inputFieldShape = */ {3, 4},
                    /*kernelShape     = */ {2, 3},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 4,
                    /*numConvGroups   = */ 1};
  params.inputTransform.paddingLower[1] = 1;
  params.outputTransform.stride[0] = 2;
  params.outputTransform.stride[1] = 2;
  // Padding means some rearrangement will happen.
  TestOptions options;
  options.allowInputRearrangement = true;
  // clang-format off
  HostTensor expect({2, 4, 1, 2}, { 48600,  73592,
                                   118488, 181880,
                                   188376, 290168,
                                   258264, 398456,
                                   // 2nd batch
                                   125400, 186488,
                                   342744, 515960,
                                   560088, 845432,
                                   777432, 1.1749e+06});
  // clang-format on
  HostTensor output =
      convolve<float>(params, makePlanConstraints(8, {0, 1}), options);
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(ExpandingBiggerShapeWithSomeTransforms) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 2,
                    /*inputFieldShape = */ {6, 6},
                    /*kernelShape     = */ {3, 3},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  params.inputTransform.paddingLower[0] = 1;
  params.inputTransform.paddingLower[1] = 1;
  params.outputTransform.stride[0] = 2;
  params.outputTransform.stride[1] = 2;
  TestOptions options;
  options.fillStrategy = FillPattern::ALL_ONES;
  options.allowInputRearrangement = true;
  // clang-format off
  HostTensor expect({2, 1, 3, 3}, {32, 48, 48,
                                   48, 72, 72,
                                   48, 72, 72,
                                   // 2nd batch
                                   32, 48, 48,
                                   48, 72, 72,
                                   48, 72, 72});
  // clang-format on
  HostTensor output =
      convolve<float>(params, makePlanConstraints(8, {0, 1}), options);
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(PopARTMultiConvBwdPassReproducer) {
  // A test-case lifted from a PopART multi-conv test, that tests unusual
  // numbers of input and output channels as well as input padding.
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {4},
                    /*kernelShape     = */ {3},
                    /*inputChannels   = */ 9,
                    /*outputChannels  = */ 6,
                    /*numConvGroups   = */ 1};
  params.inputTransform.paddingLower[0] = 2;
  params.inputTransform.paddingUpper[0] = 2;
  // The padding causes some rearrangement, but the input elements
  // shouldn't be broadcast.
  TestOptions options;
  options.allowInputRearrangement = true;
  // clang-format off
  HostTensor expect({1, 6, 6}, { 3015,  6012,  8982,  9360,  6183,  3060,
                                 7146, 14517, 22104, 23211, 15660,  7920,
                                11277, 23022, 35226, 37062, 25137, 12780,
                                15408, 31527, 48348, 50913, 34614, 17640,
                                19539, 40032, 61470, 64764, 44091, 22500,
                                23670, 48537, 74592, 78615, 53568, 27360});
  // clang-format on
  HostTensor output =
      convolve<float>(params, makePlanConstraints(8, {0}), options);
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(PopARTMultiConvFwdPassReproducerNoPadding3D) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {4, 4, 4},
                    /*kernelShape     = */ {2, 2, 2},
                    /*inputChannels   = */ 8,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  // clang-format off
  HostTensor<float> expect({1, 1, 3, 3, 3}, {
      663040, 665120, 667200,
      671360, 673440, 675520,
      679680, 681760, 683840,

      696320, 698400, 700480,
      704640, 706720, 708800,
      712960, 715040, 717120,

      729600, 731680, 733760,
      737920, 740000, 742080,
      746240, 748320, 750400,
  });
  // clang-format on
  HostTensor output = convolve<float>(params, makePlanConstraints(8, {2, 0}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(MinimalInputChannelsHalf) {
  ConvParams params{/*dataType        = */ HALF,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {2, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 4,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 1, 5},
      /* data  = */ {21520, 22048, 22576, 23104, 23632});
  HostTensor output = convolve<float>(params, makePlanConstraints(4, {1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(MinimalInputChannelsFloat) {
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {2, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 2,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 1, 5},
      /* data  = */ {2632, 2768, 2904, 3040, 3176});
  HostTensor output = convolve<float>(params, makePlanConstraints(2, {1}));
  BOOST_TEST(output == expect);
}

BOOST_AUTO_TEST_CASE(SubOptimalInputChannelsFloat) {
  // Forcing the planner to use a sub-optimal number of input channels that's
  // less than the tensors are prepared with will cause rearrangement but
  // should still work.
  ConvParams params{/*dataType        = */ FLOAT,
                    /*batchSize       = */ 1,
                    /*inputFieldShape = */ {2, 8},
                    /*kernelShape     = */ {2, 4},
                    /*inputChannels   = */ 4,
                    /*outputChannels  = */ 1,
                    /*numConvGroups   = */ 1};
  HostTensor expect = HostTensor(
      /* shape = */ {1, 1, 1, 5},
      /* data  = */ {21520, 22048, 22576, 23104, 23632});
  TestOptions options;
  options.allowInputRearrangement = true;
  HostTensor output =
      convolve<float>(params, makePlanConstraints(2, {1}), options);
  BOOST_TEST(output == expect);
}
