// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/MatrixTransforms.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/CTCInference.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <boost/multi_array.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <fstream>
#include <iomanip>
#include <random>

namespace po = boost::program_options;

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test;
using namespace poplibs_test::ctc;
using namespace poplibs_test::util;
using namespace poplibs_support;
using namespace poputil;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

template <typename FPType> struct InputSequence {
  boost::multi_array<FPType, 2> input;
  bool isLogits;
};

// TODO: Return top paths {sequence, prob}
std::vector<std::pair<std::vector<unsigned>, double>>
beamSearchIPU(std::size_t maxTime, std::size_t batchSize, unsigned blankClass,
              std::size_t numClasses, unsigned beamwidth, unsigned topPaths,
              Type inType, Type outType, const DeviceType &deviceType,
              boost::optional<unsigned> tiles) {

  auto device = createTestDevice(deviceType, 1, tiles);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);

  // Create the inputs to the beam search function
  const popnn::ctc::Plan plan = popnn::ctc_infer::plan(
      graph, inType, batchSize, maxTime, numClasses, beamwidth);

  auto data = popnn::ctc_infer::createDataInput(
      graph, inType, batchSize, maxTime, numClasses, plan, "DataInput");

  auto dataLengths = graph.addVariable(UNSIGNED_INT, {batchSize});
  graph.setTileMapping(dataLengths, 0);

  // Call both beam search functions as a placeholder test that they exist and
  // execute
  Sequence prog;
  popnn::ctc_infer::beamSearchDecoderLogits(graph, data, dataLengths, prog,
                                            blankClass, beamwidth, topPaths,
                                            plan, "BeamSearchLogits");
  popnn::ctc_infer::beamSearchDecoderLogProbabilities(
      graph, data, dataLengths, prog, blankClass, beamwidth, topPaths, plan,
      "BeamSearchLogProbabilities");
  Engine engine(graph, prog);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();
  });

  return {};
}

int main(int argc, char **argv) {
  // TODO check all these are options
  // Default input parameters.
  DeviceType deviceType = DeviceType::IpuModel2;
  boost::optional<unsigned> tiles = boost::none;

  boost::optional<unsigned> minRandomTime = boost::none;
  boost::optional<unsigned> fixedTime = boost::none;
  unsigned maxTime = 15;
  boost::optional<unsigned> minRandomLabelLength = boost::none;
  boost::optional<unsigned> fixedLabelLength = boost::none;
  unsigned maxLabelLength = 5;

  unsigned numClasses = 4;
  unsigned blankClass = 0;
  unsigned batchSize = 1;
  unsigned beamwidth = 3;
  unsigned topPaths = 2;

  Type inType = FLOAT;
  Type partialsType = FLOAT;
  Type outType = FLOAT;

  bool isLogits = false;
  unsigned seed = 42;
  unsigned verbosityLevel = 0;

  boost::optional<std::string> planConstraints;

  boost::optional<std::string> jsonProfileOut;
  boost::optional<std::string> profileFormat;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     deviceTypeHelp)
    ("tiles-per-ipu",po::value<boost::optional<unsigned>>(&tiles),
      "Number of tiles per IPU")

    ("label-length", po::value(&fixedLabelLength),
     "If set, forces every superimposed label to be of length `label-length`")
    ("min-label-length",
      po::value(&minRandomLabelLength),
     "If set, minimum randomly generated superimposed label length")
    ("max-label-length",
      po::value(&maxLabelLength)->default_value(maxLabelLength),
     "Max superimposed label length")
    ("time", po::value(&fixedTime),
     "If set, forces every sequence to be of length `time`")
    ("min-time", po::value(&minRandomTime),
     "If set, minimum randomly generated time length")
    ("max-time", po::value(&maxTime)->default_value(maxTime),
     "Max test length (time)")

    ("num-classes", po::value(&numClasses)->default_value(numClasses),
     "Classes in the alphabet including blank")
    ("blank-class", po::value(&blankClass)->default_value(blankClass),
     "Index of the blank symbol. Range 0 to (num-classes-1)")
    ("batch", po::value(&batchSize)->default_value(batchSize),
     "Batch size")
    ("beamwidth", po::value(&beamwidth)->default_value(beamwidth),
     "Number of the beams to persist at each timestep")
    ("top-paths", po::value(&topPaths)->default_value(topPaths),
     "Final number of beams to return from the operation")

    ("in-type", po::value(&inType)->default_value(inType),
     "Input data type")
    ("partials-type", po::value(&partialsType)->default_value(partialsType),
     "Input data type")
    ("out-type", po::value(&outType)->default_value(outType),
     "Output data type")

    ("logit-inputs", po::value(&isLogits)->default_value(isLogits),
     "pass logit inputs to ctc infer api, otherwise convert to logProbs prior")
    ("seed", po::value(&seed)->default_value(seed),
     "Seed used for random input generation")

    ("plan-constraints", po::value(&planConstraints),
     "JSON constraints for planner, e.g. {\"parallel\": {\"batch\": 1}}")

    ("profile", "Show profile report")
    ("profile-format",
     po::value<decltype(profileFormat)>(&profileFormat)
      ->default_value(boost::none),
     "Profile formats: v1 | experimental | unstable")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")

    ("disable-always-satisfiable-error", "Disable the check when validating time"
    " and the imposed labelLength before generating random labels to superimpose"
    " onto the input. This check ensures that the imposed label is always"
    " representable for given t and labelLength."
    "\nThe length of t required to represent a given label depends on the number"
    " of duplicate classes in the label, this check assumes the worst case where"
    " every class is a duplicate."
    "\nSpecifically:"
    "\n  2 * t - 1 >= labelLength")
    ("ignore-data", "Ignore data, to check execution time")
    ("plan-only", "Only plan the requested passes, don't build or run a graph")
    ("verbosity-level", po::value(&verbosityLevel)->default_value(verbosityLevel),
     "Level of verbosity for debug printouts");
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }

  } catch (std::exception &e) {
    std::cerr << "error parsing command line: " << e.what() << "\n";
    return 1;
  }

  const bool profile = vm.count("profile");
  (void)profile; // TODO - currently unused
  const bool ignoreData = vm.count("ignore-data");
  const bool planOnly = vm.count("plan-only");
  const bool disableAlwaysSatisfiableError =
      vm.count("disable-always-satisfiable-error");

  // Needed to set default arguments.
  po::notify(vm);

  if (blankClass >= numClasses) {
    throw poputil::poplibs_error("The blank class must be in the range 0 to "
                                 "(number of classes - 1)");
  }

  if (!minRandomTime && !fixedTime) {
    fixedTime = maxTime;
  }
  if (!minRandomLabelLength && !fixedLabelLength) {
    fixedLabelLength = maxLabelLength;
  }
  ctc::validateTimeAndLabelBounds(minRandomTime, fixedTime, maxTime,
                                  minRandomLabelLength, fixedLabelLength,
                                  maxLabelLength);

  poplar::OptionFlags planOpts;
  if (planConstraints) {
    planOpts.set("planConstraints", *planConstraints);
  }
  planOpts.set("partialsType", partialsType.toString());

  if (planOnly) {
    auto device = createTestDevice(deviceType, 1, tiles);
    const auto &target = device.getTarget();
    Graph graph(target);

    const auto plan = popnn::ctc_infer::plan(graph, inType, batchSize, maxTime,
                                             numClasses, beamwidth, planOpts);

    std::cout << plan << std::endl;
    std::cout << "No test run - plan only" << std::endl;
    return 0;
  }

  RandomUtil rand{seed};

  std::vector<InputSequence<double>> tests;
  std::vector<std::vector<std::pair<std::vector<unsigned>, double>>> references;
  for (unsigned i = 0; i < batchSize; i++) {
    const auto [t, labelLength] = getRandomSize(
        minRandomTime, fixedTime, maxTime, minRandomLabelLength,
        fixedLabelLength, maxLabelLength, disableAlwaysSatisfiableError, rand);
    auto [input, label] = getRandomTestInput<double>(
        t, maxTime, labelLength, numClasses, blankClass, isLogits, rand);

    tests.push_back({input, isLogits});

    if (verbosityLevel == 1) {
      std::cout << "\nBatch:" << i << " Time:" << t
                << " Label length:" << label.size() << "\n\n";
      if (tests[i].isLogits) {
        std::cout << "Logits in\n";
        printInput(tests[i].input, blankClass);
        std::cout << "Log Softmax Logits in\n";
        printInput(log::log(matrix::transpose(
                       log::softMax(matrix::transpose(tests[i].input)))),
                   blankClass);
      } else {
        std::cout << "Log Softmax in\n";
        printInput(tests[i].input, blankClass);
      }
      std::cout << "\n";
    }

    if (!ignoreData) {
      if (isLogits) {
        references.push_back(ctc::infer<double>(
            log::log(matrix::transpose(
                log::softMax(matrix::transpose(tests[i].input)))),
            blankClass, beamwidth, topPaths, true, verbosityLevel == 2));
      } else {
        references.push_back(ctc::infer<double>(tests[i].input, blankClass,
                                                beamwidth, topPaths, true,
                                                verbosityLevel == 2));
      }
      if (verbosityLevel == 1) {
        std::cout << "Reference output (batch " << i << "):\n";
        printBeams(references[i], blankClass);
      }
    }
  }

  auto outputs =
      beamSearchIPU(maxTime, batchSize, blankClass, numClasses, beamwidth,
                    topPaths, inType, outType, deviceType, tiles);

  for (unsigned i = 0; i < batchSize; i++) {
    if (verbosityLevel == 1) {
      std::cout << "Result output (batch " << i << "):\n";
      printBeams(outputs, blankClass);
    }
  }

  // double relativeTolerance = outType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
  // double absoluteTolerance = outType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;

  bool success = true;
  if (!ignoreData) {
    for (unsigned i = 0; i < batchSize; i++) {
      // TODO compare
      bool batchSuccess = false;
      if (!batchSuccess) {
        success = false;
      }
    }
  }
  if (!success) {
    std::cerr << "Data mismatch\n";
  }
  return !success;
}
