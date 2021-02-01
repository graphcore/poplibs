// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iomanip>
#include <random>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCLoss.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/CTCLoss.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

namespace po = boost::program_options;

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::ctc;
using namespace poplibs_test;
using namespace poplibs_test::util;
using namespace poplibs_support;
using namespace poputil;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

template <typename FPType>
boost::multi_array<FPType, 2>
transpose(const boost::multi_array<FPType, 2> &in) {
  const auto inRows = in.shape()[0];
  const auto inColumns = in.shape()[1];
  boost::multi_array<FPType, 2> out(boost::extents[inColumns][inRows]);
  for (unsigned inRow = 0; inRow < inRows; inRow++) {
    for (unsigned inColumn = 0; inColumn < inColumns; inColumn++) {
      out[inColumn][inRow] = in[inRow][inColumn];
    }
  }
  return out;
}

// Mask results that aren't valid due to the time of the batch entry
boost::multi_array<double, 2>
maskResults(const boost::multi_array<double, 2> &in, size_t validTimesteps) {
  auto out = in;
  const auto timeSteps = out.shape()[0];
  const auto symbols = out.shape()[1];
  for (unsigned sym = 0; sym < symbols; sym++) {
    for (unsigned t = validTimesteps; t < timeSteps; t++) {
      out[t][sym] = 0;
    }
  }
  return out;
}
// Print a sequence, inserting `-` for the blank symbol
void print(const std::string &prefix, const std::vector<unsigned> &symbols,
           unsigned blank, bool verbose = true) {
  if (!verbose) {
    return;
  }
  std::cout << prefix << " ";
  for (auto &i : symbols) {
    if (i == blank) {
      std::cout << "- ";
    } else {
      std::cout << i << " ";
    }
  }
  std::cout << "\n";
}

// Print the matrix `in`
template <typename FPType>
void print(const std::string &prefix, const boost::multi_array<FPType, 2> &in,
           unsigned blank, bool verbose = true) {
  if (!verbose) {
    return;
  }
  std::cout << "\n" << prefix << "\n        ";
  for (unsigned i = 0; i < in.size(); i++) {
    std::cout << "         t" << i;
  }

  for (unsigned i = 0; i < in[0].size(); i++) {
    if (i == blank) {
      std::cout << "\nIndex:-  ";
    } else {
      std::cout << "\nIndex:" << i << "  ";
    }
    for (unsigned j = 0; j < in.size(); j++) {
      std::cout << std::setw(10) << std::setprecision(4) << in[j][i] << ",";
    }
  }
  std::cout << "\n";
}

// Struct and function to return the test inputs
template <typename FPType> struct InputSequence {
  // Input, always of max size (time) but with only inputLength valid values,
  // the rest padded as blank
  boost::multi_array<FPType, 2> input;
  unsigned inputLength;
  // Labels, of the randomly chosen size for this batch
  std::vector<unsigned> labels;
};

InputSequence<double>
getRandomTestInput(boost::optional<unsigned> testTime, size_t minT, size_t maxT,
                   boost::optional<unsigned> testLabelLength,
                   size_t minLabelLength, size_t maxLabelLength,
                   unsigned numClasses, unsigned batchNo, unsigned blankClass) {

  std::mt19937 gen;
  // Seed with the batch number - resulting in repeatable pseudo random sizes
  // of each batch[n] test to run with pseudo random data content. Each batch[n]
  // entry can have a different size to the others (and different data)
  gen.seed(batchNo);

  boost::multi_array<double, 2> input(boost::extents[maxT][numClasses]);

  std::uniform_int_distribution<> randT(minT, maxT);
  std::uniform_int_distribution<> randLabels(0, numClasses - 2);
  std::uniform_int_distribution<> randInput(0, 10);

  unsigned inputLength =
      testTime.is_initialized() ? testTime.get() : randT(gen);

  // Constrain the sequence of labels to conform to the randomly chosen
  // input length - enforcing time >= 1 + 2 * labels
  size_t maxS =
      std::min(maxLabelLength, static_cast<size_t>((inputLength + 1) / 2 - 1));
  size_t minS = std::min(minLabelLength, maxS);
  std::uniform_int_distribution<> randLabelLength(minS, maxS);
  unsigned labelLength = testLabelLength.is_initialized()
                             ? testLabelLength.get()
                             : randLabelLength(gen);

  std::vector<unsigned> labels(labelLength);

  // Random label sequence of the right length
  for (size_t i = 0; i < labelLength; i++) {
    const unsigned random = randLabels(gen);
    labels[i] = static_cast<unsigned>(random >= blankClass) + random;
  }

  // Input sequence of max length
  for (size_t i = 0; i < numClasses; i++) {
    for (size_t j = 0; j < maxT; j++) {
      input[j][i] = randInput(gen);
    }
  }
  input = transpose(log::softMax(transpose(input)));

  return {input, inputLength, labels};
}

template <typename FPType>
boost::multi_array<FPType, 2> gradReference(const InputSequence<FPType> &test,
                                            unsigned blankClass,
                                            unsigned numClasses, bool verbose) {

  auto paddedSequence = extendedLabels(test.labels, blankClass);
  auto in = transpose(test.input);
  boost::multi_array<FPType, 2> logSequence(
      boost::extents[paddedSequence.size()][in.shape()[1]]);
  poplibs_test::embedding::multiSlice(in, paddedSequence, logSequence);

  auto alphaLog =
      alpha(logSequence, paddedSequence, blankClass, test.inputLength, true);
  auto betaLog =
      beta(logSequence, paddedSequence, blankClass, test.inputLength, true);

  auto expandedGradient =
      expandedGrad(logSequence, alphaLog, betaLog, paddedSequence, blankClass,
                   test.inputLength, true);
  auto gradient = grad(logSequence, alphaLog, betaLog, paddedSequence,
                       numClasses, blankClass, test.inputLength, true);
  return transpose(gradient);
}

std::vector<boost::multi_array<double, 2>>
gradIPU(const std::vector<InputSequence<double>> &inputs, unsigned maxLabels,
        unsigned blankSymbol, std::size_t numClasses, Type inType, Type outType,
        OptionFlags opts, const DeviceType &deviceType,
        boost::optional<unsigned> tiles, bool ignoreData, bool profile) {

  auto device = createTestDevice(deviceType, 1, tiles);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);

  const auto maxT = inputs[0].input.shape()[0];
  const auto batchSize = inputs.size();

  // Create the inputs to the gradient function
  const auto plan = popnn::ctc::plan(graph, inType, outType, batchSize, maxT,
                                     maxLabels, numClasses, opts);

  auto data = popnn::ctc::createDataInput(graph, inType, batchSize, maxT,
                                          numClasses, plan, "DataInput");
  auto labels = popnn::ctc::createLabelsInput(graph, UNSIGNED_SHORT, batchSize,
                                              maxLabels, plan, "LabelsInput");

  auto dataLengths = graph.addVariable(UNSIGNED_SHORT, {batchSize});
  auto labelLengths = graph.addVariable(UNSIGNED_SHORT, {batchSize});
  graph.setTileMapping(dataLengths, 0);
  graph.setTileMapping(labelLengths, 0);

  // Write the inputs
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawDataLengths, rawLabelLengths;
  std::vector<std::unique_ptr<char[]>> rawData(batchSize), rawLabels(batchSize);
  rawDataLengths = allocateHostMemoryForTensor(
      dataLengths, "dataLengths", graph, uploadProg, downloadProg, tmap);
  rawLabelLengths = allocateHostMemoryForTensor(
      labelLengths, "labelLengths", graph, uploadProg, downloadProg, tmap);

  for (unsigned i = 0; i < batchSize; i++) {
    rawData[i] = allocateHostMemoryForTensor(data.slice(i, i + 1, 1),
                                             "data_" + std::to_string(i), graph,
                                             uploadProg, downloadProg, tmap);
    rawLabels[i] = allocateHostMemoryForTensor(
        labels.slice(i, i + 1, 0), "labels_" + std::to_string(i), graph,
        uploadProg, downloadProg, tmap);
    if (!ignoreData) {
      copy(target, inputs[i].input, inType, rawData[i].get());
      copy(target, inputs[i].labels, labels.elementType(), rawLabels[i].get());
    }
  }
  std::vector<unsigned> initLabelLengths(batchSize), initDataLengths(batchSize);
  for (unsigned i = 0; i < batchSize; i++) {
    initLabelLengths[i] = inputs[i].labels.size();
    initDataLengths[i] = inputs[i].inputLength;
  }
  copy(target, initLabelLengths, labelLengths.elementType(),
       rawLabelLengths.get());
  copy(target, initDataLengths, dataLengths.elementType(),
       rawDataLengths.get());

  // Create gradient
  Sequence prog;
  const auto result =
      popnn::ctc::gradient(graph, outType, data, labels, dataLengths,
                           labelLengths, prog, blankSymbol, plan, "Gradient");

  // Create handles for reading the result
  std::vector<std::unique_ptr<char[]>> rawResult(batchSize);
  for (unsigned i = 0; i < batchSize; i++) {
    rawResult[i] = allocateHostMemoryForTensor(
        result.slice(i, i + 1, 1), "result_" + std::to_string(i), graph,
        uploadProg, downloadProg, tmap);
  }

  // Run input, gradient, output
  OptionFlags engineOptions;
  if (profile) {
    engineOptions.set("debug.instrumentCompute", "true");
  }
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);
  attachStreams(engine, tmap);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();
  });

  // Fetch the result
  std::vector<boost::multi_array<double, 2>> output(batchSize);
  if (!ignoreData) {
    for (unsigned i = 0; i < batchSize; i++) {
      output[i].resize(boost::extents[maxT][numClasses]);
      copy(target, outType, rawResult[i].get(), output[i]);
    }
  }

  if (profile && deviceType != DeviceType::Cpu) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }
  return output;
}

int main(int argc, char **argv) {
  // Default input parameters.
  DeviceType deviceType = DeviceType::IpuModel2;
  bool verbose = false;
  bool profile = false;
  boost::optional<std::string> planConstraints;
  bool ignoreData = false;
  boost::optional<unsigned> testTime = boost::none;
  unsigned minTime = 15;
  unsigned maxTime = 15;
  boost::optional<unsigned> testLabelLength = boost::none;
  unsigned blankClass = 0;
  unsigned minLabelLength = 5;
  unsigned maxLabelLength = 5;
  unsigned numClasses = 4;
  unsigned batchSize = 1;
  boost::optional<unsigned> tiles = boost::none;
  Type inType = FLOAT;
  Type outType = FLOAT;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     deviceTypeHelp)
    ("tiles-per-ipu",po::value<boost::optional<unsigned>>(&tiles),
      "Number of tiles per IPU")
    ("profile", po::value(&profile)->default_value(profile),
     "Show profile report")
    ("plan-constraints", po::value(&planConstraints),
     "JSON constraints for planner, e.g. {\"parallel\": {\"batch\": 1}}")
    ("in-type", po::value(&inType)->default_value(inType),
     "Input data type")
    ("out-type", po::value(&outType)->default_value(outType),
     "Output data type")
    ("batch", po::value(&batchSize)->default_value(batchSize),
     "Batch size")
    ("label-length", po::value<boost::optional<unsigned>>(&testLabelLength),
     "Test length (labels)")
    ("min-label-length",
      po::value(&minLabelLength)->default_value(minLabelLength),
     "Min test length (labels)")
    ("max-label-length",
      po::value(&maxLabelLength)->default_value(maxLabelLength),
     "Max test length (labels)")
    ("time", po::value<boost::optional<unsigned>>(&testTime),
     "Test length (time)")
    ("min-time", po::value(&minTime)->default_value(minTime),
     "Min test length (time)")
    ("max-time", po::value(&maxTime)->default_value(maxTime),
     "Max test length (time)")
    ("blank-class", po::value(&blankClass)->default_value(blankClass),
     "Index of the blank symbol. Range 0 to (num-classes-1)")
    ("num-classes", po::value(&numClasses)->default_value(numClasses),
     "Classes in the alphabet including blank")
    ("ignore-data", po::value(&ignoreData)->default_value(ignoreData),
     "Ignore data, to check execution time")
    ("plan-only", "Only plan the requested passes, don't build or run a graph")
    ("verbose", po::value(&verbose)->default_value(verbose),
     "Provide debug printout");
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
  const bool planOnly = vm.count("plan-only");

  // Needed to set default arguments.
  po::notify(vm);
  // Pick up on some parameters that are easy to get wrong
  if (blankClass >= numClasses) {
    throw poputil::poplibs_error("The blank class must be in the range 0 to "
                                 "(number of classes - 1)");
  }
  if (maxTime < maxLabelLength) {
    throw poputil::poplibs_error(
        "The max test time must be >= max test label length");
  }
  if (minTime < minLabelLength) {
    throw poputil::poplibs_error(
        "The min test time must be >= min test label length ");
  }
  if (maxTime < minTime) {
    throw poputil::poplibs_error("The max test time must be >= min test time");
  }
  if (maxLabelLength < minLabelLength) {
    throw poputil::poplibs_error(
        "The max test label length must be >= min test label length");
  }
  if (testTime && testTime.get() > maxTime) {
    throw poputil::poplibs_error(
        "The non random test time must be <= max test time");
  }
  if (testLabelLength && testLabelLength.get() > maxLabelLength) {
    throw poputil::poplibs_error(
        "The non random test label length must be <= max test label length");
  }
  if ((!testTime && testLabelLength) || (testLabelLength && !testTime)) {
    throw poputil::poplibs_error(
        "Use non random test time and non random label length together");
  }
  if (testTime && testLabelLength && testTime.get() < testLabelLength.get()) {
    throw poputil::poplibs_error(
        "The non random test time must be >= non random test label"
        " length");
  }

  poplar::OptionFlags opts;
  if (planConstraints) {
    opts.set("planConstraints", *planConstraints);
  }

  if (planOnly) {
    auto device = createTestDevice(deviceType, 1, tiles);
    const auto &target = device.getTarget();
    Graph graph(target);

    const auto plan =
        popnn::ctc::plan(graph, inType, outType, batchSize, maxTime,
                         maxLabelLength, numClasses, opts);

    std::cout << plan << std::endl;
    std::cout << "No test run - plan only" << std::endl;
    return 0;
  }

  // For test call the reference function for each batch input
  std::vector<InputSequence<double>> tests;
  std::vector<boost::multi_array<double, 2>> references;
  for (unsigned i = 0; i < batchSize; i++) {
    tests.push_back(getRandomTestInput(
        testTime, minTime, maxTime, testLabelLength, minLabelLength,
        maxLabelLength, numClasses, i, blankClass));

    if (verbose) {
      std::cout << "\nBatch:" << i << " Time:" << tests[i].inputLength
                << " Label length:" << tests[i].labels.size();
    }
    print(" Test sequence[" + std::to_string(tests[i].labels.size()) + "] ",
          tests[i].labels, blankClass, verbose);
    print("Input:", tests[i].input, blankClass, verbose);

    // Provide the library function with log probabilities (log of softmax)
    tests[i].input = log::log(tests[i].input);
    print("Log Softmax in", tests[i].input, blankClass, verbose);
    references.push_back(
        gradReference<double>(tests[i], blankClass, numClasses, verbose));
    references.back() = maskResults(references.back(), tests[i].inputLength);
    print("Reference gradient", references.back(), blankClass, verbose);
  }
  auto outputs = gradIPU(tests, maxLabelLength, blankClass, numClasses, inType,
                         outType, opts, deviceType, tiles, ignoreData, profile);

  for (unsigned i = 0; i < batchSize; i++) {
    outputs[i] = maskResults(outputs[i], tests[i].inputLength);
    print("Result gradient, batch:" + std::to_string(i), outputs[i], blankClass,
          verbose);
  }
  double relativeTolerance = inType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
  double absoluteTolerance = inType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;

  bool success = true;
  if (!ignoreData) {
    for (unsigned i = 0; i < batchSize; i++) {
      bool batchSuccess =
          checkIsClose("Batch:" + std::to_string(i) + " result", outputs[i],
                       references[i], relativeTolerance, absoluteTolerance);
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
