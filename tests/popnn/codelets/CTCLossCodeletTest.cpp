// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CTCLossCodeletTest
#include <iomanip>
#include <random>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCLoss.hpp>
#include <poplibs_test/CTCUtil.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/MatrixTransforms.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_test/TempDir.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <optional>

namespace po = boost::program_options;

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::ctc;
using namespace poplibs_test;
using namespace poplibs_test::matrix;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;
using namespace poputil;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

enum class TestType { ALPHA, BETA, GRAD_GIVEN_ALPHA, GRAD_GIVEN_BETA };

std::ostream &operator<<(std::ostream &os, const TestType &test) {
  if (test == TestType::ALPHA)
    return os << "alpha";
  else if (test == TestType::BETA)
    return os << "beta";
  else if (test == TestType::GRAD_GIVEN_ALPHA)
    return os << "gradGivenAlpha";
  return os << "gradGivenBeta";
}

std::istream &operator>>(std::istream &is, TestType &test) {
  std::string token;
  is >> token;
  if (token == "alpha")
    test = TestType::ALPHA;
  else if (token == "beta")
    test = TestType::BETA;
  else if (token == "gradGivenAlpha")
    test = TestType::GRAD_GIVEN_ALPHA;
  else if (token == "gradGivenBeta")
    test = TestType::GRAD_GIVEN_BETA;
  else
    throw poputil::poplibs_error("Unknown test type");
  return is;
}

boost::multi_array<double, 2>
maskResults(const boost::multi_array<double, 2> &in) {
  auto out = in;
  const auto symbols = out.shape()[0];
  const auto timeSteps = out.shape()[1];
  for (unsigned sym = 0; sym < symbols; sym++) {
    if (sym > 1) {
      out[sym][0] = 0;
    }
    if (sym < symbols - 2) {
      out[sym][timeSteps - 1] = 0;
    }
  }
  return out;
}

boost::multi_array<double, 2>
maskTimeSteps(const boost::multi_array<double, 2> &in, unsigned timeStep,
              bool invertMask) {
  auto out = in;
  const auto symbols = out.shape()[0];
  const auto timeSteps = out.shape()[1];
  for (unsigned time = 0; time < timeSteps; time++) {
    if ((invertMask && time == timeStep) || (!invertMask && time != timeStep)) {
      for (unsigned sym = 0; sym < symbols; sym++) {
        out[sym][time] = 0;
      }
    }
  }
  return out;
}

// Print a sequence, inserting `-` for the blank symbol
void print(const std::string &prefix, const std::vector<unsigned> &idx,
           unsigned blank, bool verbose = true) {
  if (!verbose) {
    return;
  }
  std::cout << "\n" << prefix << " ";
  for (auto &i : idx) {
    if (i == blank) {
      std::cout << "- ";
    } else {
      std::cout << i << " ";
    }
  }
  std::cout << "\n";
}

// Print the matrix `in`, using `paddedSequence` as row labels
template <typename FPType>
void print(const std::string &prefix, const boost::multi_array<FPType, 2> &in,
           const std::vector<unsigned> &paddedSequence, unsigned blank,
           bool verbose = true) {
  if (!verbose) {
    return;
  }
  std::cout << "\n" << prefix << "\n          ";
  for (unsigned i = 0; i < in[0].size(); i++) {
    std::cout << "         t" << i;
  }

  for (unsigned i = 0; i < in.size(); i++) {
    std::cout << "\nIndex:" << i << "  ";
    if (paddedSequence[i] == blank) {
      std::cout << "- ";
    } else {
      std::cout << paddedSequence[i] << " ";
    }
    for (unsigned j = 0; j < in[i].size(); j++) {
      std::cout << std::setw(10) << std::setprecision(4) << in[i][j] << ",";
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
  for (unsigned i = 0; i < in[0].size(); i++) {
    std::cout << "         t" << i;
  }

  for (unsigned i = 0; i < in.size(); i++) {
    if (i == blank) {
      std::cout << "\nIndex:-  ";
    } else {
      std::cout << "\nIndex:" << i << "  ";
    }
    for (unsigned j = 0; j < in[i].size(); j++) {
      std::cout << std::setw(10) << std::setprecision(4) << in[i][j] << ",";
    }
  }
  std::cout << "\n";
}

// Struct and function to return the test inputs
template <typename FPType> struct InputSequence {
  boost::multi_array<FPType, 2> input;
  std::vector<unsigned> idx;
  unsigned alphabetSizeIncBlank;
};

template <typename FPType>
boost::multi_array<FPType, 2>
gradReference(const InputSequence<FPType> &test, unsigned blankClass,
              TestType vertexToTest, bool verbose) {

  auto paddedSequence = extendedLabels(test.idx, blankClass);
  auto validTimesteps = test.input.shape()[1];

  print("Log Softmax in", test.input, blankClass, verbose);
  boost::multi_array<FPType, 2> logSequence(
      boost::extents[paddedSequence.size()][test.input.shape()[1]]);
  poplibs_test::embedding::multiSlice(test.input, paddedSequence, logSequence);

  print("Reference sequence", logSequence, paddedSequence, blankClass, verbose);

  auto alphaLog =
      alpha(logSequence, paddedSequence, blankClass, validTimesteps);
  if (vertexToTest == TestType::ALPHA) {
    print("Reference alphas", alphaLog, paddedSequence, blankClass, verbose);
    return alphaLog;
  }

  auto betaLog = beta(logSequence, paddedSequence, blankClass, validTimesteps);
  if (vertexToTest == TestType::BETA) {
    print("Reference betas", betaLog, paddedSequence, blankClass, verbose);
    return betaLog;
  }
  auto expandedGradient =
      expandedGrad(logSequence, alphaLog, betaLog, paddedSequence, blankClass,
                   validTimesteps);
  print("Expanded Reference gradient", expandedGradient, paddedSequence,
        blankClass, verbose);

  auto gradient =
      ctcGrad(logSequence, alphaLog, betaLog, paddedSequence,
              test.alphabetSizeIncBlank, blankClass, validTimesteps);
  print("Reference gradient", gradient, blankClass, verbose);

  std::cout << "\n";
  return gradient;
}

boost::multi_array<double, 2>
gradIPU(const InputSequence<double> &input, unsigned timeStep,
        unsigned blankClass, const boost::multi_array<double, 2> &initialValues,
        TestType vertexToTest, Type inType, Type outType,
        const DeviceType &deviceType, bool profile) {

  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  Sequence prog;

  const auto labelLen = input.idx.size();
  const auto extendedLabelLen = 2 * labelLen + 1;
  const auto maxT = input.input.shape()[1];
  const auto numClasses = input.input.shape()[0];

  const auto resultIsGrad = (vertexToTest == TestType::GRAD_GIVEN_ALPHA ||
                             vertexToTest == TestType::GRAD_GIVEN_BETA);
  const auto findingAlpha = (vertexToTest == TestType::ALPHA ||
                             vertexToTest == TestType::GRAD_GIVEN_BETA);

  auto probabilities =
      graph.addVariable(inType, {maxT, numClasses}, "probabilities");
  auto labels = graph.addVariable(UNSIGNED_INT, {labelLen}, "label");
  auto result = graph.addVariable(
      outType, {maxT, resultIsGrad ? numClasses : extendedLabelLen}, "result");
  auto prevTime = graph.addVariable(
      outType, {resultIsGrad ? 2u : 1u, extendedLabelLen}, "prevTime");
  graph.setTileMapping(prevTime, 0);
  auto prevLabel =
      graph.addVariable(outType, {findingAlpha ? 1u : 2u, maxT}, "prevLabel");
  graph.setTileMapping(prevLabel, 0);
  auto prevSymbol = graph.addConstant<unsigned>(UNSIGNED_INT, {}, input.idx[0]);
  graph.setTileMapping(prevSymbol, 0);
  Tensor initialAlphaOrBeta;
  if (resultIsGrad) {
    initialAlphaOrBeta = graph.addVariable(outType, {maxT, extendedLabelLen},
                                           "initialAlphaOrBeta");
    graph.setTileMapping(initialAlphaOrBeta, 0);
  }
  Tensor loss;
  if (findingAlpha) {
    loss = graph.addVariable(outType, {}, "loss");
    graph.setTileMapping(loss, 0);
  }
  auto cs = graph.addComputeSet("cs");
  std::string vertexName;
  if (vertexToTest == TestType::ALPHA) {
    vertexName =
        templateVertex("popnn::CTCAlpha", inType, outType, UNSIGNED_INT, true);
  } else if (vertexToTest == TestType::BETA) {
    vertexName =
        templateVertex("popnn::CTCBeta", inType, outType, UNSIGNED_INT, true);
  } else if (vertexToTest == TestType::GRAD_GIVEN_ALPHA) {
    vertexName = templateVertex("popnn::CTCGradGivenAlpha", inType, outType,
                                UNSIGNED_INT, true);
  } else if (vertexToTest == TestType::GRAD_GIVEN_BETA) {
    vertexName = templateVertex("popnn::CTCGradGivenBeta", inType, outType,
                                UNSIGNED_INT, true);
  }
  auto vertex = graph.addVertex(cs, vertexName);

  // TODO - Vertex features to support < maxT and < labelsLen sized
  // inputs, and accepting data from the previous label inputs
  // are only tested in the context of the whole IPU implementation.
  // Suggest that when optimising the vertices this test is improved.
  auto validLabel = graph.addConstant(UNSIGNED_INT, {}, labelLen);
  auto validTime = graph.addConstant(UNSIGNED_INT, {}, maxT);
  auto initialCount = graph.addConstant(UNSIGNED_INT, {}, timeStep);
  auto count = graph.addVariable(UNSIGNED_INT, {}, "count");
  prog.add(Copy(initialCount, count));
  graph.setTileMapping(validLabel, 0);
  graph.setTileMapping(validTime, 0);
  graph.setTileMapping(initialCount, 0);
  graph.setTileMapping(count, 0);
  graph.connect(vertex["validLabel"], validLabel);
  graph.connect(vertex["validTime"], validTime);
  graph.connect(vertex["count"], count);

  graph.setInitialValue(vertex["maxT"], maxT);
  graph.setInitialValue(vertex["numClasses"], numClasses);
  graph.setInitialValue(vertex["blankClass"], blankClass);
  graph.setInitialValue(vertex["labelOffset"], 0);
  graph.setInitialValue(vertex["timeOffset"], 0);

  auto labelWithPreviousOrNext = findingAlpha
                                     ? concat(prevSymbol.reshape({1}), labels)
                                     : concat(labels, prevSymbol.reshape({1}));
  graph.connect(vertex["label"], labelWithPreviousOrNext);
  graph.connect(vertex["probabilities"], probabilities.flatten());

  if (vertexToTest == TestType::ALPHA) {
    graph.connect(vertex["alphas"], result.flatten());
    graph.connect(vertex["alphaPrevTime"], prevTime.flatten());
    graph.connect(vertex["alphaPrevLabel"], prevLabel.flatten());
    graph.connect(vertex["loss"], loss);

    graph.connect(vertex["alphaPrevLabelOut"], prevLabel.flatten());
    graph.connect(vertex["alphaPrevLabelTime"], prevLabel.flatten());
  } else if (vertexToTest == TestType::BETA) {
    graph.connect(vertex["betas"], result.flatten());
    graph.connect(vertex["betaPrevTime"], prevTime.flatten());
    graph.connect(vertex["betaPrevLabel"], prevLabel.flatten());

    graph.connect(vertex["betaPrevLabelOut"], prevLabel.flatten());
    graph.connect(vertex["betaPrevLabelTime"], prevLabel.flatten());
  } else if (vertexToTest == TestType::GRAD_GIVEN_ALPHA) {
    graph.connect(vertex["grads"], result.flatten());
    graph.connect(vertex["alphas"], initialAlphaOrBeta.flatten());
    graph.connect(vertex["betaPrevTime"], prevTime.flatten());
    graph.connect(vertex["betaPrevLabel"], prevLabel.flatten());

    graph.connect(vertex["betaPrevPartition"], prevTime.flatten());
    graph.connect(vertex["betaPrevLabelOut"], prevLabel.flatten());
    graph.connect(vertex["betaPrevLabelTime"], prevLabel.flatten());
  } else if (vertexToTest == TestType::GRAD_GIVEN_BETA) {
    graph.connect(vertex["grads"], result.flatten());
    graph.connect(vertex["betas"], initialAlphaOrBeta.flatten());
    graph.connect(vertex["alphaPrevTime"], prevTime.flatten());
    graph.connect(vertex["alphaPrevLabel"], prevLabel.flatten());
    graph.connect(vertex["loss"], loss);

    graph.connect(vertex["alphaPrevPartition"], prevTime.flatten());
    graph.connect(vertex["alphaPrevLabelOut"], prevLabel.flatten());
    graph.connect(vertex["alphaPrevLabelTime"], prevLabel.flatten());
  }

  graph.setTileMapping(probabilities, 0);
  graph.setTileMapping(labels, 0);
  graph.setTileMapping(result, 0);
  graph.setTileMapping(vertex, 0);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;
  std::unique_ptr<char[]> rawProbabilities, rawResult, rawLabels,
      rawInitialAlphaOrBeta, rawPrevTime, rawPrevLabel;

  rawProbabilities = allocateHostMemoryForTensor(
      probabilities, "probabilities", graph, uploadProg, downloadProg, tmap);
  rawLabels = allocateHostMemoryForTensor(labels, "labels", graph, uploadProg,
                                          downloadProg, tmap);
  rawResult = allocateHostMemoryForTensor(result, "result", graph, uploadProg,
                                          downloadProg, tmap);

  copy(target, transpose(input.input), inType, rawProbabilities.get());
  copy(target, input.idx.data(), input.idx.size(), labels.elementType(),
       rawLabels.get());

  // Initialise alpha or beta if needed by the test, otherwise uninitialised
  if (resultIsGrad) {
    rawInitialAlphaOrBeta =
        allocateHostMemoryForTensor(initialAlphaOrBeta, "initialAlphaOrBeta",
                                    graph, uploadProg, downloadProg, tmap);
  } else {
    rawInitialAlphaOrBeta = allocateHostMemoryForTensor(
        result, "initialResult", graph, uploadProg, downloadProg, tmap);
  }
  copy(target, transpose(initialValues), outType, rawInitialAlphaOrBeta.get());
  boost::multi_array<double, 2> zeroResult(
      boost::extents[result.dim(0)][result.dim(1)]);
  std::fill(zeroResult.data(), zeroResult.data() + zeroResult.num_elements(),
            log::probabilityZero);
  copy(target, zeroResult, outType, rawResult.get());
  // Initialise the first Timeslice input of the vertex - in practice this could
  // be "carried over" from a previous vertex alpha or beta calculation
  rawPrevTime = allocateHostMemoryForTensor(prevTime, "prevTime", graph,
                                            uploadProg, downloadProg, tmap);
  boost::multi_array<double, 2> prevTimeInit(
      boost::extents[prevTime.dim(0)][prevTime.dim(1)]);
  std::fill(prevTimeInit.data(),
            prevTimeInit.data() + prevTimeInit.num_elements(),
            log::probabilityZero);

  if (findingAlpha) {
    // 1st symbol probability=1
    prevTimeInit[0][0] = log::probabilityOne;
    copy(target, prevTimeInit, outType, rawPrevTime.get());
  } else {
    // last symbol probability=1
    prevTimeInit[0][extendedLabelLen - 1] = log::probabilityOne;
    copy(target, prevTimeInit, outType, rawPrevTime.get());
  }

  rawPrevLabel = allocateHostMemoryForTensor(prevLabel, "prevLabel", graph,
                                             uploadProg, downloadProg, tmap);
  boost::multi_array<double, 2> prevLabelInit(
      boost::extents[prevLabel.dim(0)][prevLabel.dim(1)]);
  std::fill(prevLabelInit.data(),
            prevLabelInit.data() + prevLabelInit.num_elements(),
            log::probabilityZero);
  copy(target, prevLabelInit, outType, rawPrevLabel.get());

  prog.add(Execute(cs));
  std::optional<TempDir> tempDir;
  poplar::OptionFlags engineOptions;
  if (profile) {
    tempDir.emplace(TempDir::create());
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", tempDir->getPath());
  }
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, engineOptions);
  attachStreams(engine, tmap);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();
  });
  boost::multi_array<double, 2> output(
      boost::extents[result.dim(0)][result.dim(1)]);
  copy(target, outType, rawResult.get(), output);

  if (profile && deviceType != DeviceType::Cpu) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }
  return transpose(output);
}

int main(int argc, char **argv) {
  // Default input parameters.
  DeviceType deviceType = DeviceType::IpuModel2;
  bool verbose = false;
  bool profile = false;
  unsigned blankClass = 0;
  unsigned testTime = 15;
  unsigned testSymbols = 3;
  unsigned numClasses = 4;
  unsigned timeStep = 0;
  TestType vertexToTest = TestType::ALPHA;
  Type inType = FLOAT;
  Type outType = FLOAT;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     deviceTypeHelp)
    ("in-type", po::value(&inType)->default_value(inType),
     "Vertex input data type")
    ("out-type", po::value(&outType)->default_value(outType),
     "Vertex output data type")
    ("profile", po::value(&profile)->default_value(profile),
     "Show profile report")
    ("test", po::value(&vertexToTest)->default_value(vertexToTest),
     "Test: alpha, beta, gradGivenAlpha, gradGivenBeta")
    ("blank-class", po::value(&blankClass)->default_value(blankClass),
     "Index of the blank symbol. Range 0 to (num-classes-1)")
    ("test-symbols", po::value(&testSymbols)->default_value(testSymbols),
     "Test length (symbols)")
    ("time", po::value(&testTime)->default_value(testTime),
     "Test length (time)")
    ("num-classes", po::value(&numClasses)->default_value(numClasses),
     "Classes in the alphabet including blank")
    ("time-step", po::value(&timeStep)->default_value(timeStep),
     "The timestep (loop count) to process")
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
  // Needed to set default arguments.
  po::notify(vm);
  if (testTime < testSymbols) {
    throw poputil::poplibs_error("The test time must be >= sequence symbols");
  }
  if (blankClass >= numClasses) {
    throw poputil::poplibs_error("The blank class must be in the range 0 to "
                                 "(number of classes - 1)");
  }

  RandomUtil rand{42};
  auto [input, label] = getRandomTestInput<double>(
      testTime, testTime, testSymbols, numClasses, blankClass, true, rand);
  auto test = InputSequence<double>{transpose(input), label, numClasses};

  print("Test sequence:", test.idx, blankClass, verbose);

  // Produce a sensible input by calling the model.
  // This will contain all alpha/beta inputs for a gradGivenAlpha/Beta vertex
  // test, or all but the timeStep to be tested for alpha/beta vertices
  auto initialOutputType = (vertexToTest == TestType::ALPHA ||
                            vertexToTest == TestType::GRAD_GIVEN_ALPHA)
                               ? TestType::ALPHA
                               : TestType::BETA;
  auto initialOutput =
      gradReference<double>(test, blankClass, initialOutputType, false);
  if (vertexToTest == TestType::ALPHA || vertexToTest == TestType::BETA) {
    // For alpha, beta vertices an effective test can by made by masking out
    // the one timeslice that the vertex should be calculating
    initialOutput = maskTimeSteps(initialOutput, timeStep, true);
  }
  auto reference =
      gradReference<double>(test, blankClass, vertexToTest, verbose);
  auto output = gradIPU(test, timeStep, blankClass, initialOutput, vertexToTest,
                        inType, outType, deviceType, profile);

  double relativeTolerance = inType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
  double absoluteTolerance = inType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;

  // When finding alpha, beta some results aren't relevant. Mask them out
  if (vertexToTest == TestType::ALPHA || vertexToTest == TestType::BETA) {
    print("IPU result:", output, extendedLabels(test.idx, blankClass),
          blankClass, verbose);
    reference = maskResults(reference);
    output = maskResults(output);
  } else {
    print("IPU result:", output, blankClass, verbose);
    // Mask out all but the timestep processed.
    // TODO - This is a bit of a cheat, only the 1st (alpha) or last (beta)
    // timestep works at the moment, plus we should verify the whole output
    // is untouched.  Suggest doing this when making assembler vertices
    output = maskTimeSteps(output, timeStep, false);
    reference = maskTimeSteps(reference, timeStep, false);
  }

  bool success = checkIsClose("result", output, reference, relativeTolerance,
                              absoluteTolerance);
  if (!success) {
    std::cerr << "Data mismatch\n";
  }
  return !success;
}
