// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CTCLossCodeletTest
#include <iomanip>
#include <random>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCLoss.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

namespace po = boost::program_options;

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::ctc_loss;
using namespace poplibs_test;
using namespace poplibs_test::util;
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

// Print a sequence, inserting `-` for the blank character
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

InputSequence<double> getRandomTestInput(size_t timesteps, size_t testSymbols,
                                         unsigned alphabetSizeIncBlank,
                                         bool blankIsZero) {
  std::mt19937 gen;
  const unsigned seed = 1;
  gen.seed(seed);

  std::uniform_int_distribution<> rand(0, alphabetSizeIncBlank - 2);
  std::vector<unsigned> idx;
  for (size_t i = 0; i < testSymbols; i++) {
    unsigned symbol = static_cast<unsigned>(blankIsZero) + rand(gen);
    idx.push_back(symbol);
  }

  std::uniform_int_distribution<> randInput(0, 10);
  boost::multi_array<double, 2> input(
      boost::extents[alphabetSizeIncBlank][timesteps]);
  for (unsigned i = 0; i < input.shape()[0]; i++) {
    for (unsigned j = 0; j < input.shape()[1]; j++) {
      input[i][j] = randInput(gen);
    }
  }
  return {softMax(input), idx, alphabetSizeIncBlank};
}

template <typename FPType>
boost::multi_array<FPType, 2>
gradReference(const InputSequence<FPType> &test, unsigned blankIndex,
              TestType vertexToTest, bool verbose) {

  auto paddedSequence = extendedLabels(test.idx, blankIndex);

  print("Log Softmax in", test.input, blankIndex, verbose);
  boost::multi_array<FPType, 2> logSequence(
      boost::extents[paddedSequence.size()][test.input.shape()[1]]);
  poplibs_test::embedding::multiSlice(test.input, paddedSequence, logSequence);

  print("Reference sequence", logSequence, paddedSequence, blankIndex, verbose);

  auto alphaLog = alpha(logSequence, paddedSequence, blankIndex, true);
  if (vertexToTest == TestType::ALPHA) {
    print("Reference alphas", alphaLog, paddedSequence, blankIndex, verbose);
    return alphaLog;
  }

  auto betaLog = beta(logSequence, paddedSequence, blankIndex, true);
  if (vertexToTest == TestType::BETA) {
    print("Reference betas", betaLog, paddedSequence, blankIndex, verbose);
    return betaLog;
  }
  auto expandedGradient = expandedGrad(logSequence, alphaLog, betaLog,
                                       paddedSequence, blankIndex, true);
  print("Expanded Reference gradient", expandedGradient, paddedSequence,
        blankIndex, verbose);

  auto gradient = grad(logSequence, alphaLog, betaLog, paddedSequence,
                       test.alphabetSizeIncBlank, blankIndex, true);
  print("Reference gradient", gradient, blankIndex, verbose);

  std::cout << "\n";
  return gradient;
}

boost::multi_array<double, 2>
gradIPU(const InputSequence<double> &input, unsigned blankIndex,
        const boost::optional<boost::multi_array<double, 2>> &initialValues,
        TestType vertexToTest, Type inType, Type outType,
        const DeviceType &deviceType, bool profile) {

  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  const auto symbolsLen = input.idx.size();
  const auto extendedSymbolsLen = 2 * symbolsLen + 1;
  const auto maxT = input.input.shape()[1];
  const auto nSymbols = input.input.shape()[0];

  const auto resultIsGrad = (vertexToTest == TestType::GRAD_GIVEN_ALPHA ||
                             vertexToTest == TestType::GRAD_GIVEN_BETA);

  auto probabilities =
      graph.addVariable(inType, {maxT, nSymbols}, "probabilities");
  auto symbols = graph.addVariable(UNSIGNED_SHORT, {symbolsLen}, "symbols");
  auto result = graph.addVariable(
      outType, {maxT, resultIsGrad ? nSymbols : extendedSymbolsLen}, "result");
  auto tempAlphaOrBeta = graph.addVariable(
      outType, {resultIsGrad ? 2u : 1u, extendedSymbolsLen}, "tempAlphaOrBeta");
  graph.setTileMapping(tempAlphaOrBeta, 0);

  Tensor initialAlphaOrBeta;
  if (resultIsGrad) {
    initialAlphaOrBeta = graph.addVariable(outType, {maxT, extendedSymbolsLen},
                                           "initialAlphaOrBeta");
    graph.setTileMapping(initialAlphaOrBeta, 0);
  }

  auto cs = graph.addComputeSet("cs");
  std::string vertexName;
  if (vertexToTest == TestType::ALPHA) {
    vertexName =
        templateVertex("popnn::CTCAlpha", inType, outType, UNSIGNED_SHORT);
  } else if (vertexToTest == TestType::BETA) {
    vertexName =
        templateVertex("popnn::CTCBeta", inType, outType, UNSIGNED_SHORT);
  } else if (vertexToTest == TestType::GRAD_GIVEN_ALPHA) {
    vertexName = templateVertex("popnn::CTCGradGivenAlpha", inType, outType,
                                UNSIGNED_SHORT);
  } else if (vertexToTest == TestType::GRAD_GIVEN_BETA) {
    vertexName = templateVertex("popnn::CTCGradGivenBeta", inType, outType,
                                UNSIGNED_SHORT);
  }
  auto vertex = graph.addVertex(cs, vertexName);

  graph.setInitialValue(vertex["maxT"], maxT);
  graph.setInitialValue(vertex["nSymbols"], nSymbols);
  graph.setInitialValue(vertex["blankSymbol"], blankIndex);
  graph.connect(vertex["symbols"], symbols);
  graph.connect(vertex["probabilities"], probabilities.flatten());

  if (vertexToTest == TestType::ALPHA) {
    graph.connect(vertex["alphas"], result.flatten());
    graph.connect(vertex["alphaTemp"], tempAlphaOrBeta.flatten());
  } else if (vertexToTest == TestType::BETA) {
    graph.connect(vertex["betas"], result.flatten());
    graph.connect(vertex["betaTemp"], tempAlphaOrBeta.flatten());
  } else if (vertexToTest == TestType::GRAD_GIVEN_ALPHA) {
    graph.connect(vertex["grads"], result.flatten());
    graph.connect(vertex["alphas"], initialAlphaOrBeta.flatten());
    graph.connect(vertex["betaTemp"], tempAlphaOrBeta.flatten());
  } else if (vertexToTest == TestType::GRAD_GIVEN_BETA) {
    graph.connect(vertex["grads"], result.flatten());
    graph.connect(vertex["betas"], initialAlphaOrBeta.flatten());
    graph.connect(vertex["alphaTemp"], tempAlphaOrBeta.flatten());
  }

  graph.setTileMapping(probabilities, 0);
  graph.setTileMapping(symbols, 0);
  graph.setTileMapping(result, 0);
  graph.setTileMapping(vertex, 0);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawProbabilities, rawResult, rawSymbols,
      rawInitialAlphaOrBeta, rawTemp;

  rawProbabilities = allocateHostMemoryForTensor(
      probabilities, "probabilities", graph, uploadProg, downloadProg, tmap);
  rawSymbols = allocateHostMemoryForTensor(symbols, "symbols", graph,
                                           uploadProg, downloadProg, tmap);
  rawResult = allocateHostMemoryForTensor(result, "result", graph, uploadProg,
                                          downloadProg, tmap);

  copy(target, transpose(input.input), inType, rawProbabilities.get());
  copy(target, input.idx.data(), input.idx.size(), symbols.elementType(),
       rawSymbols.get());

  // Initialise alpha or beta if needed by the test, otherwise uninitialised
  if (initialValues) {
    rawInitialAlphaOrBeta =
        allocateHostMemoryForTensor(initialAlphaOrBeta, "initialAlphaOrBeta",
                                    graph, uploadProg, downloadProg, tmap);
    copy(target, transpose(initialValues.get()), outType,
         rawInitialAlphaOrBeta.get());
    boost::multi_array<double, 2> zeroResult(
        boost::extents[result.dim(0)][result.dim(1)]);
    std::fill(zeroResult.data(), zeroResult.data() + zeroResult.num_elements(),
              log::min);
    copy(target, zeroResult, outType, rawResult.get());
  }
  // Initialise the first Timeslice input of the vertex - in practice this could
  // be "carried over" from a previous vertex alpha or beta calculation
  rawTemp = allocateHostMemoryForTensor(tempAlphaOrBeta, "tempAlphaOrBeta",
                                        graph, uploadProg, downloadProg, tmap);
  boost::multi_array<double, 2> temp(
      boost::extents[tempAlphaOrBeta.dim(0)][tempAlphaOrBeta.dim(1)]);
  std::fill(temp.data(), temp.data() + temp.num_elements(), log::min);
  if (vertexToTest == TestType::GRAD_GIVEN_BETA ||
      vertexToTest == TestType::ALPHA) {
    // 1st symbol = 0 (probability=1)
    temp[0][0] = 0;
    copy(target, temp, outType, rawTemp.get());
  } else {
    // last symbol = 0 (probability=1)
    temp[0][extendedSymbolsLen - 1] = 0;
    copy(target, temp, outType, rawTemp.get());
  }

  Sequence(prog);
  prog.add(Execute(cs));
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
  bool blankIsZero = false;
  bool verbose = false;
  bool profile = false;
  unsigned testTime = 15;
  unsigned testSymbols = 5;
  unsigned alphabetIndices = 4;
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
    ("zero-blank", po::value(&blankIsZero)->default_value(blankIsZero),
     "Index of the blank symbol = zero.  If false it is indices-1")
    ("test-symbols", po::value(&testSymbols)->default_value(testSymbols),
     "Test length (symbols)")
    ("time", po::value(&testTime)->default_value(testTime),
     "Test length (time)")
    ("indices", po::value(&alphabetIndices)->default_value(alphabetIndices),
     "Indices in the alphabet including blank")
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

  auto test =
      getRandomTestInput(testTime, testSymbols, alphabetIndices, blankIsZero);
  const unsigned blankIndex = blankIsZero ? 0 : alphabetIndices - 1;
  test.input = ctc_loss::log(test.input);
  print("Test sequence:", test.idx, blankIndex, verbose);

  // When testing gradGivenAlpha or Beta vertices, produce a sensible input by
  // calling the model, otherwise this data is unused.
  boost::optional<boost::multi_array<double, 2>> initialOutput;

  if (vertexToTest == TestType::GRAD_GIVEN_ALPHA) {
    initialOutput =
        gradReference<double>(test, blankIndex, TestType::ALPHA, false);
  } else if (vertexToTest == TestType::GRAD_GIVEN_BETA) {
    initialOutput =
        gradReference<double>(test, blankIndex, TestType::BETA, false);
  }

  auto reference =
      gradReference<double>(test, blankIndex, vertexToTest, verbose);
  auto output = gradIPU(test, blankIndex, initialOutput, vertexToTest, inType,
                        outType, deviceType, profile);

  double relativeTolerance = inType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
  double absoluteTolerance = inType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;

  // When finding alpha, beta some results aren't relevant. Mask them out
  if (vertexToTest == TestType::ALPHA || vertexToTest == TestType::BETA) {
    print("IPU result:", output, extendedLabels(test.idx, blankIndex),
          blankIndex, verbose);
    reference = maskResults(reference);
    output = maskResults(output);
  } else {
    print("IPU result:", output, blankIndex, verbose);
  }

  bool success = checkIsClose("result", reference, output, relativeTolerance,
                              absoluteTolerance);
  if (!success) {
    std::cerr << "Data mismatch\n";
  }
  return !success;
}
