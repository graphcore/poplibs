// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <random>

#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/CTCInferenceDefs.hpp>
#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/CTCLoss.hpp>
#include <poplibs_test/CTCUtil.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/MatrixTransforms.hpp>
#include <poplibs_test/Util.hpp>
#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/random.hpp>

namespace po = boost::program_options;
namespace br = boost::random;

using namespace poplibs_support;
using namespace poplibs_test::ctc;
using namespace poplibs_test;
using namespace poplibs_test::util;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

// Print a sequence, inserting `-` for the blank class
void print(const std::string &prefix, const std::vector<unsigned> &idx,
           unsigned blankClass, bool verbose = true) {
  if (!verbose) {
    return;
  }
  std::cout << "\n" << prefix << " ";
  for (auto &i : idx) {
    if (i == blankClass) {
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
           const std::vector<unsigned> &paddedSequence, unsigned blankClass,
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
    if (paddedSequence[i] == blankClass) {
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
           unsigned blankClass, bool verbose = true) {
  if (!verbose) {
    return;
  }
  std::cout << "\n" << prefix << "\n        ";
  for (unsigned i = 0; i < in[0].size(); i++) {
    std::cout << "         t" << i;
  }

  for (unsigned i = 0; i < in.size(); i++) {
    if (i == blankClass) {
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
// Print the matrix `in` to export for comparison
template <typename FPType>
void exportPrint(const std::string &prefix,
                 const boost::multi_array<FPType, 2> &in) {
  std::cout << prefix << "\n";
  for (unsigned i = 0; i < in[0].size(); i++) {
    std::cout << "[[";
    for (unsigned j = 0; j < in.size(); j++) {
      std::cout << std::fixed << std::setw(10) << std::setprecision(4)
                << in[j][i] << ",";
    }
    std::cout << "]],\n";
  }
}

template <typename FPType>
boost::multi_array<FPType, 2> toLinear(const boost::multi_array<FPType, 2> &in,
                                       FPType scale) {
  boost::multi_array<FPType, 2> out(boost::extents[in.size()][in[0].size()]);
  for (unsigned i = 0; i < in.size(); i++) {
    for (unsigned j = 0; j < in[i].size(); j++) {
      out[i][j] = -scale * std::exp(in[i][j]);
    }
  }
  return out;
}
template <typename FPType>
boost::multi_array<FPType, 2> scale(const boost::multi_array<FPType, 2> &in,
                                    FPType scale) {
  boost::multi_array<FPType, 2> out(boost::extents[in.size()][in[0].size()]);
  for (unsigned i = 0; i < in.size(); i++) {
    for (unsigned j = 0; j < in[i].size(); j++) {
      out[i][j] = -scale * in[i][j];
    }
  }
  return out;
}
// Struct and function to return the test inputs
template <typename FPType> struct InputSequence {
  boost::multi_array<FPType, 2> input;
  std::vector<unsigned> extendedLabel;
  unsigned blankClass;
};

template <typename FPType>
void testTraining(const InputSequence<FPType> &test, bool normalise,
                  bool verbose) {
  auto blankClass = test.blankClass;
  auto paddedSequence = test.extendedLabel;
  auto validTimesteps = test.input.shape()[1];

  std::cout << "\n";

  // This can be used to copy into python/pytorch implementation for
  // comparison
  print("Log Softmax in", test.input, blankClass);
  boost::multi_array<FPType, 2> logSequence(
      boost::extents[paddedSequence.size()][test.input.shape()[1]]);
  poplibs_test::embedding::multiSlice(test.input, paddedSequence, logSequence);
  print("Sequence(log)", logSequence, paddedSequence, blankClass, verbose);

  auto alphaLog =
      alpha(logSequence, paddedSequence, blankClass, validTimesteps);
  print("Alphas(log)", alphaLog, paddedSequence, blankClass, verbose);
  print("Alphas", log::exp(alphaLog), paddedSequence, blankClass, verbose);
  auto lastTimeIndex = alphaLog.shape()[1] - 1;
  auto prodSum = log::add(alphaLog[alphaLog.size() - 1][lastTimeIndex],
                          alphaLog[alphaLog.size() - 2][lastTimeIndex]);

  auto loss = -prodSum;
  if (normalise) {
    loss = loss / test.extendedLabel.size();
  }
  std::cout << std::setprecision(6) << "\nP(sequence)=" << std::exp(prodSum)
            << "\nloss:" << loss << "\n";

  auto betaLog = beta(logSequence, paddedSequence, blankClass, validTimesteps);
  print("Betas(log)", betaLog, paddedSequence, blankClass, verbose);
  print("Betas", log::exp(betaLog), paddedSequence, blankClass, verbose);
  auto gradient =
      grad(logSequence, test.input, alphaLog, betaLog, paddedSequence,
           test.input.shape()[0], blankClass, validTimesteps, false);
  print("Gradient(log)", gradient, verbose);
  print("Gradient",
        toLinear(gradient, static_cast<FPType>(1 / std::exp(prodSum))),
        blankClass);
  std::cout << "\n";
}

template <typename FPType>
void testAllPathsInference(const InputSequence<FPType> &test,
                           bool useLogArithmetic, bool verbose) {
  unsigned timeSteps = test.input.shape()[1];
  unsigned sequenceLength = test.input.shape()[0];
  auto allPaths = findAllInputPaths(timeSteps, sequenceLength);
  print("AllInputPaths:", allPaths, UINT_MAX, verbose);
  auto allProbabilities =
      findAllInputPathProbabilities(test.input, allPaths, useLogArithmetic);
  auto allOutputPaths = inputToOutputPath(allPaths, test.blankClass);
  print("AllOutputPaths:", allOutputPaths, UINT_MAX, verbose);
  auto [paths, probabilities, instances] =
      mergePaths(allOutputPaths, allProbabilities, useLogArithmetic);
  if (verbose) {
    std::cout << "\nPaths:" << paths.size() << " probs:" << probabilities.size()
              << "\n";
  }

  // Find the index of the best output path
  auto max = std::max_element(probabilities.begin(), probabilities.end());
  const auto bestIndex = max - probabilities.begin();
  if (verbose) {
    std::cout << "Best has index:" << bestIndex << " Out sequence:";
  }
  std::vector<unsigned> bestPath;
  for (unsigned i = 0; i < allOutputPaths[paths[bestIndex]].size(); i++) {
    if (verbose) {
      std::cout << "B:" << allOutputPaths[paths[bestIndex]][i] << "\n";
    }
    if (allOutputPaths[paths[bestIndex]][i] == popnn::ctc_infer::voidSymbol) {
      break;
    }
    bestPath.push_back(allOutputPaths[paths[bestIndex]][i]);
  }
  print(bestPath, test.blankClass);
  if (useLogArithmetic) {
    std::cout << "Log(P) = " << (*max) << "\n";
    std::cout << "P = " << exp(*max) << "\n";
  } else {
    std::cout << "P = " << (*max) << "\n";
  }
  if (verbose) {
    for (unsigned i = 0; i < probabilities.size(); i++) {
      std::cout << "Path:";
      for (unsigned j = 0; j < allOutputPaths[paths[i]].size(); j++) {
        if (allOutputPaths[paths[i]][j] == popnn::ctc_infer::voidSymbol) {
          break;
        }
        std::cout << allOutputPaths[paths[i]][j] << ",";
      }
      std::cout << "    From paths:";
      for (unsigned j = 0; j < instances[i].size(); j++) {
        std::cout << instances[i][j] << ",";
      }
      std::cout << "    P:" << probabilities[i] << "\n";
    }
  }
  bestPath = extendedLabels(bestPath, test.blankClass, false);
  if (test.extendedLabel != bestPath) {
    throw std::logic_error("Incorrect result");
  }
}
template <typename FPType>
void testBeamSearchInference(
    const InputSequence<FPType> &test, bool useLogArithmetic,
    unsigned beamwidth, boost::optional<std::vector<unsigned>> expectedSequence,
    boost::optional<double> expectedLogProb, bool verbose = false) {
  auto [output, probability] =
      infer<FPType>(test.input, test.blankClass, beamwidth, 1, useLogArithmetic,
                    verbose)
          .at(0);

  std::cout << "Output:" << std::endl;
  print(output);
  std::cout << "P = " << std::exp(probability) << std::endl;
  std::cout << "Log(P) = " << probability << std::endl;

  if (expectedSequence) {
    if (!std::equal(output.begin(), output.end(), expectedSequence->begin())) {
      throw std::logic_error("Incorrect sequence");
    }
  }

  if (expectedLogProb) {
    bool matchesModel =
        checkIsClose<double>(probability, *expectedLogProb, FLOAT_REL_TOL);
    if (!matchesModel) {
      throw std::logic_error("Incorrect probability for sequence");
    }
  }
}

template <typename FPType>
InputSequence<FPType> parseInput(const std::vector<double> &input,
                                 const std::vector<unsigned> &shape,
                                 unsigned blankClass) {
  const auto numClassesIncBlank = shape.at(0);
  const auto tSize = shape.at(1);
  boost::multi_array<FPType, 2> reshapedInput(
      boost::extents[numClassesIncBlank][tSize]);
  for (unsigned i = 0; i < input.size(); i++) {
    *(reshapedInput.data() + i) = input[i];
  }
  return {reshapedInput, {}, blankClass};
}

int main(int argc, char **argv) {
  // Default input parameters.
  bool inference = false;
  bool useLogArithmetic = false;
  bool allPathsMethod = false;
  bool verbose = false;
  bool normalise = false;

  unsigned testNumber = 0;
  unsigned beamWidth = 3;
  unsigned randomTestLength = 15;
  unsigned blankClass = 0;
  unsigned numClassesIncBlank = 4;
  boost::optional<unsigned> seed = boost::none;
  boost::optional<unsigned> baseSequenceLength = boost::none;

  VectorOption<double> input;
  ShapeOption<unsigned> inputShape;
  boost::optional<VectorOption<unsigned>> expectedSequence;
  boost::optional<double> expectedLogProb;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("inference", po::value(&inference)->default_value(inference),
     "Test inference")
    ("beam-width", po::value(&beamWidth)->default_value(beamWidth),
     "Beam width for beam search (not all paths inference)")
    ("input", po::value<VectorOption<double>>(&input),
     "1-D array of softmax input")
    ("input-shape", po::value<ShapeOption<unsigned>>(&inputShape),
     "dimensions to reshape the input")
    ("expectedSequence", po::value<boost::optional<VectorOption<unsigned>>>(&expectedSequence),
     "Expected sequence")
    ("expectedLogProb", po::value<boost::optional<double>>(&expectedLogProb),
     "Expected log probability")
    ("seed", po::value(&seed),
     "If random data provide a seed, if not one is chosen and displayed")
    ("test-length", po::value(&randomTestLength)->default_value(randomTestLength),
     "Test length (t) for random test sequences")
    ("sequence-length", po::value(&baseSequenceLength),
     "Sequence length for which to increase probability for random test input"
     " Defaults to --test-length/2")
    ("num-classes", po::value(&numClassesIncBlank)->default_value(numClassesIncBlank),
     "Number of classes (including blank) to use for random test sequences")
    ("blank-class", po::value(&blankClass)->default_value(blankClass),
     "Index of the blank class")
    ("test", po::value(&testNumber)->default_value(testNumber),
     "Index of hand coded test example")
    ("all-paths", po::value(&allPathsMethod)->default_value(allPathsMethod),
     "Use all paths inference method for reference")
    ("normalise", po::value(&normalise)->default_value(normalise),
     "Normalise the loss")
    ("log", po::value(&useLogArithmetic)->default_value(useLogArithmetic),
     "Compute using log arithmetic")
    ("verbose", po::value(&verbose)->default_value(verbose),
     "More debug print");
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
  if (!baseSequenceLength) {
    baseSequenceLength = ceildiv(randomTestLength, 2u);
  }

  if (useLogArithmetic && !inference) {
    throw std::logic_error("Log arithmetic only supported with inference");
  }

  if (!seed) {
    seed = std::random_device{}();
  }
  std::cout << "Using seed: " << *seed << "\n";
  RandomUtil rand{*seed};

  const auto test = [&]() -> InputSequence<double> {
    if (input.val.empty()) {
      auto [inputProbs, label] = getRandomTestInput<double>(
          *baseSequenceLength, randomTestLength, randomTestLength,
          numClassesIncBlank, blankClass, !useLogArithmetic, rand);
      std::cout << "Input sequence: ";
      print(label, blankClass);
      auto extendedLabel = extendedLabels(label, blankClass, true);
      return {inputProbs, extendedLabel, blankClass};
    } else {
      auto test = parseInput<double>(input.val, inputShape, blankClass);
      test.input = log::log(test.input);
      return test;
    }
  }();

  if (verbose) {
    if (useLogArithmetic) {
      exportPrint("Log Softmax in", test.input);
    }
    print("Test sequence:", test.extendedLabel, test.blankClass);
    std::cout << "The blank symbol is: " << test.blankClass << "\n";
    if (useLogArithmetic) {
      print("Log Softmax input", test.input, test.blankClass);
    } else {
      print("Softmax input", test.input, test.blankClass);
    }
  }
  if (inference) {
    if (allPathsMethod) {
      testAllPathsInference<double>(test, useLogArithmetic, verbose);
    } else {
      testBeamSearchInference<double>(
          test, useLogArithmetic, beamWidth,
          expectedSequence
              ? boost::optional<std::vector<unsigned>>(expectedSequence->val)
              : boost::none,
          expectedLogProb, verbose);
    }
  } else {
    testTraining<double>(test, normalise, verbose);
  }

  return 0;
}
