// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <random>

#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/CTCLoss.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/Util.hpp>

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
  std::vector<unsigned> idx;
  unsigned blankClass;
};

template <typename FPType>
void testTraining(const InputSequence<FPType> &test, bool normalise,
                  bool verbose) {
  auto blankClass = test.blankClass;
  auto paddedSequence = test.idx;
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
    loss = loss / test.idx.size();
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
    if (allOutputPaths[paths[bestIndex]][i] == voidSymbol) {
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
        if (allOutputPaths[paths[i]][j] == voidSymbol) {
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
  if (test.idx != bestPath) {
    throw std::logic_error("Incorrect result");
  }
}
template <typename FPType>
void testBeamSearchInference(
    const InputSequence<FPType> &test, bool useLogArithmetic,
    unsigned beamwidth, boost::optional<std::vector<unsigned>> expectedSequence,
    boost::optional<double> expectedLogProb, bool verbose = false) {
  auto [probability, output] = infer<FPType>(
      test.input, test.blankClass, beamwidth, useLogArithmetic, verbose);

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
struct InputSequence<FPType> getRandomTestInput(size_t timesteps,
                                                unsigned numClassesIncBlank,
                                                unsigned seed) {

  unsigned blankClass = numClassesIncBlank - 1;
  std::vector<unsigned> idx;

  std::mt19937 gen;

  if (seed) {
    std::cout << "Using seed: " << seed << "\n";
    gen.seed(seed);
  } else {
    std::random_device rd;
    std::mt19937 seedGen(rd());
    std::uniform_int_distribution<> rand(0, INT_MAX);
    const auto randSeed = rand(seedGen);
    std::cout << "Generated seed:" << randSeed << "\n";
    gen.seed(randSeed);
  }
  std::uniform_int_distribution<> rand(0, numClassesIncBlank - 1);

  for (size_t i = 0; i < timesteps; i++) {
    unsigned symbol = rand(gen);
    idx.push_back(symbol);
  }
  const FPType pathInput = 2;
  FPType nonPathInput = 0;
  boost::multi_array<FPType, 2>
      input(boost::extents[numClassesIncBlank][idx.size()]);
  for (size_t s = 0; s < idx.size(); s++) {
    auto activeSymbolIdx = idx[s];
    for (size_t c = 0; c < numClassesIncBlank; c++) {
      if (activeSymbolIdx == c) {
        input[c][s] = pathInput;
      } else {
        input[c][s] = nonPathInput;
        nonPathInput -= 0.1;
      }
    }
  }
  std::cout << "Input sequence: ";
  print(idx, blankClass);
  return {log::softMax(input), extendedLabels(idx, blankClass, true),
          blankClass};
}

template <typename FPType>
InputSequence<FPType> parseInput(const std::vector<double> &input,
                                 const std::vector<unsigned> &shape) {
  const auto numClassesIncBlank = shape.at(0);
  const auto tSize = shape.at(1);
  boost::multi_array<FPType, 2> reshapedInput(
      boost::extents[numClassesIncBlank][tSize]);
  for (unsigned i = 0; i < input.size(); i++) {
    *(reshapedInput.data() + i) = input[i];
  }
  // Blank class always last but one for now
  return {reshapedInput, {}, shape.at(0) - 1};
}

int main(int argc, char **argv) {
  // Default input parameters.
  bool inference = false;
  bool useDoubles = true;
  bool useLogArithmetic = false;
  bool allPathsMethod = false;
  bool verbose = false;
  bool normalise = false;

  unsigned testNumber = 0;
  unsigned beamWidth = 3;
  unsigned randomTestLength = 15;
  unsigned numClassesIncBlank = 4;
  unsigned seed = 0;

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
    ("seed", po::value(&seed)->default_value(seed),
     "If random data provide a seed, if not one is chosen and displayed")
    ("test-length", po::value(&randomTestLength)->default_value(randomTestLength),
     "Test length (t) for random test sequences")
    ("num-classes", po::value(&numClassesIncBlank)->default_value(numClassesIncBlank),
     "Number of classes (including blank) to use for random test sequences")
    ("test", po::value(&testNumber)->default_value(testNumber),
     "Index of hand coded test example")
    ("all-paths", po::value(&allPathsMethod)->default_value(allPathsMethod),
     "Use all paths inference method for reference")
    ("normalise", po::value(&normalise)->default_value(normalise),
     "Normalise the loss")
    ("log", po::value(&useLogArithmetic)->default_value(useLogArithmetic),
     "Compute using log arithmetic")
    ("doubles", po::value(&useDoubles)->default_value(useDoubles),
     "Use double precision for reference")
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

  if (useLogArithmetic && !inference) {
    throw std::logic_error("Log arithmetic only supported with inference");
  }

  if (useDoubles) {
    const auto test = [&]() {
      auto result = input.val.empty()
                        ? getRandomTestInput<double>(randomTestLength,
                                                     numClassesIncBlank, seed)
                        : parseInput<double>(input.val, inputShape);

      if (useLogArithmetic) {
        result.input = log::log(result.input);
        if (verbose) {
          exportPrint("Log Softmax in", result.input);
        }
      }
      return result;
    }();
    if (verbose) {
      print("Test sequence:", test.idx, test.blankClass);
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
  } else {
    const auto test = [&]() {
      auto result = input.val.empty()
                        ? getRandomTestInput<float>(randomTestLength,
                                                    numClassesIncBlank, seed)
                        : parseInput<float>(input.val, inputShape);

      if (useLogArithmetic) {
        result.input = log::log(result.input);
        if (verbose) {
          exportPrint("Log Softmax in", result.input);
        }
      }
      return result;
    }();
    if (verbose) {
      print("Test sequence:", test.idx, test.blankClass);
      std::cout << "The blank symbol is: " << test.blankClass << "\n";
      if (useLogArithmetic) {
        print("Log Softmax input", test.input, test.blankClass);
      } else {
        print("Softmax input", test.input, test.blankClass);
      }
    }
    if (inference) {
      if (allPathsMethod) {
        testAllPathsInference<float>(test, useLogArithmetic, verbose);
      } else {
        testBeamSearchInference<float>(
            test, useLogArithmetic, beamWidth,
            expectedSequence
                ? boost::optional<std::vector<unsigned>>(expectedSequence->val)
                : boost::none,
            expectedLogProb, verbose);
      }
    } else {
      testTraining<float>(test, normalise, verbose);
    }
  }

  return 0;
}
