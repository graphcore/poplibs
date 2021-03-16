// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CTCInferenceCodeletTest

#include "CTCInferenceGenerateCandidates.hpp"
#include "CTCInferenceMergeCandidates.hpp"
#include "CTCInferenceSelectCandidates.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/CTCUtil.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/MatrixTransforms.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <iomanip>
#include <random>

namespace po = boost::program_options;

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::ctc;
using namespace poplibs_test;
using namespace poplibs_test::matrix;
using namespace poplibs_test::util;
using namespace poplibs_support;
using namespace poputil;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

enum class VertexType { GENERATE, MERGE, SELECT, UPDATE };

std::ostream &operator<<(std::ostream &os, const VertexType &test) {
  if (test == VertexType::GENERATE) {
    return os << "generate";
  } else if (test == VertexType::MERGE) {
    return os << "merge";
  } else if (test == VertexType::SELECT) {
    return os << "select";
  } else if (test == VertexType::UPDATE) {
    return os << "update";
  } else {
    throw poputil::poplibs_error("Unknown vertex type");
  }
}

std::istream &operator>>(std::istream &is, VertexType &test) {
  std::string token;
  is >> token;
  if (token == "generate") {
    test = VertexType::GENERATE;
  } else if (token == "merge") {
    test = VertexType::MERGE;
  } else if (token == "select") {
    test = VertexType::SELECT;
  } else if (token == "update") {
    test = VertexType::UPDATE;
  } else {
    throw poputil::poplibs_error(std::string{"Unknown vertex type: `"} + token +
                                 std::string{"`"});
  }
  return is;
}

// TODO move common
template <typename FPType>
void print(const boost::multi_array<FPType, 2> &in, unsigned blank) {
  std::cout << "        ";
  for (unsigned i = 0; i < in.size(); i++) {
    std::cout << std::setw(11) << (std::string{"t"} + std::to_string(i));
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

// TODO Move common
template <typename FPType>
void print(const std::vector<Candidate<FPType>> &candidates,
           unsigned voidSymbol) {
  for (const auto &candidate : candidates) {
    std::cout << "(Beam=" << candidate.beam;
    std::cout << ", addend: ";
    if (candidate.addend == voidSymbol) {
      std::cout << " ";
    } else {
      std::cout << candidate.addend;
    }
    std::cout << std::fixed << std::setprecision(4)
              << " [pnb: " << candidate.pnb << ", pb: " << candidate.pb
              << "])\n";
  }
}

// TODO abs tolerance in poputil
template <typename ActualFPType, typename ExpectedFPType>
bool candidateIsClose(const Candidate<ActualFPType> &actual,
                      const Candidate<ExpectedFPType> &expected,
                      double relativeTolerance) {
  return (actual.addend == expected.addend) && (actual.beam == expected.beam) &&
         checkIsClose<ExpectedFPType>(actual.pnb, expected.pnb,
                                      relativeTolerance) &&
         checkIsClose<ExpectedFPType>(actual.pb, expected.pb,
                                      relativeTolerance);
}

template <typename ActualFPType, typename ExpectedFPType>
bool candidatesAreClose(const std::vector<Candidate<ActualFPType>> &actual,
                        const std::vector<Candidate<ExpectedFPType>> &expected,
                        double relativeTolerance) {
  // TODO: better comparison
  if (actual.size() != expected.size()) {
    return false;
  }
  auto isClose = true;
  for (size_t i = 0; i < actual.size(); i++) {
    isClose &= candidateIsClose(actual[i], expected[i], relativeTolerance);
  }
  return isClose;
}

template <typename FPType>
std::pair<boost::multi_array<FPType, 2>, std::vector<unsigned>>
getRandomTestInput(unsigned maxT, unsigned baseSequenceLength,
                   unsigned numClassesIncBlank, unsigned blankClass,
                   RandomUtil &rand) {
  auto [input, label] = provideInputWithPath<FPType>(
      baseSequenceLength, maxT, maxT, numClassesIncBlank, blankClass, rand);

  return {log::log(transpose(log::softMax(transpose(input)))), label};
}

int main(int argc, char **argv) {
  unsigned seed = 42;
  DeviceType deviceType = DeviceType::IpuModel2;
  VertexType vertexType = VertexType::GENERATE;
  Type inType = FLOAT;
  Type partialsType = FLOAT;
  Type outType = FLOAT;
  unsigned maxT = 15;
  unsigned numClassesIncBlank = 5;
  unsigned beamwidth = 2;
  unsigned blankClass = 0;
  unsigned timestep = 0;
  boost::optional<unsigned> baseSequenceLength = boost::none;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     deviceTypeHelp)
    ("seed", po::value(&seed)->default_value(seed),
     "Seed used for random number generators")
    ("vertex-type", po::value(&vertexType)->default_value(vertexType),
     "Vertex type to test: generate, merge, select, update")
    ("in-type", po::value(&inType)->default_value(inType),
     "Vertex input data type")
    ("partials-type", po::value(&partialsType)->default_value(partialsType),
     "Vertex partials data type")
    ("out-type", po::value(&outType)->default_value(outType),
     "Vertex output data type")
    ("max-time", po::value(&maxT)->default_value(maxT),
     "Maximum length of time that is planned for the op")
    ("sequence-length", po::value(&baseSequenceLength),
     "Sequence length for which to increase probability for random test input"
     " Defaults to --max-time/2")
    ("num-classes", po::value(&numClassesIncBlank)->default_value(numClassesIncBlank),
     "Classes in the alphabet including blank")
    ("beamwidth", po::value(&beamwidth)->default_value(beamwidth),
     "Beamwidth to use for the op")
    ("blank-class", po::value(&blankClass)->default_value(blankClass),
     "Index of the blank symbol. Range 0 to (num-classes - 1)")
    ("timestep", po::value(&timestep)->default_value(timestep),
     "The timestep (loop count) to process")
    ("profile", "Show profile report")
    ("verbose", "Provide debug printout");
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
  po::notify(vm);
  if (!baseSequenceLength) {
    baseSequenceLength = ceildiv(maxT, 2u);
  }
  const bool profile = vm.count("profile");
  const bool verbose = vm.count("verbose");

  // TODO: stop repeating defn
  const auto voidSymbol = std::numeric_limits<unsigned>::max();

  RandomUtil rand{seed};
  auto [logProbs, label] = getRandomTestInput<float>(
      maxT, *baseSequenceLength, numClassesIncBlank, blankClass, rand);

  if (verbose) {
    std::cout << "\nLabel:\n";
    print(label, blankClass);
    std::cout << "\nInput:\n";
    print(logProbs, blankClass);
  }

  // TODO refactor common Initial state
  BeamHistory beamHistory{beamwidth, maxT};
  std::vector<BeamProbability<float>> beamProbs{};
  beamProbs.push_back({log::probabilityOne, log::probabilityZero});
  for (size_t i = 1; i < beamwidth; i++) {
    beamProbs.push_back({log::probabilityZero, log::probabilityZero});
  }

  for (unsigned i = 0; i < timestep; i++) {
    const auto candidates = generateCandidates(
        transpose(logProbs), i, beamProbs, beamHistory, blankClass, true);
    const auto mergedCandidates =
        mergeEquivalentCandidates(candidates, beamHistory, true);
    const auto sortedCandidates = sortCandidates(mergedCandidates, true);
    const auto prunedCandidates =
        pruneCandidates(sortedCandidates, beamwidth, true);
    applyCandidates(beamHistory, beamProbs, prunedCandidates, true);
    if (verbose) {
      std::cout << "\nTimestep " << i << ":\n";
      std::cout << "\nBeam history:\n";
      print(beamHistory);
      std::cout << "\nAppended candidates:\n";
      print(prunedCandidates, voidSymbol);
    }
  }

  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);

  // TODO half
  const auto relTolerance = FLOAT_REL_TOL;

  const auto modelCandidates = generateCandidates(
      transpose(logProbs), timestep, beamProbs, beamHistory, blankClass, true);
  const auto modelMergedCandidates =
      mergeEquivalentCandidates(modelCandidates, beamHistory, true);
  const auto modelSortedCandidates =
      sortCandidates(modelMergedCandidates, true);
  const auto modelPrunedCandidates =
      pruneCandidates(modelSortedCandidates, beamwidth, true);
  // TODO this function mutates state so can't be run here!
  // applyCandidates(beamHistory, beamProbs, prunedCandidates, true);

  if (vertexType == VertexType::GENERATE) {
    const auto candidates = runGenerateCandidatesCodelet<float, float>(
        graph, device, deviceType, inType, partialsType, logProbs, timestep,
        beamProbs, beamHistory, blankClass, profile);

    if (verbose) {
      std::cout << "\nOutput:\n";
      print(candidates, voidSymbol);
      std::cout << "\nModel:\n";
      print(modelCandidates, voidSymbol);
    }

    if (!candidatesAreClose(candidates, modelCandidates, relTolerance)) {
      if (!verbose) {
        std::cerr << "\nMismatch:\n";
        std::cerr << "\nActual:\n";
        print(candidates, voidSymbol);
        std::cerr << "\nExpected:\n";
        print(modelCandidates, voidSymbol);
      } else {
        std::cerr << "Data mismatch\n";
      }
      return 1;
    }
    return 0;
  }

  if (vertexType == VertexType::MERGE) {
    auto candidates = runMergeCandidatesCodelet<float>(
        graph, device, deviceType, inType, partialsType, modelCandidates,
        timestep, beamHistory, blankClass, profile);

    if (verbose) {
      std::cout << "\nGenerated Candidates:\n";
      print(modelCandidates, voidSymbol);
      std::cout << "\nOutput:\n";
      print(candidates, voidSymbol);
      std::cout << "\nModel:\n";
      print(modelMergedCandidates, voidSymbol);
    }

    auto paddedModelMergedCandidates = modelMergedCandidates;
    // TODO better comparison to avoid filling in missing candidates
    if (candidates.size() != paddedModelMergedCandidates.size()) {
      for (unsigned i = 0; i < candidates.size(); i++) {
        if ((paddedModelMergedCandidates.size() <= i) ||
            (candidates[i].addend != paddedModelMergedCandidates[i].addend) ||
            (candidates[i].beam != paddedModelMergedCandidates[i].beam)) {
          paddedModelMergedCandidates.insert(
              paddedModelMergedCandidates.begin() + i,
              {candidates[i].beam, candidates[i].addend, log::probabilityZero,
               log::probabilityZero});
        }
      }
    }

    if (!candidatesAreClose(candidates, paddedModelMergedCandidates,
                            relTolerance)) {
      if (!verbose) {
        std::cerr << "\nMismatch:\n";
        std::cerr << "\nActual:\n";
        print(candidates, voidSymbol);
        std::cerr << "\nExpected:\n";
        print(paddedModelMergedCandidates, voidSymbol);
      } else {
        std::cerr << "Data mismatch\n";
      }
      return 1;
    }
    return 0;
  }

  if (vertexType == VertexType::SELECT) {
    const auto sortedCandidates = runSelectCandidatesCodelet<float>(
        graph, device, deviceType, partialsType, modelMergedCandidates,
        beamwidth, profile);

    if (verbose) {
      std::cerr << "\nPreviously merged candidates:\n";
      print(modelMergedCandidates, voidSymbol);
      std::cout << "\nOutput:\n";
      print(sortedCandidates, voidSymbol);
      std::cout << "\nModel:\n";
      print(modelPrunedCandidates, voidSymbol);
    }

    if (!candidatesAreClose(sortedCandidates, modelPrunedCandidates,
                            relTolerance)) {
      if (!verbose) {
        std::cerr << "\nMismatch:\n";
        std::cerr << "\nActual:\n";
        print(sortedCandidates, voidSymbol);
        std::cerr << "\nExpected:\n";
        print(modelPrunedCandidates, voidSymbol);
      } else {
        std::cerr << "Data mismatch\n";
      }
      return 1;
    }
    return 0;
  }

  return 0;
}
