// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CTCInferenceCodeletTest

#include "CTCInferenceCodeletTestConnection.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/CTCInferenceDefs.hpp>
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
using namespace poplar_test;
using namespace poplibs_support;
using namespace popnn::ctc_infer;
using namespace poputil;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

enum class VertexType {
  GENERATE_COPY,
  GENERATE_EXTEND,
  MERGE,
  RANK,
  REDUCE,
  UPDATE,
  OUTPUT
};

std::ostream &operator<<(std::ostream &os, const VertexType &test) {
  if (test == VertexType::GENERATE_COPY) {
    return os << "generate_copy";
  } else if (test == VertexType::GENERATE_EXTEND) {
    return os << "generate_extend";
  } else if (test == VertexType::MERGE) {
    return os << "merge";
  } else if (test == VertexType::RANK) {
    return os << "rank";
  } else if (test == VertexType::REDUCE) {
    return os << "reduce";
  } else if (test == VertexType::UPDATE) {
    return os << "update";
  } else if (test == VertexType::OUTPUT) {
    return os << "output";
  } else {
    throw poputil::poplibs_error("Unknown vertex type");
  }
}

std::istream &operator>>(std::istream &is, VertexType &test) {
  std::string token;
  is >> token;
  if (token == "generate_copy") {
    test = VertexType::GENERATE_COPY;
  } else if (token == "generate_extend") {
    test = VertexType::GENERATE_EXTEND;
  } else if (token == "merge") {
    test = VertexType::MERGE;
  } else if (token == "rank") {
    test = VertexType::RANK;
  } else if (token == "reduce") {
    test = VertexType::REDUCE;
  } else if (token == "update") {
    test = VertexType::UPDATE;
  } else if (token == "output") {
    test = VertexType::OUTPUT;
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
              << ", pTotal: " << candidate.pTotal << "])\n";
  }
}
bool beamHistoryIsClose(const BeamHistory &actual,
                        const BeamHistory &expected) {
  if (actual.nextIndexToAssign != expected.nextIndexToAssign) {
    return false;
  }
  for (unsigned t = 0; t < expected.nextIndexToAssign; t++) {
    for (unsigned b = 0; b < expected.symbols.size(); b++) {
      if (expected.symbols[b][t] != actual.symbols[b][t] ||
          *expected.parents[b][t] != *actual.parents[b][t]) {
        return false;
      }
    }
  }
  return true;
}

template <typename ActualFPType, typename ExpectedFPType>
bool beamProbabilitiesAreClose(
    const std::vector<BeamProbability<ActualFPType>> &actual,
    const std::vector<BeamProbability<ExpectedFPType>> &expected,
    double relativeTolerance) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (unsigned i = 0; i < expected.size(); i++) {
    bool pbClose = checkIsClose<ExpectedFPType>(actual[i].pb, expected[i].pb,
                                                relativeTolerance);
    bool pnbClose = checkIsClose<ExpectedFPType>(actual[i].pnb, expected[i].pnb,
                                                 relativeTolerance);
    if (!pnbClose || !pbClose) {
      return false;
    }
  }
  return true;
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
                                      relativeTolerance) &&
         checkIsClose<ExpectedFPType>(actual.pTotal, expected.pTotal,
                                      relativeTolerance);
}

template <typename ActualFPType, typename ExpectedFPType>
bool candidatesAreClose(const std::vector<Candidate<ActualFPType>> &actual,
                        const std::vector<Candidate<ExpectedFPType>> &expected,
                        double relativeTolerance) {
  auto isClose = true;
  if (actual.size() != expected.size()) {
    isClose = false;
  } else {
    for (size_t i = 0; i < actual.size(); i++) {
      isClose &= candidateIsClose(actual[i], expected[i], relativeTolerance);
    }
  }
  return isClose;
}

template <typename ActualFPType, typename ExpectedFPType>
void debugPrint(const std::vector<Candidate<ActualFPType>> &actual,
                const std::vector<Candidate<ExpectedFPType>> &expected,
                bool isClose, bool verbose) {

  if (verbose) {
    std::cout << "\nOutput:\n";
    print(actual, voidSymbol);
    std::cout << "\nModel:\n";
    print(expected, voidSymbol);
  }

  if (!verbose && !isClose) {
    std::cerr << "\nMismatch:\n";
    std::cerr << "\nActual:\n";
    print(actual, voidSymbol);
    std::cerr << "\nExpected:\n";
    print(expected, voidSymbol);
  }
}

template <typename ExpectedFPType>
std::vector<Candidate<ExpectedFPType>>
selectCopyCandidates(const std::vector<Candidate<ExpectedFPType>> &expected,
                     unsigned beam) {
  std::vector<Candidate<ExpectedFPType>> selected;
  for (const auto candidate : expected) {
    if (candidate.addend == voidSymbol) {
      if (candidate.beam == beam) {
        selected.push_back(candidate);
      }
    }
  }
  return selected;
}

template <typename ExpectedFPType>
std::vector<Candidate<ExpectedFPType>>
selectExtendCandidates(const std::vector<Candidate<ExpectedFPType>> &expected,
                       unsigned addendClass) {
  std::vector<Candidate<ExpectedFPType>> selected;
  for (const auto candidate : expected) {
    if (candidate.addend == addendClass) {
      selected.push_back(candidate);
    }
  }
  return selected;
}

template <typename ExpectedFPType>
std::vector<Candidate<ExpectedFPType>>
selectMergeCandidates(const std::vector<Candidate<ExpectedFPType>> &actual) {
  std::vector<Candidate<ExpectedFPType>> selected;
  // Don't compare those that are close to zero (close to as when adding 2
  // log::probabilityZero we don't quite get zero)
  const ExpectedFPType thresholdForZeroProb = log::probabilityZero + 100;
  for (const auto candidate : actual) {
    if (candidate.pb > thresholdForZeroProb ||
        candidate.pnb > thresholdForZeroProb) {
      selected.push_back(candidate);
    }
  }
  return selected;
}

template <typename ExpectedFPType>
std::vector<Candidate<ExpectedFPType>>
createReduceCandidates(const std::vector<Candidate<ExpectedFPType>> &actual,
                       unsigned rank, unsigned partitions,
                       unsigned activePartition) {

  std::vector<Candidate<ExpectedFPType>> toReduce(partitions);
  for (unsigned i = 0; i < partitions; i++) {
    if (i == activePartition) {
      toReduce[i] = actual[rank];
    } else {
      toReduce[i] = {0, 0, 0.0, 0.0, 0.0};
    }
  }
  return toReduce;
}

template <typename FPType>
std::pair<Candidate<FPType>, std::vector<Candidate<FPType>>>
getCopyAndExtendCandidates(const std::vector<Candidate<FPType>> &candidates,
                           unsigned mergedCopyBeam, unsigned mergedExtendBeam) {
  Candidate<FPType> copyCandidate;
  std::vector<Candidate<FPType>> extendCandidates;
  for (const auto candidate : candidates) {
    if (candidate.addend == voidSymbol && candidate.beam == mergedCopyBeam) {
      copyCandidate = candidate;
    }
    if (candidate.addend != voidSymbol && candidate.beam == mergedExtendBeam) {
      extendCandidates.push_back(candidate);
    }
  }
  return std::make_pair(copyCandidate, extendCandidates);
}

BeamHistory toIpuFormat(const BeamHistory &beamHistory, unsigned timestep) {
  const auto beamwidth = beamHistory.symbols.size();
  const auto maxT = beamHistory.symbols[0].size();
  // There is an extra timestep in the IPU format to provide an initial state
  auto ipuHistory = BeamHistory(beamwidth, maxT + 1);
  // First timestep:
  // beam 0 has parent = 0, symbol = void
  // Other beams have parent = 0, and symbol = unique symbol not in the set
  // of allowed symbols
  ipuHistory.incrementIndex();
  ipuHistory.parents[0][0] = 0;
  ipuHistory.symbols[0][0] = voidSymbol;
  for (unsigned i = 1; i < beamwidth; i++) {
    ipuHistory.parents[i][0] = 0;
    ipuHistory.symbols[i][0] = voidSymbol - i;
  }

  for (unsigned i = 0; i < timestep + 1; i++) {
    for (unsigned j = 0; j < beamwidth; j++) {
      const unsigned idx = i + 1;
      if (beamHistory.symbols[j][i] == voidSymbol) {
        // For a void symbol, reference the previous parent (already in the
        // ipu history) and repeat the symbol that is there with it
        const auto previous = *beamHistory.parents[j][i];
        ipuHistory.symbols[j][idx] = ipuHistory.symbols[previous][idx - 1];
        ipuHistory.parents[j][idx] = *ipuHistory.parents[previous][idx - 1];
      } else {
        // For a non void symbol, reference the parent, converting to a flat
        // index and copy the symbol
        const auto previousTimestepOffset = i * beamwidth;
        ipuHistory.parents[j][idx] =
            *beamHistory.parents[j][i] + previousTimestepOffset;
        ipuHistory.symbols[j][idx] = beamHistory.symbols[j][i];
      }
    }
    ipuHistory.incrementIndex();
  }
  return ipuHistory;
}

std::vector<unsigned> currentBeamOutputLengths(const BeamHistory &beamHistory,
                                               unsigned timestep) {
  const auto beamwidth = beamHistory.symbols.size();
  std::vector<unsigned> expectedOutputLength(2 * beamwidth);
  for (unsigned i = 0; i < beamwidth; i++) {
    if (timestep & 1) {
      expectedOutputLength[i + beamwidth] =
          beamHistory.getOutputSequence(i).size();
    } else {
      expectedOutputLength[i] = beamHistory.getOutputSequence(i).size();
    }
  }
  return expectedOutputLength;
}

int main(int argc, char **argv) {
  unsigned seed = 42;
  DeviceType deviceType = DeviceType::IpuModel2;
  VertexType vertexType = VertexType::GENERATE_COPY;
  Type inType = FLOAT;
  Type partialsType = FLOAT;
  Type outType = FLOAT;
  unsigned maxT = 15;
  unsigned numClassesIncBlank = 5;
  unsigned beamwidth = 2;
  unsigned blankClass = 0;
  unsigned timestep = 0;
  unsigned beam = 0;
  boost::optional<unsigned> baseSequenceLength = boost::none;
  boost::optional<unsigned> addendClass = boost::none;
  boost::optional<unsigned> mergeCopyBeam = boost::none;
  boost::optional<unsigned> mergeExtendBeam = boost::none;
  boost::optional<unsigned> reducePartitions = boost::none;
  boost::optional<unsigned> reduceActivePartition = boost::none;
  boost::optional<unsigned> updateInvalidCandidate = boost::none;

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
     "Vertex type to test: generate, merge, rank, reduce, update")
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
    ("beam", po::value(&beam)->default_value(beam),
     "The beam to output when testing the GenerateOutput vertex.\n"
     "The beam to make a copy candidate from when testing the "
     "GenerateCopyCandidate vertex.")
    ("blank-class", po::value(&blankClass)->default_value(blankClass),
     "Index of the blank symbol. Range 0 to (num-classes - 1)")
    ("addend-class", po::value(&addendClass),
     "Symbol to make the addend when extending or making copy candidates."
     " Range 0 to (num-classes - 1) avoiding the blank-class")
    ("merge-copy-beam", po::value(&mergeCopyBeam),
     "Parent beam of the copy candidate to choose when merging candidates."
     " Range 0 to (beamwidth-1")
    ("merge-extend-beam", po::value(&mergeExtendBeam),
     "Parent beam of the extend candidates to choose when merging candidates."
     " Range 0 to (beamwidth-1")
    ("reduce-partitions", po::value(&reducePartitions),
     "Number of partitions to split the reduction vertex test input into")
    ("reduce-active-partition", po::value(&reduceActivePartition),
     "The active partition of the input to test the ranking vertex with."
     "  Defaults to the last partition")
    ("update-invalid-candidate", po::value(&updateInvalidCandidate),
     "The index of invalid candidate to test the update vertex")
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
  if (addendClass && (mergeCopyBeam || mergeExtendBeam)) {
    std::cerr << "Specifying addend-class and merge-beam(s) -"
                 " use either addend-class or both merge-beams\n";
    return 1;
  }
  if (addendClass && *addendClass == blankClass) {
    std::cerr << "addend-class must not equal blank-class\n";
    return 1;
  }
  if (mergeCopyBeam != mergeExtendBeam) {
    std::cerr << "Select either both or neither of merge-copy-beam and"
                 " merge-extend-beam\n";
    return 1;
  }
  if (vertexType == VertexType::REDUCE) {
    if (!reducePartitions) {
      std::cerr << "Number of partitions to use when testing reduce vertex is"
                   " not defined\n";
      return 1;
    }
    if (reduceActivePartition && *reduceActivePartition >= reducePartitions) {
      std::cerr << "The reduce partition specified must be less than the"
                   " number of reduce partitions\n";
      return 1;
    }
  }
  if (vertexType != VertexType::UPDATE && updateInvalidCandidate) {
    std::cerr << "Specifying an invalid candidate is specific to the update"
                 " vertex test\n";
    return 1;
  }
  if (!addendClass) {
    // Pick an arbitrary addend that isn't a blank
    addendClass = blankClass == 0 ? 1 : 0;
  }
  const bool profile = vm.count("profile");
  const bool verbose = vm.count("verbose");

  if (timestep >= maxT) {
    throw poputil::poplibs_error("The test timestep must be < maxT");
  }

  RandomUtil rand{seed};

  auto [logProbs, label] =
      getRandomTestInput<float>(maxT, maxT, *baseSequenceLength,
                                numClassesIncBlank, blankClass, false, rand);

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

    if (verbose && i == timestep - 1 && vertexType != VertexType::OUTPUT) {
      std::cout << "\nTimestep " << i << " (Before vertex under test):\n";
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

  if (vertexType == VertexType::GENERATE_COPY ||
      vertexType == VertexType::GENERATE_EXTEND) {
    if (verbose) {
      std::cout << "\nAll Generated Candidates:\n";
      print(modelCandidates, voidSymbol);
    }
    const auto testGenerateCopyVertex = vertexType == VertexType::GENERATE_COPY;
    const auto candidates = runGenerateCandidatesCodelet<float, float>(
        graph, device, deviceType, inType, partialsType, logProbs, timestep,
        beamProbs, beamHistory, *addendClass, beam, blankClass,
        testGenerateCopyVertex, profile);

    const auto compareCandidates =
        vertexType == VertexType::GENERATE_COPY
            ? selectCopyCandidates(modelCandidates, beam)
            : selectExtendCandidates(modelCandidates, *addendClass);

    auto isClose =
        candidatesAreClose(candidates, compareCandidates, relTolerance);
    debugPrint(candidates, compareCandidates, isClose, verbose);
    if (!isClose) {
      std::cerr << "Data mismatch\n";
      return 1;
    }
    return 0;
  }

  if (vertexType == VertexType::MERGE) {
    // Merge events are fairly unusual so if there is one and the addend wasn't
    // selected, choose the addend that corresponds to the first merge that
    // happened
    const auto mergedPairs =
        listMergeableCandidates(modelCandidates, beamHistory);

    const auto [mergedCopy, mergedExtend] = [&]() {
      boost::optional<unsigned> copy;
      boost::optional<unsigned> extend;
      if (mergedPairs.size() >= 1) {
        copy = mergedPairs[0].first;
        extend = mergedPairs[0].second;
      }
      return std::tuple<boost::optional<unsigned>, boost::optional<unsigned>>(
          copy, extend);
    }();

    auto pickMergeBeam = [&](const boost::optional<unsigned> beam,
                             const boost::optional<unsigned> candidate,
                             unsigned defaultBeam) {
      if (beam) {
        return *beam;
      }
      if (candidate) {
        return modelCandidates[*candidate].beam;
      }
      return defaultBeam;
    };

    const auto chosenMergeCopyBeam =
        pickMergeBeam(mergeCopyBeam, mergedCopy, 0);
    const auto chosenMergeExtendBeam =
        pickMergeBeam(mergeExtendBeam, mergedExtend, 1);

    auto [copyCandidate, extendCandidates] = getCopyAndExtendCandidates(
        modelCandidates, chosenMergeCopyBeam, chosenMergeExtendBeam);

    auto expectedOutputLength = currentBeamOutputLengths(beamHistory, timestep);

    const auto lastBeamOutputSym =
        beamHistory.getLastOutput(copyCandidate.beam);
    auto candidates = runMergeCandidatesCodelet<float>(
        graph, device, deviceType, inType, partialsType, extendCandidates,
        copyCandidate, timestep + 1, blankClass,
        toIpuFormat(beamHistory, timestep - 1), expectedOutputLength,
        lastBeamOutputSym, numClassesIncBlank, profile);

    extendCandidates.insert(extendCandidates.begin(), copyCandidate);
    const auto modelMergedCandidates =
        mergeEquivalentCandidates(extendCandidates, beamHistory, true);

    if (verbose) {
      if (mergedCopy && mergedExtend) {
        std::cout << "Merge on the timestep for test - copy candidate index:"
                  << *mergedCopy << " extend candidate index:" << *mergedExtend
                  << "\n";
      }
      std::cout << "\nAll model candidates:\n";
      print(modelCandidates, voidSymbol);
      std::cout << "\nCandidates selected to merge:\n";
      print(extendCandidates, voidSymbol);
    }
    auto candidatesToCompare = selectMergeCandidates(candidates);
    auto modelCandidatesToCompare =
        selectMergeCandidates(modelMergedCandidates);
    auto isClose = candidatesAreClose(candidatesToCompare,
                                      modelCandidatesToCompare, relTolerance);
    debugPrint(candidatesToCompare, modelCandidatesToCompare, isClose, verbose);
    if (!isClose) {
      std::cerr << "Data mismatch\n";
      return 1;
    }
    return 0;
  }

  if (vertexType == VertexType::RANK) {
    const auto sortedCandidates = runRankCandidatesCodelet<float>(
        graph, device, deviceType, partialsType, modelMergedCandidates,
        beamwidth, timestep + 1, profile);

    if (verbose) {
      std::cerr << "\nPreviously merged candidates:\n";
      print(modelMergedCandidates, voidSymbol);
    }

    auto isClose = candidatesAreClose(sortedCandidates, modelPrunedCandidates,
                                      relTolerance);
    debugPrint(sortedCandidates, modelPrunedCandidates, isClose, verbose);
    if (!isClose) {
      std::cerr << "Data mismatch\n";
      return 1;
    }
    return 0;
  }
  if (vertexType == VertexType::REDUCE) {
    if (!reduceActivePartition) {
      reduceActivePartition = *reducePartitions - 1;
    }
    const auto candidatesToReduce = createReduceCandidates(
        modelPrunedCandidates, 0, *reducePartitions, *reduceActivePartition);
    const auto reducedCandidates = runReduceCandidatesCodelet<float>(
        graph, device, deviceType, partialsType, candidatesToReduce, beamwidth,
        timestep + 1, profile);
    std::vector<Candidate<float>> expectedReducedCandidate(1);
    expectedReducedCandidate[0] = modelPrunedCandidates[0];

    if (verbose) {
      std::cerr << "\nCandidates to reduce:\n";
      print(candidatesToReduce, voidSymbol);
    }

    auto isClose = candidatesAreClose(reducedCandidates,
                                      expectedReducedCandidate, relTolerance);
    debugPrint(reducedCandidates, expectedReducedCandidate, isClose, verbose);
    if (!isClose) {
      std::cerr << "Data mismatch\n";
      return 1;
    }
    return 0;
  }

  if (vertexType == VertexType::UPDATE) {
    const auto previousOutputLength =
        currentBeamOutputLengths(beamHistory, timestep);
    auto candidatesToUpdate = modelPrunedCandidates;
    if (updateInvalidCandidate) {
      if (*updateInvalidCandidate >= candidatesToUpdate.size()) {
        // The input random data has resulted in a pruned
        // candidate list that is too small to complete the test.
        std::cerr << "Index of the invalid candidate ("
                  << *updateInvalidCandidate << ") is greater than or equal to "
                  << "the number of candidates (" << candidatesToUpdate.size()
                  << ")\n";
        return 1;
      }
      candidatesToUpdate[*updateInvalidCandidate].addend = invalidSymbol;
    }
    auto [ipuBeamHistory, ipuBeamProbs, ipuBeamLength] =
        runUpdateCodelet<float>(
            graph, device, deviceType, inType, partialsType, candidatesToUpdate,
            timestep + 1, toIpuFormat(beamHistory, timestep - 1),
            previousOutputLength, beamProbs, blankClass, profile);
    // Complete the model implementation for comparison, now the beam history
    // has been used by the codelet
    applyCandidates(beamHistory, beamProbs, candidatesToUpdate, true);
    // Convert to the same form that we expect the IPU output to be in
    if (updateInvalidCandidate) {
      beamProbs[*updateInvalidCandidate].pb = log::probabilityZero;
      beamProbs[*updateInvalidCandidate].pnb = log::probabilityZero;
    }
    auto modelBeamHistory = toIpuFormat(beamHistory, timestep);
    auto modelOutputLength =
        currentBeamOutputLengths(beamHistory, timestep + 1);
    for (unsigned i = 0; i < modelOutputLength.size(); i++) {
      modelOutputLength[i] += previousOutputLength[i];
    }
    if (verbose) {
      std::cout << "\nPruned Candidates:\n";
      print(modelPrunedCandidates, voidSymbol);
      std::cout << "\nOutput:\n";
      print(ipuBeamHistory);
      std::cout << "\nModel:\n";
      print(modelBeamHistory);
      for (unsigned i = 0; i < beamProbs.size(); i++) {
        std::cout << "Model: [pnb: " << beamProbs[i].pnb
                  << ", pb: " << beamProbs[i].pb << "]";
        std::cout << " Output: [pnb: " << ipuBeamProbs[i].pnb
                  << ", pb: " << ipuBeamProbs[i].pb << "]\n";
      }
      std::cout << "Previous lengths: ";
      for (unsigned i = 0; i < previousOutputLength.size(); i++) {
        std::cout << previousOutputLength[i] << ", ";
      }
      std::cout << "\nModel lengths:    ";
      for (unsigned i = 0; i < modelOutputLength.size(); i++) {
        std::cout << modelOutputLength[i] << ", ";
      }
      std::cout << "\nOutput lengths:   ";
      for (unsigned i = 0; i < ipuBeamLength.size(); i++) {
        std::cout << ipuBeamLength[i] << ", ";
      }
      std::cout << "\n";
    }
    if (!beamHistoryIsClose(modelBeamHistory, ipuBeamHistory) ||
        !beamProbabilitiesAreClose(beamProbs, ipuBeamProbs, relTolerance) ||
        !checkEqual("Lengths", ipuBeamLength.data(), {ipuBeamLength.size()},
                    modelOutputLength.data(), modelOutputLength.size())) {
      std::cerr << "Data mismatch\n";
      return 1;
    }
  }

  if (vertexType == VertexType::OUTPUT) {
    // Complete the model implementation for comparison
    applyCandidates(beamHistory, beamProbs, modelPrunedCandidates, true);
    if (verbose) {
      std::cout << "\nBeam history:\n";
      print(beamHistory);
    }
    const auto expectedOutput = beamHistory.getOutputSequence(beam);

    const auto ipuOutput = runGenerateOutputCodelet(
        graph, device, deviceType, timestep + 1,
        toIpuFormat(beamHistory, timestep), expectedOutput.size(), beam,
        numClassesIncBlank, profile);
    if (verbose) {
      std::cout << "Actual:  ";
      print(ipuOutput);
      std::cout << "Expected:";
      print(expectedOutput);
      std::cout << "\n";
    }
    if (ipuOutput != expectedOutput) {
      std::cerr << "Data mismatch\n";
      return 1;
    }
  }

  return 0;
}
