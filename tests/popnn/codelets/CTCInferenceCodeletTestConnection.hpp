// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef poplibs_test_ctc_inference_codelet_test_connection_hpp
#define poplibs_test_ctc_inference_codelet_test_connection_hpp

#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/Util.hpp>

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>

namespace poplibs_test {
namespace ctc {

struct CandidateHandles {
  std::unique_ptr<char[]> parent;
  std::unique_ptr<char[]> addend;
  std::unique_ptr<char[]> probNonBlank;
  std::unique_ptr<char[]> probBlank;
  boost::optional<std::unique_ptr<char[]>> probTotal = boost::none;
};

struct BeamHandles {
  std::unique_ptr<char[]> pnb;
  std::unique_ptr<char[]> pb;
  std::unique_ptr<char[]> pTotal;
  std::unique_ptr<char[]> lastOutput;
};

enum class BeamScalars { BLANK, NON_BLANK, BLANK_AND_NON_BLANK };

template <typename InputType, typename PartialsType>
std::vector<Candidate<PartialsType>> runGenerateCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type inType,
    poplar::Type partialsType,
    const boost::multi_array<InputType, 2> &logProbsIn, unsigned timestep,
    const std::vector<BeamProbability<PartialsType>> &beamProbs,
    const BeamHistory &beamHistory, unsigned classToMakeAddend, unsigned beam,
    unsigned blankClass, bool testGenerateCopyVertex, bool profile);

template <typename PartialsType>
std::vector<Candidate<PartialsType>> runMergeCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type inType,
    poplar::Type partialsType,
    const std::vector<Candidate<PartialsType>> &candidates,
    const Candidate<PartialsType> &copyCandidate, unsigned timestep,
    unsigned blankClass, const BeamHistory &beamHistory,
    const poplar::ArrayRef<unsigned> &outputLengths, unsigned lastBeamOutputSym,
    unsigned numClasses, bool profile);

template <typename PartialsType>
std::vector<Candidate<PartialsType>> runRankCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<PartialsType>> &candidates, unsigned beamwidth,
    unsigned timestep, bool profile);

template <typename PartialsType>
std::vector<Candidate<PartialsType>> runReduceCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<PartialsType>> &candidates, unsigned beamwidth,
    unsigned timestep, bool profile);

template <typename PartialsType>
std::tuple<BeamHistory, std::vector<BeamProbability<PartialsType>>,
           std::vector<unsigned>>
runUpdateCodelet(poplar::Graph &graph, poplibs_support::TestDevice &device,
                 poplibs_support::DeviceType deviceType, poplar::Type inType,
                 poplar::Type partialsType,
                 const std::vector<Candidate<PartialsType>> &candidates,
                 unsigned timestep, const BeamHistory &beamHistory,
                 const std::vector<unsigned> &beamLengthIn,
                 const std::vector<BeamProbability<PartialsType>> &beamProbs,
                 unsigned blankClass, bool profile);

std::vector<unsigned> runGenerateOutputCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, unsigned timestep,
    const BeamHistory &beamHistory, unsigned beamOutLength, unsigned outputBeam,
    unsigned numClassesIncBlank, bool profile);

CandidateHandles createAndConnectCandidates(
    poplar::Graph &graph, const poplar::VertexRef &vertex,
    const std::string &prefix, const poplar::Type &partialsType,
    const poplar::ArrayRef<std::size_t> &shape,
    poplar::program::Sequence &uploadProg,
    poplar::program::Sequence &downloadProg,
    std::vector<std::pair<std::string, poplar_test::HostMemory>> &tmap,
    bool includeTotalAndBlank = true);

BeamHandles createAndConnectBeamProbs(
    poplar::Graph &graph, const poplar::VertexRef &vertex,
    const poplar::Type &probsType, const poplar::ArrayRef<std::size_t> &shape,
    BeamScalars selectBlank, poplar::program::Sequence &uploadProg,
    poplar::program::Sequence &downloadProg,
    std::vector<std::pair<std::string, poplar_test::HostMemory>> &tmap);

} // namespace ctc
} // namespace poplibs_test

#endif
