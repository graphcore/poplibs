// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDefUtil.hpp>
#include <popnn/codelets.hpp>
#include "poputil/VertexTemplates.hpp"
#include "poplibs_test/NonLinearity.hpp"
#include "poplibs_test/Util.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <boost/program_options.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popnn;
using namespace poputil;
using namespace poplibs_test::util;

#define TOL 0.1 //tolerance of 0.1%
#define FLOAT_ATOL 1e-20
#define HALF_ATOL 1e-7

namespace {

// Generated with python: "{0:-,.6f}".format(random.gauss(0.0,5.0))
constexpr static std::size_t randomDataSize = 80;
constexpr static double randomData[randomDataSize] = {
  -0.737873, 1.337075, 8.437973, -10.654166, -4.325676,
  9.487553, 8.889291, 1.709754, -4.326430, -0.426860,
  -2.501665, -0.299281, 0.831253, 0.549082, -9.290357,
  -6.706331, 4.499326, 12.251322, -1.957114, 0.592880,
  4.177108, -2.815978, -5.566706, 1.880811, 6.710350,
  9.380615, 4.871368, 2.216989, 2.792609, 2.474920,
  3.169892, -1.546323, -6.200352, -5.285461, -4.760159,
  0.523091, 1.635472, 3.255665, 1.603656, -2.059597,
  -3.134368, -6.930725, 1.465806, -2.113396, -4.939822,
  -1.009831, -5.018216, -4.275927, -5.259424, 1.409921,
  7.581671, -3.363830, -7.479782, -0.867524, 4.391104,
  -5.352319, 2.532336, 3.769678, 7.958588, -8.574877,
  -1.584704, -6.517893, 8.309027, -3.836431, 3.404698,
  -2.815457, -2.627222, 0.508437, -8.954427, 5.862103,
  1.590211, -0.167946, 9.862661, -0.853928, 1.191383,
  -6.853818, 5.237251, -1.092760, 0.851126, -1.905695
};

constexpr static std::size_t randomData2Size = 80;
constexpr static double randomData2[randomData2Size] = {
  -4.436526, 3.106630, 1.293025, -6.476578, -6.447246,
  -5.495446, 2.864524, 5.506601, -1.118131, -1.967583,
  8.106135, 1.792813, -0.594200, 4.334451, -1.202842,
  6.522413, 0.696654, -1.322245, -4.257638, -8.676723,
  -7.290822, -0.062777, -10.468264, 6.252668, 0.612885,
  -6.255880, -1.359411, -5.116864, 7.431683, 5.348917,
  0.253156, 0.977697, 4.575610, 1.229057, -0.919852,
  6.160684, -0.223279, -8.867977, -6.295284, -1.953921,
  -2.577365, 14.365989, 2.315338, -6.359479, -3.551872,
  -0.330124, 7.465863, -8.430678, 3.860931, -4.650391,
  -11.242988, -5.981461, 6.576061, 1.947509, -11.811246,
  0.936855, -5.088598, -3.710995, -7.859832, -2.770553,
  1.481630, -8.869824, 1.375009, 0.838196, -6.784614,
  5.024838, -1.321908, -0.647572, -8.963158, 4.039382,
  3.718576, 2.268444, -3.604734, -4.731974, 15.841928,
  -1.603268, -1.221412, -1.057963, -11.249434, 1.300697
};

struct SliceDesc {
  std::size_t offset;
  std::size_t numElements;
};

void doTest(const DeviceType &deviceType,
            const Type &dataType, const NonLinearityType &nlType) {
  Device device = createTestDevice(deviceType);
  const auto &target = device.getTarget();
  Graph graph(device);
  popnn::addCodelets(graph);

  const auto vectorWidth = target.getVectorWidth(dataType);
  const auto numWorkers = target.getNumWorkerContexts();

  // Once we have hit 3 full 64-bit loops for all workers +
  // up to 32-bits leading we will have covered all paths through the codelet.
  const auto maxElements =
    (vectorWidth * numWorkers) * 3 +
    (vectorWidth / 2);
  assert(randomDataSize >= maxElements);
  assert(randomData2Size >= maxElements);

  auto acts = graph.addVariable(dataType, {maxElements});
  auto outgrad = graph.addVariable(dataType, {maxElements});
  auto ingrad = graph.addVariable(dataType, {maxElements});
  graph.setTileMapping(acts, 0);
  graph.setTileMapping(outgrad, 0);
  graph.setTileMapping(ingrad, 0);

  // Generate some test data
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto rawHostActsIn =
    allocateHostMemoryForTensor(acts, "test acts", graph, uploadProg,
                                downloadProg, tmap);
  auto rawHostGradOut =
    allocateHostMemoryForTensor(outgrad, "test outgrad", graph, uploadProg,
                                downloadProg, tmap);
  auto rawHostGradIn =
    allocateHostMemoryForTensor(ingrad, "test ingrad", graph, uploadProg,
                                downloadProg, tmap);

  boost::multi_array<double, 1>
    hostActsIn(boost::extents[maxElements]),
    hostGradOut(boost::extents[maxElements]);
  std::copy(&randomData[0], &randomData[maxElements], hostActsIn.data());
  std::copy(&randomData2[0], &randomData2[maxElements], hostGradOut.data());

  boost::multi_array<double, 1>
    modelActsOut(boost::extents[maxElements]),
    modelGradIn(boost::extents[maxElements]);
  poplibs_test::nonLinearity(
      nlType, hostActsIn.data(), modelActsOut.data(), maxElements);
  std::copy_n(hostGradOut.data(), maxElements, modelGradIn.data());
  poplibs_test::bwdNonLinearity(
      nlType, hostActsIn.data(), modelGradIn.data(), maxElements);

  const auto fwdVertexClass =
    templateVertex("popnn::NonLinearitySupervisor", dataType, nlType);
  const auto bwdVertexClass =
    templateVertex("popnn::NonLinearityGradSupervisor", dataType, nlType);

  std::vector<Program> programs;
  std::vector<SliceDesc> programSlices;
  // For each possible offset up to multiples of 8 byte alignment
  // for this type. Because we have guaranteed 4-byte start alignment
  // only bother testing at 4-byte offsets.
  const auto elementSize = target.getTypeSize(dataType);
  for (std::size_t offset = 0;
       offset < vectorWidth;
       offset += std::max(1ul, 4 / elementSize)) {
    for (std::size_t numElements = 1;
         offset + numElements < maxElements;
         ++numElements) {
      // Get a slice into our common tensor to perform the non-linearity on
      auto actsTestSlice = acts.slice(offset, offset + numElements);
      auto outgradTestSlice = outgrad.slice(offset, offset + numElements);
      auto ingradTestSlice = ingrad.slice(offset, offset + numElements);

      auto fwdCS = graph.addComputeSet("cs_fwd_" + std::to_string(offset) +
                                       "_" + std::to_string(numElements));
      auto fwdV = graph.addVertex(fwdCS, fwdVertexClass);
      graph.setTileMapping(fwdV, 0);
      graph.connect(fwdV["data"], actsTestSlice);
      graph.setInitialValue(fwdV["n"], numElements);

      auto bwdCS = graph.addComputeSet("cs_bwd_" + std::to_string(offset) +
                                       "_" + std::to_string(numElements));
      auto bwdV = graph.addVertex(bwdCS, bwdVertexClass);
      graph.setTileMapping(bwdV, 0);
      graph.connect(bwdV["out"], actsTestSlice);
      graph.connect(bwdV["outGrad"], outgradTestSlice);
      graph.connect(bwdV["inGrad"], ingradTestSlice);
      graph.setInitialValue(bwdV["n"], numElements);

      programs.push_back(
        Sequence(
          Execute(bwdCS),
          Execute(fwdCS)
        ));
      programSlices.push_back(SliceDesc{
        offset,
        numElements
      });
    }
  }
  const auto numTests = programs.size();
  const auto uploadProgIndex = programs.size();
  programs.push_back(uploadProg);
  const auto downloadProgIndex = programs.size();
  programs.push_back(downloadProg);

  Engine e(graph, programs, OptionFlags{
    { "target.workerStackSizeInBytes", "0x100" }
  });
  e.load(device);
  attachStreams(e, tmap);

  boost::multi_array<double, 1>
    hostActsOut(boost::extents[maxElements]);
  boost::multi_array<double, 1>
    hostGradIn(boost::extents[maxElements]);
  const auto relativeTolerance = TOL;
  const auto absoluteTolerance =
    dataType == FLOAT ? FLOAT_ATOL
                      : HALF_ATOL;
  for (std::size_t testId = 0; testId < numTests; ++testId) {
    copy(target, hostActsIn, dataType, rawHostActsIn.get());
    copy(target, hostGradOut, dataType, rawHostGradOut.get());
    e.run(uploadProgIndex);
    e.run(testId);
    e.run(downloadProgIndex);
    copy(target, dataType, rawHostActsIn.get(), hostActsOut);
    copy(target, dataType, rawHostGradIn.get(), hostGradIn);

    auto &testDesc = programSlices[testId];
    bool validation = true;
    validation &=
      checkIsClose("fwd_" + std::to_string(testDesc.offset) +
                   "_" + std::to_string(testDesc.numElements),
                   hostActsOut.data() + testDesc.offset,
                   {testDesc.numElements},
                   modelActsOut.data() + testDesc.offset,
                   testDesc.numElements,
                   relativeTolerance, absoluteTolerance);
    validation &=
      checkIsClose("bwd_" + std::to_string(testDesc.offset) +
                   "_" + std::to_string(testDesc.numElements),
                   hostGradIn.data() + testDesc.offset,
                   {testDesc.numElements},
                   modelGradIn.data() + testDesc.offset,
                   testDesc.numElements,
                   relativeTolerance, absoluteTolerance);
    if (!validation)
      throw std::runtime_error("Results validation failed");
  }
}


} // end anonymous namespace

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type dataType;
  NonLinearityType nlType;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("data-type",
     po::value<Type>(&dataType)->required(),
     "Data type for the non-linearity")
    ("nl-type",
     po::value<NonLinearityType>(&nlType)->required(),
     "Non-linearity type");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  try {
    doTest(deviceType, dataType, nlType);
  } catch (std::exception &e) {
    std::cerr << "Test failed: " << e.what() << "\n";
    return 1;
  }
}
