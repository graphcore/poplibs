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
constexpr static std::size_t randomDataSize = 54;
constexpr static double randomData[randomDataSize] = {
  -1.363063, 6.590855, -1.808181, -7.897238, -4.266652,
  4.064484, -12.879016, -8.521756, -3.760860, 13.313155,
  9.388029, -3.281752, -0.848680, -3.044330, -5.974442,
  -4.652190, 0.683414, 0.468458, -2.227848, 3.479158,
  2.000612, -9.059770, -11.321158, 3.265349, 7.258343,
  -5.725096, 2.390649, -5.182225, 2.477468, -1.410790,
  5.631778, 1.608285, -3.547397, 4.153984, 7.495993,
  1.061097, 3.186139, -1.215145, 5.905051, 0.284197,
  3.458912, -10.597435, 3.889679, -4.992706, 8.237274,
  -3.864746, -2.701962, 9.659042, -2.789558, -0.937477,
  11.091992, 8.830758, -3.798772
};

constexpr static std::size_t randomData2Size = 54;
constexpr static double randomData2[randomData2Size] = {
  6.063585, -1.603059, 3.668168, -3.164358, -5.669303,
  6.042537, -1.827524, 3.141997, 7.839743, -0.449503,
  11.704780, 4.080705, 6.092508, 1.941723, -4.315962,
  -3.851170, -4.832400, -2.369191, 4.311729, 12.653507,
  0.090832, -5.663074, 11.635682, -1.027572, -1.115758,
  3.978761, 7.596279, -2.769141, 4.046201, -3.727586,
  3.798045, 4.005900, -0.328572, -2.739621, 5.095390,
  4.176589, 6.089753, 5.570235, 6.978266, 2.231976,
  -0.563166, 2.612484, -2.182849, 6.476901, 5.321058,
  3.173620, 1.060784, 7.694968, -3.347791, -8.521936,
  -2.760514, 6.953309, 2.192015
};

struct SliceDesc {
  std::vector<Interval> intervals;
};

bool doTest(const DeviceType &deviceType,
            const Type &dataType, const NonLinearityType &nlType) {
  Device device = createTestDevice(deviceType);
  const auto &target = device.getTarget();
  Graph graph(device);
  popnn::addCodelets(graph);

  const auto vectorWidth = target.getVectorWidth(dataType);

  constexpr auto maxRegions = 3;
  // In each inner region of the 2D tensor, test up to at least 3 full
  // 64-bit loops to catch all inner loop paths plus up to 64-bits of
  // trailing elements plus up to 32-bits of leading elements.
  const auto maxElementsPerRegion =
    (vectorWidth * 3) + (vectorWidth + (vectorWidth / 2));
  const auto maxElements =
    maxElementsPerRegion * maxRegions;
  assert(maxElements <= randomDataSize);
  assert(maxElements <= randomData2Size);

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
    allocateHostMemoryForTensor(acts, "test in", graph, uploadProg,
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
    modelActsOut(boost::extents[maxElements]);
  boost::multi_array<double, 1>
    modelGradIn(boost::extents[maxElements]);

  const auto fwdVertexClass =
    templateVertex("popnn::NonLinearity2D",
                   dataType, nlType);
  const auto bwdVertexClass =
    templateVertex("popnn::NonLinearityGrad2D",
                   dataType, nlType);

  std::vector<Program> programs;
  std::vector<SliceDesc> programSlices;
  unsigned uid = 0;
  for (std::size_t nRegions = 1; nRegions <= maxRegions; ++nRegions) {
    std::size_t currRegion = 0;
    std::vector<Interval> intervals(nRegions);
    bool filled = false;
    for (std::size_t offset = 0; offset < vectorWidth; ++offset) {
      for (std::size_t n = 1; n < maxElementsPerRegion - offset; ++n) {
        const auto regionOffset = currRegion * maxElementsPerRegion;
        intervals[currRegion] =
          {regionOffset + offset, regionOffset + offset + n};
        currRegion = (currRegion + 1) % nRegions;
        filled |= (currRegion == 0);

        // Only test if we've filled all the regions
        if (filled) {
          // Get slices into our variable
          auto actTestSlices = acts.slices(intervals);
          auto outgradTestSlices = outgrad.slices(intervals);
          auto ingradTestSlices = ingrad.slices(intervals);

          auto fwdCS = graph.addComputeSet("cs_fwd_" + std::to_string(uid));
          auto fwdV = graph.addVertex(fwdCS, fwdVertexClass);
          graph.setTileMapping(fwdV, 0);
          graph.connect(fwdV["data"], actTestSlices);

          auto bwdCS = graph.addComputeSet("cs_bwd_" + std::to_string(uid));
          auto bwdV = graph.addVertex(bwdCS, bwdVertexClass);
          graph.setTileMapping(bwdV, 0);
          graph.connect(bwdV["out"], actTestSlices);
          graph.connect(bwdV["outGrad"], outgradTestSlices);
          graph.connect(bwdV["inGrad"], ingradTestSlices);

          programs.emplace_back(
            Sequence(
              Execute(bwdCS),
              Execute(fwdCS)
            ));
          programSlices.emplace_back(SliceDesc{intervals});
          uid++;
        }
      }
    }
  }
  const auto numTests = programs.size();
  const auto uploadProgIndex = programs.size();
  programs.push_back(uploadProg);
  const auto downloadProgIndex = programs.size();
  programs.push_back(downloadProg);

  // The multiple levels of function calls and loops in the NonLinearity2D
  // vertices manage to overflow the stack sometimes in the C++ codelets at
  // present.
  Engine e(graph, programs, OptionFlags{
    { "target.textSectionSizeInBytes", "0x9000" },
    { "target.workerStackSizeInBytes", "0x100" }
  });
  e.load(device);
  attachStreams(e, tmap);

  boost::multi_array<double, 1>
    hostActsOut(boost::extents[maxElements]),
    hostGradIn(boost::extents[maxElements]);
  const auto relativeTolerance = TOL;
  const auto absoluteTolerance =
    dataType == FLOAT ? FLOAT_ATOL
                      : HALF_ATOL;

  bool success = true;
  for (std::size_t testId = 0; testId < numTests; ++testId) {
    // Fill out the model data, applying the non-linearity to regions to be
    // processed by the vertex. This allows us to detect over/underwrites
    // by the vertex as a bonus
    std::copy(hostActsIn.data(), hostActsIn.data() + hostActsIn.num_elements(),
              modelActsOut.data());
    // fill areas of output that should be untouched by backward vertex with
    // values that should never be output by the vertex
    std::fill_n(modelGradIn.data(), modelGradIn.num_elements(), 50.0);
    for (const auto &region : programSlices[testId].intervals) {
      const auto offset = region.begin();
      const auto n = region.size();
      std::copy(hostGradOut.data() + offset,
                hostGradOut.data() + offset + n,
                modelGradIn.data() + offset);
      poplibs_test::bwdNonLinearity(nlType,
                                    hostActsIn.data() + offset,
                                    modelGradIn.data() + offset,
                                    n);
      poplibs_test::nonLinearity(nlType,
                                 hostActsIn.data() + offset,
                                 modelActsOut.data() + offset,
                                 n);
    }

    copy(target, hostActsIn, dataType, rawHostActsIn.get());
    copy(target, hostGradOut, dataType, rawHostGradOut.get());
    // fill areas of output that should be untouched by backward vertex with
    // values that should never be output by the vertex
    std::fill_n(hostGradIn.data(), hostGradIn.num_elements(), 50.0);
    copy(target, hostGradIn, dataType, rawHostGradIn.get());
    e.run(uploadProgIndex);
    e.run(testId);
    e.run(downloadProgIndex);
    copy(target, dataType, rawHostActsIn.get(), hostActsOut);
    copy(target, dataType, rawHostGradIn.get(), hostGradIn);

    success &=
      checkIsClose("fwd_" + std::to_string(testId),
                   hostActsOut.data(), {hostActsOut.num_elements()},
                   modelActsOut.data(), modelActsOut.num_elements(),
                   relativeTolerance, absoluteTolerance);
    success &=
      checkIsClose("bwd_" + std::to_string(testId),
                   hostGradIn.data(), {hostGradIn.num_elements()},
                   modelGradIn.data(), modelGradIn.num_elements(),
                   relativeTolerance, absoluteTolerance);
  }
  return success;
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

  if (!doTest(deviceType, dataType, nlType))
    return 1;
  return 0;
}
