// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Tests one or more of the cast codelets:
//
//     Cast
//     CastSupervisor
//     Cast2d
//
// One or more combinations of source and destination types can be specified.
//
// There is also an option to compare cycles reported when running the vertex
// on two different devices.
//
// See description string in main() for details and examples of usage.

#include "UnaryCodeletsTest.hpp"

#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <algorithm>
#include <cfenv>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <optional>
#include <sstream>
#include <type_traits>

using namespace poplar::program;
using namespace poputil;
using namespace popops;
using boost::format;
using std::to_string;

// The name of the compute set where we run the vertex under test.
const static std::string computeSetName = "vertexComputeSet";

const std::vector<std::string> verticesNames = {
    "Cast",
    "CastSupervisor",
    "Cast2d",
};

//*************************************************************************
// This contains the name of the vertex under test and various flags that
// characterise it, used in different places during the test.
struct VertexDesc {
  std::string name;

  Type srcType;
  Type dstType;

  std::string vClass;    // full name with template params for addVertex()
  std::string vClassFmt; // vClass, formatted for display

  bool is2D;
  bool isSupervisor;

  VertexDesc(const std::string &vertexName, const Type &srcType,
             const Type &dstType)
      : name(vertexName), srcType(srcType), dstType(dstType) {

    // Extract the flags by looking at the name
    is2D = vertexName == "Cast2d";
    isSupervisor = vertexName == "CastSupervisor";

    // Get the vertex class
    std::string vName = "popops::" + name;
    vClass = templateVertex(vName, srcType, dstType);

    // Format the name for display, removing namespaces
    vClassFmt = vClass;
    boost::erase_all(vClassFmt, "popops::");
    vClassFmt += std::string(60 - vClassFmt.size(), ' '); // pad for display
  }
};

// Contains information relative to the test for one single vertex
struct TestRecord {
  SizeDesc size;
  std::unique_ptr<VertexDesc> vertex;

  std::unique_ptr<char[]> hostRawBuf;
  std::unique_ptr<char[]> hostOutBuf;

  // Stream names used to transfer the host data and the output. Must be
  // different for each test run in the same graph.
  std::string writeName;
  std::string readName;

  /// \param[in] v      The vertex (with operation and data type) to test.
  /// \param[in] seq    A sequential index, different for each test
  /// \param[in] tSizes The data sizes to use for the test.
  TestRecord(std::unique_ptr<VertexDesc> v, unsigned seq, const SizeDesc &sz)
      : size(sz), vertex(std::move(v)) {
    writeName = "src_" + to_string(seq);
    readName = "dst_" + to_string(seq);
    if (size.isRowsByCols) {
      size.isRowsByCols = false;
      unsigned rows = size.val.at(0);
      unsigned cols = size.val.at(1);
      size.val.clear();
      if (vertex->is2D) {
        size.val.resize(rows, cols);
      } else {
        size.val.push_back(rows * cols);
      }
    } else {
      if (!vertex->is2D) {
        size.val.resize(1);
      }
    }
  };
};

//*************************************************************************
// Return true if the combination of src and dst type is valid
bool isValidTypes(const VertexDesc &vertex) {
  const Type &src = vertex.srcType;
  const Type &dst = vertex.dstType;

  // if BOTH src and destination are among these types, the combination is
  // valid, with the exclusion of:
  //   1. Supervisor vertices for INT <=> UNSIGNED_INT are not defined (?)
  //   2. Only vertices with SRC != DST are defined.
  const static std::array types{FLOAT,        HALF,           INT,
                                UNSIGNED_INT, UNSIGNED_SHORT, BOOL};
  if (std::find(types.begin(), types.end(), src) != types.end() &&
      std::find(types.begin(), types.end(), dst) != types.end()) {
    if (vertex.isSupervisor && ((src == INT && dst == UNSIGNED_INT) ||
                                (src == UNSIGNED_INT && dst == INT))) {
      return false;
    }
    return src != dst;
  }

  // We can also cast from FLOAT or HALF into all char types, and vice-versa.
  if ((src == FLOAT || src == HALF) &&
      (dst == CHAR || dst == SIGNED_CHAR || dst == UNSIGNED_CHAR)) {
    return true;
  }
  if ((src == CHAR || src == SIGNED_CHAR || src == UNSIGNED_CHAR) &&
      (dst == FLOAT || dst == HALF)) {
    return true;
  }
  return false;
}

//*************************************************************************
/// Verifies if the results of the cast performed on the device match
/// with the one the host
///
/// \param isIpuModel          Is IpuModel or IpuModel2?.
/// \param srcType             The data type used (float, half).
/// \param inHost              Data buffer for the operand.
/// \param outHost             Data for result, obtained from
///                            device and converted to host types.
/// \param operation           Operation performed on device.
template <typename HostDataType, typename HostOutType>
bool verifyTest(const Target &target, bool isIpuModel, const TestRecord &test) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &sizes = test.size.val;

  // Convert the device data in host format. Also convert back the input data
  // in host format.
  const unsigned nElems =
      vertex.is2D ? std::accumulate(sizes.begin(), sizes.end(), 0) : sizes[0];
  std::vector<HostDataType> inHost(nElems);
  std::vector<HostOutType> outHost(nElems);
  copy(target, vertex.srcType, test.hostRawBuf.get(), inHost.data(),
       inHost.size());
  copy(target, vertex.dstType, test.hostOutBuf.get(), outHost.data(),
       outHost.size());

  if (isIpuModel) {
    std::fesetround(FE_TOWARDZERO);
  } else {
    // Currently the IPU has only 1 rounding mode for floating point to int
    // conversions (f32toi32/f32toui32 instructions): Round-To-Nearest,
    // Ties-To-Even (see use of 'nearbyint()' in 'performCast'
    std::fesetround(FE_TONEAREST);
  }

  unsigned errCount = 0; // how many mismatched elements we find

  // Loop sequentially over the operand linear data buffer
  for (unsigned i = 0; i < inHost.size(); i++) {
    HostDataType val = inHost[i]; // operands

    HostOutType actual = outHost[i]; // result from device

    HostOutType expected = 0; // result for verification
    performCast(isIpuModel, val, expected, vertex.srcType, vertex.dstType);

    if (!equalValues(isIpuModel, vertex.dstType, vertex.srcType, expected,
                     actual)) {
      std::cerr << "out[" << i << "] = "
                << "cast<" << vertex.dstType.toString() << ">("
                << convertToString(val)
                << ") =>  expected:" << convertToString(expected)
                << ";  actual:" << convertToString(actual) << "\n";
      errCount++;
    }
  }

  if (errCount > 0) {
    std::cerr << "Failed: mismatch on " << errCount << " value(s)\n";
  }
  return errCount == 0;
}

using StreamMap = std::vector<std::pair<std::string, char *>>;

//*************************************************************************
/// Setup one vertex test.
///
/// \tparam HostDataType   Type to use on the host for srcType.
/// \tparam HostOutType    Type to use on the host for dstType.
/// \param[in]    target   Which target.
/// \param[inout] graph    The graph.
/// \param[inout] upload   A Sequence where we will add the uploading of the
//                         data for this vertex (from the host to the device)
/// \param[inout] cs       The compute set to add the vertex to.
/// \param[inout] download A Sequence where we will add the downloading of the
///                        result data (from device to host)
/// \param[inout] test     Describes the test to setup. Pointers to data and
///                        output buffers are setup here.
/// \param[in]  randomSeed Random seed for data (0 = don't use random values).
/// \param[in]  tile       Which tile to run this test on.
///
template <typename HostDataType, typename HostOutType>
static void setupTest(const Target &target, Graph &graph, Sequence &upload,
                      ComputeSet &cs, Sequence &download, StreamMap &streamMap,
                      TestRecord &test, unsigned randomSeed, unsigned tile) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &sizes = test.size.val;
  const Type &srcType = vertex.srcType;
  const Type &dstType = vertex.dstType;

  // Check for various possible inconsistencies in the combinations of
  // parameters

  if (std::find(std::begin(verticesNames), std::end(verticesNames),
                vertex.name) == std::end(verticesNames)) {
    throw std::runtime_error(vertex.name +
                             " is not a valid vertex name for this test");
  }

  // === Allocate and initialise host buffers with appropriate values.
  const unsigned nElems =
      vertex.is2D ? std::accumulate(sizes.begin(), sizes.end(), 0) : sizes[0];
  std::vector<HostDataType> inHost(nElems);
  fillHostBuffer(vertex.dstType, srcType, randomSeed, inHost);

  // === Create graph variables.
  Tensor in = graph.addVariable(srcType, {nElems}, "src");
  graph.setTileMapping(in, tile);

  Tensor out = graph.addVariable(dstType, {nElems}, "dst");
  graph.setTileMapping(out, tile);

  // === Create the vertex
  auto v = graph.addVertex(cs, vertex.vClass);
  graph.setTileMapping(v, tile);

  test.hostRawBuf = allocateHostMemoryForTensor(in, test.writeName, graph,
                                                upload, boost::none, streamMap);
  test.hostOutBuf = allocateHostMemoryForTensor(
      out, test.readName, graph, boost::none, download, streamMap);
  copy(target, inHost.data(), inHost.size(), srcType, test.hostRawBuf.get());

  // === Connect the edges appropriately, depending on the vertex variant
  auto connectOperand2D = [&](const std::string &name,
                              const std::vector<unsigned> &sizes, Tensor &in) {
    graph.setFieldSize(v[name], sizes.size());
    unsigned rowSum = 0;
    for (unsigned i = 0; i < sizes.size(); i++) {
      graph.connect(v[name][i], in.slice(rowSum, rowSum + sizes[i]));
      rowSum += sizes[i];
    }
  };
  if (vertex.is2D) {
    connectOperand2D("src", sizes, in);
    connectOperand2D("dst", sizes, out);
  } else {
    graph.connect(v["src"], in);
    graph.connect(v["dst"], out);
    if (vertex.isSupervisor) {
      // Computing the bitfields for the 'partitionParams' word. See the codelet
      // C++ definition for the meaning of the fields.
      unsigned grainSize = 4;
      unsigned numGrains = (nElems + grainSize - 1) / grainSize;
      unsigned numWorkerContexts = target.getNumWorkerContexts();
      unsigned workerCount = numWorkerContexts;
      unsigned grainsPerWorker = 1;
      unsigned workerLast = numWorkerContexts - 1;
      if (numGrains <= numWorkerContexts) {
        workerCount = numGrains;
        workerLast = workerCount - 1;
      } else {
        grainsPerWorker = numGrains / workerCount;
        unsigned rem = numGrains % workerCount;
        if (rem > 0) {
          workerCount = rem;
          grainsPerWorker += 1;
        }
      }
      unsigned workerElems = grainsPerWorker * grainSize;
      unsigned deltaLast =
          workerCount * workerElems +
          (numWorkerContexts - workerCount) * (workerElems - grainSize) -
          nElems;
      unsigned partitionParams = (workerElems << 9) | (workerCount << 6) |
                                 (workerLast << 3) | deltaLast;
      graph.setInitialValue(v["partitionParams"], partitionParams);
    } else {
      graph.setInitialValue(v["numElems"], nElems);
    }
  }
}

//*************************************************************************
// Calls the appropriate version of setupTest using the template parameters
// relevant to the data/output types.
// See 'setupTest()' for parameters
static void doSetupTest(const Target &target, Graph &graph, Sequence &upload,
                        ComputeSet &cs, Sequence &download,
                        StreamMap &streamMap, TestRecord &test,
                        const unsigned randomSeed, unsigned tile) {
  VertexDesc &vertex = *test.vertex;
  // Call the appropriate instantiation of the templated function
#define SELECT_ONE(IPU_SRC_TYPE, IPU_DST_TYPE, HOST_SRC_TYPE, HOST_DST_TYPE)   \
  if (vertex.srcType == IPU_SRC_TYPE && vertex.dstType == IPU_DST_TYPE) {      \
    setupTest<HOST_SRC_TYPE, HOST_DST_TYPE>(target, graph, upload, cs,         \
                                            download, streamMap, test,         \
                                            randomSeed, tile);                 \
    return;                                                                    \
  }
  SELECT_BY_TYPES()
  throw invalid_types(vertex.srcType, vertex.dstType);
#undef SELECT_ONE
}

//*************************************************************************
// Calls the appropriate version of verifyTest using the template parameters
// relevant to the data/output types.
// See 'verifyTest()' for parameters
bool doVerifyTest(const Target &target, bool isIpuModel, TestRecord &test) {
  VertexDesc &vertex = *test.vertex;

#define SELECT_ONE(IPU_SRC_TYPE, IPU_DST_TYPE, HOST_SRC_TYPE, HOST_DST_TYPE)   \
  if (vertex.srcType == IPU_SRC_TYPE && vertex.dstType == IPU_DST_TYPE) {      \
    return verifyTest<HOST_SRC_TYPE, HOST_DST_TYPE>(target, isIpuModel, test); \
  }
  SELECT_BY_TYPES()
  throw invalid_types(vertex.srcType, vertex.dstType);
#undef SELECT_ONE
}

//*************************************************************************
/// Runs the tests specified by 'tests[offs:offs+numTests]', all in a single
/// compute set.
///
/// \param[inout] tests      A vector of tests to run. This function will modify
///                          the test records that are run, adding the data and
///                          output buffers.
/// \param[in]    offs       Offset into 'tests' specifying the first test to
///                          run
/// \param[in]    numTests   How many tests from 'tests[offs]' we want to run
/// \param[in]    deviceType What type of device to run the tests on
/// \param[in]    ignoreData Do we want to verify the results?
/// \param[in]    doReport   Do we want to print a report on stdout?
/// \param[in]    verbose    Do we want to print a description of each test as
///                          it is setup/verified?
/// \return   number of FAILED tests.
unsigned runTests(std::vector<TestRecord> &tests, unsigned offs,
                  unsigned numTests, const DeviceType deviceType,
                  bool ignoreData, bool doReport, bool disableFpExceptions,
                  bool verbose, unsigned randomSeed,
                  uint64_t *cycles = nullptr) {

  // local function to print a concise description for the test
  auto describeTest = [&](const TestRecord &test) {
    std::cout << test.vertex->vClassFmt << test.size << "\n";
  };
  TestDevice device = createTestDevice(deviceType, 1, numTests + 1);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  ComputeSet cs = graph.addComputeSet(computeSetName);
  Sequence program, upload, download;

  StreamMap streamMap;
  for (unsigned i = 0; i < numTests; i++) {
    // If running only one test, we print the info about that test before
    // doing the setup. This will help debug any problem arising during the
    // setup itself. The same is if we don't run the data verification.
    if (verbose && (numTests == 1 || ignoreData))
      describeTest(tests[offs + i]);
    doSetupTest(target, graph, upload, cs, download, streamMap, tests[offs + i],
                randomSeed, i);
  }
  program.add(Execute(cs));

  // === Run the program
  OptionFlags engOpts;
  if (doReport || cycles) {
    engOpts.set("debug.instrumentCompute", "true");
  }
  if (disableFpExceptions) {
    engOpts.set("debug.floatPointOpException", "false");
  }
  Engine engine(graph, Sequence(upload, program, download), engOpts);
  attachStreams(engine, streamMap);

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();

    if (doReport) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");
      engine.printProfileSummary(std::cout, opt);
    }
  });
  unsigned errCount = 0;
  if (ignoreData) {
    std::cout << "Result not checked for correctness\n";
  } else {
    for (unsigned i = offs; i < offs + numTests; i++) {
      // If we are running grouped tests, we print the test description before
      // verification of each test. This helps finding out which specific test
      // might have failed
      if (verbose && numTests > 1)
        describeTest(tests[i]);

      if (doVerifyTest(target, isIpuModel(deviceType), tests[i]) == false)
        errCount++;
    }
  }
  if (cycles) {
    // Get the cycles by searching the "simulation"/"steps" vector for an
    // "OnTileExecute" element having the compute set name we used.
    poplar::ProfileValue execProfile = engine.getExecutionProfile();
    for (auto s : execProfile["simulation"]["steps"].asVector()) {
      if (s["type"] == "OnTileExecute" && s["name"] == computeSetName) {
        *cycles = s["cycles"].asUint();
      }
    }
  }
  return errCount;
}

/// Compare cycles obtained by running one vertex with the two specified
/// devices. Prints result on standard output.
///
/// \param[in] devPair          Two devices for which we will run the test
/// \param[in] tests            A vector of tests that must contain a single
///                             test record.
/// \param[in] randomSeed       Random seed for the data (0=data is not random)
/// \param[in] compareThreshold A percentage threshold of the difference in
///                             cycles, compared to the cycles from the first
///                             device, to determine success/failure.
///
/// \return true   if both run returned successfully and the difference is less
///                than 'compareThreshold' % of the run with the first device.
static bool compareCycles(const std::array<DeviceType, 2> devPair,
                          std::vector<TestRecord> &tests,
                          const bool disableFpExceptions,
                          const unsigned randomSeed,
                          const unsigned compareThreshold) {
  assert(tests.size() == 1);
  VertexDesc &vertex = *tests[0].vertex;

  std::stringstream devName[2]; // To get strings with the name of selected dev
  std::cout << vertex.vClassFmt << std::flush;

  bool ok[2];
  uint64_t cycles[2];
  // Run with the two devices and get the cycles
  for (unsigned i = 0; i < 2; i++) {
    devName[i] << devPair[i];
    uint64_t cyc = 0;
    ok[i] = runTests(tests, 0, 1, devPair[i], false, false, disableFpExceptions,
                     false, randomSeed, &cyc) == 0;
    if (!ok[i]) {
      std::cout << "Failed on device " << devName[i].str() << " (see stderr)\n";
      return false;
    }
    cycles[i] = cyc;
  }
  float diff = static_cast<float>(cycles[1]) - static_cast<float>(cycles[0]);
  float diffPerc = diff / cycles[0] * 100;
  bool compareOk = abs(diffPerc) < compareThreshold;
  std::cout << format("%s:%8u;  %s:%8u;   diff =%4u  %7.2f%%%s\n") %
                   devName[0].str() % cycles[0] % devName[1].str() % cycles[1] %
                   diff % diffPerc % (compareOk ? "" : " <<== FAIL");
  return compareOk;
}

//*************************************************************************
int main(int argc, char **argv) {

  DeviceType deviceType;
  std::vector<Type> srcTypes;

  //  Sizes of the operand. For a Supervisor vertex,
  std::vector<SizeDesc> sizes = {{false, {25, 12, 21}}};

  std::vector<std::string> dstTypeStrs;
  std::vector<std::string> vertices;
  bool doReport = false;
  bool disableFpExceptions = false;
  bool ignoreData = false;
  unsigned randomSeed = 1; // we use '0' to mean 'not random'
  unsigned groupTests = 1;
  boost::optional<std::string> cycleCompareDevice;
  unsigned cycleCompareThreshold = 10; // percent

  // clang-format off
  const static std::string description =
  "Tests one or more of the Cast vertices with one or more combinations of\n"
  "source and destination type(s), with a default data size, or a specified\n"
  "one.\n"
  "If no vertex is specified, all will be tested. Same for source and\n"
  "destination types\n"
  "Using the --compare-cycles option, cycles reported when running the vertex\n"
  "on two different devices can be compared."
  "Examples of usages:\n"
  "\n"
  " A Supervisor vertex, with an operand of 5000 half to be casted as ints:\n"
  "   CastCodeletsTest --vertex CastSupervisor --data-type half --cast int \\\n"
  "                    --size 5000\n"
  "\n"
  "\n"
  " As above, but with multiple casts (int and unsigned short):\n"
  "   CastCodeletsTest --vertex CastSupervisor --data-type half \\\n"
  "                     --cast int ushort --size 5000\n"
  "\n"
  "\n"
  "Details of options are:";

  po::options_description poDesc(description);

  poDesc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(DeviceType::Sim2),
     "Device type")
    ("vertex",
     po::value<std::vector<std::string>>(&vertices)->multitoken(),
     "Vertices to test, one or more of: Cast, CastSupervisor, Cast2d")
    ("cast",
     po::value<std::vector<std::string>>(&dstTypeStrs)->multitoken(),
     "Destination type(s) for the cast(s) to perform")
    ("data-type",
     po::value<std::vector<Type>>(&srcTypes)->multitoken(),
     "Data type: one or more of half, float, int, uint, short, ushort, bool, "
     "char, schar, uchar")
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex, "
     "multiple values for a 2D vertex")
    ("compare-cycles",
     po::value<boost::optional<std::string>>(&cycleCompareDevice)->
                                         implicit_value(std::string("default")),
     "For each specified vertex, compare the cycles reported by the device ("
     "--device-type option) and another device specified by this option")
    ("cycle-threshold",
     po::value<unsigned>(&cycleCompareThreshold)->
                                          default_value(cycleCompareThreshold),
     "Percent threshold when running the --compare-cycle option. An (absolute) "
     "cycle difference greater than this threshold will make the test fail.")
    ("report",
     po::value<bool>(&doReport)->implicit_value(true),
     "Provide a poplar report")
    ("disable-fp-exceptions",
     po::value<bool>(&disableFpExceptions)->default_value(disableFpExceptions),
     "Disable floating point exceptions when running on device.")
    ("options-file",
     po::value<std::string>(),
     "A file containing options, with the same syntax as the command line; "
     "can be also specified with '@options_file_name'")
    ("random-seed",
     po::value<unsigned>(&randomSeed)->implicit_value(randomSeed),
     "Seed for random data. Value of 0 means 'no random data'")
    ("ignore-data",
     po::value<bool>(&ignoreData)->implicit_value(true),
     "Do not check correctness of result, useful for benchmarking without "
     "overhead of host-side computation")
    ("group-tests",
     po::value<unsigned>(&groupTests)->implicit_value(100),
     "Run multiple tests together in a single graph and single compute set, "
     "each test on a separate tile, to increase execution speed")
    ;
  // clang-format on
  parseOptions(argc, argv, poDesc);

  // === Some parameter checks
  if (cycleCompareDevice && groupTests > 1) {
    std::cout << "When running with --compare-cycle option, the --group-tests "
                 "is ignored\n";
  }

  // === If no vertices specified, test 'em all
  if (vertices.empty()) {
    vertices = verticesNames;
  }

  // All types for which at least one cast to or from is defined.
  const static std::array allTypes = {HALF,           FLOAT, INT,  UNSIGNED_INT,
                                      UNSIGNED_SHORT, BOOL,  CHAR, SIGNED_CHAR,
                                      UNSIGNED_CHAR};

  // === If no destination type specified, test 'em all
  std::vector<Type> dstTypes;
  if (dstTypeStrs.empty()) {
    dstTypes.insert(dstTypes.begin(), allTypes.begin(), allTypes.end());
  } else {
    // Convert strings to UnaryOpType
    for (auto str : dstTypeStrs) {
      Type type;
      std::istringstream is(str);
      is >> type;
      dstTypes.push_back(type);
    }
  }

  // === If no src type specified, test 'em all
  if (srcTypes.empty()) {
    srcTypes.insert(srcTypes.begin(), allTypes.begin(), allTypes.end());
  }

  // If we are comparing cycles, create an array with the 2 devices
  std::array<DeviceType, 2> devPair;
  if (cycleCompareDevice) {
    devPair = {deviceType,
               getCycleCompareDevice(deviceType, *cycleCompareDevice)};
  }

  std::vector<TestRecord> tests;
  unsigned numTests = 0;
  unsigned errCount = 0;
  // Loop over all vertices, src type and dst type
  for (std::string vertexName : vertices) {
    for (auto dstType : dstTypes) {
      for (auto srcType : srcTypes) {
        for (auto sz : sizes) {
          auto vertex =
              std::make_unique<VertexDesc>(vertexName, srcType, dstType);
          if (isValidTypes(*vertex)) {
            numTests++;
            tests.emplace_back(std::move(vertex), tests.size(), sz);
            // If we run a cycle comparison, or with groupTests<=1, we run
            // the one test straight away here (tests[] will be then cleared so
            // it will always contain a single record), otherwise we accumulate
            // all test records in 'tests[]' to be run afterwards.
            if (cycleCompareDevice) {
              compareCycles(devPair, tests, disableFpExceptions, randomSeed,
                            cycleCompareThreshold);
              tests.clear();
            } else if (groupTests <= 1) {
              errCount +=
                  runTests(tests, 0, 1, deviceType, ignoreData, doReport,
                           disableFpExceptions, true, randomSeed);
              tests.clear();
            }
          }
        }
      }
    }
  }
  if (numTests == 0) {
    throw std::runtime_error(
        "The specified data types do not match any valid combination");
  }
  // Do we need to run all records in tests[]?
  if (groupTests > 1) {
    // Run the tests in batches of up to groupTests together on a single graph/
    // single CS, each test on a different tile. When running on the simulators,
    // if too many tiles are used, execution is slower.
    // Run all of 'tests', grouping them in up to MAX_GROUP to be run in a
    // single graph (and compute set)
    unsigned offs = 0, n = numTests;
    while (n) {
      unsigned l = n > groupTests ? groupTests : n;
      errCount += runTests(tests, offs, l, deviceType, ignoreData, doReport,
                           disableFpExceptions, true, randomSeed);
      offs += l;
      n -= l;
    }
  }
  if (numTests > 1) {
    std::cout << "CastCodeletsTest: " << numTests << " tests run in total; "
              << ((errCount == 0) ? "All passed\n"
                                  : to_string(errCount) + " failed\n");
  }
  return (errCount == 0) ? 0 : 1; // returning 1 means an error.
}
