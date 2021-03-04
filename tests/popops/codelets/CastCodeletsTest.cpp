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
#include <poputil/TileMapping.hpp>

#include <algorithm>
#include <cfenv>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <optional>
#include <sstream>
#include <type_traits>

using namespace poputil;
using namespace popops;

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

    const unsigned FMT_LEN = 60;
    unsigned len = vClassFmt.size();
    unsigned padLen = (len < FMT_LEN) ? (FMT_LEN - len) : 1;
    vClassFmt += std::string(padLen, ' '); // pad for display
  }
};

//*************************************************************************
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
  }
  TestRecord(TestRecord &&) = default;

  std::string toString() { return vertex->vClassFmt + size.toString(); }
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
template <typename HostDataType, typename HostOutType>
bool verifyTest(const Target &target, bool isIpuModel, const TestRecord &test,
                const MiscOptions &options) {
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
  if (options.printBuffers) {
    printBuffer("out", outHost, vertex.dstType, sizes);
  }

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
/// \param[inout] streamMap Used to pass the appropriate streams for upload/
///                        download when running on the device.
/// \param[inout] test     Describes the test to setup. Pointers to data and
///                        output buffers are setup here.
/// \param[in]    tile     Which tile to run this test on.
/// \param[in]    options  Global options.
///
template <typename HostDataType, typename HostOutType>
static void setupTest(const Target &target, Graph &graph, Sequence &upload,
                      ComputeSet &cs, Sequence &download, StreamMap &streamMap,
                      TestRecord &test, unsigned tile,
                      const MiscOptions &options) {
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
  fillHostBuffer(vertex.dstType, srcType, options.randomSeed, inHost);
  if (options.printBuffers) {
    printBuffer("in", inHost, srcType, sizes);
  }

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
                        StreamMap &streamMap, TestRecord &test, unsigned tile,
                        const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;
  // Call the appropriate instantiation of the templated function
#define SELECT_ONE(IPU_SRC_TYPE, IPU_DST_TYPE, HOST_SRC_TYPE, HOST_DST_TYPE)   \
  if (vertex.srcType == IPU_SRC_TYPE && vertex.dstType == IPU_DST_TYPE) {      \
    setupTest<HOST_SRC_TYPE, HOST_DST_TYPE>(                                   \
        target, graph, upload, cs, download, streamMap, test, tile, options);  \
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
bool doVerifyTest(const Target &target, bool isIpuModel, TestRecord &test,
                  const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;

#define SELECT_ONE(IPU_SRC_TYPE, IPU_DST_TYPE, HOST_SRC_TYPE, HOST_DST_TYPE)   \
  if (vertex.srcType == IPU_SRC_TYPE && vertex.dstType == IPU_DST_TYPE) {      \
    return verifyTest<HOST_SRC_TYPE, HOST_DST_TYPE>(target, isIpuModel, test,  \
                                                    options);                  \
  }
  SELECT_BY_TYPES()
  throw invalid_types(vertex.srcType, vertex.dstType);
#undef SELECT_ONE
}

//*************************************************************************
int main(int argc, char **argv) {

  DeviceType deviceType;
  std::vector<Type> srcTypes;

  //  Sizes of the operand. For a Supervisor vertex,
  std::vector<SizeDesc> sizes = {{false, {25, 12, 21}}};

  std::vector<std::string> dstTypeStrs;
  std::vector<std::string> vertices;
  unsigned groupTests = 1;
  boost::optional<std::string> cycleCompareDevice;

  MiscOptions options;

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
    ("vertex",
     po::value<std::vector<std::string>>(&vertices)->multitoken(),
     "Vertices to test, one or more of: Cast, CastSupervisor, Cast2d")
    ("cast",
     po::value<std::vector<std::string>>(&dstTypeStrs)->multitoken(),
     "Destination type(s) for the cast(s) to perform")
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex or a "
     "square bracket, comma separated list of values for a 2D vertex")
    ;
  // clang-format on
  addCommonOptions(poDesc, deviceType, cycleCompareDevice, srcTypes, groupTests,
                   options);

  parseOptions(argc, argv, poDesc);

  // === Some parameter checks
  if (cycleCompareDevice && groupTests > 1) {
    std::cout << "When running with --compare-cycle option, the --group-tests "
                 "option is ignored\n";
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

  // If we are comparing cycles, we need a vector with the 2 devices to compare
  std::vector<DeviceType> devices = {deviceType};
  if (cycleCompareDevice) {
    devices.push_back(getCycleCompareDevice(deviceType, *cycleCompareDevice));
  }

  std::optional<std::vector<std::shared_ptr<TestRecord>>> tests;
  if (!cycleCompareDevice && groupTests > 1) {
    tests.emplace();
  }
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
            auto testRec =
                std::make_shared<TestRecord>(std::move(vertex), numTests, sz);
            addOneTest<TestRecord, VertexDesc>(tests, testRec, devices,
                                               errCount, options);
          }
        }
      }
    }
  }
  runAllTests<TestRecord>(tests, numTests, groupTests, deviceType, errCount,
                          options);
  return (errCount == 0) ? 0 : 1; // returning 1 means an error.
}
