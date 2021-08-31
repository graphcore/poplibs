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

  Type dataType;
  Type outputType;

  std::string vClass;    // full name with template params for addVertex()
  std::string vClassFmt; // vClass, formatted for display

  // Name of the operand field in the vertex
  std::string inName = "src";
  std::string outName = "dst";

  bool is2D;
  bool isSupervisor;

  VertexDesc(const std::string &vertexName, const Type &srcType,
             const Type &dstType)
      : name(vertexName), dataType(srcType), outputType(dstType) {

    // Extract the flags by looking at the name
    is2D = vertexName == "Cast2d";
    isSupervisor = vertexName == "CastSupervisor";

    // Get the vertex class
    std::string vName = "popops::" + name;
    vClass = templateVertex(vName, dataType, outputType);

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
// Return true if the combination of src and dst type is valid
bool isValidTypes(const VertexDesc &vertex) {
  const Type &src = vertex.dataType;
  const Type &dst = vertex.outputType;

  // if BOTH src and destination are among these types, the combination is
  // valid, with the exclusion of:
  //   1. Supervisor vertices for INT <=> UNSIGNED_INT are not defined (?)
  //   2. Only vertices with SRC != DST are defined.
  const static std::array types{FLOAT,        HALF,           INT,
                                UNSIGNED_INT, UNSIGNED_SHORT, BOOL,
                                CHAR,         SIGNED_CHAR,    UNSIGNED_CHAR};
  if (std::find(types.begin(), types.end(), src) != types.end() &&
      std::find(types.begin(), types.end(), dst) != types.end()) {
    if (vertex.isSupervisor && ((src == INT && dst == UNSIGNED_INT) ||
                                (src == UNSIGNED_INT && dst == INT))) {
      return false;
    }
    return src != dst;
  }

  if ((dst == UNSIGNED_LONGLONG) || (dst == LONGLONG)) {
    return src == INT || src == UNSIGNED_INT || src == CHAR ||
           src == UNSIGNED_CHAR || src == UNSIGNED_SHORT || src == SHORT ||
           src == BOOL;
  }

  // conversion from 64-bit to any other type is not yet supported
  if ((src == UNSIGNED_LONGLONG) || (src == LONGLONG)) {
    return false;
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
bool verifyTest(const Target &target, bool isIpuModel,
                const TestRecord<VertexDesc> &test,
                const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &sizes = test.size.val;

  // Convert the device data in host format. Also convert back the input data
  // in host format.
  std::vector<HostDataType> inHost(test.in.totalElems);
  std::vector<HostOutType> outHost(test.out.totalElems);
  copy(target, vertex.dataType, test.in.rawBuf.get(), inHost.data(),
       inHost.size());
  copy(target, vertex.outputType, test.out.rawBuf.get(), outHost.data(),
       outHost.size());
  if (options.printBuffers) {
    printBuffer(vertex.outName, outHost, vertex.outputType, sizes,
                test.out.offsets);
  }

  // Check for mismatches on computed values
  unsigned errCount = 0; // how many mismatched elements we find
  unsigned numRows = sizes.size();
  Operation castOp(vertex.outputType);
  for (unsigned row = 0; row < numRows; row++) {
    for (unsigned i = 0; i < sizes[row]; i++) {
      HostDataType val = inHost[test.in.offsets[row] + i]; // operands

      // result from device
      HostOutType actual = outHost[test.out.offsets[row] + i];

      HostOutType expected = 0; // result for verification
      performCast(isIpuModel, val, expected, vertex.dataType,
                  vertex.outputType);

      if (!equalValues(isIpuModel, castOp, vertex.dataType, expected, actual)) {
        std::string posStr =
            vertex.is2D ? to_string(row) + "," + to_string(i) : to_string(i);
        std::cerr << vertex.outName << "[" << posStr << "] = cast<"
                  << vertex.outputType.toString() << ">("
                  << convertToString(val)
                  << ") =>  expected:" << convertToString(expected)
                  << ";  actual:" << convertToString(actual) << "\n";
        errCount++;
      }
    }
  }
  if (errCount > 0) {
    std::cerr << "Failed: mismatch on " << errCount << " value(s)\n";
  }

  // Check for overwrites past the end of each row
  auto overwriteCount = test.out.checkPadBytes(target, vertex.is2D, isIpuModel);

  return (errCount == 0) && (overwriteCount == 0);
}

//*************************************************************************
/// Setup one vertex test.
///
/// \tparam HostDataType   Type to use on the host for dataType.
/// \tparam HostOutType    Type to use on the host for outputType.
/// \param[in]    target   Which target.
/// \param[inout] graph    The graph.
/// \param[inout] upload   A Sequence where we will add the uploading of the
///                        data for this vertex (from the host to the device)
/// \param[inout] alignCS  Compute set containing the 'alignment' vertices.
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
static void setupTest(const Target &target, bool isIpuModel, Graph &graph,
                      Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                      Sequence &download, StreamMap &streamMap,
                      TestRecord<VertexDesc> &test, unsigned tile,
                      const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &sizes = test.size.val;
  const Type &srcType = vertex.dataType;
  const Type &dstType = vertex.outputType;

  // Check for various possible inconsistencies in the combinations of
  // parameters

  if (std::find(std::begin(verticesNames), std::end(verticesNames),
                vertex.name) == std::end(verticesNames)) {
    throw std::runtime_error(vertex.name +
                             " is not a valid vertex name for this test");
  }

  // === Setup offsets for padding
  test.in.setup(target, srcType, sizes, options.alignStart);
  test.out.setup(target, dstType, sizes, options.alignStart);

  // === Allocate and initialise host buffers with appropriate values.
  std::vector<HostDataType> inHost(test.in.totalElems);
  Operation castOp(dstType);
  fillHostBuffer(castOp, srcType, options.randomSeed, inHost);
  if (options.printBuffers) {
    printBuffer(vertex.inName, inHost, srcType, sizes, test.in.offsets);
  }

  // === Create graph variables.
  Tensor in = graph.addVariable(srcType, {test.in.totalElems}, vertex.inName);
  graph.setTileMapping(in, tile);

  Tensor out =
      graph.addVariable(dstType, {test.out.totalElems}, vertex.outName);
  graph.setTileMapping(out, tile);

  // === Create the auxiliary vertices required to align the tensors to 8 bytes
  createAlignVertex(graph, alignCS, in, tile);
  createAlignVertex(graph, alignCS, out, tile);

  // === Create the vertex
  auto v = graph.addVertex(cs, vertex.vClass);
  graph.setTileMapping(v, tile);

  test.in.rawBuf = allocateHostMemoryForTensor(in, test.writeName, graph,
                                               upload, boost::none, streamMap);
  test.out.rawBuf = allocateHostMemoryForTensor(out, test.readName, graph,
                                                upload, download, streamMap);
  copy(target, inHost.data(), inHost.size(), srcType, test.in.rawBuf.get());

  // Fill the padding space in the input buffer (with NaNs) for overprocessing
  // detection (only for floating point types). Also fill the output buffer
  // padding, for overrun detection.
  if (srcType == FLOAT || srcType == HALF) {
    test.in.setPadBytes(target, isIpuModel, true);
  }
  test.out.setPadBytes(target, isIpuModel);

  // === Connect the edges appropriately, depending on the vertex variant
  TestOperand::OperandType opType = vertex.is2D
                                        ? TestOperand::OperandType::is2D
                                        : TestOperand::OperandType::is1D;
  test.in.connectOperand(graph, v, opType, in, vertex.inName);
  test.out.connectOperand(graph, v, opType, out, vertex.outName);

  if (!vertex.is2D) {
    unsigned totalElems = sizes[0];
    if (vertex.isSupervisor) {
      // Computing the bitfields for the 'partitionParams' word. See the codelet
      // C++ definition for the meaning of the fields.
      unsigned grainSize = 4;
      unsigned numGrains = (totalElems + grainSize - 1) / grainSize;
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
          totalElems;
      unsigned partitionParams = (workerElems << 9) | (workerCount << 6) |
                                 (workerLast << 3) | deltaLast;
      graph.setInitialValue(v["partitionParams"], partitionParams);
    } else {
      graph.setInitialValue(v["numElems"], totalElems);
    }
  }
}

//*************************************************************************
// Calls the appropriate version of setupTest using the template parameters
// relevant to the data/output types.
// See 'setupTest()' for parameters
static void doSetupTest(const Target &target, bool isIpuModel, Graph &graph,
                        Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                        Sequence &download, StreamMap &streamMap,
                        TestRecord<VertexDesc> &test, unsigned tile,
                        const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;
  // Call the appropriate instantiation of the templated function
#define SELECT_ONE(IPU_SRC_TYPE, IPU_DST_TYPE, HOST_SRC_TYPE, HOST_DST_TYPE)   \
  if (vertex.dataType == IPU_SRC_TYPE && vertex.outputType == IPU_DST_TYPE) {  \
    setupTest<HOST_SRC_TYPE, HOST_DST_TYPE>(target, isIpuModel, graph, upload, \
                                            alignCS, cs, download, streamMap,  \
                                            test, tile, options);              \
    return;                                                                    \
  }
  SELECT_BY_TYPES()
  throw invalid_types(vertex.dataType, vertex.outputType);
#undef SELECT_ONE
}

//*************************************************************************
// Calls the appropriate version of verifyTest using the template parameters
// relevant to the data/output types.
// See 'verifyTest()' for parameters
bool doVerifyTest(const Target &target, bool isIpuModel,
                  TestRecord<VertexDesc> &test, const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;

#define SELECT_ONE(IPU_SRC_TYPE, IPU_DST_TYPE, HOST_SRC_TYPE, HOST_DST_TYPE)   \
  if (vertex.dataType == IPU_SRC_TYPE && vertex.outputType == IPU_DST_TYPE) {  \
    return verifyTest<HOST_SRC_TYPE, HOST_DST_TYPE>(target, isIpuModel, test,  \
                                                    options);                  \
  }
  SELECT_BY_TYPES()
  throw invalid_types(vertex.dataType, vertex.outputType);
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
    ("data-type",
     po::value<std::vector<Type>>(&srcTypes)->multitoken(),
     "Data type: one or more of half, float, int, uint, short, ushort, bool, "
     "char, schar, uchar, ulonglong, longlong")
    ("cast",
     po::value<std::vector<std::string>>(&dstTypeStrs)->multitoken(),
     "Destination type(s) for the cast(s) to perform")
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex or a "
     "square bracket, comma separated list of values for a 2D vertex")
    ;
  // clang-format on
  addCommonOptions(poDesc, deviceType, options);

  parseOptions(argc, argv, poDesc);

  // === If no vertices specified, test 'em all
  if (vertices.empty()) {
    vertices = verticesNames;
  }

  // All types for which at least one cast to or from is defined.
  const static std::array allTypes = {
      HALF, FLOAT,       INT,           UNSIGNED_INT,      UNSIGNED_SHORT, BOOL,
      CHAR, SIGNED_CHAR, UNSIGNED_CHAR, UNSIGNED_LONGLONG, LONGLONG};

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

  std::vector<std::shared_ptr<TestRecord<VertexDesc>>> tests;
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
            auto testRec = std::make_shared<TestRecord<VertexDesc>>(
                std::move(vertex), numTests, sz);
            addOneTest<TestRecord<VertexDesc>, VertexDesc>(
                tests, testRec, deviceType, errCount, options);
          }
        }
      }
    }
  }
  runAllTests<TestRecord<VertexDesc>>(tests, numTests, deviceType, errCount,
                                      options);
  return (errCount == 0) ? 0 : 1; // returning 1 means an error.
}
