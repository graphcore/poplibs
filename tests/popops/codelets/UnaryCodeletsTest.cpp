// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

//
// Tests one or more of the element-wise unary codelets:
//
//     UnaryOp1DSupervisor
//     UnaryOp1DInPlaceSupervisor
//     UnaryOp2D
//     UnaryOp2DInPlace
//
// One or more combinations of operation/data type can be specified for the
// vertices under test.
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
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <type_traits>

using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;

const std::vector<std::string> verticesNames = {
    "UnaryOp1DSupervisor",
    "UnaryOp1DInPlaceSupervisor",
    "UnaryOp2D",
    "UnaryOp2DInPlace",
};

//*************************************************************************
// This contains the name of the vertex under test and various flags that
// characterise it, used in different places during the test.
struct VertexDesc {
  std::string name;

  UnaryOpType op;
  Type dataType;
  Type outputType;

  std::string vClass;    // full name with template params for addVertex()
  std::string vClassFmt; // vClass, formatted for display

  // Name of the operand field in the vertex
  std::string inName;

  bool is2D;
  bool inPlace;

  VertexDesc(const std::string &vertexName, UnaryOpType &op,
             const Type &dataType)
      : name(vertexName), op(op), dataType(dataType) {

    // Extract the flags by looking at the name
    is2D = vertexName.find("2D") != std::string::npos;
    inPlace = vertexName.find("InPlace") != std::string::npos;

    outputType = dataType;
    if (isBoolOp(op)) {
      outputType = BOOL;
    }

    inName = inPlace ? "inOut" : "in";

    // Get the vertex class
    std::string vName = "popops::" + name;
    vClass = templateVertex(vName, op, dataType);

    // Shorten the name for display, removing namespaces
    vClassFmt = vClass;
    boost::erase_all(vClassFmt, "popops::");
    boost::erase_all(vClassFmt, "expr::UnaryOpType::");
    const unsigned FMT_LEN = 70;
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
    writeName = vertex->inName + "_" + to_string(seq);
    readName = "out_" + to_string(seq);
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
// Return true if the combination of vertex, operation and data type is valid
bool isValidCombination(const VertexDesc &vertex) {
  const UnaryOpType &op = vertex.op;
  const Type &type = vertex.dataType;
  if (op == UnaryOpType::LOGICAL_NOT) {
    return type == BOOL;
  }
  if (vertex.inPlace &&
      (op == UnaryOpType::IS_FINITE || op == UnaryOpType::IS_INF ||
       op == UnaryOpType::IS_NAN)) {
    return false;
  }
  if (op == UnaryOpType::BITWISE_NOT) {
    return type == INT || type == UNSIGNED_INT || type == SHORT ||
           type == UNSIGNED_SHORT;
  }
  if (op == UnaryOpType::POPCOUNT || op == UnaryOpType::COUNT_LEADING_ZEROS) {
    return type == INT || type == UNSIGNED_INT;
  }
  if (op == UnaryOpType::ABSOLUTE || op == UnaryOpType::NEGATE ||
      op == UnaryOpType::SIGNUM || op == UnaryOpType::SQRT) {
    return type == HALF || type == FLOAT || type == INT;
  }
  if (op == UnaryOpType::SQUARE) {
    return type == HALF || type == FLOAT || type == INT || type == UNSIGNED_INT;
  }
  return type == HALF || type == FLOAT;
}

//*************************************************************************
/// Verifies if the results of the operation performed on the device match
/// with the one the host
template <typename HostDataType, typename HostOutType>
bool verifyTest(const Target &target, bool isIpuModel, const TestRecord &test,
                const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &sizes = test.size.val;
  const UnaryOpType &op = vertex.op;

  // Convert the device data in host format. Also convert back the input data
  // in host format.
  const unsigned nElems =
      vertex.is2D ? std::accumulate(sizes.begin(), sizes.end(), 0) : sizes[0];
  std::vector<HostDataType> inHost(nElems);
  std::vector<HostOutType> outHost(nElems);
  copy(target, vertex.dataType, test.hostRawBuf.get(), inHost.data(),
       inHost.size());
  copy(target, vertex.outputType, test.hostOutBuf.get(), outHost.data(),
       outHost.size());
  if (options.printBuffers) {
    printBuffer("out", outHost, vertex.outputType, sizes);
  }

  unsigned errCount = 0; // how many mismatched elements we find

  // Loop sequentially over the operand linear data buffer
  for (unsigned i = 0; i < inHost.size(); i++) {
    HostDataType val = inHost[i]; // operands

    HostOutType actual = outHost[i]; // result from device

    HostOutType expected = 0; // result for verification
    performOp(op, val, expected);

    if (!equalValues(isIpuModel, op, vertex.outputType, expected, actual)) {
      std::cerr << "out[" << i << "] = " << unaryOpToString.at(op) << "("
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
/// \tparam HostDataType Type to use on the host for dataType1, datatType2.
/// \tparam HostOutType  Type to use on the host for outputType.
/// \param[in]    target    Which target.
/// \param[inout] graph     The graph.
/// \param[inout] upload    A Sequence where we will add the uploading of the
//                          data for this vertex (from the host to the device)
/// \param[inout] cs        The compute set to add the vertex to.
/// \param[inout] download  A Sequence where we will add the downloading of the
///                         result data (from device to host)
/// \param[inout] streamMap Used to pass the appropriate streams for upload/
///                         download when running on the device.
/// \param[inout] test      Describes the test to setup. Pointers to data and
/// \param[in]    tile      Which tile to run this test on.
/// \param[in]    options   Global options.
///
template <typename HostDataType, typename HostOutType>
static void setupTest(const Target &target, Graph &graph, Sequence &upload,
                      ComputeSet &cs, Sequence &download, StreamMap &streamMap,
                      TestRecord &test, unsigned tile,
                      const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &sizes = test.size.val;
  const Type &dataType = vertex.dataType;
  const Type &outputType = vertex.outputType;

  // Check for various possible inconsistencies in the combinations of
  // parameters

  if (std::find(std::begin(verticesNames), std::end(verticesNames),
                vertex.name) == std::end(verticesNames)) {
    throw std::runtime_error(vertex.name +
                             " is not a valid vertex name for this test. Maybe "
                             "you wanted to use the --vertexRE option for a "
                             "regular expression?");
  }

  if (vertex.inPlace && (outputType != dataType)) {
    throw std::runtime_error("For in place operations, the data and output "
                             "types must be the same (specified data type=" +
                             dataType.toString() + ", specified output type=" +
                             outputType.toString() + ")");
  }

  if (isIntOp(vertex.op) &&
      (dataType == HALF || dataType == FLOAT || dataType == BOOL)) {
    throw std::runtime_error(unaryOpToString.at(vertex.op) +
                             " requires data "
                             "of integer type (specified  data type=" +
                             dataType.toString() + ")");
  }

  // === Allocate and initialise host buffers with appropriate values.
  const unsigned nElems =
      vertex.is2D ? std::accumulate(sizes.begin(), sizes.end(), 0) : sizes[0];
  std::vector<HostDataType> inHost(nElems);
  fillHostBuffer(vertex.op, dataType, options.randomSeed, inHost);
  if (options.printBuffers) {
    printBuffer("in", inHost, dataType, sizes);
  }

  // === Create graph variables.
  Tensor in, out;
  in = graph.addVariable(dataType, {nElems}, vertex.inName);
  graph.setTileMapping(in, tile);
  if (vertex.inPlace) {
    out = in;
  } else {
    out = graph.addVariable(outputType, {nElems}, vertex.inName);
    graph.setTileMapping(out, tile);
  }

  // === Create the vertex
  auto v = graph.addVertex(cs, vertex.vClass);
  graph.setTileMapping(v, tile);

  if (vertex.inPlace) {
    test.hostRawBuf =
        allocateHostMemoryForTensor(target, out, graph.getReplicationFactor());
    test.hostOutBuf = allocateHostMemoryForTensor(out, test.writeName, graph,
                                                  upload, download, streamMap);
    copy(target, inHost.data(), inHost.size(), dataType, test.hostRawBuf.get());
    copy(target, inHost.data(), inHost.size(), outputType,
         test.hostOutBuf.get());
  } else {
    test.hostRawBuf = allocateHostMemoryForTensor(
        in, test.writeName, graph, upload, boost::none, streamMap);
    test.hostOutBuf = allocateHostMemoryForTensor(
        out, test.readName, graph, boost::none, download, streamMap);
    copy(target, inHost.data(), inHost.size(), dataType, test.hostRawBuf.get());
  }

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
    connectOperand2D(vertex.inName, sizes, in);
    if (!vertex.inPlace)
      connectOperand2D("out", sizes, out);
  } else {
    graph.connect(v[vertex.inName], in);
    if (!vertex.inPlace)
      graph.connect(v["out"], out);
  }

  // The in place vertices for these operators have an additional field
  if (!vertex.is2D && vertex.inPlace &&
      (vertex.op == UnaryOpType::RELU || vertex.op == UnaryOpType::SIGMOID ||
       vertex.op == UnaryOpType::TANH)) {
    graph.setInitialValue(v["n"], nElems);
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
#define SELECT_ONE(IPU_DATA_TYPE, IPU_OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE) \
  if (vertex.dataType == IPU_DATA_TYPE && vertex.outputType == IPU_OUT_TYPE) { \
    setupTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(                                  \
        target, graph, upload, cs, download, streamMap, test, tile, options);  \
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
bool doVerifyTest(const Target &target, bool isIpuModel, TestRecord &test,
                  const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;

#define SELECT_ONE(IPU_DATA_TYPE, IPU_OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE) \
  if (vertex.dataType == IPU_DATA_TYPE && vertex.outputType == IPU_OUT_TYPE) { \
    return verifyTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(target, isIpuModel, test, \
                                                     options);                 \
  }
  SELECT_BY_TYPES()
  throw invalid_types(vertex.dataType, vertex.outputType);
#undef SELECT_ONE
}

//*************************************************************************
int main(int argc, char **argv) {

  DeviceType deviceType;
  std::vector<Type> dataTypes;

  //  Sizes of the operand. For a Supervisor vertex,
  std::vector<SizeDesc> sizes = {{false, {25, 12, 21}}};

  std::vector<std::string> operationStr;
  std::vector<std::string> vertices;
  std::string vertexRE; // regular expression for vertex name
  unsigned groupTests = 1;
  boost::optional<std::string> cycleCompareDevice;

  MiscOptions options;

  // clang-format off
  const static std::string description =
  "Tests one or more of the unary vertices with one or more operations/data \n"
  "type(s), with a default data size, or a specified one.\n"
  "If no vertex is specified, all will be tested. Same for operation and data\n"
  "type.\n"
  "Using the --compare-cycles option, cycles reported when running the vertex\n"
  "on two different devices can be compared."
  "Examples of usages:\n"
  "\n"
  " A Supervisor vertex, with an operand of 5000 floats:\n"
  "   UnaryCodeletsTest --vertex UnaryOp1DSupervisor --operation EXPONENT \\\n"
  "                      --data-type float --size 5000\n"
  "\n"
  " As above, but with multiple data types:\n"
  "   UnaryCodeletsTest --vertex UnaryOp1DSupervisor --operation EXPONENT \\\n"
  "                      --data-type float half int --size 5000\n"
  "\n"
  " A 2D vertex, where the operand is a 2D vector of vectors of: [[300]\n"
  " [48] [100]] floats:\n"
  "   UnaryCodeletsTest --vertex UnaryOp2DInPlace --operation SQRT \\\n"
  "                      --data-type float --size 300 48 100\n"
  "\n"
  "Compare cycles reported between Sim and IpuModel when running a specific\n"
  "vertex:\n"
  "   UnaryCodeletsTest --vertex UnaryOp1DSupervisor --operation SQUARE \\\n"
  "                      --data-type float --size 5000 --device-type Sim \\\n"
  "                      --compare-cycles IpuModel\n"
  "\n"
  "\n"
  "Details of options are:";

  po::options_description poDesc(description);

  poDesc.add_options()
    ("vertex",
     po::value<std::vector<std::string>>(&vertices)->multitoken(),
     "Vertices to test, one or more of: UnaryOp1DSupervisor, "
     "UnaryOp1DInPlaceSupervisor, UnaryOp2D, UnaryOp2DInPlace")
    ("vertexRE",
     po::value<std::string>(&vertexRE),
     "Regular expression to specify vertex names (alternative to --vertex)")
    ("operation",
     po::value<std::vector<std::string>>(&operationStr)->multitoken(),
     ("Operation(s) to perform, one or more of: " + allOpsStr()).c_str())
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex, "
     "multiple values for a 2D vertex")
    ;
  // clang-format on
  addCommonOptions(poDesc, deviceType, cycleCompareDevice, dataTypes,
                   groupTests, options);

  parseOptions(argc, argv, poDesc);

  // === Some parameter checks
  if (!vertexRE.empty() && !vertices.empty()) {
    throw std::runtime_error(
        "Cannot specify both --vertexRE and --vertex option");
  }
  if (cycleCompareDevice && groupTests > 1) {
    std::cout << "When running with --compare-cycle option, the --group-tests "
                 "option is ignored\n";
  }

  // === If no vertices specified, test 'em all
  if (vertices.empty()) {
    vertices = verticesNames;
  }

  // === If no operators specified, test 'em all
  std::vector<UnaryOpType> operations;
  if (operationStr.empty()) {
    setAllOps(operations);
  } else {
    // Convert strings to UnaryOpType
    for (auto opStr : operationStr)
      operations.push_back(stringToUnaryOp(opStr));
  }

  // === If no data type specified, test 'em all
  if (dataTypes.empty()) {
    dataTypes = {HALF, FLOAT, INT, UNSIGNED_INT, SHORT, UNSIGNED_SHORT, BOOL};
  }

  std::regex vertexRegEx(vertexRE);

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
  // Loop over all vertices, operations, data types
  for (std::string vertexName : vertices) {
    // If a regex was specified, see if it matches
    if (vertexRE.empty() || std::regex_search(vertexName, vertexRegEx)) {
      for (auto op : operations) {
        for (auto type : dataTypes) {
          for (auto sz : sizes) {
            auto vertex = std::make_unique<VertexDesc>(vertexName, op, type);
            if (isValidCombination(*vertex)) {
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
  }
  runAllTests<TestRecord>(tests, numTests, groupTests, deviceType, errCount,
                          options);
  return (errCount == 0) ? 0 : 1; // returning 1 means an error.
}
