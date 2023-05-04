// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

//
// Tests one or more of the element-wise unary codelets:
//
//     UnaryOp1D
//     UnaryOp1DInPlace
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

const std::vector<std::string> verticesNames = {
    "UnaryOp1D",
    "UnaryOp1DInPlace",
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
  std::string outName = "out"; // not-in-place only

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
           type == UNSIGNED_SHORT || type == LONGLONG ||
           type == UNSIGNED_LONGLONG;
  }
  if (op == UnaryOpType::POPCOUNT || op == UnaryOpType::COUNT_LEADING_ZEROS) {
    return type == INT || type == UNSIGNED_INT;
  }
  if (op == UnaryOpType::ABSOLUTE || op == UnaryOpType::NEGATE) {
    return type == HALF || type == FLOAT || type == INT || type == LONGLONG;
  }
  if (op == UnaryOpType::SIGNUM || op == UnaryOpType::SQRT) {
    return type == HALF || type == FLOAT || type == INT;
  }

  if (op == UnaryOpType::SQUARE) {
    return type == HALF || type == FLOAT || type == INT || type == UNSIGNED_INT;
  }
  return type == HALF || type == FLOAT;
}

//*************************************************************************
/// Verifies if the results of the operation performed on the device match
/// with the one the host. Also check for overwrite at the end of the data
///
/// \return true if the verification is passed
template <typename HostDataType, typename HostOutType>
bool verifyTest(const Target &target, bool isIpuModel,
                const TestRecord<VertexDesc> &test,
                const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &sizes = test.size.val;
  const UnaryOpType &op = vertex.op;

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
  for (unsigned row = 0; row < numRows; row++) {
    for (unsigned i = 0; i < sizes[row]; i++) {
      HostDataType val = inHost[test.in.offsets[row] + i]; // operands

      // result from device
      HostOutType actual = outHost[test.out.offsets[row] + i];

      HostOutType expected = 0; // result for verification
      performOp(op, val, expected);

      if (!equalValues(isIpuModel, op, vertex.dataType, expected, actual)) {
        std::string posStr =
            vertex.is2D ? to_string(row) + "," + to_string(i) : to_string(i);
        std::cerr << vertex.outName << "[" << posStr
                  << "] : " << unaryOpToString.at(op) << "("
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
  auto overwriteCount = test.out.checkPadBytes(target, vertex.is2D, isIpuModel,
                                               test.padOutWithNan);

  return (errCount == 0) && (overwriteCount == 0);
}

//*************************************************************************
/// Setup one vertex test.
///
/// \tparam HostDataType Type to use on the host for dataType1, datatType2.
/// \tparam HostOutType  Type to use on the host for outputType.
/// \param[in]    target    Which target.
/// \param[inout] graph     The graph.
/// \param[inout] upload    A Sequence where we will add the uploading of the
///                         data for this vertex (from the host to the device)
/// \param[inout] alignCS   Compute set containing the 'alignment' vertices.
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
static void setupTest(const Target &target, bool isIpuModel, Graph &graph,
                      Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                      Sequence &download, StreamMap &streamMap,
                      TestRecord<VertexDesc> &test, unsigned tile,
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

  // === Setup offsets for padding
  test.in.setup(target, dataType, sizes, test.startPadBytes);
  test.out.setup(target, outputType, sizes, test.startPadBytes);

  // === Allocate and initialise host buffers with appropriate values.
  std::vector<HostDataType> inHost(test.in.totalElems);
  fillHostBuffer(vertex.op, dataType, options.randomSeed, inHost);
  if (options.printBuffers) {
    printBuffer(vertex.inName, inHost, dataType, sizes, test.in.offsets);
  }

  // === Create graph variables.
  Tensor in, out;
  in = graph.addVariable(dataType, {test.in.totalElems}, vertex.inName);
  graph.setTileMapping(in, tile);
  if (vertex.inPlace) {
    out = in;
  } else {
    out = graph.addVariable(outputType, {test.out.totalElems}, vertex.outName);
    graph.setTileMapping(out, tile);
  }

  // === Create the auxiliary vertices required to align the tensors to 8 bytes
  createAlignVertex(graph, alignCS, in, tile);
  if (!vertex.inPlace) {
    createAlignVertex(graph, alignCS, out, tile);
  }

  // === Create the vertex under test
  auto v = graph.addVertex(cs, vertex.vClass);
  graph.setTileMapping(v, tile);

  if (vertex.inPlace) {
    test.in.rawBuf =
        allocateHostMemoryForTensor(target, out, graph.getReplicationFactor());
    test.out.rawBuf = allocateHostMemoryForTensor(out, test.writeName, graph,
                                                  upload, download, streamMap);
    // Copy the input data from inHost to:
    //  test.out.rawBuf, because that's what we upload to the device
    //  test.in.rawBuf,  because we need it for verification
    copy(target, inHost.data(), inHost.size(), dataType, test.in.rawBuf.get());
    copy(target, inHost.data(), inHost.size(), outputType,
         test.out.rawBuf.get());
  } else {
    test.in.rawBuf = allocateHostMemoryForTensor(
        in, test.writeName, graph, upload, boost::none, streamMap);
    test.out.rawBuf = allocateHostMemoryForTensor(out, test.readName, graph,
                                                  upload, download, streamMap);
    // Copy the input data from inHost to test.in.rawBuf to upload to the device
    copy(target, inHost.data(), inHost.size(), dataType, test.in.rawBuf.get());
  }

  // Fill the padding space in the input buffer (with NaNs) for overprocessing
  // detection (only for floating point types). Also fill the output buffer
  // padding, for overrun detection. For inPlace, 'in' and 'out' are the same.
  if (dataType == FLOAT || dataType == HALF) {
    if (vertex.inPlace) {
      test.padOutWithNan = true;
    } else {
      test.in.setPadBytes(target, isIpuModel, true);
    }
  }
  test.out.setPadBytes(target, isIpuModel, test.padOutWithNan);

  // === Connect the edges appropriately, depending on the vertex variant
  TestOperand::OperandType opType = vertex.is2D
                                        ? TestOperand::OperandType::is2D
                                        : TestOperand::OperandType::is1D;
  test.in.connectOperand(graph, v, opType, in, vertex.inName);
  if (!vertex.inPlace)
    test.out.connectOperand(graph, v, opType, out, vertex.outName);

  // The in place vertices for these operators have an additional field
  if (!vertex.is2D && vertex.inPlace &&
      (vertex.op == UnaryOpType::RELU || vertex.op == UnaryOpType::SIGMOID ||
       vertex.op == UnaryOpType::GELU_ERF || vertex.op == UnaryOpType::TANH)) {
    graph.setInitialValue(v["n"], sizes[0]);
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
#define SELECT_ONE(IPU_DATA_TYPE, IPU_OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE) \
  if (vertex.dataType == IPU_DATA_TYPE && vertex.outputType == IPU_OUT_TYPE) { \
    setupTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(target, isIpuModel, graph,        \
                                             upload, alignCS, cs, download,    \
                                             streamMap, test, tile, options);  \
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

  //  Sizes of the operand. For a 1D vertex, only the first value is used
  std::vector<SizeDesc> sizes = {{false, {25, 12, 21}}};

  std::vector<std::string> operationStr;
  std::vector<std::string> vertices;
  std::string vertexRE; // regular expression for vertex name

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
  " A 1D vertex, with an operand of 5000 floats:\n"
  "   UnaryCodeletsTest --vertex UnaryOp1D --operation EXPONENT \\\n"
  "                      --data-type float --size 5000\n"
  "\n"
  " As above, but with multiple data types:\n"
  "   UnaryCodeletsTest --vertex UnaryOp1D --operation EXPONENT \\\n"
  "                      --data-type float half int --size 5000\n"
  "\n"
  " A 2D vertex, where the operand is a 2D vector of vectors of: [[300]\n"
  " [48] [100]] floats:\n"
  "   UnaryCodeletsTest --vertex UnaryOp2DInPlace --operation SQRT \\\n"
  "                      --data-type float --size 300 48 100\n"
  "\n"
  "Compare cycles reported between Sim and IpuModel when running a specific\n"
  "vertex:\n"
  "   UnaryCodeletsTest --vertex UnaryOp1D --operation SQUARE \\\n"
  "                      --data-type float --size 5000 --device-type Sim \\\n"
  "                      --compare-cycles IpuModel\n"
  "\n"
  "\n"
  "Details of options are:";

  po::options_description poDesc(description);

  poDesc.add_options()
    ("vertex",
     po::value<std::vector<std::string>>(&vertices)->multitoken(),
     "Vertices to test, one or more of: UnaryOp1D, UnaryOp1DInPlace, "
     "UnaryOp2D, UnaryOp2DInPlace")
    ("vertexRE",
     po::value<std::string>(&vertexRE),
     "Regular expression to specify vertex names (alternative to --vertex)")
    ("data-type",
     po::value<std::vector<Type>>(&dataTypes)->multitoken(),
     "Data type: one or more of half, float, int, uint, short, ushort, bool, "
     "char, schar, uchar")
    ("operation",
     po::value<std::vector<std::string>>(&operationStr)->multitoken(),
     ("Operation(s) to perform, one or more of: " + allOpsStr()).c_str())
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex, "
     "multiple values for a 2D vertex")
    ;
  // clang-format on
  addCommonOptions(poDesc, deviceType, options);

  parseOptions(argc, argv, poDesc);

  // === Some parameter checks
  if (!vertexRE.empty() && !vertices.empty()) {
    throw std::runtime_error(
        "Cannot specify both --vertexRE and --vertex option");
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
    dataTypes = {HALF,    FLOAT,          INT,  UNSIGNED_INT,
                 SHORT,   UNSIGNED_SHORT, BOOL, UNSIGNED_LONGLONG,
                 LONGLONG};
  }

  std::regex vertexRegEx(vertexRE);

  std::vector<std::shared_ptr<TestRecord<VertexDesc>>> tests;
  unsigned numTests = 0;
  unsigned errCount = 0;
  // Loop over all vertices, operations, data types
  for (std::string vertexName : vertices) {
    // If a regex was specified, see if it matches
    if (vertexRE.empty() || std::regex_search(vertexName, vertexRegEx)) {
      for (auto op : operations) {
        for (auto type : dataTypes) {
          auto vertex = std::make_shared<VertexDesc>(vertexName, op, type);
          if (isValidCombination(*vertex)) {
            for (auto sz : sizes) {
              for (auto startPadBytes : options.startPadBytes) {
                numTests++;
                auto testRec = std::make_shared<TestRecord<VertexDesc>>(
                    vertex, numTests, sz, startPadBytes);
                addOneTest<TestRecord<VertexDesc>, VertexDesc>(
                    tests, testRec, deviceType, errCount, options);
              }
            }
          }
        }
      }
    }
  }
  runAllTests<TestRecord<VertexDesc>>(tests, numTests, deviceType, errCount,
                                      options);
  return (errCount == 0) ? 0 : 1; // returning 1 means an error.
}
