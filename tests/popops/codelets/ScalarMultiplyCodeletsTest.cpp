// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Tests the ScalarMultiply codelets:
//
// One or more combinations of operation/data type can be specified for the
// vertices under test.
//
// There is also an option to compare cycles reported when running the vertex
// on two different devices.
//
// See description string in main() for details and examples of usage.

#include "BinaryCodeletsTest.hpp"
#include <poplar/Engine.hpp>
#include <regex>
#include <spdlog/fmt/fmt.h>

using namespace poputil;

// All vertices that can be tested by this code
const std::vector<std::string> verticesNames = {
    "ScalarMultiply1D",
    "ScalarMultiply1DInplace",
    "ScalarMultiply2D",
    "ScalarMultiply2DInplace",
};

//*************************************************************************
// This contains the name of the vertex under test and various flags that
// characterise it, used in different places during the test.
struct VertexDesc {
  std::string name;

  Type dataType, scaleType;

  std::string vClass;    // Full name with template params for addVertex()
  std::string vClassFmt; // vClass, formatted for display

  // Names of the operand fields in the vertex
  std::string in1Name;
  const std::string in2Name = "in2";
  const std::string outName = "out";

  bool is2D, inPlace;

  VertexDesc(const std::string &vertexName, const Type &dataType,
             const Type &scaleType)
      : name(vertexName), dataType(dataType), scaleType(scaleType) {

    // Extract the flags by looking at the name
    is2D = vertexName.find("2D") != std::string::npos;
    inPlace = vertexName.find("Inplace") != std::string::npos;

    // Build the vertex class
    std::string vName = "popops::" + name;
    vClass = templateVertex(vName, dataType, scaleType);

    in1Name = inPlace ? "in1Out" : "in1";

    // Shorten the name for display, removing namespaces
    vClassFmt = vClass;
    boost::erase_all(vClassFmt, "popops::");
    const unsigned FMT_LEN = 37;
    unsigned len = vClassFmt.size();
    unsigned padLen = (len < FMT_LEN) ? (FMT_LEN - len) : 1;
    vClassFmt += std::string(padLen, ' '); // pad for display
  }
};

//*************************************************************************
// Contains information relative to the test for one single vertex
struct TestRecord {
  SizeDesc size;
  std::shared_ptr<VertexDesc> vertex;

  TestOperand in1, in2, out;

  float tolerance;

  unsigned startPadBytes; // see definition in MiscOptions

  // Stream names used to transfer the host data, for the two operands and the
  // output. Must be different for each test that is run in the same graph/CS.
  std::string writeName1, writeName2, readName;

  // Is the output buffer padding made up of Nan values?
  bool padOutWithNan = false;

  /// \param[in] v      The vertex (with data and scale type) to test.
  /// \param[in] seq    A sequential index, different for each test
  /// \param[in] sz     Describes generically the size to use for the test,
  ///                   needs to be adjusted for this specific vertex.
  /// \param[in] tolerance The value in the vertex field
  TestRecord(std::shared_ptr<VertexDesc> v, unsigned seq, const SizeDesc &sz,
             float tolerance, unsigned startPadBytes)
      : vertex(std::move(v)), tolerance(tolerance),
        startPadBytes(startPadBytes) {
    writeName1 = vertex->in1Name + "_" + to_string(seq);
    writeName2 = vertex->in2Name + "_" + to_string(seq);
    readName = vertex->outName + "_" + to_string(seq);
    size = sz.adjust(vertex->is2D);
  };
  TestRecord(TestRecord &&) = default;

  std::string toString() {
    return fmt::format("{} toler. = {:<9}   {}{}", vertex->vClassFmt, tolerance,
                       size.toString(), formatStartPadBytes(startPadBytes));
  }
};

const static std::vector<Type> validDataTypes = {HALF};
const static std::vector<Type> validScaleTypes = {FLOAT};

//*************************************************************************
// Return true if the information in vertex (vertex name, data and scale
// type) is for a valid vertex type.
bool isValidCombination(const VertexDesc &vertex) {
  return (vertex.dataType == HALF && vertex.scaleType == FLOAT);
}

//*************************************************************************
/// Verifies the results of the test.
/// Input data and device results are stored in 'test'.
///
/// \tparam HostDataType     Type used on the host for dataType1, datatType2.
/// \tparam HostScaleType    Type used on the host for scaleType.
/// \param[in]    target     Which target.
/// \param[in]    isIpuModel Was the device and IpuModel?
/// \param[in]    test       Describes the test to setup.
///
/// \return true if the values returned by the device match (with appropriate
///         tolerances) the one computed on the host
///
template <typename HostDataType, typename HostScaleType>
bool verifyTest(const Target &target, bool isIpuModel, const TestRecord &test,
                const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &size = test.size.val;

  // Convert the device data in host format. Also convert back the input data
  // in host format.
  std::vector<HostDataType> in1Host(test.in1.totalElems);
  std::vector<HostScaleType> in2Host(test.in2.totalElems);
  std::vector<HostDataType> outHost(test.out.totalElems);
  copy(target, vertex.dataType, test.in1.rawBuf.get(), in1Host.data(),
       in1Host.size());
  copy(target, vertex.scaleType, test.in2.rawBuf.get(), in2Host.data(),
       in2Host.size());
  copy(target, vertex.dataType, test.out.rawBuf.get(), outHost.data(),
       outHost.size());
  if (options.printBuffers) {
    printBuffer(vertex.outName, outHost, vertex.dataType, size,
                test.out.offsets);
  }

  // Check for mismatches on computed values

  // Second operand is a single value
  HostDataType val2 = in2Host[test.in2.offsets[0]];

  unsigned errCount = 0; // how many mismatched elements we find
  unsigned numRows = size.size();
  for (unsigned row = 0; row < numRows; row++) {
    for (unsigned i = 0; i < size[row]; i++) {
      HostDataType val1 = in1Host[test.in1.offsets[row] + i];

      // Result from device
      HostDataType actual = outHost[test.out.offsets[row] + i];

      // Result for verification
      HostDataType expected;
      performOp(BinaryOpType::MULTIPLY, val1, val2, expected);

      if (!equalValues(isIpuModel, BinaryOpType::MULTIPLY, vertex.dataType,
                       expected, actual)) {
        std::cerr << format("%s[%s] : %s %s %s  =>  expected:%s;  "
                            "actual:%s\n") %
                         vertex.outName %
                         // If its is 2D, we want to show row and column where
                         // it failed, not just the linear index.
                         (vertex.is2D ? to_string(row) + "][" + to_string(i)
                                      : to_string(i)) %
                         convertToString(val1) % " x " % convertToString(val2) %
                         convertToString(expected) % convertToString(actual);
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
/// \tparam HostDataType    Type to use on the host for dataType1, datatType2.
/// \tparam HostScaleType   Type to use on the host for scaleType.
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
///                         output buffers are setup here.
/// \param[in]    tile      Which tile to run this test on.
/// \param[in]    options   Global options.
///
template <typename HostDataType, typename HostScaleType>
static void setupTest(const Target &target, bool isIpuModel, Graph &graph,
                      Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                      Sequence &download, StreamMap &streamMap,
                      TestRecord &test, unsigned tile,
                      const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const std::vector<unsigned> &size = test.size.val;
  const Type &dataType = vertex.dataType;
  const Type &scaleType = vertex.scaleType;

  // Check for inconsistencies in the combinations of parameters
  if (std::find(std::begin(verticesNames), std::end(verticesNames),
                vertex.name) == std::end(verticesNames)) {
    throw std::runtime_error(vertex.name +
                             " is not a valid vertex name. Maybe you wanted "
                             "to use the --vertexRE option for a regular "
                             "expression?");
  }

  // === Setup offsets for padding
  test.in1.setup(target, dataType, size, test.startPadBytes);
  test.in2.setup(target, scaleType, {1}, test.startPadBytes);
  test.out.setup(target, dataType, size, test.startPadBytes);

  // === Allocate and initialise host buffers with appropriate values.
  std::vector<HostDataType> in1Host(test.in1.totalElems);
  std::vector<HostScaleType> in2Host(test.in2.totalElems);

  // Using a specific random generator means that we get the same random values
  // on different platforms.
  RandomEngine rndEng;
  if (options.randomSeed != 0 || dataType == BOOL)
    rndEng = std::minstd_rand(options.randomSeed);

  // Limit the range differently for different types
  HostDataType absMax = (dataType == HALF) ? 200.0 : 32000.0;
  fillBuffer(dataType, rndEng, in1Host, 100, -absMax, absMax, true);
  fillBuffer(dataType, rndEng, in2Host, 255, -absMax, absMax, true);

  // If requested, print the buffers
  if (options.printBuffers) {
    printBuffer(vertex.in1Name, in1Host, dataType, size, test.in1.offsets);
    printBuffer(vertex.in2Name, in2Host, scaleType, {1}, test.in2.offsets);
  }

  // === Create graph variables.
  Tensor in1, in2, out;

  in1 = graph.addVariable(dataType, {test.in1.totalElems}, vertex.in1Name);
  graph.setTileMapping(in1, tile);
  in2 = graph.addVariable(scaleType, {test.in2.totalElems}, vertex.in2Name);
  graph.setTileMapping(in2, tile);
  createAlignVertex(graph, alignCS, in1, tile);
  createAlignVertex(graph, alignCS, in2, tile);
  if (vertex.inPlace) {
    out = in1;
  } else {
    out = graph.addVariable(vertex.dataType, {test.out.totalElems},
                            vertex.outName);
    graph.setTileMapping(out, tile);
    createAlignVertex(graph, alignCS, out, tile);
  }

  // === Create the vertex
  auto v = graph.addVertex(cs, vertex.vClass);
  graph.setTileMapping(v, tile);

  if (vertex.inPlace) {
    // In the inPlace case we copy the data in the 'test.out.rawBuf' from where
    // it would be both written to the device and read back with the result.
    // But we also copy it in the 'test.in1.rawBuf' to be used later for the
    // verification.
    test.in1.rawBuf =
        allocateHostMemoryForTensor(target, out, graph.getReplicationFactor());
    test.out.rawBuf = allocateHostMemoryForTensor(out, test.writeName1, graph,
                                                  upload, download, streamMap);
    copy(target, in1Host.data(), in1Host.size(), dataType,
         test.in1.rawBuf.get());
    copy(target, in1Host.data(), in1Host.size(), dataType,
         test.out.rawBuf.get());
  } else {
    test.in1.rawBuf = allocateHostMemoryForTensor(
        in1, test.writeName1, graph, upload, boost::none, streamMap);
    test.out.rawBuf = allocateHostMemoryForTensor(out, test.readName, graph,
                                                  upload, download, streamMap);
    copy(target, in1Host.data(), in1Host.size(), dataType,
         test.in1.rawBuf.get());
  }
  test.in2.rawBuf = allocateHostMemoryForTensor(in2, test.writeName2, graph,
                                                upload, boost::none, streamMap);
  copy(target, in2Host.data(), in2Host.size(), scaleType,
       test.in2.rawBuf.get());

  // Fill the padding space in the input buffers (with NaNs) for overprocessing
  // detection (only for floating point types). Also fill the output buffer
  // padding, for overrun detection. For inPlace, 'in1' and 'out' are the same.
  if (dataType == FLOAT || dataType == HALF) {
    if (vertex.inPlace) {
      test.padOutWithNan = true;
    } else {
      test.in1.setPadBytes(target, isIpuModel, true);
    }
    test.in2.setPadBytes(target, isIpuModel, true);
  }
  test.out.setPadBytes(target, isIpuModel, test.padOutWithNan);

  // Connect the operands
  TestOperand::OperandType opType = vertex.is2D
                                        ? TestOperand::OperandType::is2D
                                        : TestOperand::OperandType::is1D;
  test.in1.connectOperand(graph, v, opType, in1, vertex.in1Name);

  if (!vertex.inPlace)
    test.out.connectOperand(graph, v, opType, out, vertex.outName);

  test.in2.connectOperand(graph, v, TestOperand::OperandType::is1D, in2,
                          vertex.in2Name);

  graph.setInitialValue(v["tolerance"], test.tolerance);
}

// A macro to match the device data and output types with the types used
// on the host. This is used in doSetupTest and doVerifyTest.
// Currently only (HALF, FLOAT) is used
#define SELECT_BY_TYPES()                                                      \
  /* Note that for both HALF and FLOAT the host buffers are 'float' */         \
  SELECT_ONE(HALF, FLOAT, float, float)                                        \
  SELECT_ONE(FLOAT, FLOAT, float, float)                                       \
  /* The combination of 'dataType'+'scaleType' was not specified above */      \
  throw invalid_types(vertex.dataType, vertex.scaleType);

//*************************************************************************
// Calls the appropriate version of setupTest using the template parameters
// relevant to the data/output types.
// See 'setupTest()' for parameters
static void doSetupTest(const Target &target, bool isIpuModel, Graph &graph,
                        Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                        Sequence &download, StreamMap &streamMap,
                        TestRecord &test, unsigned tile,
                        const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;

  // Call the appropriate instantiation of the templated function
#define SELECT_ONE(IPU_DATA_TYPE, IPU_SCALE_TYPE, HOST_DATA_TYPE,              \
                   HOST_SCALE_TYPE)                                            \
  if (vertex.dataType == IPU_DATA_TYPE &&                                      \
      vertex.scaleType == IPU_SCALE_TYPE) {                                    \
    setupTest<HOST_DATA_TYPE, HOST_SCALE_TYPE>(                                \
        target, isIpuModel, graph, upload, alignCS, cs, download, streamMap,   \
        test, tile, options);                                                  \
    return;                                                                    \
  }
  SELECT_BY_TYPES()
#undef SELECT_ONE
}

//*************************************************************************
// Calls the appropriate version of verifyTest using the template parameters
// relevant to the data/output types.
// See 'verifyTest()' for parameters
bool doVerifyTest(const Target &target, bool isIpuModel, TestRecord &test,
                  const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;

#define SELECT_ONE(IPU_DATA_TYPE, IPU_SCALE_TYPE, HOST_DATA_TYPE,              \
                   HOST_SCALE_TYPE)                                            \
  if (vertex.dataType == IPU_DATA_TYPE &&                                      \
      vertex.scaleType == IPU_SCALE_TYPE) {                                    \
    return verifyTest<HOST_DATA_TYPE, HOST_SCALE_TYPE>(target, isIpuModel,     \
                                                       test, options);         \
  }
  SELECT_BY_TYPES()
#undef SELECT_ONE
}

//*************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  std::vector<Type> dataTypes;
  std::vector<Type> scaleTypes;
  std::vector<float> tolerances = {0.01};

  std::vector<SizeDesc> sizes = {{false, {25, 12, 21}}};

  std::vector<std::string> vertices;
  std::string vertexRE; // regular expression for vertex name
  MiscOptions options;

  // clang-format off
  const static std::string description =
  "Tests one or more of the ScalarMultiply vertices, with a default data\n"
  "size, or specified ones.\n"
  "If no vertex is specified, all will be tested.\n"
  "Using the --compare-cycles option, cycles reported when running the vertex\n"
  "on two different devices can be compared."
  "Examples of usages:\n"
  "\n"
  "   ScalarMultiplyTest --vertex ScalarMultiply1D  --size 5000\n"
  "\n"
  "   ScalarMultiplyTest --vertexRE 2D  --size 5000 [10,40,50] 300 [23,24]\n"
   "\n"
  "\n"
  "Details of options are:";

  po::options_description poDesc(description);

  // Get a string with all vertices names, comma separated
  std::string allVerticesStr;
  for (auto &n : verticesNames)
    allVerticesStr += (allVerticesStr.empty()? "" : ", ") + n;

  poDesc.add_options()
    ("vertex",
     po::value<std::vector<std::string>>(&vertices)->multitoken(),
     ("Vertices to test, one or more of: " + allVerticesStr).c_str())
    ("vertexRE",
     po::value<std::string>(&vertexRE),
     "Regular expression to specify vertex names (alternative to --vertex)")
    ("data-type",
     po::value<std::vector<Type>>(&dataTypes)->multitoken(),
     "Data type (for first operand): one or more of half, float, int, uint, "
     "short, ushort, bool, char, schar, uchar, ulonglong, longlong")
    ("scale-type",
     po::value<std::vector<Type>>(&scaleTypes)->multitoken(),
     "Scale type (for second operand): one or more of half, float")
    ("tolerance",
     po::value<std::vector<float>>(&tolerances)->multitoken(),
     "Value(s) for the 'tolerance' field")
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex, "
     "multiple values, comma separated, in square brackets, for a 2D vertex")
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

  // === If no data or scale type specified, test valid ones
  if (dataTypes.empty()) {
    dataTypes = validDataTypes;
  }
  if (scaleTypes.empty()) {
    scaleTypes = validScaleTypes;
  }

  std::regex vertexRegEx(vertexRE);

  std::vector<std::shared_ptr<TestRecord>> tests;

  unsigned numTests = 0;
  unsigned errCount = 0;
  unsigned nSizes = sizes.size();
  // Loop over every specified: vertex names, data types, scale types,
  // tolerances and sizes.
  for (std::string vertexName : vertices) {
    // If a regex was specified, see if it matches this name
    if (vertexRE.empty() || std::regex_search(vertexName, vertexRegEx)) {
      for (auto dataType : dataTypes) {
        for (auto scaleType : scaleTypes) {
          auto vertex =
              std::make_shared<VertexDesc>(vertexName, dataType, scaleType);
          for (auto tolerance : tolerances) {
            for (unsigned i = 0; i < nSizes; i++) {
              for (auto startPadBytes : options.startPadBytes) {
                // Finally do the deed: check if valid vertex, add
                // the test to the 'to-be-run' list (or run it straight away,
                // depending on options).
                if (isValidCombination(*vertex)) {
                  numTests++;
                  auto testRec = std::make_shared<TestRecord>(
                      vertex, numTests, sizes[i], tolerance, startPadBytes);
                  addOneTest<TestRecord, VertexDesc>(tests, testRec, deviceType,
                                                     errCount, options);
                }
              }
            }
          }
        }
      }
    }
  }
  runAllTests<TestRecord>(tests, numTests, deviceType, errCount, options);
  return (errCount == 0) ? 0 : 1; // returning 1 means an error.
}