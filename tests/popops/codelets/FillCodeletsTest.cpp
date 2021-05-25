// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Tests one or both of the Fill codelets (Fill, Fill2d):
//
// One or more data types can be specified for the vertices under test.
//
// There is also an option to compare cycles reported when running the vertex
// on two different devices.
//
// See description string in main() for details and examples of usage.

#include "CodeletsTestsCommon.hpp"

#include <poplar/Engine.hpp>

#include <exception>

using namespace poputil;

const std::vector<std::string> verticesNames = {"Fill", "Fill2d"};

//*************************************************************************
// This contains the name of the vertex under test and some info about it.
struct VertexDesc {
  std::string name;
  Type dataType;
  std::string vClass;    // full name with template params for addVertex()
  std::string vClassFmt; // vClass, formatted for display
  bool is2D;

  VertexDesc(const std::string &vertexName, const Type &dataType)
      : name(vertexName), dataType(dataType) {

    is2D = vertexName.find("2d") != std::string::npos;

    std::string vName = "popops::" + name;
    vClass = templateVertex(vName, dataType);

    // Shorten the name for display, removing namespaces
    vClassFmt = vClass;
    boost::erase_all(vClassFmt, "popops::");
    const unsigned FMT_LEN = 40;
    unsigned len = vClassFmt.size();
    unsigned padLen = (len < FMT_LEN) ? (FMT_LEN - len) : 1;
    vClassFmt += std::string(padLen, ' '); // pad for display
  }
};

//*************************************************************************
// Return true if the data type is valid for the fill vertices
bool isValidCombination(const VertexDesc &vertex) {
  const Type &type = vertex.dataType;
  return type == HALF || type == FLOAT || type == INT || type == UNSIGNED_INT ||
         type == BOOL || type == CHAR || type == UNSIGNED_CHAR ||
         type == SIGNED_CHAR;
}

//*************************************************************************
// Contains information relative to a single test (one specific vertex,
// and one SizeDesc value)
struct TestRecord {
  SizeDesc size;
  std::shared_ptr<VertexDesc> vertex;
  TestOperand out; // Describes the single tensor of the vertex
  float fillValue; // Note: is 'float', but is used for integer types as well

  // Stream name used to transfer in and out the tensor data. Must be
  // different for each test that is run in the same graph/compute set.
  std::string streamName;

  /// \param[in] v    The vertex (with operation and data type) to test.
  /// \param[in] seq  A sequential index, different for each test (tile number)
  /// \param[in] size The data sizes to use for the test, from command line
  /// \param[in] fillValue The value to fill the tensor with
  TestRecord(std::shared_ptr<VertexDesc> v, unsigned seq, const SizeDesc &sz,
             float fillValue)
      : vertex(std::move(v)), fillValue(fillValue) {
    streamName = "out_" + to_string(seq);
    // Adjust size specified on cmd line to vertex type
    size = sz.adjust(vertex->is2D);
  }
  TestRecord(TestRecord &&) = default;

  std::string toString() { return vertex->vClassFmt + size.toString(); }
};

//*************************************************************************
/// Verifies if the buffer has been filled. Also checks for overwrites
///
/// \return true if the verification is passed
template <typename HostDataType>
bool verifyTest(const Target &target, bool isIpuModel, const TestRecord &test,
                const MiscOptions &options) {
  const VertexDesc &v = *test.vertex;
  const std::vector<unsigned> &sizes = test.size.val;

  // Convert the device data in host format.
  std::vector<HostDataType> outH(test.out.totalElems);
  copy(target, v.dataType, test.out.rawBuf.get(), outH.data(), outH.size());
  if (options.printBuffers) {
    printBuffer("out", outH, v.dataType, sizes, test.out.offsets);
  }

  HostDataType fillValue = test.fillValue;
  // Check for mismatches on returned values
  unsigned errCount = 0; // how many mismatched elements we find
  unsigned numRows = sizes.size();
  for (unsigned row = 0; row < numRows; row++) {
    for (unsigned i = 0; i < sizes[row]; i++) {
      // value from device
      HostDataType actual = outH[test.out.offsets[row] + i];

      if (actual != fillValue) {
        std::string posStr =
            v.is2D ? to_string(row) + "," + to_string(i) : to_string(i);
        std::cerr << "out[" << posStr
                  << "] - expected:" << convertToString(fillValue)
                  << ";  actual:" << convertToString(actual) << "\n";
        errCount++;
      }
    }
  }
  if (errCount > 0) {
    std::cerr << "Failed: mismatch on " << errCount << " value(s)\n";
  }

  // Check for overwrites past the end of each row
  auto overwriteCount = test.out.checkPadBytes(target, v.is2D, isIpuModel);

  return (errCount == 0) && (overwriteCount == 0);
}

//*************************************************************************
/// Setup one vertex test.
///
/// \tparam HostDataType Type to use on the host for dataType1, datatType2.
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
template <typename HostDataType>
static void setupTest(const Target &target, bool isIpuModel, Graph &graph,
                      Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                      Sequence &download, StreamMap &streamMap,
                      TestRecord &test, unsigned tile,
                      const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;

  // Is the vertex name correct?
  if (std::find(std::begin(verticesNames), std::end(verticesNames),
                vertex.name) == std::end(verticesNames)) {
    throw std::runtime_error(vertex.name + " not a valid vertex name");
  }

  // === Setup offsets for padding in output tensor
  test.out.setup(target, vertex.dataType, test.size.val, options.alignStart);

  // === Create output tensor.
  Tensor out = graph.addVariable(vertex.dataType, {test.out.totalElems}, "out");
  graph.setTileMapping(out, tile);

  // === Create the auxiliary vertex required to align the tensor to 8 bytes
  createAlignVertex(graph, alignCS, out, tile);

  // === Create the vertex under test
  auto v = graph.addVertex(cs, vertex.vClass);
  graph.setTileMapping(v, tile);

  // Note that allocateHostMemoryForTensor fills the buffer with zeros
  test.out.rawBuf = allocateHostMemoryForTensor(out, test.streamName, graph,
                                                upload, download, streamMap);

  // Fill the output buffer padding, for overrun detection.
  test.out.setPadBytes(target, isIpuModel);

  // === Connect the edges appropriately, depending on the vertex variant
  test.out.connectOperand(graph, v,
                          vertex.is2D ? TestOperand::OperandType::is2D
                                      : TestOperand::OperandType::is1D,
                          out, "out");

  graph.setInitialValue(v["in"], static_cast<HostDataType>(test.fillValue));
}

#define SELECT_BY_TYPES()                                                      \
  SELECT_ONE(HALF, float)                                                      \
  SELECT_ONE(FLOAT, float)                                                     \
  SELECT_ONE(INT, int)                                                         \
  SELECT_ONE(UNSIGNED_INT, unsigned int)                                       \
  SELECT_ONE(SHORT, short)                                                     \
  SELECT_ONE(UNSIGNED_SHORT, unsigned short)                                   \
  SELECT_ONE(BOOL, unsigned char)                                              \
  SELECT_ONE(CHAR, char)                                                       \
  SELECT_ONE(UNSIGNED_CHAR, unsigned char)                                     \
  SELECT_ONE(SIGNED_CHAR, signed char)                                         \
  throw std::runtime_error("data type (" + test.vertex->dataType.toString() +  \
                           ") not supported");

//*************************************************************************
// Calls the appropriate version of setupTest using the template parameters
// relevant to the data/output types.
// See 'setupTest()' for parameters
static void doSetupTest(const Target &target, bool isIpuModel, Graph &graph,
                        Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                        Sequence &download, StreamMap &streamMap,
                        TestRecord &test, unsigned tile,
                        const MiscOptions &options) {
#define SELECT_ONE(IPU_DATA_TYPE, HOST_DATA_TYPE)                              \
  if (test.vertex->dataType == IPU_DATA_TYPE) {                                \
    setupTest<HOST_DATA_TYPE>(target, isIpuModel, graph, upload, alignCS, cs,  \
                              download, streamMap, test, tile, options);       \
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
#define SELECT_ONE(IPU_DATA_TYPE, HOST_DATA_TYPE)                              \
  if (test.vertex->dataType == IPU_DATA_TYPE) {                                \
    return verifyTest<HOST_DATA_TYPE>(target, isIpuModel, test, options);      \
  }
  SELECT_BY_TYPES()
#undef SELECT_ONE
}

//*************************************************************************
int main(int argc, char **argv) {
  DeviceType deviceType;
  std::vector<Type> dataTypes;
  std::vector<SizeDesc> sizes = {{false, {25, 12, 21}}}; //  Sizes of output.
  std::vector<std::string> vertices; // vertices that will be tested
  MiscOptions options;
  options.randomSeed = 0;
  float fillValue = 66;

  // clang-format off
  const static std::string description =
  "Tests Fill vertices with one or more data type(s), with a default data\n"
  "size, or a specified one.\n"
  "If no vertex is specified, both will be tested. If no data type specified,\n"
  "all valid types will be tested."
  "Using the --compare-cycles option, cycles reported when running the vertex\n"
  "on two different devices can be compared."
  "Examples of usages:\n"
  "\n"
  "Test Fill<half>, with an operand of size 5000 (use default fill value):\n"
  "   FillCodeletsTest --vertex Fill --data-type half --size 5000\n"
  "\n"
  "Test Fill<char> and Fill2d<char>, default operand size with value -12:\n"
  "   FillCodeletsTest --data-type char --fill-value -12\n"
"\n"
  "\n"
  "Details of options are:";

  po::options_description poDesc(description);

  poDesc.add_options()
    ("vertex",
     po::value<std::vector<std::string>>(&vertices)->multitoken(),
     "Vertices to test, one or both of: Fill, Fill2d")
    ("data-type",
     po::value<std::vector<Type>>(&dataTypes)->multitoken(),
     "Data type: one or more of half,float,int,uint,bool,char,schar,uchar")
    ("fill-value",
     po::value<float>(&fillValue)->default_value(fillValue),
     "fill value to use")
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex or a "
     "square bracket, comma separated list of values for a 2D vertex")
    ;
  // clang-format on
  addCommonOptions(poDesc, deviceType, options);
  parseOptions(argc, argv, poDesc);

  // === Some parameter checks
  if (options.randomSeed) {
    std::cout << "--random-seed option is ignored by this test\n";
  }

  // === If no vertices specified, test 'em all
  if (vertices.empty()) {
    vertices = verticesNames;
  }

  // === If no data type specified, test 'em all
  if (dataTypes.empty()) {
    dataTypes = {HALF, FLOAT, INT,         UNSIGNED_INT,
                 BOOL, CHAR,  SIGNED_CHAR, UNSIGNED_CHAR};
  }

  std::vector<std::shared_ptr<TestRecord>> tests;
  unsigned numTests = 0;
  unsigned errCount = 0;
  // Loop over all vertices, operations, data types
  for (std::string vertexName : vertices) {
    for (auto type : dataTypes) {
      auto vertex = std::make_shared<VertexDesc>(vertexName, type);
      for (auto sz : sizes) {
        if (isValidCombination(*vertex)) {
          numTests++;
          auto testRec =
              std::make_shared<TestRecord>(vertex, numTests, sz, fillValue);
          addOneTest<TestRecord, VertexDesc>(tests, testRec, deviceType,
                                             errCount, options);
        }
      }
    }
  }
  runAllTests<TestRecord>(tests, numTests, deviceType, errCount, options);
  return (errCount == 0) ? 0 : 1; // returning 1 means an error.
}
