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
#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <type_traits>

using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;
using boost::format;
using std::to_string;

// The name of the compute set where we run the vertex under test.
const static std::string computeSetName = "vertexComputeSet";

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

  std::string vClass;      // full name with template params for addVertex()
  std::string vClassShort; // vClass, abbreviated for display

  // Name of the operand field in the vertex
  std::string inName;

  bool is2D;
  bool inPlace;

  VertexDesc(const std::string &vertexName) {
    name = vertexName;

    // Extract the flags by looking at the name
    is2D = vertexName.find("2D") != std::string::npos;
    inPlace = vertexName.find("InPlace") != std::string::npos;

    inName = inPlace ? "inOut" : "in";
  }

  void setVertexClass(const UnaryOpType op, const Type &dataType,
                      const Type &outputType) {
    // Get the vertex class
    std::string vName = "popops::" + name;
    vClass = templateVertex(vName, op, dataType);

    // Shorten the name for display, removing namespaces
    vClassShort = vClass;
    boost::erase_all(vClassShort, "popops::");
    boost::erase_all(vClassShort, "expr::UnaryOpType::");
  }
};

//*************************************************************************
// Return true if the combination of vertex, operation and data type is valid
bool isValidCombination(const VertexDesc &vertex, const UnaryOpType op,
                        const Type &type) {
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
///
/// \param isIpuModel          Is IpuModel or IpuModel2?.
/// \param dataType            The data type used (float, half).
/// \param inHost              Data buffer for the operand.
/// \param outHost             Data for result, obtained from
///                            device and converted to host types.
/// \param operation           Operation performed on device.
template <typename HOST_DATA_TYPE, typename HOST_OUT_TYPE>
static bool verifyResult(const bool isIpuModel, const VertexDesc &vertex,
                         const UnaryOpType op, const Type &dataType,
                         const std::vector<HOST_DATA_TYPE> &inHost,
                         const std::vector<HOST_OUT_TYPE> &outHost) {
  unsigned errCount = 0; // how many mismatched elements we find

  // Loop sequentially over the  operand linear data buffer
  for (unsigned i = 0; i < inHost.size(); i++) {
    HOST_DATA_TYPE val = inHost[i]; // operands

    HOST_OUT_TYPE actual = outHost[i]; // result from device

    HOST_OUT_TYPE expected = 0; // result for verification
    performOp(op, val, expected);

    if (!equalValues(isIpuModel, op, dataType, expected, actual)) {
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
/// Run one vertex test.
///
/// \tparam HOST_DATA_TYPE   Type to use on the host for dataType.
/// \tparam HOST_OUT_TYPE    Type to use on the host for outputType.
/// \param  deviceType       The device used.
/// \param  vertex           Which vertex.
/// \param  op               Operation to perform
/// \param  dataType         The data type used for the operand.
/// \param  outputType       The type for the result of the operation.
/// \param  sizes            Describes the size of the operands.
/// \param  randomSeed       Random seed (0 = don't use random values).
/// \param  ignoreData       Do not verify results.
/// \param  doReport         Print poplar report on stdout.
/// \param  cycles           If non-empty, will be populated with the cycles
///                          used by the compute set that runs the vertex.
///
/// \return true if the results from the device passed verification
template <typename HOST_DATA_TYPE, typename HOST_OUT_TYPE>
static bool doTest(const DeviceType &deviceType, const VertexDesc &vertex,
                   const UnaryOpType op, const Type &dataType,
                   const Type &outputType, const std::vector<unsigned> &sizes,
                   const unsigned randomSeed, const bool ignoreData,
                   const bool doReport, std::optional<uint64_t> &cycles) {

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

  if (isIntOp(op) &&
      (dataType == HALF || dataType == FLOAT || dataType == BOOL)) {
    throw std::runtime_error(unaryOpToString.at(op) +
                             " requires data "
                             "of integer type (specified  data type=" +
                             dataType.toString() + ")");
  }

  TestDevice device = createTestDevice(deviceType, 1, 1);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // === Allocate and initialise host buffers with appropriate values.
  const unsigned nElems =
      vertex.is2D ? std::accumulate(sizes.begin(), sizes.end(), 0) : sizes[0];
  std::vector<HOST_DATA_TYPE> inHost(nElems);
  fillHostBuffer(op, dataType, randomSeed, inHost);

  // === Create graph variables.
  Tensor in, out;
  in = graph.addVariable(dataType, {nElems}, vertex.inName);
  graph.setTileMapping(in, 0);
  if (vertex.inPlace) {
    out = in;
  } else {
    out = graph.addVariable(outputType, {nElems}, vertex.inName);
    graph.setTileMapping(out, 0);
  }

  // === Make a program to run the vertex
  Sequence prog;

  ComputeSet cs = graph.addComputeSet(computeSetName);

  auto v = graph.addVertex(cs, vertex.vClass);
  graph.setTileMapping(v, 0);
  prog.add(Execute(cs));

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
      (op == UnaryOpType::RELU || op == UnaryOpType::SIGMOID ||
       op == UnaryOpType::TANH)) {
    graph.setInitialValue(v["n"], nElems);
  }

  // === Create host 'transfer' buffers with the right size for the device type
  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::unique_ptr<char[]> inHostRaw;
  std::unique_ptr<char[]> outHostRaw;
  char *outHostRawPtr = nullptr;
  inHostRaw = allocateHostMemoryForTensor(in, vertex.inName, graph, uploadProg,
                                          downloadProg, tmap);
  if (!vertex.inPlace) {
    outHostRaw = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                             downloadProg, tmap);
    outHostRawPtr = outHostRaw.get();
  } else {
    outHostRawPtr = inHostRaw.get();
  }

  // === Copy and convert the data from the initialised buffers to the transfer
  // === buffers (still on host)
  auto copyBuffer = [&](Type type, std::vector<HOST_DATA_TYPE> &buf,
                        std::unique_ptr<char[]> &rawBuf) {
    copy(target, buf.data(), buf.size(), type, rawBuf.get());
    // For HALF, we copy and convert back into the (float) host buffers so that
    // the host buffers contain the exact HALF values (which are exactly
    // representable in float). This helps with the validation for the
    // comparison operators
    if (type == HALF)
      copy(target, type, rawBuf.get(), buf.data(), buf.size());
  };
  copyBuffer(dataType, inHost, inHostRaw);

  // === Run the program
  OptionFlags engOpts;
  if (doReport || cycles) {
    engOpts.set("debug.instrumentCompute", "true");
  }
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engOpts);
  attachStreams(engine, tmap);

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();

    if (doReport) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");
      engine.printProfileSummary(std::cout, opt);
    }
    if (cycles) {
      // Get the cycles by searching the "simulation"/"steps" vector for an
      // "OnTileExecute" element having the compute set name we used.
      poplar::ProfileValue execProfile = engine.getExecutionProfile();
      for (auto s : execProfile["simulation"]["steps"].asVector()) {
        if (s["type"] == "OnTileExecute" && s["name"] == computeSetName) {
          cycles = s["cycles"].asUint();
        }
      }
    }
  });

  // === Check the result
  if (ignoreData) {
    std::cout << "Result not checked for correctness\n";
  } else {
    // Get the result out of the device
    std::vector<HOST_OUT_TYPE> outHost(nElems);
    copy(target, outputType, outHostRawPtr, outHost.data(), outHost.size());
    return verifyResult<HOST_DATA_TYPE, HOST_OUT_TYPE>(
        isIpuModel(deviceType), vertex, op, outputType, inHost, outHost);
  }
  return true;
}

//*************************************************************************
// Calls doTest with the appropriate template parameters for the
// data/output types.
// See 'doTest()' for parameters
static bool doVertexTest(const DeviceType &deviceType, VertexDesc &vertex,
                         const UnaryOpType op, const Type &dataType,
                         const std::vector<unsigned> &sizes,
                         const unsigned randomSeed, const bool verbose,
                         const bool ignoreData, const bool doReport,
                         std::optional<uint64_t> &cycles) {
  // Get right output type for vertex, operator and data type
  Type outputType = dataType;
  if (isBoolOp(op))
    outputType = BOOL;

  vertex.setVertexClass(op, dataType, outputType);

  if (verbose) {
    // Make a string that describes the size of the operand
    std::string sizesStr = "[";
    if (vertex.is2D) {
      for (auto e : sizes)
        sizesStr += "[" + to_string(e) + "] ";
      sizesStr.erase(sizesStr.size() - 1); // remove last space
    } else {
      sizesStr += to_string(sizes[0]);
    }
    sizesStr += ']';

    std::cout << boost::format("%-70s %s\n") % vertex.vClassShort % sizesStr;
  }

  // Call the appropriate instantiation of the templated function
#define DO_TEST(DATA_TYPE, OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE)            \
  if (dataType == DATA_TYPE && outputType == OUT_TYPE) {                       \
    return doTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(                              \
               deviceType, vertex, op, dataType, outputType, sizes,            \
               randomSeed, ignoreData, doReport, cycles)                       \
               ? true                                                          \
               : false;                                                        \
  }

  // Note that for both HALF and FLOAT the host buffers are 'float'
  DO_TEST(BOOL, BOOL, HostBool, HostBool)

  DO_TEST(HALF, HALF, float, float)
  DO_TEST(HALF, BOOL, float, HostBool)

  DO_TEST(FLOAT, FLOAT, float, float)
  DO_TEST(FLOAT, BOOL, float, HostBool)

  DO_TEST(HALF, FLOAT, float, float)
  DO_TEST(FLOAT, HALF, float, float)

  DO_TEST(INT, INT, int, int)
  DO_TEST(INT, BOOL, int, HostBool)

  DO_TEST(UNSIGNED_INT, UNSIGNED_INT, unsigned, unsigned)
  DO_TEST(UNSIGNED_INT, BOOL, unsigned, HostBool)

  DO_TEST(SHORT, SHORT, short, short)
  DO_TEST(UNSIGNED_SHORT, UNSIGNED_SHORT, unsigned short, unsigned short)

  // Reaching here means the combination of 'dataType' and 'outputType' was
  // invalid.
  throw std::runtime_error("Combination of data type/operator not supported");
  return false;
}

/// Compare cycles obtained by running the vertex with the two specified
/// devices. Prints result on standard output.
///
/// \return true   if both run returned successfully and the difference is less
///                than 'tolerance' % of the run with the first device.
static bool compareCycles(const std::array<DeviceType, 2> devPair,
                          VertexDesc &vertex, const UnaryOpType op,
                          const Type &dataType,
                          const std::vector<unsigned> &sizes,
                          const unsigned randomSeed,
                          const unsigned compareThreshold) {
  std::stringstream devName[2]; // To get strings with the name of selected dev
  bool ok[2];
  uint64_t cycles[2];
  // Run with the two devices and get the cycles
  for (unsigned i = 0; i < 2; i++) {
    devName[i] << devPair[i];
    std::optional<uint64_t> cyclesOpt = uint64_t(0);
    ok[i] = doVertexTest(devPair[i], vertex, op, dataType, sizes, randomSeed,
                         false, false, false, cyclesOpt);
    cycles[i] = *cyclesOpt;
  }

  float diffPerc = 0;
  if (ok[0] && ok[1]) {
    float diff = static_cast<float>(cycles[1]) - static_cast<float>(cycles[0]);
    diffPerc = diff / cycles[0] * 100;
    std::cout << boost::format(
                     "%-70s - %s:%8u;  %s:%8u;   diff = %u  %7.2f%%%s\n") %
                     vertex.vClassShort % devName[0].str() % cycles[0] %
                     devName[1].str() % cycles[1] % diff % diffPerc %
                     ((abs(diffPerc) < compareThreshold) ? "" : " <<== FAIL");
  } else {
    std::cout << boost::format("%-74s - Failed\n") % vertex.vClassShort;
  }
  return ok[0] && ok[1] && abs(diffPerc) < compareThreshold;
}

//*************************************************************************
int main(int argc, char **argv) {

  DeviceType deviceType;
  std::vector<Type> dataTypes;

  //  Sizes of the operand. For a Supervisor vertex,
  std::vector<unsigned> sizes = {25, 12, 21};

  std::vector<std::string> operationStr;
  std::vector<std::string> vertices;
  std::string vertexRE; // regular expression for vertex name
  bool doReport = false;
  bool ignoreData = false;
  unsigned randomSeed = 1; // we use '0' to mean 'not random'
  boost::optional<std::string> cycleCompareDevice;
  unsigned cycleCompareThreshold = 10; // percent

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
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(DeviceType::Sim2),
     "Device type")
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
    ("data-type",
     po::value<std::vector<Type>>(&dataTypes)->multitoken(),
     "Data type: one or more of half, float, int, uint, short, ushort, bool")
    ("size",
     po::value<std::vector<unsigned>>(&sizes)->multitoken(),
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
    ;
  // clang-format on
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
    dataTypes = {HALF, FLOAT, INT, UNSIGNED_INT, BOOL, SHORT, UNSIGNED_SHORT};
  }

  std::regex vertexRegEx(vertexRE);

  // If we are comparing cycles, create an array with the 2 devices. If the
  // specific type of device to compare was unspecified ("default") we try to
  // match Sim1 with IpuModel1 or Sim2 with IpuModel2, otherwise we just use
  // the specified device.
  std::array<DeviceType, 2> devPair;
  if (cycleCompareDevice) {
    std::optional<DeviceType> compDev;
    if (*cycleCompareDevice == "default") {
      if (deviceType == DeviceType::Sim1) {
        compDev = DeviceType::IpuModel1;
      } else if (deviceType == DeviceType::Sim2) {
        compDev = DeviceType::IpuModel2;
      }
    }
    if (!compDev) {
      std::istringstream is(*cycleCompareDevice);
      is >> *compDev;
    }
    devPair = {deviceType, *compDev};
  }

  // Loop over all vertices, operations, data types
  bool allOk = true;
  unsigned count = 0, errCount = 0;
  std::optional<uint64_t> cycles = std::nullopt; // not interested in cycles
  for (std::string vertexName : vertices) {
    // If a regex was specified, see if it matches
    if (vertexRE.empty() || std::regex_search(vertexName, vertexRegEx)) {
      VertexDesc vertex(vertexName);
      for (auto op : operations) {
        for (auto type : dataTypes) {
          if (isValidCombination(vertex, op, type)) {
            bool ok = cycleCompareDevice
                          ? compareCycles(devPair, vertex, op, type, sizes,
                                          randomSeed, cycleCompareThreshold)
                          : doVertexTest(deviceType, vertex, op, type, sizes,
                                         randomSeed, true, ignoreData, doReport,
                                         cycles);
            allOk = allOk & ok;
            count++;
            errCount += ok ? 0 : 1;
          }
        }
      }
    }
  }
  if (count == 0) {
    throw std::runtime_error("The specified vertex, operand(s) and data "
                             "type(s) do not match any valid combination");
  } else if (count > 1) {
    std::cout << "UnaryCodeletsTest: " << count << " tests run in total; "
              << ((errCount == 0) ? "All passed"
                                  : to_string(errCount) + " failed");
  }
  return allOk ? 0 : 1; // returning 1 means an error.
}
