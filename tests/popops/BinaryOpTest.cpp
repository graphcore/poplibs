// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
//
// Performs a binary operation between two tensors with any desired shape, each
// mapped in any desired way among tiles.

#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>

#include <boost/format.hpp>

#include "codelets/BinaryCodeletsTest.hpp"

#include <poplibs_test/TempDir.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <optional>
#include <type_traits>

using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplar_test;

//*************************************************************************
/// Verifies if the results of the operation performed on the device match
/// with the one the host
///
/// \param deviceType          The device used.
/// \param dataType            The data type used (float, half).
/// \param in1Host             Data buffer for the first operand.
/// \param shape1Ext           Shape for first operand, rank-extended.
/// \param in2Host, shape1Ext  Data and shape for second operand.
/// \param outHost, outShape   Data (and shape) for result, obtained from
///                            device and converted to host types.
/// \param operation           Operation performed on device.
template <typename HostDataType, typename HostOutType>
static bool verifyResult(const bool isIpuModel, const Type &dataType,
                         const std::vector<HostDataType> &in1Host,
                         const std::vector<size_t> &shape1Ext,
                         const std::vector<HostDataType> &in2Host,
                         const std::vector<size_t> &shape2Ext,
                         const std::vector<HostOutType> &outHost,
                         const std::vector<size_t> &shapeOut,
                         const BinaryOpType op) {
  unsigned errCount = 0; // how many mismatched elements we find

  unsigned n = shapeOut.size(); // How many dimensions we have

  // Perform the specified 'op'eration element-wise between in1 and in2 (on
  // the host) and compare with what is returned by the device.
  // We need to nest a variable number ('n') of loops to scan through all
  // 'n' dimensions. We use a recursive function the will recur once for each
  // nested loop ('n' times).
  // The vector 'i[]' contains the indices into 'in1', 'in2, 'out'
  std::vector<unsigned> i(n);
  // Cannot use 'auto' for the type, because it's a recursive function.
  std::function<void(unsigned)> loopOn = [&](unsigned k) {
    // Run the k-th nested loop
    for (i[k] = 0; i[k] < shapeOut[k]; i[k]++) {
      if (k == n - 1) {
        // This is the "innermost loop"; we need to compute:
        // expected[ i[0], i[1],... ] =
        //                in1[ i[0], i[1],... ] *OP*  in2[ i[0], i[1],... ]
        // and compare with the actual value from the device

        HostOutType actual = get(outHost.data(), shapeOut, i); // from device

        HostDataType val1 = get(in1Host.data(), shape1Ext, i);
        HostDataType val2 = get(in2Host.data(), shape2Ext, i);

        HostOutType expected = 0;

        performOp(op, val1, val2, expected);

        if (!equalValues(isIpuModel, op, dataType, expected, actual)) {
          std::cerr << "out[" << i[0];
          for (unsigned j = 1; j < n; j++)
            std::cerr << "," << i[j];
          std::cerr << "] = " << convertToString(val1) << " "
                    << binaryOpToString.at(op) << " " << convertToString(val2)
                    << " =>  expected:" << convertToString(expected)
                    << ";  actual:" << convertToString(actual) << "\n";
          errCount++;
        }
      } else {
        loopOn(k + 1); // recur to go down to next nested loop
      }
    }
  };
  loopOn(0);

  if (errCount > 0) {
    std::cerr << "Failed: mismatch on " << errCount << " value(s)\n";
  }
  return errCount == 0;
}

//*************************************************************************
/// Do a binary operation, where the two operands are described by 'desc1' and
/// 'desc2'. The shape that the output will have has already been computed
/// using broadcasting rules ('shapeOut').
///
/// \param deviceType            The device used.
/// \param dataType              The data type used for the two opernds.
/// \param outputType            The type for the result of the operation.
/// \param desc1                 Description for first operand.
/// \param desc2                 Description for second operand.
/// \param outShape              Shape of result (computed from shape1, shape2).
/// \param tiles                 How many tiles to allocate.
/// \param mapLinearly           If yes, we map linearly the two operands on
///                              all tiles.
/// \param operation             Operation performed on device.
/// \param inPlace               Is the operation to be done in place?
/// \param doReport              Print poplar report.
/// \param doPrintTensors        Print the tensors (for verification).
/// \param ignoreData            Do not verify results.
/// \param enableOptimisations   Enable broadcasted vector op optimisations.
template <typename HostDataType, typename HostOutType>
static bool doBinaryOpTest(
    const DeviceType &deviceType, const Type &dataType, const Type &outputType,
    const OperandDescriptor &desc1, const OperandDescriptor &desc2,
    const std::vector<size_t> &shapeOut, const unsigned tiles,
    const bool mapLinearly, const BinaryOpType operation, const bool inPlace,
    const bool doReport, const bool doPrintTensors, const unsigned randomSeed,
    const bool ignoreData, const bool enableOptimisations) {

  bool in1IsConst = desc1.map.size() > 0 && desc1.map[0].isConst;
  bool in2IsConst = desc2.map.size() > 0 && desc2.map[0].isConst;

  auto nElems1 =
      std::accumulate(desc1.shape.begin(), desc1.shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  auto nElems2 =
      std::accumulate(desc2.shape.begin(), desc2.shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  if (in1IsConst && inPlace) {
    throw std::runtime_error("For in-place operations, first operand cannot "
                             "be a constant");
  }

  if (in1IsConst && in2IsConst) {
    throw std::runtime_error("The two operands cannot be both constants");
  }

  if (in1IsConst && nElems1 != 1) {
    throw std::runtime_error("The first operand is specified as a constant "
                             "but also has more than one element");
  }

  if (in2IsConst && nElems2 != 1) {
    throw std::runtime_error("The second operand is specified as a constant "
                             "but also has more than one element");
  }

  if (inPlace && (outputType != dataType)) {
    throw std::runtime_error("For in place operations, the data and output "
                             "types must be the same (specified data type=" +
                             dataType.toString() + ", specified output type=" +
                             outputType.toString() + ")");
  }

  if (isIntOp(operation) && (dataType == HALF || dataType == FLOAT)) {
    throw std::runtime_error(binaryOpToString.at(operation) +
                             " requires data "
                             "of integer type (specified  data type=" +
                             dataType.toString() + ")");
  }

  auto nElemsOut =
      std::accumulate(shapeOut.begin(), shapeOut.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  // Allocate and initialise host buffers with appropriate values.
  std::vector<HostDataType> in1Host(nElems1);
  std::vector<HostDataType> in2Host(nElems2);
  fillHostBuffers(operation, dataType, randomSeed, in1Host, in2Host);

  // Setup const value based on user's input
  if (!desc1.map.empty() && desc1.map[0].constVal)
    in1Host[0] = *desc1.map[0].constVal;
  else if (!desc2.map.empty() && desc2.map[0].constVal)
    in2Host[0] = *desc2.map[0].constVal;

  // Create Graph object, target and device
  auto device = createTestDevice(deviceType, 1, tiles);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  Tensor in1, in2;
  std::vector<poplar::Tensor> tensors;
  expr::BinaryOp binOp = [&]() {
    if (in1IsConst) {
      in2 = graph.addVariable(dataType, desc2.shape, "in2");
      mapTensor(graph, in2, mapLinearly, desc2.map);
      tensors = {in2};
      return expr::BinaryOp(operation, expr::Const(in1Host[0]), expr::_1);
    } else if (in2IsConst) {
      in1 = graph.addVariable(dataType, desc1.shape, "in1");
      mapTensor(graph, in1, mapLinearly, desc1.map);
      tensors = {in1};
      return expr::BinaryOp(operation, expr::_1, expr::Const(in2Host[0]));
    } else {
      in1 = graph.addVariable(dataType, desc1.shape, "in1");
      in2 = graph.addVariable(dataType, desc2.shape, "in2");
      mapTensor(graph, in1, mapLinearly, desc1.map);
      mapTensor(graph, in2, mapLinearly, desc2.map);
      tensors = {in1, in2};
      return expr::BinaryOp(operation, expr::_1, expr::_2);
    }
  }();

  OptionFlags opOpts{{"enableVectorBroadcastOptimisations",
                      (enableOptimisations ? "true" : "false")}};

  // Make a program sequence to run the operation
  Sequence prog;
  Tensor out;
  if (inPlace) {
    mapInPlace(graph, binOp, tensors, prog, "", opOpts);
    out = in1;
  } else {
    out = map(graph, binOp, tensors, prog, "", opOpts);
  }

  // Create host 'transfer' buffers with the right size for the device type
  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::unique_ptr<char[]> in1HostRaw;
  std::unique_ptr<char[]> in2HostRaw;
  std::unique_ptr<char[]> outHostRaw;
  char *outHostRawPtr = nullptr;
  if (!in1IsConst)
    in1HostRaw = allocateHostMemoryForTensor(in1, "in1", graph, uploadProg,
                                             downloadProg, tmap);
  if (!in2IsConst)
    in2HostRaw = allocateHostMemoryForTensor(in2, "in2", graph, uploadProg,
                                             downloadProg, tmap);
  if (!inPlace) {
    outHostRaw = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                             downloadProg, tmap);
    outHostRawPtr = outHostRaw.get();
  } else {
    outHostRawPtr = in1HostRaw.get();
  }

  // Copy and convert the data from the initialised buffers to the transfer
  // buffers (still on host)
  auto copyBuffer = [&](std::vector<HostDataType> &buf,
                        std::unique_ptr<char[]> &rawBuf) {
    copy(target, buf.data(), buf.size(), dataType, rawBuf.get());
    // For HALF, we copy and convert back into the (float) host buffers so
    // that the host buffers contain the exact HALF values (which are exactly
    // representable in float). This helps with the validation for the
    // comparison operators
    if (dataType == HALF)
      copy(target, dataType, rawBuf.get(), buf.data(), buf.size());
  };
  if (!in1IsConst)
    copyBuffer(in1Host, in1HostRaw);
  if (!in2IsConst)
    copyBuffer(in2Host, in2HostRaw);

  if (doPrintTensors) {
    if (!in1IsConst)
      prog.add(PrintTensor("in1", in1));
    if (!in2IsConst)
      prog.add(PrintTensor("in2", in2));
    if (!inPlace)
      prog.add(PrintTensor("out", out));
  }

  // Run sequences
  std::optional<TempDir> tempDir;
  poplar::OptionFlags engineOptions;
  if (doReport) {
    tempDir.emplace(TempDir::create());
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", tempDir->getPath());
  }
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, engineOptions);
  attachStreams(engine, tmap);

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();

    if (doReport) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");
      engine.printProfileSummary(std::cout, opt);
    }
  });

  // Check the result
  if (ignoreData) {
    std::cout << "Result not checked for correctness\n";
  } else {
    // Get the result out of the device
    std::vector<HostOutType> outHost(nElemsOut);
    copy(target, outputType, outHostRawPtr, outHost.data(), outHost.size());
    return verifyResult<HostDataType, HostOutType>(
        isIpuModel(deviceType), dataType, in1Host, desc1.shapeExt, in2Host,
        desc2.shapeExt, outHost, shapeOut, operation);
  }
  return true;
}

//*************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type dataType;

  std::string operation;
  bool doReport = false;
  bool doPrintTensors = false;
  bool inPlace = false;
  unsigned tiles = 0;
  bool ignoreData = false;
  unsigned randomSeed = 1; // we use '0' to mean 'not random'
  bool enableOptimisations = true;
  ShapeOption<size_t> shape1;
  ShapeOption<size_t> shape2;
  OperandDescriptor desc1, desc2;

  // clang-format off
  const static std::string description =
  "Perform a binary operation between two tensors having any specified shape,\n"
  "each mapped in any desired way among tiles.\n"
  "\n"
  "Options are:";
  po::options_description poDesc(description);

  poDesc.add_options()
    ("help", "Print help")
    ("report",
     po::value<bool>(&doReport)->implicit_value(true),
     "Provide a poplar report")
    ("options-file",
     po::value<std::string>(),
     "A file containing options, with the same syntax as the command line; "
     "can be also specified with '@options_file_name'")
    ("print",
     po::value<bool>(&doPrintTensors)->implicit_value(true),
     "Print the tensors")
    ("random-seed",
     po::value<unsigned>(&randomSeed)->implicit_value(randomSeed),
     "Seed for random data. Value of 0 means 'no random data'")
    ("ignore-data",
     po::value<bool>(&ignoreData)->implicit_value(true),
     "Do not check correctness of result, useful for benchmarking without "
     "overhead of host-side computation")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(DeviceType::Sim2),
     "Device Type")
    ("data-type",
     po::value<Type>(&dataType)->default_value(HALF),
     "Data Type: half, float, int, unsigned, bool, ulonglong, longlong")
    ("in-place",
     po::value<bool>(&inPlace)->implicit_value(true),
     "Do the specified operation in place")
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use for linearly mapping the operands. If "
     "unspecified, or 0, do not map lineraly the operands (use only the "
     "explicit mapping specified by --map1, --map2)")
    ("shape1",
     po::value<ShapeOption<size_t>>(&shape1)->multitoken()->required(),
     "Shape for first operand, curly bracket delimited:  {d1,d2,...}.")
    ("map1",
     po::value<std::vector<MappingDesc>>(&desc1.map)->multitoken(),
     "Tile mapping for first operand; a sequence of one or more: "
     "T:{d1,d2,...} ... , where T is the tile number and {d1,d2,...} is the "
     "slice mapped on T. If not specified, the operand is mapped linearly on "
     "the allocated tiles.")
    ("shape2",
     po::value<ShapeOption<size_t>>(&shape2)->multitoken()->required(),
     "Shape for second operand; see --shape1")
    ("map2",
     po::value<std::vector<MappingDesc>>(&desc2.map)->multitoken(),
     "Tile mapping for second operand; see --map1.")
    ("operation",
     po::value<std::string>(&operation)->required(),
     ("Operation to perform, one of: " + allOpsStr()).c_str())
    ("enable-optimisations",
     po::value<bool>(&enableOptimisations)->default_value(enableOptimisations),
     "Enable broadcast operation optimisations")
    ;
  // clang-format on
  parseOptions(argc, argv, poDesc);

  expr::BinaryOpType opType = stringToBinaryOp(operation);

  // Find the shape of the output, applying the broadcasting rules
  // First, get the 'extended to the left' operand shapes; for instance, if
  // the two operands have shapes {9,8,7,6} and {7,1}, the second is
  // 'extended' to {1,1,7,1}
  desc1.shape = shape1.val;
  desc2.shape = shape2.val;
  unsigned n1 = desc1.shape.size();
  unsigned n2 = desc2.shape.size();
  unsigned n = std::max(n1, n2);
  desc1.shapeExt = extendShape(desc1.shape, n);
  desc2.shapeExt = extendShape(desc2.shape, n);

  std::vector<size_t> shapeOut(n);
  for (unsigned i = 0; i < n; i++) {
    size_t d1 = desc1.shapeExt[i];
    size_t d2 = desc2.shapeExt[i];

    // If the dimensions are different, one of them must be '1'
    if ((d1 != d2) && (d1 != 1) && (d2 != 1)) {
      std::cerr << "Error: shapes incompatible for broadcasting\n";
      return 1;
    }
    shapeOut[i] = std::max(d1, d2);
  }
  if (inPlace && (shapeOut != shape1.val)) {
    std::cerr << "Error: cannot specify '--in-place:true' if shape of output "
                 "is not the same as shape of first operand\n";
    return 1;
  }

  bool mapLinearly = tiles > 0;

  if (tiles == 0) {
    // Find the highest tile number in the tile mapping for the two operands
    for (auto m : desc1.map) {
      tiles = std::max(tiles, m.tile);
    }
    for (auto m : desc2.map) {
      tiles = std::max(tiles, m.tile);
    }
    tiles++;
  }

  Type outputType = isBoolOp(opType) ? BOOL : dataType;

#define SELECT_ONE(DATA_TYPE, OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE)         \
  if (dataType == DATA_TYPE && outputType == OUT_TYPE) {                       \
    return doBinaryOpTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(                      \
               deviceType, dataType, outputType, desc1, desc2, shapeOut,       \
               tiles, mapLinearly, opType, inPlace, doReport, doPrintTensors,  \
               randomSeed, ignoreData, enableOptimisations)                    \
               ? 0                                                             \
               : 1;                                                            \
  } // nonzero value = error

  SELECT_ONE(BOOL, BOOL, unsigned char, unsigned char)
  SELECT_ONE(SHORT, BOOL, short, unsigned char)
  SELECT_ONE(SHORT, SHORT, short, short)
  SELECT_ONE(UNSIGNED_SHORT, BOOL, unsigned short, unsigned char)
  SELECT_ONE(UNSIGNED_SHORT, UNSIGNED_SHORT, unsigned short, unsigned short)
  SELECT_ONE(HALF, BOOL, float, unsigned char)
  SELECT_ONE(HALF, HALF, float, float)
  SELECT_ONE(HALF, FLOAT, float, float)
  SELECT_ONE(FLOAT, BOOL, float, unsigned char)
  SELECT_ONE(FLOAT, HALF, float, float)
  SELECT_ONE(FLOAT, FLOAT, float, float)
  SELECT_ONE(INT, BOOL, int, unsigned char)
  SELECT_ONE(INT, INT, int, int)
  SELECT_ONE(UNSIGNED_INT, BOOL, unsigned, unsigned char)
  SELECT_ONE(UNSIGNED_INT, UNSIGNED_INT, unsigned, unsigned)

  SELECT_ONE(LONGLONG, BOOL, long long, unsigned char)
  SELECT_ONE(LONGLONG, LONGLONG, long long, long long)
  SELECT_ONE(UNSIGNED_LONGLONG, BOOL, unsigned long long, unsigned char)
  SELECT_ONE(UNSIGNED_LONGLONG, UNSIGNED_LONGLONG, unsigned long long,
             unsigned long long)

  // Reaching here means the combination of 'dataType' and 'outputType' was
  // invalid.
  throw invalid_types(dataType, outputType);
}
