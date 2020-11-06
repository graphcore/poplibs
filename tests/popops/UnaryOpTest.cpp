// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs a unary operation between on a tensor with any desired shape,
// mapped in any desired way among tiles.

#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include <boost/format.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include "codelets/UnaryCodeletsTest.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <type_traits>

using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;

const poplar::OptionFlags options{{"debug.instrumentCompute", "true"}};

// A descriptor to keep information about which tile to store a slice of
// a tensor on
struct MappingDesc {
  bool isConst = false;
  unsigned tile;
  std::vector<size_t> slice;
};

// This extends to rank 'n' a given tensor shape.
// Returns a shape having rank 'n', obtained by prepending '1's at the left
// ('n' must be >= shape.size()).
// I.e. if shape is {6,1} and 'n' is 4, it returns {1,1,6,1}.
static std::vector<size_t> extendShape(const std::vector<size_t> &shape,
                                       unsigned n) {
  unsigned m = shape.size();
  assert(n >= m);
  std::vector<size_t> shapeExt(n, 1);
  for (unsigned k = 0; k < m; k++) {
    shapeExt[n - m + k] = shape[k];
  }
  return shapeExt;
}

// Given a linear array 'data' (one of the host buffers) which represent a
// tensor with specified shape, get the element with indices specified by
// 'i[]', using broadcasting rules.
// Basically this returns:  data[ i[0], i[1], ... ].
template <typename T>
T get(const T data[], const std::vector<size_t> shape,
      const std::vector<unsigned> i) {
  unsigned offs = 0;
  for (unsigned k = 0; k < i.size(); k++) {
    // Need to keep into account broadcasting rules: if a certain
    // dimension is 1, then the corresponding index does not matter (i.e.
    // the effective index to use is 0)
    offs = offs * shape[k] + ((shape[k] == 1) ? 0 : i[k]);
  }
  return data[offs];
}

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
template <typename HOST_DATA_TYPE, typename HOST_OUT_TYPE>
static bool verifyResult(const DeviceType &deviceType, const Type &dataType,
                         const std::vector<HOST_DATA_TYPE> &inHost,
                         const std::vector<size_t> &shapeExt,
                         const std::vector<HOST_OUT_TYPE> &outHost,
                         const UnaryOpType op) {
  unsigned errCount = 0; // how many mismatched elements we find

  unsigned n = shapeExt.size(); // How many dimensions we have

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
    for (i[k] = 0; i[k] < shapeExt[k]; i[k]++) {
      if (k == n - 1) {
        // This is the "innermost loop"; we need to compute:
        // expected[ i[0], i[1],... ] =
        //                in1[ i[0], i[1],... ] *OP*  in2[ i[0], i[1],... ]
        // and compare with the actual value from the device

        HOST_OUT_TYPE actual = get(outHost.data(), shapeExt, i); // from device

        HOST_DATA_TYPE val = get(inHost.data(), shapeExt, i);

        HOST_OUT_TYPE expected = 0;

        performOp(op, val, expected);

        if (!equalValues(isIpuModel(deviceType), op, dataType, expected,
                         actual)) {
          std::cerr << "out[" << i[0];
          for (unsigned j = 1; j < n; j++)
            std::cerr << "," << i[j];
          std::cerr << "] = " << unaryOpToString.at(op) << " "
                    << convertToString(val)
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

// This collects together information about the operand
struct OperandDescriptor {
  std::vector<size_t> shape;    // Shape, as defined on command line.
  std::vector<size_t> shapeExt; // Shape, rank-extended.
  std::vector<MappingDesc> map; // Indicates where to map this operand
};

//*************************************************************************
/// Do a unary operation, where the operand is described by 'desc1'.
///
/// \param deviceType            The device used.
/// \param dataType              The data type used for the two operands.
/// \param outputType            The type for the result of the operation.
/// \param desc                  Description for first operand.
/// \param tiles                 How many tiles to allocate.
/// \param mapLinearly           If yes, we map linearly the two operands on
///                              all tiles.
/// \param operation             Operation performed on device.
/// \param inPlace               Is the operation to be done in place?
/// \param doReport              Print poplar report.
/// \param doPrintTensors        Print the tensor (for verification).
/// \param ignoreData            Do not verify results.
/// \param enableOptimisations   Enable broadcasted vector op optimisations.
template <typename HOST_DATA_TYPE, typename HOST_OUT_TYPE>
static bool doUnaryOpTest(const DeviceType &deviceType, const Type &dataType,
                          const Type &outputType, const OperandDescriptor &desc,
                          const unsigned tiles, const bool mapLinearly,
                          const UnaryOpType operation, const bool inPlace,
                          const bool doReport, const bool doPrintTensors,
                          const unsigned randomSeed, const bool ignoreData,
                          const bool enableOptimisations) {

  auto nElems = std::accumulate(desc.shape.begin(), desc.shape.end(),
                                std::size_t(1), std::multiplies<std::size_t>());

  if (inPlace && (outputType != dataType)) {
    throw std::runtime_error("For in place operations, the data and output "
                             "types must be the same (specified data type=" +
                             dataType.toString() + ", specified output type=" +
                             outputType.toString() + ")");
  }

  if (isIntOp(operation) && (dataType == HALF || dataType == FLOAT)) {
    throw std::runtime_error(unaryOpToString.at(operation) +
                             " requires data "
                             "of integer type (specified  data type=" +
                             dataType.toString() + ")");
  }

  // Allocate and initialise host buffers with appropriate values.
  std::vector<HOST_DATA_TYPE> inHost(nElems);
  fillHostBuffer(operation, dataType, randomSeed, inHost);

  // Create Graph object, target and device
  auto device = createTestDevice(deviceType, 1, tiles);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // Map operands on tiles. First it is mapped linearly on all tiles,
  // then the mappings specified by --mapX are applied.
  // Note that each mapping can/will override the previous. This makes easier
  // to obtain arbitrary mappings.
  auto mapTensor = [&](const Tensor &t,
                       const std::vector<MappingDesc> &mapping) {
    if (mapLinearly || (mapping.size() == 0)) {
      mapTensorLinearly(graph, t);
    }
    for (auto m : mapping) {
      if (m.slice.size() == 0) {
        graph.setTileMapping(t, m.tile);
      } else {
        std::vector<size_t> ends;
        for (auto i : m.slice) {
          ends.push_back(i + 1);
        }
        graph.setTileMapping(t.slice(m.slice, ends), m.tile);
      }
    }
  };

  std::vector<poplar::Tensor> tensors;
  Tensor in = graph.addVariable(dataType, desc.shape, "in");
  mapTensor(in, desc.map);
  expr::UnaryOp unaryOp = expr::UnaryOp(operation, expr::_1);

  OptionFlags opOpts{{"enableVectorBroadcastOptimisations",
                      (enableOptimisations ? "true" : "false")}};

  // Make a program sequence to run the operation
  Sequence prog;
  Tensor out;
  if (inPlace) {
    mapInPlace(graph, unaryOp, {in}, prog, "", opOpts);
    out = in;
  } else {
    out = map(graph, unaryOp, {in}, prog, "", opOpts);
  }

  // Create host 'transfer' buffers with the right size for the device type
  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::unique_ptr<char[]> inHostRaw;
  std::unique_ptr<char[]> outHostRaw;
  char *outHostRawPtr = nullptr;
  inHostRaw = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                          downloadProg, tmap);
  if (!inPlace) {
    outHostRaw = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                             downloadProg, tmap);
    outHostRawPtr = outHostRaw.get();
  } else {
    outHostRawPtr = inHostRaw.get();
  }

  // Copy and convert the data from the initialised buffers to the transfer
  // buffers (still on host)
  auto copyBuffer = [&](std::vector<HOST_DATA_TYPE> &buf,
                        std::unique_ptr<char[]> &rawBuf) {
    copy(target, buf.data(), buf.size(), dataType, rawBuf.get());
    // For HALF, we copy and convert back into the (float) host buffers so that
    // the host buffers contain the exact HALF values (which are exactly
    // representable in float). This helps with the validation for the
    // comparison operators
    if (dataType == HALF)
      copy(target, dataType, rawBuf.get(), buf.data(), buf.size());
  };
  copyBuffer(inHost, inHostRaw);

  if (doPrintTensors) {
    prog.add(PrintTensor("in", in));
    if (!inPlace)
      prog.add(PrintTensor("out", out));
  }

  // Run sequences
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
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
    std::vector<HOST_OUT_TYPE> outHost(nElems);
    copy(target, outputType, outHostRawPtr, outHost.data(), outHost.size());
    return verifyResult<HOST_DATA_TYPE, HOST_OUT_TYPE>(
        deviceType, dataType, inHost, desc.shapeExt, outHost, operation);
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
  ShapeOption<size_t> shape;
  OperandDescriptor opDesc;

  // clang-format off
  const static std::string description =
  "Performs a unary operation between two tensors having any specified \n"
  "shape, each mapped in any desired way among tiles.\n"
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
     "Data Type: half, float, int, unsigned, bool")
    ("in-place",
     po::value<bool>(&inPlace)->implicit_value(true),
     "Do the specified operation in place")
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use for linearly mapping the operands. If "
     "unspecified, or 0, do not map lineraly the operands (use only the "
     "explicit mapping specified by --map1, --map2)")
    ("shape",
     po::value<ShapeOption<size_t>>(&shape)->multitoken()->required(),
     "Shape for the operand, curly bracket delimited:  {d1,d2,...}.")
    ("map",
     po::value<std::vector<MappingDesc>>(&opDesc.map)->multitoken(),
     "Tile mapping for the operand; a sequence of one or more: "
     "T:{d1,d2,...} ... , where T is the tile number and {d1,d2,...} is the "
     "slice mapped on T. If not specified, the operand is mapped linearly on "
     "the allocated tiles.")
    ("operation",
     po::value<std::string>(&operation)->required(),
     ("Operation to perform, one of: " + allOpsStr()).c_str())
    ("enable-optimisations",
     po::value<bool>(&enableOptimisations)->default_value(enableOptimisations),
     "Enable broadcast operation optimisations")
    ;
  // clang-format on
  parseOptions(argc, argv, poDesc);

  expr::UnaryOpType opType = stringToUnaryOp(operation);

  // Find the shape of the output, applying the broadcasting rules
  // First, get the 'extended to the left' operand shapes; for instance, if
  // the two operands have shapes {9,8,7,6} and {7,1}, the second is
  // 'extended' to {1,1,7,1}
  opDesc.shape = shape.val;
  unsigned n = opDesc.shape.size();
  opDesc.shapeExt = extendShape(opDesc.shape, n);

  bool mapLinearly = tiles > 0;

  if (tiles == 0) {
    // Find the highest tile number in the tile mapping for the two operands
    for (auto m : opDesc.map) {
      tiles = std::max(tiles, m.tile);
    }
    tiles++;
  }

  Type outputType = isBoolOp(opType) ? BOOL : dataType;

#define DO_TEST(DATA_TYPE, OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE)            \
  if (dataType == DATA_TYPE && outputType == OUT_TYPE) {                       \
    return doUnaryOpTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(                       \
               deviceType, dataType, outputType, opDesc, tiles, mapLinearly,   \
               opType, inPlace, doReport, doPrintTensors, randomSeed,          \
               ignoreData, enableOptimisations)                                \
               ? 0                                                             \
               : 1;                                                            \
  } // nonzero value = error

  // Note that for HALF and FLOAT the host buffers are 'float'
  DO_TEST(BOOL, BOOL, HostBool, HostBool)

  DO_TEST(HALF, HALF, float, float)
  DO_TEST(HALF, BOOL, float, HostBool)

  DO_TEST(FLOAT, FLOAT, float, float)
  DO_TEST(FLOAT, BOOL, float, HostBool)

  DO_TEST(INT, INT, int, int)
  DO_TEST(INT, BOOL, int, HostBool)

  DO_TEST(UNSIGNED_INT, UNSIGNED_INT, unsigned, unsigned)
  DO_TEST(UNSIGNED_INT, BOOL, unsigned, HostBool)

  // Reaching here means the combination of 'dataType' and 'outputType' was
  // invalid.
  std::cerr << "Combination of data type and operator not supported\n";
  return 1;
}

// Utility function to read a MappingDesc from a stream
std::istream &operator>>(std::istream &in, MappingDesc &md) {
  char c = in.peek();
  if (c == 'c') {
    in >> c; // flush the peeked char
    md.isConst = true;
  } else {
    in >> md.tile;
    in >> c;
    if (c != ':') {
      throw std::runtime_error("Invalid shape; expected ':'after tile number'");
    }
    ShapeOption<size_t> slice;
    in >> slice;
    md.slice = slice.val;
  }
  return in;
}

// Utility function to write a MappingDesc to a stream
std::ostream &operator<<(std::ostream &os, const MappingDesc &md) {
  if (md.isConst) {
    return os << "const";
  } else {
    os << md.tile << ":{";
    for (auto x : md.slice)
      os << x << ",";
    return os << "}";
  }
}
