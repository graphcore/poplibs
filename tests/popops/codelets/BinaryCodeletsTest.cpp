// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

//
// Tests one or more of the element-wise binary/broadcast codelets:
//
//     BinaryOp1D[InPlace]Supervisor
//     BinaryOp2D[InPlace]
//
//     BroadcastScalar1D[InPlace]Supervisor
//     BroadcastScalar2Types1DSupervisor
//     BroadcastScalar2DData[InPlace]
//     BroadcastScalar2Types2DData
//     BroadcastScalar2D[InPlace]
//
//     BroadcastVectorInner[InPlace]Supervisor
//     BroadcastVectorInner2D[InPlace]
//
//     BroadcastVectorOuterByRow[InPlace]Supervisor
//     BroadcastVectorOuterByColumn[InPlace]Supervisor
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
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <optional>
#include <regex>
#include <type_traits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;
using namespace poplibs_support;
using boost::format;
using popops::expr::BinaryOpType;
using std::to_string;

// The name of the compute set where we run the vertex under test.
const static std::string computeSetName = "vertexComputeSet";

const std::vector<std::string> verticesNames = {
    "BinaryOp1DSupervisor",
    "BinaryOp1DInPlaceSupervisor",
    "BinaryOp2D",
    "BinaryOp2DInPlace",

    "BroadcastScalar1DSupervisor",
    "BroadcastScalar1DInPlaceSupervisor",
    "BroadcastScalar2Types1DSupervisor",
    "BroadcastScalar2DData",
    "BroadcastScalar2DDataInPlace",
    "BroadcastScalar2Types2DData",
    "BroadcastScalar2D",
    "BroadcastScalar2DInPlace",

    "BroadcastVectorInnerSupervisor",
    "BroadcastVectorInnerInPlaceSupervisor",
    "BroadcastVectorInner2D",
    "BroadcastVectorInner2DInPlace",

    "BroadcastVectorOuterByRowSupervisor",
    "BroadcastVectorOuterByRowInPlaceSupervisor",
    "BroadcastVectorOuterByColumnSupervisor",
    "BroadcastVectorOuterByColumnInPlaceSupervisor",
};

// Returns a string with all vertices names, comma separated. Used for the help
// message
const std::string allVerticesStr() {
  std::string result;
  for (auto &n : verticesNames) {
    if (!result.empty()) // add a comma between names
      result += ", ";
    result += n;
  }
  return result;
}

//*************************************************************************
// This contains the vertex names and various flags that characterise the
// vertex, used in different places during the test.
struct VertexDesc {
  std::string name;

  std::string vClass;      // full name with template params foer addVertex()
  std::string vClassShort; // vClass, abbreviated for display

  // Names of the two operand fields in the vertex
  std::string in1Name;
  std::string in2Name;

  bool isBinaryOp;
  bool is2D;
  bool inPlace;
  bool isVectorInner;
  bool isVectorInner2D;
  bool isVectorOuter;
  bool is2Types; // is the vertex a special one with different types?
  bool isBroadcastScalar;
  bool isBroadcastScalar2D;
  bool op2isSingleElem; // must the second operand have a single element?

  bool allowMisaligned = true; // For VectorOuter vertices

  VertexDesc(const std::string &vertexName) {
    name = vertexName;

    // Extract the flags by looking at the name
    isBinaryOp = vertexName.find("BinaryOp") != std::string::npos;
    is2D = vertexName.find("2D") != std::string::npos;
    inPlace = vertexName.find("InPlace") != std::string::npos;
    isVectorInner = vertexName.find("VectorInner") != std::string::npos;
    isVectorInner2D = isVectorInner && is2D;
    isVectorOuter = vertexName.find("VectorOuter") != std::string::npos;
    is2Types = vertexName.find("2Types") != std::string::npos;
    isBroadcastScalar = vertexName.find("BroadcastScalar") != std::string::npos;
    isBroadcastScalar2D = (vertexName == "BroadcastScalar2D" ||
                           vertexName == "BroadcastScalar2DInPlace");
    op2isSingleElem = isBroadcastScalar && !isBroadcastScalar2D;

    if (isBinaryOp) {
      in1Name = inPlace ? "in1Out" : "in1";
      in2Name = "in2";
    } else {
      in1Name = "data";
      in2Name = "B";
    }
  }

  void setVertexClass(const BinaryOpType op, const Type &dataType,
                      const Type &outputType) {
    // Get the vertex class
    std::string vName = "popops::" + name;
    if (isVectorOuter) {
      vClass = templateVertex(vName, op, dataType, allowMisaligned);
    } else if (is2Types) {
      vClass = templateVertex(vName, op, dataType, outputType);
    } else {
      vClass = templateVertex(vName, op, dataType);
    }

    // Shorten the name for display, removing namespaces
    vClassShort = vClass;
    boost::erase_all(vClassShort, "popops::");
    boost::erase_all(vClassShort, "expr::BinaryOpType::");
  }
};

// Maps specifying valid (operation, dataType) pairs for the generic binary
// and broadcast vertices.
const static std::map<expr::BinaryOpType, const std::set<Type>>
    binaryBroadcastCombinations = {
        {BinaryOpType::ADD, {FLOAT, HALF, INT, UNSIGNED_INT}},
        {BinaryOpType::ATAN2, {FLOAT, HALF}},
        {BinaryOpType::BITWISE_AND, {INT, UNSIGNED_INT, SHORT, UNSIGNED_SHORT}},
        {BinaryOpType::BITWISE_OR, {INT, UNSIGNED_INT, SHORT, UNSIGNED_SHORT}},
        {BinaryOpType::BITWISE_XOR, {INT, UNSIGNED_INT, SHORT, UNSIGNED_SHORT}},
        {BinaryOpType::BITWISE_XNOR,
         {INT, UNSIGNED_INT, SHORT, UNSIGNED_SHORT}},
        {BinaryOpType::DIVIDE, {FLOAT, HALF, INT, UNSIGNED_INT}},
        {BinaryOpType::LOGICAL_AND, {BOOL}},
        {BinaryOpType::LOGICAL_OR, {BOOL}},
        {BinaryOpType::MAXIMUM, {FLOAT, HALF, INT, UNSIGNED_INT}},
        {BinaryOpType::MINIMUM, {FLOAT, HALF, INT, UNSIGNED_INT}},
        {BinaryOpType::MULTIPLY, {FLOAT, HALF, INT, UNSIGNED_INT}},
        {BinaryOpType::POWER, {FLOAT, HALF}},
        {BinaryOpType::REMAINDER, {FLOAT, HALF, INT, UNSIGNED_INT}},
        {BinaryOpType::SHIFT_LEFT, {INT, UNSIGNED_INT}},
        {BinaryOpType::SHIFT_RIGHT, {INT, UNSIGNED_INT}},
        {BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, {INT}},
        {BinaryOpType::SUBTRACT, {FLOAT, HALF, INT, UNSIGNED_INT}},
        {BinaryOpType::EQUAL,
         {FLOAT, HALF, INT, UNSIGNED_INT, BOOL, SHORT, UNSIGNED_SHORT}},
        {BinaryOpType::GREATER_THAN, {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::GREATER_THAN_EQUAL,
         {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::LESS_THAN, {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::LESS_THAN_EQUAL, {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::NOT_EQUAL,
         {FLOAT, HALF, INT, UNSIGNED_INT, BOOL, SHORT, UNSIGNED_SHORT}},
};

// Return true if the combination of vertex, operation and data type is valid
bool isValidCombination(const VertexDesc &vertex, const BinaryOpType op,
                        const Type &type) {

  // The "2Types" vertices and the STD_DEV <=> VARIANCE operators are special
  if (vertex.is2Types && (op != BinaryOpType::INV_STD_DEV_TO_VARIANCE) &&
      (op != BinaryOpType::VARIANCE_TO_INV_STD_DEV))
    return false;
  if (op == BinaryOpType::INV_STD_DEV_TO_VARIANCE) {
    if (vertex.is2Types) {
      return type == HALF;
    } else {
      return vertex.isBroadcastScalar && (type == HALF || type == FLOAT);
    }
  } else if (op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) {
    if (vertex.is2Types) {
      return type == FLOAT;
    } else {
      return vertex.isBroadcastScalar && (type == HALF || type == FLOAT);
    }
  }

  // VectorInner and Outer vertices have a restricted set of operation/types
  if (vertex.isVectorInner) {
    return ((op == BinaryOpType::ADD || op == BinaryOpType::DIVIDE ||
             op == BinaryOpType::MULTIPLY || op == BinaryOpType::SUBTRACT) &&
            (type == HALF || type == FLOAT));
  } else if (vertex.isVectorOuter) {
    return ((op == BinaryOpType::ADD || op == BinaryOpType::SUBTRACT ||
             op == BinaryOpType::MULTIPLY) &&
            (type == HALF || type == FLOAT));
  }

  if (isBoolOp(op) && vertex.inPlace) {
    return type == BOOL;
  }
  // The combinations for the BinaryOp and BroadcastScalar are specified by
  // binaryBroadcastCombinations
  return binaryBroadcastCombinations.at(op).count(type) == 1;
}

//*************************************************************************
// Describes the sizes of two operands (and the output).
// The fields that are used for different vertex types.
struct TensorSizes {
  // For 2D vertices: size of each row
  // For 1D vertices: the first element is size of whole 1st operand
  std::vector<unsigned> rowSizes;

  // For BroadcastVectorInner2D vertices: size of each row for 2nd operand
  // Each element must be an exact divisor of corresponding row of rowSizes
  std::vector<unsigned> op2RowSizes;

  // Only for "VectorOuter" vertices
  unsigned rows, columns;

  // Total number of elements in first operand ("in1") and output. For 2D
  // vertices is the sum of all 'rowSizes'. For 1D vertices is the same as
  // 'rowSizes[0]'
  // For "VectorOuter" vertices is rows x columns
  unsigned nElems1;

  // Total number of elements in second operand ("in2")
  unsigned nElems2;

  // A string that describes the sizes of the operands
  std::string operandStr;

  TensorSizes()
      : rowSizes({25, 12, 21}), op2RowSizes({5, 3, 7}), rows(10), columns(23),
        nElems1(0), nElems2(0){};

  // Given a vertex, adjust the field as required by that vertex
  void adjustForVertex(const VertexDesc &vertex) {
    std::string op1Str, op2Str;

    // Convert a vector to a string
    auto vector2str = [](std::vector<unsigned> v) {
      std::string s;
      for (auto e : v)
        s += "[" + to_string(e) + "] ";
      s.erase(s.size() - 1); // remove last space
      return s;
    };

    // Adjust size for first operand
    if (vertex.isVectorOuter) {
      if (rows == 0 || columns == 0) {
        throw std::runtime_error(vertex.name +
                                 " must have nonzero rows and columns");
      }
      nElems1 = rows * columns;
    } else {
      if (rowSizes.size() == 0) {
        throw std::runtime_error(vertex.name +
                                 " must have at least one row size specified");
      }
      if (vertex.is2D) {
        nElems1 = std::accumulate(rowSizes.begin(), rowSizes.end(), 0);
        rows = rowSizes.size();
        op1Str = vector2str(rowSizes);
      } else {
        nElems1 = rowSizes[0];
      }
    }

    // Adjust size for second operand
    if (vertex.isBinaryOp) {
      nElems2 = nElems1;
      if (vertex.is2D) {
        op2RowSizes = rowSizes;
        op2Str = op1Str;
      }
    } else {
      if (vertex.op2isSingleElem) {
        nElems2 = 1;
        op2Str = to_string(nElems2);
      } else if (vertex.isBroadcastScalar2D) {
        nElems2 = rows;
        op2Str = to_string(nElems2);
      } else if (vertex.isVectorInner) {
        if (vertex.is2D) {
          // if too many rows for second operand, just trim it
          if (op2RowSizes.size() > rows) {
            op2RowSizes = std::vector<unsigned>(op2RowSizes.begin(),
                                                op2RowSizes.begin() + rows);
          }
          op2Str = vector2str(op2RowSizes);
          nElems2 = std::accumulate(op2RowSizes.begin(), op2RowSizes.end(), 0);
          for (unsigned i = 0; i < rows; i++) {
            if ((rowSizes[i] % op2RowSizes[i]) != 0) {
              throw std::runtime_error(
                  (format("%s's second operand sizes for row %u is %u but"
                          " it must be an exact divisor of the size of "
                          "the corresponding row of first operand (%u)") %
                   vertex.name % i % op2RowSizes[i] % rowSizes[i])
                      .str());
            }
          }
        } else {
          nElems2 = op2RowSizes[0] > 0 ? op2RowSizes[0] : 1;
          if ((nElems1 % nElems2) != 0) {
            throw std::runtime_error(
                (format("%s's second operand size is %u but it must be an"
                        " exact divisor of the first operand size (%u)") %
                 vertex.name % nElems2 % nElems1)
                    .str());
          }
        }
      } else if (vertex.isVectorOuter) {
        if (op2RowSizes.size() != 0) {
          nElems2 = op2RowSizes[0];
        }
        if (nElems2 == 0 || (rows % nElems2) != 0) {
          throw std::runtime_error(
              (format("%s's second operand size is %u but it must be an"
                      " exact divisor of of the number of rows of the first "
                      "operand (%u)") %
               vertex.name % nElems2 % nElems1)
                  .str());
        }
      }
    }

    if (op1Str.empty())
      op1Str = to_string(nElems1);
    if (op2Str.empty())
      op2Str = to_string(nElems2);
    operandStr = vertex.in1Name + ": [" + op1Str + "];  " + vertex.in2Name +
                 ": [" + op2Str + "]";
  }
};

//*************************************************************************
/// Verifies if the results of the operation performed on the device match
/// with the one the host
///
/// \param isIpuModel          Is IpuModel or IpuModel2?.
/// \param dataType            The data type used (float, half).
/// \param dataType            The data type used (float, half).
/// \param in1Host             Data buffer for the first operand.
/// \param in2Host             Data for second operand.
/// \param outHost             Data for result, obtained from
///                            device and converted to host types.
/// \param operation           Operation performed on device.
template <typename HOST_DATA_TYPE, typename HOST_OUT_TYPE>
static bool verifyResult(const bool isIpuModel, const VertexDesc &vertex,
                         const BinaryOpType op, const Type &dataType,
                         const std::vector<HOST_DATA_TYPE> &in1Host,
                         const std::vector<HOST_DATA_TYPE> &in2Host,
                         const std::vector<HOST_OUT_TYPE> &outHost,
                         const TensorSizes sizes) {
  unsigned errCount = 0; // how many mismatched elements we find

  // For 2D vertices, running partial sums of elements for the rows we have
  // already done, needed for the broadcasting. For 1st and 2nd operand
  unsigned rowOffs = 0;
  unsigned rowOffs2 = 0;

  unsigned row = 0; // For 2D vertices, index row of first operand

  // Loop sequentially over the 1st operand linear data buffer, selecting the
  // correct element from the second, according to the broadcasting rule of the
  // vertex.
  unsigned i = 0; // linear index into 1st operand buffer
  unsigned j = 0; // linear index into 2nd operand buffer
  do {
    HOST_DATA_TYPE val1 = in1Host[i]; // operands
    HOST_DATA_TYPE val2 = in2Host[j];

    HOST_OUT_TYPE actual = outHost[i]; // result from device

    HOST_OUT_TYPE expected = 0; // result for verification
    performOp(op, val1, val2, expected);

    if (!equalValues(isIpuModel, op, dataType, expected, actual)) {
      std::cerr << format("out[%u] = %s %s %s  =>  expected:%s;  actual:%s\n") %
                       i % convertToString(val1) % binaryOpToString.at(op) %
                       convertToString(val2) % convertToString(expected) %
                       convertToString(actual);
      errCount++;
    }

    // Update index into the two operands according to the broadcasting rules
    // of this vertex and sizes combination

    // for first operand/output, always increment
    i++;

    // Update row index when we change row
    if (i == rowOffs + sizes.rowSizes[row]) {
      rowOffs += sizes.rowSizes[row];
      if (vertex.isVectorInner2D) {
        rowOffs2 += sizes.op2RowSizes[row];
      }
      row++;
    }

    // For second operand, need to check which broadcast pattern we are using
    if (vertex.isBinaryOp) {
      j++;
    } else if (vertex.isBroadcastScalar2D) {
      j = row;
    } else if (vertex.isVectorInner2D) {
      if (row < sizes.rows) // Need to avoid division by zero on last iteration
        j = rowOffs2 + ((i - rowOffs) % sizes.op2RowSizes[row]);
    } else if (vertex.isVectorInner) {
      j = i % sizes.nElems2;
    } else if (vertex.isVectorOuter) {
      j = (i / sizes.columns) % sizes.nElems2;
    }
  } while (i < outHost.size());

  if (errCount > 0) {
    std::cerr << "Failed: mismatch on " << errCount << " value(s)\n";
  }
  return errCount == 0;
}

//*************************************************************************
/// Run one vertex test.
///
/// \tparam HOST_DATA_TYPE   Type to use on the host for dataType1, datatType2.
/// \tparam HOST_OUT_TYPE    Type to use on the host for outputType.
/// \param  deviceType       The device used.
/// \param  vertex           Which vertex.
/// \param  op               Operation to perform
/// \param  dataType         The data type used for operands.
/// \param  outputType       The type for the result of the operation.
/// \param  sizes            Describes the sizes of the operands.
/// \param  randomSeed       Random seed (0 = don't use random values).
/// \param  ignoreData       Do not verify results.
/// \param  doReport         Print poplar report on stdout.
/// \param  cycles           If non-empty, will be populated with the cycles
///                          used by the compute set that runs the vertex.
///
/// \return true if the results from the device passed verification
template <typename HOST_DATA_TYPE, typename HOST_OUT_TYPE>
static bool doTest(const DeviceType &deviceType, const VertexDesc &vertex,
                   const BinaryOpType op, const Type &dataType,
                   const Type &outputType, const TensorSizes &sizes,
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
    throw std::runtime_error(binaryOpToString.at(op) +
                             " requires data "
                             "of integer type (specified  data type=" +
                             dataType.toString() + ")");
  }

  TestDevice device = createTestDevice(deviceType, 1, 1);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // === Allocate and initialise host buffers with appropriate values.
  std::vector<HOST_DATA_TYPE> in1Host(sizes.nElems1);
  std::vector<HOST_DATA_TYPE> in2Host(sizes.nElems2);
  fillHostBuffers(op, dataType, randomSeed, in1Host, in2Host);

  // === Create graph variables.
  Tensor in1, in2, out;
  in1 = graph.addVariable(dataType, {sizes.nElems1}, vertex.in1Name);
  graph.setTileMapping(in1, 0);
  if (vertex.inPlace) {
    out = in1;
  } else {
    out = graph.addVariable(outputType, {sizes.nElems1}, vertex.in1Name);
    graph.setTileMapping(out, 0);
  }
  in2 = graph.addVariable(dataType, {sizes.nElems2}, vertex.in2Name);
  graph.setTileMapping(in2, 0);

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
    connectOperand2D(vertex.in1Name, sizes.rowSizes, in1);
    if (!vertex.inPlace)
      connectOperand2D("out", sizes.rowSizes, out);
    if (vertex.isBinaryOp)
      connectOperand2D(vertex.in2Name, sizes.rowSizes, in2);
    else if (vertex.op2isSingleElem)
      graph.connect(v[vertex.in2Name], in2[0]);
    else if (vertex.isVectorInner)
      connectOperand2D(vertex.in2Name, sizes.op2RowSizes, in2);
    else
      graph.connect(v[vertex.in2Name], in2);
  } else {
    graph.connect(v[vertex.in1Name], in1);
    if (!vertex.inPlace)
      graph.connect(v["out"], out);
    if (vertex.op2isSingleElem)
      graph.connect(v[vertex.in2Name], in2[0]);
    else
      graph.connect(v[vertex.in2Name], in2);
  }

  // VectorOuter and VectorInner have additional fields
  if (vertex.isVectorOuter) {
    graph.setInitialValue(v["columns"], sizes.columns);
    graph.setInitialValue(v["rows"], sizes.rows);
  } else if (vertex.isVectorInner) {
    if (vertex.is2D) {
      graph.setInitialValue(v["n"], sizes.rows);

      std::vector<std::uint16_t> BLen(sizes.rows);
      std::vector<std::uint16_t> dataBlockCount(sizes.rows);
      for (unsigned i = 0; i < sizes.rows; i++) {
        BLen[i] = sizes.op2RowSizes[i];
        dataBlockCount[i] = sizes.rowSizes[i] / sizes.op2RowSizes[i];
      }
      graph.setInitialValue(v["BLen"], std::move(BLen));
      graph.setInitialValue(v["dataBlockCount"], dataBlockCount);
    } else {
      unsigned n = sizes.nElems1 / sizes.nElems2;
      unsigned nWorkers = target.getNumWorkerContexts();
      std::uint16_t dataBlockCountPacked = ((n / nWorkers) << 3) | n % nWorkers;
      graph.setInitialValue(v["dataBlockCountPacked"], dataBlockCountPacked);
    }
  }

  // === Create host 'transfer' buffers with the right size for the device type
  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::unique_ptr<char[]> in1HostRaw;
  std::unique_ptr<char[]> in2HostRaw;
  std::unique_ptr<char[]> outHostRaw;
  char *outHostRawPtr = nullptr;
  in1HostRaw = allocateHostMemoryForTensor(in1, vertex.in1Name, graph,
                                           uploadProg, downloadProg, tmap);
  in2HostRaw = allocateHostMemoryForTensor(in2, vertex.in2Name, graph,
                                           uploadProg, downloadProg, tmap);
  if (!vertex.inPlace) {
    outHostRaw = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                             downloadProg, tmap);
    outHostRawPtr = outHostRaw.get();
  } else {
    outHostRawPtr = in1HostRaw.get();
  }

  // === Copy and convert the data from the initialised buffers to the transfer
  // === buffers (still on host)
  auto copyBuffer = [&](Type type, std::vector<HOST_DATA_TYPE> &buf,
                        std::unique_ptr<char[]> &rawBuf) {
    copy(target, buf.data(), buf.size(), type, rawBuf.get());
    // For HALF, we copy and convert back into the (float) host buffers so
    // that the host buffers contain the exact HALF values (which are exactly
    // representable in float). This helps with the validation for the
    // comparison operators
    if (type == HALF)
      copy(target, type, rawBuf.get(), buf.data(), buf.size());
  };
  copyBuffer(dataType, in1Host, in1HostRaw);
  copyBuffer(dataType, in2Host, in2HostRaw);

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
    std::vector<HOST_OUT_TYPE> outHost(sizes.nElems1);
    copy(target, outputType, outHostRawPtr, outHost.data(), outHost.size());

    return verifyResult<HOST_DATA_TYPE, HOST_OUT_TYPE>(
        isIpuModel(deviceType), vertex, op, outputType, in1Host, in2Host,
        outHost, sizes);
  }
  return true;
}

//*************************************************************************
// Calls doTest with the appropriate template parameters for the
// data/output types.
// See 'doTest()' for parameters
static bool doVertexTest(const DeviceType &deviceType, VertexDesc &vertex,
                         const BinaryOpType op, const Type &dataType,
                         const TensorSizes &sizes, const unsigned randomSeed,
                         const bool verbose, const bool ignoreData,
                         const bool doReport, std::optional<uint64_t> &cycles) {
  // Get right output type for vertex, operator and data type
  Type outputType = dataType;
  if (isBoolOp(op))
    outputType = BOOL;
  else if (vertex.is2Types) {
    outputType = (dataType == HALF) ? FLOAT : HALF;
  }

  // Adjust the operand sizes for vertex
  TensorSizes sizesAdj = sizes;
  sizesAdj.adjustForVertex(vertex);

  vertex.setVertexClass(op, dataType, outputType);

  if (verbose) {
    std::cout << format("%-70s %s\n") % vertex.vClassShort %
                     sizesAdj.operandStr;
  }

  // Call the appropriate instantiation of the templated function
#define DO_TEST(DATA_TYPE, OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE)            \
  if (dataType == DATA_TYPE && outputType == OUT_TYPE) {                       \
    return doTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(                              \
               deviceType, vertex, op, dataType, outputType, sizesAdj,         \
               randomSeed, ignoreData, doReport, cycles)                       \
               ? true                                                          \
               : false;                                                        \
  }

  // Note that for both HALF and FLOAT the host buffers are 'float'
  DO_TEST(BOOL, BOOL, unsigned char, unsigned char)

  DO_TEST(HALF, HALF, float, float)
  DO_TEST(HALF, BOOL, float, unsigned char)

  DO_TEST(FLOAT, FLOAT, float, float)
  DO_TEST(FLOAT, BOOL, float, unsigned char)

  DO_TEST(HALF, FLOAT, float, float)
  DO_TEST(FLOAT, HALF, float, float)

  DO_TEST(INT, INT, int, int)
  DO_TEST(INT, BOOL, int, unsigned char)

  DO_TEST(UNSIGNED_INT, UNSIGNED_INT, unsigned, unsigned)
  DO_TEST(UNSIGNED_INT, BOOL, unsigned, unsigned char)

  DO_TEST(SHORT, SHORT, short, short)
  DO_TEST(UNSIGNED_SHORT, UNSIGNED_SHORT, unsigned short, unsigned short)

  DO_TEST(SHORT, BOOL, short, unsigned char)
  DO_TEST(UNSIGNED_SHORT, BOOL, unsigned short, unsigned char)

  // Reaching here means the combination of 'dataType' and 'outputType' was
  // invalid.
  throw std::runtime_error("Combination of data type and operator not "
                           "supported");
  return false;
}

/// Compare cycles obtained by running the vertex with the two specified
/// devices. Prints result on standard output.
///
/// \return true   if both run returned successfully and the difference is less
///                than 'compareThreshold' % of the run with the first device.
static bool compareCycles(const std::array<DeviceType, 2> devPair,
                          VertexDesc &vertex, const BinaryOpType op,
                          const Type &dataType, const TensorSizes &sizes,
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
  bool compareOk = false;
  if (ok[0] && ok[1]) {
    float diff = static_cast<float>(cycles[1]) - static_cast<float>(cycles[0]);
    diffPerc = diff / cycles[0] * 100;
    compareOk = abs(diffPerc) < compareThreshold;
    std::cout << format("%-70s - %s:%8u;  %s:%8u;   diff = %u  %7.2f%%%s\n") %
                     vertex.vClassShort % devName[0].str() % cycles[0] %
                     devName[1].str() % cycles[1] % diff % diffPerc %
                     (compareOk ? "" : " <<== FAIL");

  } else {
    std::cout << format("%-74s - Failed\n") % vertex.vClassShort;
  }
  return ok[0] && ok[1] && compareOk;
  ;
}

//*************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  std::vector<Type> dataTypes;

  TensorSizes sizes;

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
  "Tests one or more of the binary/broadcast vertices with one or more\n"
  "operations/data type, with a default data size, or a specified one.\n"
  "If no vertex is specified, all will be tested. Same for operation and data\n"
  "type.\n"
  "Note that the size of the second operand does not need to be specified\n"
  "when it can be deducted from the vertex variant and the size of the first\n"
  "operand\n"
  "Using the --compare-cycles option, cycles reported when running the vertex\n"
  "on two different devices can be compared."
  "Examples of usages:\n"
  "\n"
  " A binary Supervisor vertex, with operands of 5000 floats; the second\n"
  " operand is automatically set to same size:\n"
  "   BinaryCodeletsTest --vertex BinaryOp1DSupervisor --operation ADD \\\n"
  "                      --data-type float --size 5000\n"
  "\n"
  " As above, but with multiple data types:\n"
  "   BinaryCodeletsTest --vertex BinaryOp1DSupervisor --operation ADD \\\n"
  "                      --data-type float half int --size 5000\n"
  "\n"
  " A binary 2D vertex, where operands are 2D vector of vectors of: [[300]\n"
  " [48] [100]] floats (second operand set to same size):\n"
  "   BinaryCodeletsTest --vertex BinaryOp2D --operation ADD \\\n"
  "                      --data-type float --size 300 48 100\n"
  "\n"
  " A BroadcastScalar2DData vertex, with first operand [[300] [48] [100]]\n"
  " floats; second operand set automatically to size 1:\n"
  "   BinaryCodeletsTest --vertex BroadcastScalar2DData --operation ADD \\\n"
  "                      --data-type float --size 300 48 100\n"
  "\n"
  " A BroadcastVectorOuterByRowSupervisor, with first operand of 30 x 60\n"
  " floats and second operand of 6 floats:\n"
  "   BinaryCodeletsTest --vertex BroadcastVectorOuterByRowSupervisor \\\n"
  "                      --operation ADD --data-type float --row 30\\\n"
  "                      --columns 60 --size 6\n"
  "\n"
  " A BroadcastVectorInner2D, with first operand of [[300] [48] [100]] floats\n"
  " and second operand of [[10] [3] [5]] floats:\n"
  "   BinaryCodeletsTest --vertex BroadcastVectorInner2D --operation ADD\\\n"
  "                      --data-type float --size 300 48 100 --size2 10 3 5\n"
  "\n"
  "Compare cycles reported between Sim and IpuModel when running a specific\n"
  "vertex:\n"
  "   BinaryCodeletsTest --vertex BinaryOp1DSupervisor --operation ADD \\\n"
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
     ("Vertices to test, one or more of: " + allVerticesStr()).c_str())
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
     po::value<std::vector<unsigned>>(&sizes.rowSizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex, "
     "multiple values for a 2D vertex")
    ("size2",
     po::value<std::vector<unsigned>>(&sizes.op2RowSizes)->multitoken(),
     "Size(s) for rows of seconds operand. Single or multiple values depending "
     "on vertex variant")
    ("rows",
     po::value<unsigned>(&sizes.rows)->default_value(sizes.rows),
     "Only for VectorOuter vertices: number of rows")
    ("columns",
     po::value<unsigned>(&sizes.columns)->default_value(sizes.columns),
     "Only for VectorOuter vertices: number of columns")
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
  std::vector<BinaryOpType> operations;
  if (operationStr.empty()) {
    setAllOps(operations);
  } else {
    // Convert strings to BinaryOpType
    for (auto opStr : operationStr)
      operations.push_back(stringToBinaryOp(opStr));
  }

  // === If no data type specified, test 'em all
  if (dataTypes.empty()) {
    dataTypes = {HALF, FLOAT, INT, UNSIGNED_INT, SHORT, UNSIGNED_SHORT, BOOL};
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
  std::optional<uint64_t> cycles =
      std::nullopt; // not interested in cycles here
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
