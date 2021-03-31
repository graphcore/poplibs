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
#include <poputil/TileMapping.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <optional>
#include <regex>
#include <type_traits>

using namespace poplar;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;
using namespace poplibs_support;

static const unsigned TARGET_DATAPATH_WIDTH = 64;
static const unsigned HALF_VECTOR_ELEMS = TARGET_DATAPATH_WIDTH / 16;
static const unsigned FLOAT_VECTOR_ELEMS = TARGET_DATAPATH_WIDTH / 32;

// All vertices that can be tested by this code
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

//*************************************************************************
// This contains the name of the vertex under test and various flags that
// characterise it, used in different places during the test.
struct VertexDesc {
  std::string name;

  BinaryOpType op;
  Type dataType;
  Type outputType;

  std::optional<bool> allowMisaligned; // For VectorOuter vertices only

  std::string vClass;    // Full name with template params for addVertex()
  std::string vClassFmt; // vClass, formatted for display

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

  VertexDesc(const std::string &vertexName, BinaryOpType &op,
             const Type &dataType, std::optional<bool> allowMisaligned)
      : name(vertexName), op(op), dataType(dataType),
        allowMisaligned(allowMisaligned) {

    // Extract the flags by looking at the name
    isBinaryOp = vertexName.find("BinaryOp") != std::string::npos;
    is2D = vertexName.find("2D") != std::string::npos;
    inPlace = vertexName.find("InPlace") != std::string::npos;
    isVectorInner = vertexName.find("VectorInner") != std::string::npos;
    isVectorInner2D = isVectorInner && is2D;
    isVectorOuter = vertexIsVectorOuter(vertexName);
    is2Types = vertexName.find("2Types") != std::string::npos;
    isBroadcastScalar = vertexName.find("BroadcastScalar") != std::string::npos;
    isBroadcastScalar2D = (vertexName == "BroadcastScalar2D" ||
                           vertexName == "BroadcastScalar2DInPlace");
    op2isSingleElem = isBroadcastScalar && !isBroadcastScalar2D;

    outputType = dataType;
    if (isBoolOp(op)) {
      outputType = BOOL;
    } else if (is2Types) {
      outputType = (dataType == HALF) ? FLOAT : HALF;
    }

    if (isBinaryOp) {
      in1Name = inPlace ? "in1Out" : "in1";
      in2Name = "in2";
    } else {
      in1Name = "data";
      in2Name = "B";
    }

    // Get the vertex class
    std::string vName = "popops::" + name;
    if (isVectorOuter) {
      vClass = templateVertex(vName, op, dataType, *allowMisaligned);
    } else if (is2Types) {
      vClass = templateVertex(vName, op, dataType, outputType);
    } else {
      vClass = templateVertex(vName, op, dataType);
    }

    // Shorten the name for display, removing namespaces
    vClassFmt = vClass;
    boost::erase_all(vClassFmt, "popops::");
    boost::erase_all(vClassFmt, "expr::BinaryOpType::");
    const unsigned FMT_LEN = 70;
    unsigned len = vClassFmt.size();
    unsigned padLen = (len < FMT_LEN) ? (FMT_LEN - len) : 1;
    vClassFmt += std::string(padLen, ' '); // pad for display
  }
  // We have a separate static function for this, because it is called also
  // before instantiating a VertexDesc
  static bool vertexIsVectorOuter(const std::string &vertexName) {
    return vertexName.find("VectorOuter") != std::string::npos;
  }
};

//*************************************************************************
// Describes the sizes of the two operands (and the output) for a specific
// vertex. The individual fields are used for different vertex types.
struct TensorSizes {
  // For 2D vertices: size of each row
  // For 1D vertices: single element (size of whole 1st operand)
  // For VectorOuter: single element (size of whole 1st operand)
  std::vector<unsigned> rowSizes;

  // For BroadcastVectorInner2D vertices: size of each row for 2nd operand
  // Each element must be an exact divisor of corresponding row of rowSizes
  std::vector<unsigned> op2RowSizes;

  unsigned rows = 0;    // Only for "VectorOuter" vertices, or 2D vertices
  unsigned columns = 0; // Only for "VectorOuter" vertices

  // A string that describes the sizes of the operands, for display
  std::string operandStr;

  // The constructor is given the vertex to test and the sizes specified on the
  // command line. It will try very hard to make sense of those sizes for the
  // specified vertex
  TensorSizes(const VertexDesc &vertex, const SizeDesc size,
              const std::optional<SizeDesc> size2) {
    std::string op1Str, op2Str; // text descriptions of size of operands

    // Convert a vector to a string
    auto vector2str = [](std::vector<unsigned> v) {
      std::string s;
      if (v.size() > 0) {
        s += to_string(v[0]);
        for (unsigned i = 1; i < v.size(); i++)
          s += "," + to_string(v[i]);
      }
      return s;
    };

    // Total number of elements in first operand ("in1") and output. For 2D
    // vertices is the sum of all 'rowSizes'. For 1D vertices is the same as
    // 'rowSizes[0]'
    // For "VectorOuter" vertices is rows x columns
    unsigned nElems1 = 0;

    // Total number of elements in second operand ("in2")
    unsigned nElems2 = 0;

    // =========== First Operand ===========
    if (vertex.isVectorOuter) {
      // -------- Vector Outer --------
      if (size.isRowsByCols) {
        rows = size.val[0];
        columns = size.val[1];
      } else {
        rows = findDivisor(size.val[0]);
        columns = size.val[0] / rows;
      }
      // If allowMisaligned is false, each row must contain a multiple of atoms
      unsigned atomSize =
          (vertex.dataType == HALF) ? HALF_VECTOR_ELEMS : FLOAT_VECTOR_ELEMS;
      if (vertex.allowMisaligned == false && (columns % atomSize) != 0) {
        columns = roundUp(columns, atomSize);
      }
      nElems1 = rows * columns;
      rowSizes.resize(1, nElems1);
      op1Str = to_string(rows) + "x" + to_string(columns);
    } else {
      // -------- VectorInner, BroadcastScalar, BinaryOp --------
      if (vertex.is2D) {
        if (size.isRowsByCols) {
          rows = size.val[0];
          rowSizes.resize(rows, size.val[1]);
          nElems1 = rows * size.val[1];
          op1Str = to_string(rows) + "x" + to_string(size.val[1]);
        } else {
          rows = size.val.size();
          rowSizes = size.val;
          nElems1 = std::accumulate(rowSizes.begin(), rowSizes.end(), 0);
          op1Str = vector2str(rowSizes);
        }
      } else {
        if (size.isRowsByCols) {
          rowSizes.resize(size.val[0], size.val[1]);
          nElems1 = size.val[0] * size.val[1];
        } else {
          rowSizes = {size.val[0]};
          nElems1 = size.val[0];
        }
      }
    }

    // =========== Second Operand ===========
    if (vertex.isVectorOuter) {
      // -------- Vector Outer --------
      nElems2 = size2 ? size2->val[0] : ((rows == 1) ? 1 : rows - 1);
      op2RowSizes = {nElems2};
    } else if (vertex.isVectorInner) {
      if (vertex.is2D) {
        // -------- Vector Inner 2D --------
        if (size2) {
          if (size2->isRowsByCols) {
            op2RowSizes.resize(size2->val[0], size2->val[1]);
          } else {
            op2RowSizes = size2->val;
          }
        }
        // if wrong number of rows for second operand, adjust it
        if (op2RowSizes.size() != rows) {
          op2RowSizes.resize(rows);
          for (unsigned i = 0; i < rows; i++) {
            op2RowSizes[i] = findDivisor(rowSizes[i]);
          }
        }
        // Verify row by row: is length of second operand a divisor of first?
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
        nElems2 = std::accumulate(op2RowSizes.begin(), op2RowSizes.end(), 0);
        op2Str = vector2str(op2RowSizes);
      } else {
        // === Vector Inner Supervisor ===
        if (size2) {
          if (size2->isRowsByCols) {
            nElems2 = size2->val[0] * size2->val[1];
          } else {
            nElems2 = size2->val[0];
          }
        } else {
          nElems2 = findDivisor(nElems1);
        }
        op2RowSizes = {nElems2};
        // Is length of second operand a divisor of first?
        if ((nElems1 % nElems2) != 0) {
          throw std::runtime_error(
              (format("%s's second operand size is %u but it must be an"
                      " exact divisor of the first operand size (%u)") %
               vertex.name % nElems2 % nElems1)
                  .str());
        }
      }
    } else if (vertex.isBroadcastScalar2D) {
      op2RowSizes = {rows};
      nElems2 = rows;
    } else if (vertex.isBroadcastScalar) {
      op2RowSizes = {1};
      nElems2 = 1;
    } else {
      op2RowSizes = rowSizes;
      nElems2 = nElems1;
      if (vertex.is2D) {
        op2Str = vector2str(op2RowSizes);
      }
    }

    if (op1Str.empty()) {
      op1Str = to_string(nElems1);
    }
    if (op2Str.empty()) {
      op2Str = to_string(nElems2);
    }
    operandStr = vertex.in1Name + ": [" + op1Str + "];  " + vertex.in2Name +
                 ": [" + op2Str + "]";
  }
};

//*************************************************************************
// Contains information relative to the test for one single vertex
struct TestRecord {
  TensorSizes sizes;
  std::unique_ptr<VertexDesc> vertex;

  // If not empty, offset in bytes in the device memory between the START of 1st
  // and the 2nd operand; if '0', it means place the two operands back-to-back.
  boost::optional<unsigned> operandOffset;

  TestOperand in1;
  TestOperand in2;
  TestOperand out;

  // Stream names used to transfer the host data, for the two operands and the
  // output. Must be different for each test that is run in the same graph/CS.
  std::string writeName1;
  std::string writeName2;
  std::string readName;

  // Is the output buffer padding made up of Nan values?
  bool padOutWithNan = false;

  /// \param[in] v      The vertex (with operation and data type) to test.
  /// \param[in] seq    A sequential index, different for each test
  /// \param[in] tSizes Describes generically the sizes to use for the test,
  ///                   needs to be adjusted for this specific vertex.
  TestRecord(std::unique_ptr<VertexDesc> v, unsigned seq,
             const TensorSizes &tSizes, boost::optional<unsigned> operandOffset)
      : sizes(tSizes), vertex(std::move(v)), operandOffset(operandOffset) {
    writeName1 = vertex->in1Name + "_" + to_string(seq);
    writeName2 = vertex->in2Name + "_" + to_string(seq);
    readName = "out_" + to_string(seq);
  };
  TestRecord(TestRecord &&) = default;

  std::string toString() { return vertex->vClassFmt + sizes.operandStr; }
};

//*************************************************************************
// Return true if the information in vertex (vertex name, operation and data
// type) is valid.
bool isValidCombination(const VertexDesc &vertex) {
  const BinaryOpType &op = vertex.op;
  const Type &type = vertex.dataType;
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
/// Verifies the results of the test.
/// Input data and device results are stored in 'test'.
///
/// \tparam HostDataType Type used on the host for dataType1, datatType2.
/// \tparam HostOutType  Type used on the host for outputType.
/// \param[in]    target     Which target.
/// \param[in]    isIpuModel Was the device and IpuModel?
/// \param[in]    test       Describes the test to setup.
///
/// \return true if the values returned by the device match (with appropriate
///         tolerances) the one computed on the host
///
template <typename HostDataType, typename HostOutType>
bool verifyTest(const Target &target, bool isIpuModel, const TestRecord &test,
                const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const TensorSizes &sizes = test.sizes;
  const BinaryOpType &op = vertex.op;

  // Convert the device data in host format. Also convert back the input data
  // in host format.
  std::vector<HostDataType> in1Host(test.in1.totalElems);
  std::vector<HostDataType> in2Host(test.in2.totalElems);
  std::vector<HostOutType> outHost(test.out.totalElems);
  copy(target, vertex.dataType, test.in1.rawBuf.get(), in1Host.data(),
       in1Host.size());
  copy(target, vertex.dataType, test.in2.rawBuf.get(), in2Host.data(),
       in2Host.size());
  copy(target, vertex.outputType, test.out.rawBuf.get(), outHost.data(),
       outHost.size());
  if (options.printBuffers) {
    printBuffer("out", outHost, vertex.outputType, sizes.rowSizes,
                test.out.offsets);
  }

  // Check for mismatches on computed values
  auto &rowSizes = test.out.rowSizes; // same as test.in.rowSizes
  unsigned errCount = 0;              // how many mismatched elements we find
  unsigned numRows = rowSizes.size();
  for (unsigned row = 0; row < numRows; row++) {
    for (unsigned i = 0; i < rowSizes[row]; i++) {

      // First operand (has the same size as result)
      HostDataType val1 = in1Host[test.in1.offsets[row] + i];
      // Second operand is more complex
      HostDataType val2;
      if (vertex.isBinaryOp) {
        val2 = in2Host[test.in2.offsets[row] + i];
      } else if (vertex.op2isSingleElem) {
        val2 = in2Host[test.in2.offsets[0]];
      } else if (vertex.isBroadcastScalar2D) {
        val2 = in2Host[test.in2.offsets[0] + row];
      } else if (vertex.isVectorInner) {
        unsigned j = i % test.in2.rowSizes[row];
        val2 = in2Host[test.in2.offsets[row] + j];
      } else if (vertex.isVectorOuter) {
        unsigned voRow = i / sizes.columns;
        val2 = in2Host[test.in2.offsets[row] + voRow % test.in2.rowSizes[0]];
      } else {
        std::cerr << "Unhandled vertex type\n";
        errCount++;
        continue;
      }

      // Result from device
      HostOutType actual = outHost[test.out.offsets[row] + i];

      HostOutType expected = 0; // result for verification
      performOp(op, val1, val2, expected);

      if (!equalValues(isIpuModel, op, vertex.outputType, expected, actual)) {
        // If its is 2D, we want to show row and column where it failed, not
        // just the linear index.
        std::cerr << format(
                         "out[%s] = %s %s %s  =>  expected:%s;  actual:%s\n") %
                         (vertex.is2D ? to_string(row) + "][" + to_string(i)
                                      : to_string(i)) %
                         convertToString(val1) % binaryOpToString.at(op) %
                         convertToString(val2) % convertToString(expected) %
                         convertToString(actual);
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
/// \tparam HostOutType     Type to use on the host for outputType.
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
template <typename HostDataType, typename HostOutType>
static void setupTest(const Target &target, bool isIpuModel, Graph &graph,
                      Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                      Sequence &download, StreamMap &streamMap,
                      TestRecord &test, unsigned tile,
                      const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const TensorSizes &sizes = test.sizes;
  const Type &dataType = vertex.dataType;
  const Type &outputType = vertex.outputType;

  assert(TARGET_DATAPATH_WIDTH == target.getDataPathWidth());

  // Check for various possible inconsistencies in the combinations of
  // parameters

  if (std::find(std::begin(verticesNames), std::end(verticesNames),
                vertex.name) == std::end(verticesNames)) {
    throw std::runtime_error(vertex.name +
                             " is not a valid vertex name. Maybe you wanted "
                             "to use the --vertexRE option for a regular "
                             "expression?");
  }

  if (vertex.inPlace && (outputType != dataType)) {
    throw std::runtime_error("For in place operations, the data and output "
                             "types must be the same (specified data type=" +
                             dataType.toString() + ", specified output type=" +
                             outputType.toString() + ")");
  }

  if (isIntOp(vertex.op) &&
      (vertex.dataType == HALF || dataType == FLOAT || dataType == BOOL)) {
    throw std::runtime_error(binaryOpToString.at(vertex.op) +
                             " requires data "
                             "of integer type (specified  data type=" +
                             dataType.toString() + ")");
  }

  // === Setup offsets for padding
  test.in1.setup(target, dataType, sizes.rowSizes, options.alignStart);
  test.in2.setup(target, dataType, sizes.op2RowSizes, options.alignStart);
  test.out.setup(target, outputType, sizes.rowSizes, options.alignStart);

  // === Allocate and initialise host buffers with appropriate values.
  std::vector<HostDataType> in1Host(test.in1.totalElems);
  std::vector<HostDataType> in2Host(test.in2.totalElems);
  fillHostBuffers(vertex.op, dataType, options.randomSeed, in1Host, in2Host);
  // If requested, print the buffers, selecting the correct sizes
  if (options.printBuffers) {
    printBuffer(vertex.in1Name, in1Host, dataType, sizes.rowSizes,
                test.in1.offsets);
    printBuffer(vertex.in2Name, in2Host, dataType, sizes.op2RowSizes,
                test.in1.offsets);
  }

  // === Create graph variables.
  // The two operands can be created in two ways, based on the value of
  // operandOffs:
  //
  // 1. If operandOffs contains a value (is not 'none') a single graph variable
  //    will be created, that is then sliced in two, one slice for each operand.
  //    This allows controlling the offset in device memory between the two
  //    operands, as required by some tests (as long as poplar doesn't do
  //    rearrangements!).
  //    Doing this can slow down significantly the execution when running tests
  //    in groups.
  //    If the offset is specified as 0, the two slices will be placed back-to-
  //    back (no padding, apart from a possible 8 byte alignment).
  //    If the offset (in bytes) is > 0, it must be a multiple of 8 bytes and
  //    we will add padding between the two slices, so that the distance between
  //    the START of first operand and the START of the second is equal to the
  //    specified offset (so it must be greater than the size of the first
  //    operand).
  //
  // 2. If operandOffs is empty ('none') the operands will be created as two
  //    separate graph variables. This let poplar place the operands in memory
  //    as it sees fit and it is much faster.

  Tensor in1, in2, out;

  if (!test.operandOffset) {
    // Optional not specified ('none'), let poplar place the operands in memory
    in1 = graph.addVariable(dataType, {test.in1.totalElems}, vertex.in1Name);
    graph.setTileMapping(in1, tile);
    in2 = graph.addVariable(dataType, {test.in2.totalElems}, vertex.in2Name);
    graph.setTileMapping(in2, tile);
    createAlignVertex(graph, alignCS, in1, tile);
    createAlignVertex(graph, alignCS, in2, tile);
    if (vertex.inPlace) {
      out = in1;
    } else {
      out = graph.addVariable(vertex.outputType, {test.out.totalElems}, "out");
      graph.setTileMapping(out, tile);
      createAlignVertex(graph, alignCS, out, tile);
    }
  } else {
    // Offset optional specified, check it's ok (must be aligned and big enough)
    const static unsigned ALIGN = 8; // align sizes/offsets on this num. bytes
    const unsigned dataTypeSize = target.getTypeSize(dataType);
    assert((ALIGN % dataTypeSize) == 0);
    const unsigned nBytes1 =
        test.in1.totalElems * dataTypeSize; // Operand1 size bytes
    const unsigned nBytes1Aligned = roundUp(nBytes1, ALIGN);

    unsigned offs = *test.operandOffset;
    if ((offs % ALIGN) != 0) {
      throw std::runtime_error("The specified --offset-operands (" +
                               to_string(offs) + ") should be multiple of " +
                               to_string(ALIGN));
    } else if (offs == 0) {
      // '0' means "no padding" between operands, just make sure op2 is aligned
      offs = nBytes1Aligned;
    } else if (offs < nBytes1) {
      // offset is the distance between the START of the two operands, so cannot
      // be smaller than the size of the first one.
      throw std::runtime_error("The specified --offset-operands (" +
                               to_string(offs) +
                               ") is smaller than the size of the first "
                               "operand (" +
                               to_string(nBytes1) + ")");
    }

    // Create one single tensor to be sliced in two
    unsigned offs2 = std::max(nBytes1Aligned, offs); // Where op2/out starts
    unsigned offs2Elems = offs2 / dataTypeSize;      // Now in elems, not bytes

    if (vertex.op2isSingleElem && !vertex.inPlace && dataType == outputType) {
      unsigned totSize = offs2Elems + test.out.totalElems;
      Tensor in = graph.addVariable(dataType, {totSize}, "in1_out");
      graph.setTileMapping(in, tile);
      createAlignVertex(graph, alignCS, in, tile);
      in1 = in.slice(0, test.in1.totalElems);
      out = in.slice(offs2Elems, offs2Elems + test.out.totalElems);

      in2 = graph.addVariable(dataType, {test.in2.totalElems}, vertex.in2Name);
      graph.setTileMapping(in2, tile);
      createAlignVertex(graph, alignCS, in2, tile);
    } else {
      unsigned totSize = offs2Elems + test.in2.totalElems;
      Tensor in = graph.addVariable(dataType, {totSize}, "in1_in2");
      graph.setTileMapping(in, tile);
      createAlignVertex(graph, alignCS, in, tile);

      in1 = in.slice(0, test.in1.totalElems);
      in2 = in.slice(offs2Elems, offs2Elems + test.in2.totalElems);
      if (vertex.inPlace) {
        out = in1;
      } else {
        out =
            graph.addVariable(vertex.outputType, {test.out.totalElems}, "out");
        graph.setTileMapping(out, tile);
        createAlignVertex(graph, alignCS, out, tile);
      }
    }
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
    copy(target, in1Host.data(), in1Host.size(), outputType,
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
  copy(target, in2Host.data(), in2Host.size(), dataType, test.in2.rawBuf.get());

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
    test.out.connectOperand(graph, v, opType, out, "out");
  // Second operand is more complex
  if (vertex.op2isSingleElem) {
    opType = TestOperand::OperandType::isScalar;
  } else if ((vertex.isBinaryOp && vertex.is2D) || vertex.isVectorInner2D) {
    opType = TestOperand::OperandType::is2D;
  } else {
    opType = TestOperand::OperandType::is1D;
  }
  test.in2.connectOperand(graph, v, opType, in2, vertex.in2Name);

  // VectorOuter and VectorInner have additional fields
  if (vertex.isVectorOuter) {
    graph.setInitialValue(v["columns"], sizes.columns);
    graph.setInitialValue(v["rows"], sizes.rows);
  } else if (vertex.isVectorInner) {
    if (vertex.is2D) {
      std::vector<std::uint16_t> BLen(sizes.rows);
      std::vector<std::uint16_t> dataBlockCount(sizes.rows);
      std::vector<std::uint16_t> workList;
      workList.push_back(sizes.rows - 1);
      for (unsigned i = 0; i < sizes.rows; i++) {
        workList.push_back(sizes.op2RowSizes[i]);
        workList.push_back(sizes.rowSizes[i] / sizes.op2RowSizes[i]);
      }

      auto workListTensor =
          graph.addConstant(UNSIGNED_SHORT, {workList.size()}, workList.data());
      graph.setTileMapping(workListTensor, 0);
      graph.connect(v["workList"], workListTensor);
    } else {
      unsigned n = test.in1.rowSizes[0] / test.in2.rowSizes[0];
      unsigned nWorkers = target.getNumWorkerContexts();
      std::uint16_t dataBlockCountPacked = ((n / nWorkers) << 3) | n % nWorkers;
      graph.setInitialValue(v["dataBlockCountPacked"], dataBlockCountPacked);
    }
  }
}

// A macro to match the device data and output types with the types used
// on the host. This is used in doSetupTest and doVerifyTest
#define SELECT_BY_TYPES()                                                      \
  /* Note that for both HALF and FLOAT the host buffers are 'float' */         \
  SELECT_ONE(BOOL, BOOL, unsigned char, unsigned char)                         \
  SELECT_ONE(SHORT, BOOL, short, unsigned char)                                \
  SELECT_ONE(SHORT, SHORT, short, short)                                       \
  SELECT_ONE(UNSIGNED_SHORT, BOOL, unsigned short, unsigned char)              \
  SELECT_ONE(UNSIGNED_SHORT, UNSIGNED_SHORT, unsigned short, unsigned short)   \
  SELECT_ONE(HALF, BOOL, float, unsigned char)                                 \
  SELECT_ONE(HALF, HALF, float, float)                                         \
  SELECT_ONE(HALF, FLOAT, float, float)                                        \
  SELECT_ONE(FLOAT, BOOL, float, unsigned char)                                \
  SELECT_ONE(FLOAT, HALF, float, float)                                        \
  SELECT_ONE(FLOAT, FLOAT, float, float)                                       \
  SELECT_ONE(INT, BOOL, int, unsigned char)                                    \
  SELECT_ONE(INT, INT, int, int)                                               \
  SELECT_ONE(UNSIGNED_INT, BOOL, unsigned, unsigned char)                      \
  SELECT_ONE(UNSIGNED_INT, UNSIGNED_INT, unsigned, unsigned)                   \
  /* The combination of 'dataType'+'outputType' was not specified above */     \
  throw invalid_types(vertex.dataType, vertex.outputType);

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
#define SELECT_ONE(IPU_DATA_TYPE, IPU_OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE) \
  if (vertex.dataType == IPU_DATA_TYPE && vertex.outputType == IPU_OUT_TYPE) { \
    setupTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(target, isIpuModel, graph,        \
                                             upload, alignCS, cs, download,    \
                                             streamMap, test, tile, options);  \
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

#define SELECT_ONE(IPU_DATA_TYPE, IPU_OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE) \
  if (vertex.dataType == IPU_DATA_TYPE && vertex.outputType == IPU_OUT_TYPE) { \
    return verifyTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(target, isIpuModel, test, \
                                                     options);                 \
  }
  SELECT_BY_TYPES()
#undef SELECT_ONE
}

//*************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  std::vector<Type> dataTypes;

  std::vector<SizeDesc> sizes = {{false, {25, 12, 21}}};
  std::vector<SizeDesc> sizes2;

  std::vector<std::string> operationStr;
  std::vector<bool> allowMisaligned;
  std::vector<std::string> vertices;
  std::string vertexRE; // regular expression for vertex name
  unsigned groupTests = 1;
  boost::optional<std::string> cycleCompareDevice;

  boost::optional<unsigned> operandOffset;
  MiscOptions options;

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
     "Data type: one or more of half, float, int, uint, short, ushort, bool, "
     "char, schar, uchar")
    ("operation",
     po::value<std::vector<std::string>>(&operationStr)->multitoken(),
     ("Operation(s) to perform, one or more of: " + allOpsStr()).c_str())
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex, "
     "multiple values for a 2D vertex")
    ("size2",
     po::value<std::vector<SizeDesc>>(&sizes2)->multitoken(),
     "Size(s) for rows of seconds operand. Single or multiple values depending "
     "on vertex variant")
    ("allow-misaligned",
     po::value<std::vector<bool>>(&allowMisaligned)->multitoken(),
     "For VectorOuter vertices only, value(s) for the 'allowMisaligned "
     "template parameters")
    ("offset-operands",
     po::value<boost::optional<unsigned>>(&operandOffset)->implicit_value(0),
     "The 2nd operand will be placed in memory so that its start is at the "
     "specified number of bytes from the start of the 1st operand. If 0, the "
     "two operands will be allocated back-to-back. If not present, placement "
     "in memory will be left to poplar")
    ;
  // clang-format on
  addCommonOptions(poDesc, deviceType, cycleCompareDevice, groupTests, options);

  parseOptions(argc, argv, poDesc);

  // === Some parameter checks
  if (!vertexRE.empty() && !vertices.empty()) {
    throw std::runtime_error(
        "Cannot specify both --vertexRE and --vertex option");
  }
  if (cycleCompareDevice && (groupTests > 1)) {
    std::cout << "When running with --compare-cycle option, the --group-tests "
                 "option is ignored\n";
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

  // === If allowMisaligned not specified, test both (only for VectorOuter)
  if (allowMisaligned.empty()) {
    allowMisaligned = {false, true};
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
  unsigned nSizes = sizes.size();
  unsigned nSizes2 = sizes2.size();
  // Loop over all vertices, operations, data types
  for (std::string vertexName : vertices) {
    // If a regex was specified, see if it matches
    if (vertexRE.empty() || std::regex_search(vertexName, vertexRegEx)) {
      for (auto op : operations) {
        for (auto type : dataTypes) {
          for (unsigned i = 0; i < nSizes; i++) {
            // For VectorOuter we have to select the right 'allowMisaligned'
            // template parameter value(s) (false, true or both). If not
            // VectorOuter, we just ignore it (set it to 'nullopt')
            std::vector<std::optional<bool>> misalignedList;
            if (VertexDesc::vertexIsVectorOuter(vertexName)) {
              for (auto m : allowMisaligned)
                misalignedList.push_back(m);
            } else {
              misalignedList.push_back(std::nullopt);
            }
            for (auto &misaligned : misalignedList) {
              auto vertex = std::make_unique<VertexDesc>(vertexName, op, type,
                                                         misaligned);
              if (isValidCombination(*vertex)) {
                numTests++;
                std::optional<SizeDesc> sz2;
                if (i < nSizes2) {
                  sz2 = sizes2[i];
                }
                auto testRec = std::make_shared<TestRecord>(
                    std::move(vertex), numTests,
                    TensorSizes(*vertex, sizes[i], sz2), operandOffset);
                addOneTest<TestRecord, VertexDesc>(tests, testRec, devices,
                                                   errCount, options);
              }
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
