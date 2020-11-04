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

#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include "popops/ElementWise.hpp"
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/token_functions.hpp>
#include <boost/tokenizer.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <type_traits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;
using namespace poplibs_support;
using popops::expr::BinaryOpType;
using std::to_string;

// The name of the compute set where we run the vertex under test.
const static std::string computeSetName = "vertexComputeSet";

// We use 'unsigned char' on the host instead of 'bool', because
// std::vector<bool> gets specialised with 1 bit per element instead of 1 byte
typedef unsigned char HostBool;

// A map that, given a BinaryOpType, returns a string with its name
const std::map<BinaryOpType, const std::string> binaryOpToString = {
#define ONE_OP(opId)                                                           \
  { BinaryOpType::opId, #opId }
    ONE_OP(ADD),
    ONE_OP(ATAN2),
    ONE_OP(BITWISE_AND),
    ONE_OP(BITWISE_OR),
    ONE_OP(BITWISE_XOR),
    ONE_OP(BITWISE_XNOR),
    ONE_OP(DIVIDE),
    ONE_OP(EQUAL),
    ONE_OP(GREATER_THAN_EQUAL),
    ONE_OP(GREATER_THAN),
    ONE_OP(INV_STD_DEV_TO_VARIANCE),
    ONE_OP(LESS_THAN_EQUAL),
    ONE_OP(LOGICAL_AND),
    ONE_OP(LOGICAL_OR),
    ONE_OP(LESS_THAN),
    ONE_OP(MAXIMUM),
    ONE_OP(MINIMUM),
    ONE_OP(MULTIPLY),
    ONE_OP(NOT_EQUAL),
    ONE_OP(POWER),
    ONE_OP(REMAINDER),
    ONE_OP(SHIFT_LEFT),
    ONE_OP(SHIFT_RIGHT),
    ONE_OP(SHIFT_RIGHT_SIGN_EXTEND),
    ONE_OP(SUBTRACT),
    ONE_OP(VARIANCE_TO_INV_STD_DEV),
#undef ONE_OP
};

// Given a string, returns the corresponding BinaryOpType. The string can be
// uppercase or lowercase and be just the start of the name of the operator, as
// long as it uniquely identifies it (i.e. "ad" ==> ADD, but "shift_r" is
// not enough, as we have both SHIFT_RIGHT and SHIFT_RIGHT_SIGN_EXTEND.
// But note that if the string matches *exactly* the name, then it is considered
// valid; for instance, "shift_right" => SHIFT_RIGHT, even if we have
// SHIFT_RIGHT_SIGN_EXTEND
BinaryOpType stringToBinaryOp(const std::string &s) {
  std::optional<BinaryOpType> opFound(std::nullopt);
  std::string us = s;
  boost::to_upper(us);
  for (auto &e : binaryOpToString) {
    auto op = e.first;
    auto opName = e.second;
    if (opName == us) {
      return op; // Exact match. Return straight away
    }
    if (opName.rfind(us, 0) != std::string::npos) {
      // Partial match. We need to scan them all to see if more than 1 match
      if (opFound) {
        throw std::runtime_error("<" + s +
                                 "> does not uniquely define a "
                                 "poplar::expr::BinaryOpType");
      } else {
        opFound = op;
      }
    }
  }
  if (opFound) {
    return *opFound;
  } else {
    throw std::runtime_error("<" + s +
                             "> is not a valid poplar::expr::BinaryOpType");
  }
}

// Fills a vector with all operations
void setAllOps(std::vector<BinaryOpType> &ops) {
  for (auto &e : binaryOpToString) {
    ops.push_back(e.first);
  }
}

// Returns a string with all operations, comma separated. Used for the help
// message
const std::string allOpsStr() {
  std::vector<std::string> ops;
  for (auto &e : binaryOpToString) {
    ops.push_back(e.second);
  }
  sort(ops.begin(), ops.end());
  std::string result = ops[0];
  std::for_each(ops.begin() + 1, ops.end(),
                [&](auto &s) { result += ", " + s; });
  return result;
}

// Is 'op' one of the comparison (relational) operators that return booleans?
bool isComparisonOp(BinaryOpType op) {
  switch (op) {
  case BinaryOpType::EQUAL:
  case BinaryOpType::GREATER_THAN_EQUAL:
  case BinaryOpType::GREATER_THAN:
  case BinaryOpType::LESS_THAN_EQUAL:
  case BinaryOpType::LESS_THAN:
  case BinaryOpType::LOGICAL_AND:
  case BinaryOpType::LOGICAL_OR:
  case BinaryOpType::NOT_EQUAL:
    return true;
  default:
    return false;
  }
}

// Is 'op' one of the operators that can only take integer operands?
bool isIntOp(BinaryOpType op) {
  switch (op) {
  case BinaryOpType::BITWISE_AND:
  case BinaryOpType::BITWISE_OR:
  case BinaryOpType::BITWISE_XOR:
  case BinaryOpType::BITWISE_XNOR:
  case BinaryOpType::SHIFT_LEFT:
  case BinaryOpType::SHIFT_RIGHT:
  case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
    return true;
  default:
    return false;
  }
}

// Definitions for an overloaded 'performOp' function to execute the operation
// on the host (for verification). Note that the result value is a function
// parameter (passed by reference) and not the return value of the function, to
// allow for full overload resolution (return type is not used for overload
// resolution in C++).

// A macro to reduce to the minimum the typing to define each operation
#define ONE_OP(opId, expr)                                                     \
  if (op == BinaryOpType::opId) {                                              \
    result = expr;                                                             \
    return;                                                                    \
  }

// These operations are common to floating point and integer types.
#define COMMON_OPS()                                                           \
  ONE_OP(ADD, a + b);                                                          \
  ONE_OP(DIVIDE, a / b);                                                       \
  ONE_OP(MAXIMUM, (a > b) ? a : b);                                            \
  ONE_OP(MINIMUM, (a < b) ? a : b);                                            \
  ONE_OP(MULTIPLY, a *b);                                                      \
  ONE_OP(SUBTRACT, a - b)

// These operations are common to INT and UNSIGNED_INT types
#define INTEGER_OPS()                                                          \
  COMMON_OPS();                                                                \
  ONE_OP(BITWISE_AND, a &b);                                                   \
  ONE_OP(BITWISE_OR, a | b);                                                   \
  ONE_OP(BITWISE_XOR, a ^ b);                                                  \
  ONE_OP(BITWISE_XNOR, ~(a ^ b));                                              \
  ONE_OP(LOGICAL_AND, a &&b);                                                  \
  ONE_OP(LOGICAL_OR, a || b);                                                  \
  ONE_OP(REMAINDER, a - (a / b) * b);                                          \
  ONE_OP(SHIFT_LEFT, a << b);                                                  \
  ONE_OP(SHIFT_RIGHT, (unsigned)a >> b;);                                      \
  ONE_OP(SHIFT_RIGHT_SIGN_EXTEND, a >> b;);

// Do the operation specified by 'op' on 'a' and 'b', where the operands are
// floating point types ('float' on the host, HALF or FLOAT on the device)
void performOp(BinaryOpType op, float a, float b, float &result) {
  COMMON_OPS();
  ONE_OP(ATAN2, atan2(a, b));
  ONE_OP(INV_STD_DEV_TO_VARIANCE, 1 / (a * a) - b);
  ONE_OP(POWER, pow(a, b));
  ONE_OP(REMAINDER, std::fmod(a, b));
  ONE_OP(VARIANCE_TO_INV_STD_DEV, 1 / (std::sqrt(a + b)));
  throw std::logic_error(to_string(unsigned(op)) +
                         " is not a valid operator for floating point types");
}

// Do the operation specified by 'op' on 'a' and 'b' where the operands are
// INT types.
void performOp(BinaryOpType op, int a, int b, int &result) {
  INTEGER_OPS();
  throw std::logic_error(to_string(unsigned(op)) +
                         " is not a valid operator for signed integer");
}

// Do the operation specified by 'op' on 'a' and 'b' where the operands are
// UNSIGNED_INT types.
void performOp(BinaryOpType op, unsigned a, unsigned b, unsigned &result) {
  INTEGER_OPS();
  throw std::logic_error(to_string(unsigned(op)) +
                         " is not a valid operator for unsigned integer");
}

// Do the operation specified by 'op' on 'a' and 'b' where the 'op' is one of
// the operators that return a boolean. This works for floating point, integer
// and boolean data types.
template <typename T>
void performOp(BinaryOpType op, T a, T b, HostBool &result) {
  ONE_OP(EQUAL, a == b);
  ONE_OP(GREATER_THAN_EQUAL, a >= b);
  ONE_OP(GREATER_THAN, a > b);
  ONE_OP(LESS_THAN_EQUAL, a <= b);
  ONE_OP(LESS_THAN, a < b);
  ONE_OP(LOGICAL_AND, a && b);
  ONE_OP(LOGICAL_OR, a || b);
  ONE_OP(NOT_EQUAL, a != b);
  throw std::logic_error(to_string(unsigned(op)) +
                         " is not a boolean operator");
}

#undef COMMON_OPS
#undef ONE_OP

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
        {BinaryOpType::BITWISE_AND, {INT, UNSIGNED_INT}},
        {BinaryOpType::BITWISE_OR, {INT, UNSIGNED_INT}},
        {BinaryOpType::BITWISE_XOR, {INT, UNSIGNED_INT}},
        {BinaryOpType::BITWISE_XNOR, {INT, UNSIGNED_INT}},
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
        {BinaryOpType::EQUAL, {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::GREATER_THAN, {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::GREATER_THAN_EQUAL,
         {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::LESS_THAN, {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::LESS_THAN_EQUAL, {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
        {BinaryOpType::NOT_EQUAL, {FLOAT, HALF, INT, UNSIGNED_INT, BOOL}},
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
  if (vertex.isVectorInner || vertex.isVectorOuter) {
    return ((op == BinaryOpType::ADD || op == BinaryOpType::SUBTRACT ||
             op == BinaryOpType::MULTIPLY) &&
            (type == HALF || type == FLOAT));
  }

  if (isComparisonOp(op) && vertex.inPlace) {
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
          op2Str = vector2str(op2RowSizes);
          nElems2 = std::accumulate(op2RowSizes.begin(), op2RowSizes.end(), 0);
          for (unsigned i = 0; i < rows; i++) {
            if ((rowSizes[i] % op2RowSizes[i]) != 0) {
              throw std::runtime_error(vertex.name +
                                       "'s second operand size must be an "
                                       "exact divisor of the size of each row "
                                       "of first operand");
            }
          }
        } else {
          nElems2 = op2RowSizes[0] > 0 ? op2RowSizes[0] : 1;
          if ((nElems1 % nElems2) != 0) {
            throw std::runtime_error(vertex.name +
                                     "'s second operand size is " +
                                     to_string(nElems2) +
                                     ", but it must be an exact "
                                     "divisor of first operand size (" +
                                     to_string(nElems1) + ")");
          }
        }
      } else if (vertex.isVectorOuter) {
        if (op2RowSizes.size() != 0) {
          nElems2 = op2RowSizes[0];
        }
        if (nElems2 == 0 || (rows % nElems2) != 0) {
          throw std::runtime_error(vertex.name + "'s second operand size is " +
                                   to_string(nElems2) +
                                   ", but it must be an exact "
                                   "divisor of the number of rows (" +
                                   to_string(rows) + ")");
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

// Overloaded & templated convertToString functions to print correctly
// from inside the templated 'verifyResult()'  function

std::string convertToString(unsigned val) {
  std::stringstream ss;
  ss << "0x" << std::hex << val;
  return ss.str();
}

std::string convertToString(HostBool val) {
  return convertToString(unsigned(val));
}

std::string convertToString(int val) { return convertToString(unsigned(val)); }

template <typename T> std::string convertToString(T val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

/// Check if two values are 'equal enough', for verification. This is the
/// overloaded version for floating point type
bool equalValues(const bool isIpuModel, const BinaryOpType op,
                 const Type &dataType, float expected, float actual) {
  // For single precision floating types there are some operators where we
  // expect the result from the device to be bit exact with the one from the
  // host
  if (dataType == FLOAT &&
      (op == BinaryOpType::ADD || op == BinaryOpType::SUBTRACT ||
       op == BinaryOpType::MULTIPLY || op == BinaryOpType::DIVIDE ||
       op == BinaryOpType::REMAINDER)) {
    return expected == actual;
  } else {

    double tolerance = 0.000001;

    // Horrible contortions to verify result for halves. We should really
    // have a half bit-exact computation library for the host.
    if (dataType == HALF) {
      float clipTreshHalf =
          isIpuModel ? std::numeric_limits<float>::infinity() : 65504.0f;
      float clipValueHalf = 65488.0f;
      if (actual >= clipTreshHalf) {
        return expected >= clipValueHalf;
      } else if (actual <= -clipTreshHalf) {
        return expected <= -clipValueHalf;
      }

      // POWER in half is not very precise
      tolerance = (op == BinaryOpType::POWER) ? 0.012 : 0.003;
    }

    bool isEqual = false;
    if (expected == 0) {
      isEqual = (expected == actual);
    } else {
      double delta = std::abs(expected - actual);
      delta = delta / expected;
      isEqual = (delta <= tolerance);
    }
    return isEqual;
  }
}

/// Check if two values are 'equal enough'.
/// For int/unsigned/boolean values, results must be bit exact
template <typename T>
bool equalValues(const bool isIpuModel, const BinaryOpType op,
                 const Type &dataType, float expected, float actual) {
  return expected == actual;
}

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
      std::cerr << "out[" << i << "] = " << convertToString(val1) << " "
                << binaryOpToString.at(op) << " " << convertToString(val2)
                << " =>  expected:" << convertToString(expected)
                << ";  actual:" << convertToString(actual) << "\n";
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

// A 'random generator' engine which might not be there (in which case it means
// that we are not using random values to fill the data buffers).
using RandomEngine = boost::optional<std::minstd_rand>;

// Filling one of the operand buffers, for boolean data. We always fill it with
// random booleans, (ignoring 'i', 'min', 'max' and 'nonZero')
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<HostBool> &buf, int i, HostBool min, HostBool max,
                bool nonZero) {
  std::bernoulli_distribution d(0.5);

  for (auto &x : buf)
    x = d(*rndEng);
}

// Filling one of the operand buffers for int data.
void fillBufferInt(const Type &dataType, RandomEngine &rndEng, int *data,
                   unsigned n, int i, int min, int max, bool nonZero) {
  std::uniform_int_distribution<int> d(min, max);
  for (unsigned k = 0; k < n; k++) {
    if (rndEng) {
      do {
        data[k] = d(*rndEng);
      } while (nonZero && data[k] == 0);
    } else {
      if (max != 0 && i > max)
        i = 0;
      data[k] = i++;
    }
  }
}

void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<int> &buf, int i, int min, int max, bool nonZero) {
  fillBufferInt(dataType, rndEng, buf.data(), buf.size(), i, min, max, nonZero);
}

void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<unsigned> &buf, unsigned i, unsigned min,
                unsigned max, bool nonZero) {
  // The 'int' filling is good for 'unsigned' as well
  fillBufferInt(dataType, rndEng, (int *)(buf.data()), buf.size(), i, min, max,
                nonZero);
}

// Filling one of the operand buffers for FLOAT and HALF data (both use 'float'
// buffers on the host)
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<float> &buf, float i, float min, float max,
                bool nonZero) {
  std::uniform_real_distribution<float> d(min, max);
  for (auto &x : buf) {
    if (rndEng) {
      do {
        x = d(*rndEng);
      } while (nonZero && x == 0);
    } else {
      if (i > max)
        i = 0;
      x = i;
      i += 1.0;
    }
  }
}

//*************************************************************************
/// Fills the host buffers with values appropriate to the data type and the
/// operation being performed.
template <typename HOST_DATA_TYPE>
void fillBuffers(BinaryOpType op, const Type &dataType, unsigned randomSeed,
                 std::vector<HOST_DATA_TYPE> &buf1,
                 std::vector<HOST_DATA_TYPE> &buf2) {
  bool nonZero = false;

  // Using a specific random generator means that we get the same random values
  // on different platforms.
  RandomEngine rndEng;
  if (randomSeed != 0 || dataType == BOOL)
    rndEng = std::minstd_rand(randomSeed);

  // For integer types we generate values in the closed interval [min, max]
  // For floating point types we generate values in the open interval [min, max)
  HOST_DATA_TYPE min1 = 0, max1 = 0;
  HOST_DATA_TYPE min2 = 0, max2 = 0;

  if (std::is_floating_point<HOST_DATA_TYPE>::value) {

    // For floating point, we limit the range
    HOST_DATA_TYPE absMax;
    absMax = (dataType == HALF) ? 200.0 : 32000.0;
    min1 = min2 = -absMax;
    max1 = max2 = absMax;

    // For the power operator we want values in a small positive range, to
    // avoid having mostly overflows/underflows
    if (op == BinaryOpType::POWER) {
      min1 = min2 = 1.0;
      max1 = max2 = 5.0;
    }

    if (dataType == HALF) {
      // In case of ADD,HALF we make both value positive, while for HALF,
      // SUBTRACT the first is positive and the second negative, because
      // subtracting two values that are very similar can gives results that
      // are very inaccurate [for instance (1.0) + (-1.00000001)].
      if (op == BinaryOpType::ADD) {
        min1 = min2 = 0;
      } else if (op == BinaryOpType::SUBTRACT) {
        min1 = max2 = 0;
      } else if (op == BinaryOpType::DIVIDE) {
        // In case of DIVIDE, HALF we must avoid overflow, so we choose a
        // limited range for the divisor
        min2 = 0.5;
        max2 = 600;
      }
    }

    // VARIANCE_TO_INV_STD_DEV is: 1/sqrt(a+b), so (a+b) must be non-negative.
    // To simplify, we force both operands to be non-negative
    if (op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) {
      min1 = min2 = 0;
    }
  } else {
    // Non floating point case (INT, UNSIGNED)./ For BOOL these are ignored
    min1 = min2 = std::numeric_limits<HOST_DATA_TYPE>::min();
    max1 = max2 = std::numeric_limits<HOST_DATA_TYPE>::max();
  }

  // Shifting more than 31 can give different results on different platforms
  if (op == BinaryOpType::SHIFT_LEFT || op == BinaryOpType::SHIFT_RIGHT ||
      op == BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND) {
    min2 = 0;
    max2 = 31;
  }

  // If we are dealing with integer divide, we limit the size of the second
  // operand, just to avoid having a lot of zeros in the results (note that this
  // influences heavily the timing!)
  if (std::is_same<HOST_DATA_TYPE, int>::value ||
      std::is_same<HOST_DATA_TYPE, unsigned>::value) {
    if (op == BinaryOpType::DIVIDE || op == BinaryOpType::REMAINDER) {
      min2 = (dataType == UNSIGNED_INT) ? 0 : -32767;
      max2 = 32767;
    }
  }

  // These operations must have a second operand != 0
  if (op == BinaryOpType::DIVIDE || op == BinaryOpType::ATAN2 ||
      op == BinaryOpType::REMAINDER) {
    nonZero = true;
  }

  fillBuffer(dataType, rndEng, buf1, 100, min1, max1, nonZero);
  fillBuffer(dataType, rndEng, buf2, 500, min2, max2, nonZero);

  // If comparing for equality/inequality, we make sure we have a few values
  // that are equal to test the 'equal' path
  if (rndEng && (op == BinaryOpType::EQUAL || op == BinaryOpType::NOT_EQUAL ||
                 op == BinaryOpType::GREATER_THAN_EQUAL ||
                 op == BinaryOpType::LESS_THAN_EQUAL)) {
    // We want to make 1 in 5 elements the same
    std::uniform_int_distribution<int> d(1, 5);
    unsigned n1 = buf1.size();
    unsigned n2 = buf2.size();
    if (n1 == n2) {
      // Same number of elements, i.e. one of the BinaryOp vertices. Select
      // random positions and copy the value from second operand into the first
      for (unsigned i = 0; i < n1; i++) {
        if (d(*rndEng) == 1)
          buf1[i] = buf2[i];
      }
    } else {
      // Second operand is smaller than the first, i.e. one of the broadcast
      // vertices. We copy the first element of the second operand in random
      // positions of the first. This works fine for broadcast scalar with a
      // single element, while for BroadcastScalar2D and VectorOuter/Inner works
      // only for the first row.
      for (unsigned i = 0; i < n1; i++) {
        if (d(*rndEng) == 1)
          buf1[i] = buf2[0];
      }
    }
  }
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
                             " is not a valid vertex name for this test");
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
  fillBuffers(op, dataType, randomSeed, in1Host, in2Host);

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
    // For HALF, we copy and convert back into the (float) host buffers so that
    // the host buffers contain the exact HALF values (which are exactly
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
  if (isComparisonOp(op))
    outputType = BOOL;
  else if (vertex.is2Types) {
    outputType = (dataType == HALF) ? FLOAT : HALF;
  }

  // Adjust the operand sizes for vertex
  TensorSizes sizesAdj = sizes;
  sizesAdj.adjustForVertex(vertex);

  vertex.setVertexClass(op, dataType, outputType);

  if (verbose) {
    std::cout << boost::format("%-70s %s\n") % vertex.vClassShort %
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
///                than 'tolerance' % of the run with the first device.
static bool compareCycles(const std::array<DeviceType, 2> devPair,
                          VertexDesc &vertex, const BinaryOpType op,
                          const Type &dataType, const TensorSizes &sizes,
                          const unsigned randomSeed) {
  const float tolerance = 10.0; // percent

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
                     "%-70s - %s:%8u;  %s:%8u;   diff = %7.2f%%%s\n") %
                     vertex.vClassShort % devName[0].str() % cycles[0] %
                     devName[1].str() % cycles[1] % diffPerc %
                     ((abs(diffPerc) < tolerance) ? "" : " <<====");
  } else {
    std::cout << boost::format("%-74s - Failed\n") % vertex.vClassShort;
  }
  return ok[0] && ok[1] && abs(diffPerc) < tolerance;
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
  boost::optional<DeviceType> cycleCompareDevice;

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

  po::options_description desc(description);

  desc.add_options()
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
     "Data type: one or more of half, float, int, unsigned, bool")
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
     po::value<boost::optional<DeviceType>>(&cycleCompareDevice)->
                                         implicit_value(DeviceType::IpuModel2),
     "Compare cycles reported for the vertex between the device specified by "
     "--device-type and another device specified by this option")
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
  po::variables_map vm;
  try {
    // Additional command line parser to interpret an argument '@filename' as
    // a option "config-file" with the value "filename"
    auto at_option_parser = [](std::string const &s) {
      if ('@' == s[0]) {
        return std::make_pair(std::string("options-file"), s.substr(1));
      } else {
        return std::pair<std::string, std::string>();
      }
    };

    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .extra_parser(at_option_parser)
                  .run(),
              vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    // If there is a file to read the options from, do it
    if (vm.count("options-file")) {
      std::string filename = vm["options-file"].as<std::string>();
      std::ifstream ifs(filename.c_str());
      if (!ifs) {
        throw std::runtime_error("Could not open options file <" + filename +
                                 ">");
      }
      // Read the whole file into a stringstream
      std::stringstream ss;
      ss << ifs.rdbuf();
      // Split the file content into tokens, using spaces/newlines/tabs
      boost::char_separator<char> sep(" \t\n\r");
      std::string sstr = ss.str();
      boost::tokenizer<boost::char_separator<char>> tok(sstr, sep);
      std::vector<std::string> args;
      std::copy(tok.begin(), tok.end(), back_inserter(args));
      // Parse the file and store the options
      po::store(po::command_line_parser(args).options(desc).run(), vm);
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

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
    dataTypes = {HALF, FLOAT, INT, UNSIGNED_INT, BOOL};
  }

  std::regex vertexRegEx(vertexRE);

  // If we are comparing cycles, create an array with the 2 devices
  std::array<DeviceType, 2> devPair;
  if (cycleCompareDevice)
    devPair = {deviceType, *cycleCompareDevice};

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
                                          randomSeed)
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
  if (count > 1) {
    std::cout << boost::format(
                     "BinaryCodeletsTest: %u tests run in total; %u Failed\n") %
                     count % errCount;
  }
  return allOk ? 0 : 1; // returning 1 means an error.
}
