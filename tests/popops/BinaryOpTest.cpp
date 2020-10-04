// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BinaryOpTest
//
// Performs a binary operation between two tensors with any desired shape, each
// mapped in any desired way among tiles.

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
#include <boost/program_options.hpp>
#include <boost/token_functions.hpp>
#include <boost/tokenizer.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <type_traits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;
using namespace poplibs_support;
using popops::expr::BinaryOpType;

const poplar::OptionFlags options{{"debug.instrumentCompute", "true"}};

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
  if (opFound)
    return *opFound;
  else
    throw std::runtime_error("<" + s +
                             "> is not a valid poplar::expr::BinaryOpType");
}

// Returns a string with all operations, comma separated. Used for the help
// message
const std::string allOps() {
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
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for floating point types");
}

// Do the operation specified by 'op' on 'a' and 'b' where the operands are
// INT types.
void performOp(BinaryOpType op, int a, int b, int &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for signed integer");
}

// Do the operation specified by 'op' on 'a' and 'b' where the operands are
// UNSIGNED_INT types.
void performOp(BinaryOpType op, unsigned a, unsigned b, unsigned &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
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
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a boolean operator");
}

#undef COMMON_OPS
#undef ONE_OP

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
/// overloaded for floating point trype
bool equalValues(const DeviceType &deviceType, const BinaryOpType op,
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
      float clipTreshHalf = (isIpuModel(deviceType))
                                ? std::numeric_limits<float>::infinity()
                                : 65504.0f;
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
bool equalValues(const DeviceType &deviceType, const BinaryOpType op,
                 const Type &dataType, float expected, float actual) {
  return expected == actual;
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
                         const std::vector<HOST_DATA_TYPE> &in1Host,
                         const std::vector<size_t> &shape1Ext,
                         const std::vector<HOST_DATA_TYPE> &in2Host,
                         const std::vector<size_t> &shape2Ext,
                         const std::vector<HOST_OUT_TYPE> &outHost,
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

        HOST_OUT_TYPE actual = get(outHost.data(), shapeOut, i); // from device

        HOST_DATA_TYPE val1 = get(in1Host.data(), shape1Ext, i);
        HOST_DATA_TYPE val2 = get(in2Host.data(), shape2Ext, i);

        HOST_OUT_TYPE expected = 0;

        performOp(op, val1, val2, expected);

        if (!equalValues(deviceType, op, dataType, expected, actual)) {
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

// This collects together information about each one of the two operands
struct OperandDescriptor {
  std::vector<size_t> shape;    // Shape, as defined on command line.
  std::vector<size_t> shapeExt; // Shape, rank-extended.
  std::vector<MappingDesc> map; // Indicates where to map this operand
};

// A 'random generator' engine which might not be there (in which case it means
// that we are not using random values to fill the data buffers).
using RandomEngine = std::optional<std::minstd_rand>;

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
    absMax = (dataType == HALF) ? 300.0 : 32000.0;
    min1 = min2 = -absMax;
    max1 = max2 = absMax;

    // For the power operator we want values in a small positive range, to
    // avoid having mostly overflows/underflows
    if (op == BinaryOpType::POWER) {
      min1 = min2 = 1.0;
      max1 = max2 = 5.0;
    }

    // In case of ADD,HALF we make both value positive, while for HALF,SUBTRACT
    // the first is positive and the second negative, because subtracting two
    // values that are very similar can gives results that are very inaccurate
    // [for instance (1.0) + (-1.00000001)].
    if (dataType == HALF) {
      if (op == BinaryOpType::ADD) {
        min1 = min2 = 0;
      } else if (op == BinaryOpType::SUBTRACT) {
        min1 = max2 = 0;
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

  fillBuffer(dataType, rndEng, buf1, 10, min1, max1, nonZero);
  fillBuffer(dataType, rndEng, buf2, 100, min2, max2, nonZero);

  // If comparing for equality/disequality, we make sure we have a few values
  // that are equal to test the 'equal' path
  if (rndEng && (op == BinaryOpType::EQUAL || op == BinaryOpType::NOT_EQUAL ||
                 op == BinaryOpType::GREATER_THAN_EQUAL ||
                 op == BinaryOpType::LESS_THAN_EQUAL)) {
    std::uniform_int_distribution<int> d(1, 6);
    unsigned n1 = buf1.size();
    unsigned n2 = buf2.size();
    if (n1 == n2) {
      for (unsigned i = 0; i < n1; i++) {
        if (d(*rndEng) == 1)
          buf1[i] = buf2[i];
      }
    } else if (n2 == 1) {
      // Broadcast scalar (second operand is broadcasted)
      for (unsigned i = 0; i < n1; i++) {
        if (d(*rndEng) == 1)
          buf1[i] = buf2[0];
      }
    } else if (n1 == 1) {
      // Broadcast scalar (first operand is broadcasted)
      for (unsigned i = 0; i < n2; i++) {
        if (d(*rndEng) == 1)
          buf2[i] = buf1[0];
      }
    }
  }
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
template <typename HOST_DATA_TYPE, typename HOST_OUT_TYPE>
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
  std::vector<HOST_DATA_TYPE> in1Host(nElems1);
  std::vector<HOST_DATA_TYPE> in2Host(nElems2);
  fillBuffers(operation, dataType, randomSeed, in1Host, in2Host);

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

  Tensor in1, in2;
  std::vector<poplar::Tensor> tensors;
  expr::BinaryOp binOp = [&]() {
    if (in1IsConst) {
      in2 = graph.addVariable(dataType, desc2.shape, "in2");
      mapTensor(in2, desc2.map);
      tensors = {in2};
      return expr::BinaryOp(operation, expr::Const(in1Host[0]), expr::_1);
    } else if (in2IsConst) {
      in1 = graph.addVariable(dataType, desc1.shape, "in1");
      mapTensor(in1, desc1.map);
      tensors = {in1};
      return expr::BinaryOp(operation, expr::_1, expr::Const(in2Host[0]));
    } else {
      in1 = graph.addVariable(dataType, desc1.shape, "in1");
      in2 = graph.addVariable(dataType, desc2.shape, "in2");
      mapTensor(in1, desc1.map);
      mapTensor(in2, desc2.map);
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
    std::vector<HOST_OUT_TYPE> outHost(nElemsOut);
    copy(target, outputType, outHostRawPtr, outHost.data(), outHost.size());
    return verifyResult<HOST_DATA_TYPE, HOST_OUT_TYPE>(
        deviceType, dataType, in1Host, desc1.shapeExt, in2Host, desc2.shapeExt,
        outHost, shapeOut, operation);
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

  po::options_description desc("Perform a binary operation between two tensors "
                               "having any specified shape, each mapped in any "
                               "desired way among tiles.\nOptions are:");

  // clang-format off
  desc.add_options()
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
     po::value<DeviceType>(&deviceType)->default_value(DeviceType::Sim),
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
     ("Operation to perform, one of: " + allOps()).c_str())
    ("enable-optimisations",
     po::value<bool>(&enableOptimisations)->default_value(enableOptimisations),
     "Enable broadcast operation optimisations")
    ;
  // clang-format on
  po::variables_map vm;
  try {
    // Additional command line parser to interpret an argument '@filename' as
    // a option "config-file" with the value "filename"
    auto at_option_parser = [](std::string const &s) {
      if ('@' == s[0])
        return std::make_pair(std::string("options-file"), s.substr(1));
      else
        return std::pair<std::string, std::string>();
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

  Type outputType = isComparisonOp(opType) ? BOOL : dataType;

#define DO_TEST(DATA_TYPE, OUT_TYPE, HOST_DATA_TYPE, HOST_OUT_TYPE)            \
  if (dataType == DATA_TYPE && outputType == OUT_TYPE) {                       \
    return doBinaryOpTest<HOST_DATA_TYPE, HOST_OUT_TYPE>(                      \
               deviceType, dataType, outputType, desc1, desc2, shapeOut,       \
               tiles, mapLinearly, opType, inPlace, doReport, doPrintTensors,  \
               randomSeed, ignoreData, enableOptimisations)                    \
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
