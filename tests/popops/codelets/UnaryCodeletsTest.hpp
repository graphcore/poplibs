// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_UnaryCodeletsTest_hpp
#define popops_UnaryCodeletsTest_hpp

// Definitions/declarations used in test code for element-wise unary or cast
// operation.

#include "CodeletsTestsCommon.hpp"
#include <poplibs_test/Util.hpp>
#include <poputil/exceptions.hpp>

#include <regex>
#include <sstream>
#include <tuple>
#include <variant>

using popops::expr::UnaryOpType;

// An 'Operation' is either a UnaryOpType op, or a cast to a data type
using Operation = std::variant<UnaryOpType, Type>;
bool isCast(const Operation &op) { return std::holds_alternative<Type>(op); }

// A map that, given a UnaryOpType, returns a string with its name
const std::map<UnaryOpType, const std::string> unaryOpToString = {
#define ONE_OP(opId)                                                           \
  { UnaryOpType::opId, #opId }
    ONE_OP(ABSOLUTE),
    ONE_OP(ASIN),
    ONE_OP(BITWISE_NOT),
    ONE_OP(CBRT),
    ONE_OP(CEIL),
    ONE_OP(COS),
    ONE_OP(COUNT_LEADING_ZEROS),
    ONE_OP(ERF),
    ONE_OP(EXPONENT),
    ONE_OP(EXPONENT_MINUS_ONE),
    ONE_OP(FLOOR),
    ONE_OP(INVERSE),
    ONE_OP(IS_FINITE),
    ONE_OP(IS_INF),
    ONE_OP(IS_NAN),
    ONE_OP(LOGARITHM),
    ONE_OP(LOGARITHM_ONE_PLUS),
    ONE_OP(LOGICAL_NOT),
    ONE_OP(NEGATE),
    ONE_OP(POPCOUNT),
    ONE_OP(RELU),
    ONE_OP(SIGNUM),
    ONE_OP(SIN),
    ONE_OP(TAN),
    ONE_OP(TANH),
    ONE_OP(ROUND),
    ONE_OP(SQRT),
    ONE_OP(SQUARE),
    ONE_OP(SIGMOID),
    ONE_OP(RSQRT),
#undef ONE_OP
};

// Given a string, returns the corresponding UnaryOpType. The string can be
// uppercase or lowercase and be just the start of the name of the operator, as
// long as it uniquely identifies it (i.e. "as" ==> ASIN, but "exp" is
// not enough, as we have both EXPONENT and EXPONENT_MINUS_ONE.
// But note that if the string matches *exactly* the name, then it is considered
// valid; for instance, "logarithm" => LOGARITHM, even if we have
// LOGARITHM_ONE_PLUS
UnaryOpType stringToUnaryOp(const std::string &s) {
  std::optional<UnaryOpType> opFound(std::nullopt);
  std::string us = s;
  boost::to_upper(us);
  for (auto &e : unaryOpToString) {
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
                                 "poplar::expr::UnaryOpType");
      } else {
        opFound = op;
      }
    }
  }
  if (opFound)
    return *opFound;
  else
    throw std::runtime_error("<" + s +
                             "> is not a valid poplar::expr::UnaryOpType");
}

// Fills a vector with all UnaryOpType operations
void setAllOps(std::vector<UnaryOpType> &ops) {
  for (auto &e : unaryOpToString) {
    ops.push_back(e.first);
  }
}

// Returns a string with all UnaryOpType operations, comma separated. Used for
// the help message
const std::string allOpsStr() {
  std::vector<std::string> ops;
  for (auto &e : unaryOpToString) {
    ops.push_back(e.second);
  }
  sort(ops.begin(), ops.end());
  std::string result = ops[0];
  std::for_each(ops.begin() + 1, ops.end(),
                [&](auto &s) { result += ", " + s; });
  return result;
}

// Does 'op' one returns booleans?
bool isBoolOp(UnaryOpType op) {
  switch (op) {
  case UnaryOpType::LOGICAL_NOT:
  case UnaryOpType::IS_FINITE:
  case UnaryOpType::IS_INF:
  case UnaryOpType::IS_NAN:
    return true;
  default:
    return false;
  }
}

// Is 'op' one of the operators that can only take integer operands?
bool isIntOp(UnaryOpType op) {
  switch (op) {
  case UnaryOpType::BITWISE_NOT:
  case UnaryOpType::COUNT_LEADING_ZEROS:
  case UnaryOpType::POPCOUNT:
    return true;
  default:
    return false;
  }
}

// Given a string that describes a UnaryOpType or a cast, returns a tuple
// of [Operation, outputType]
std::tuple<Operation, Type> stringToOperation(const std::string &str,
                                              Type dataType) {
  Type outputType;
  Operation operation;
  try {
    std::istringstream is(str);
    std::regex castRE("cast<([^)]+)>");
    std::smatch m;
    if (std::regex_search(str, m, castRE)) {
      is.str(m[1]);
    }
    is >> outputType;
    operation = outputType;
  } catch (poputil::poplibs_error &e) {
    UnaryOpType opType = stringToUnaryOp(str);
    operation = opType;
    outputType = isBoolOp(opType) ? BOOL : dataType;
  }
  return {operation, outputType};
}

// Definitions for an overloaded 'performOp' function to execute a UnaryOpType
// on the host (for verification). Note that the result value is a function
// parameter (passed by reference) and not the return value of the function, to
// allow for full overload resolution (return type is not used for overload
// resolution in C++).

// A macro to reduce to the minimum the typing to define each operation
#define ONE_OP(opId, expr)                                                     \
  if (op == UnaryOpType::opId) {                                               \
    result = expr;                                                             \
    return;                                                                    \
  }

// These operations are common to floating point and integer types.
#define COMMON_OPS() ONE_OP(SQUARE, a *a);

// These operations are common to INT and UNSIGNED_INT types
#define INTEGER_OPS()                                                          \
  ONE_OP(BITWISE_NOT, ~a);                                                     \
  ONE_OP(COUNT_LEADING_ZEROS, __builtin_clz((unsigned)a));                     \
  ONE_OP(POPCOUNT, __builtin_popcount((unsigned)a));                           \
  COMMON_OPS();

#define LONGLONG_OPS() ONE_OP(BITWISE_NOT, ~a);

// Do the operation specified by 'op' on 'a' and 'b', where the operands are
// floating point types ('float' on the host, HALF or FLOAT on the device)
void performOp(UnaryOpType op, float a, float &result) {
  COMMON_OPS();
  ONE_OP(ABSOLUTE, std::abs(a));
  ONE_OP(ASIN, std::asin(a));
  ONE_OP(CBRT, std::cbrt(a));
  ONE_OP(CEIL, std::ceil(a));
  ONE_OP(COS, std::cos(a));
  ONE_OP(ERF, std::erf(a));
  ONE_OP(EXPONENT, std::exp(a));
  ONE_OP(EXPONENT_MINUS_ONE, std::expm1(a));
  ONE_OP(FLOOR, std::floor(a));
  ONE_OP(INVERSE, 1.0 / a);
  ONE_OP(LOGARITHM, std::log(a));
  ONE_OP(LOGARITHM_ONE_PLUS, std::log1p(a));
  ONE_OP(NEGATE, -a);
  ONE_OP(RELU, a > 0 ? a : 0);
  ONE_OP(RSQRT, 1.0 / std::sqrt(a));
  ONE_OP(ROUND, std::round(a))
  ONE_OP(SIGNUM, (0 < a) - (a < 0));
  ONE_OP(SIGMOID, (std::tanh(a * 0.5) + 1) * 0.5);
  ONE_OP(SIN, std::sin(a));
  ONE_OP(SQRT, std::sqrt(a));
  ONE_OP(TAN, std::tan(a));
  ONE_OP(TANH, std::tanh(a));
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for floating point types");
}

// Do the operation specified by 'op' on 'a', where 'a' is of various integer
// types.

void performOp(UnaryOpType op, int a, int &result) {
  ONE_OP(ABSOLUTE, std::abs(a));
  ONE_OP(NEGATE, -a);
  ONE_OP(SIGNUM, (0 < a) - (a < 0));
  ONE_OP(SQRT, (int)(std::sqrt(a)));
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for signed integer");
}

void performOp(UnaryOpType op, long long a, long long &result) {
  ONE_OP(ABSOLUTE, std::abs(a));
  ONE_OP(NEGATE, -a);
  LONGLONG_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for signed long long");
}

void performOp(UnaryOpType op, unsigned long long a,
               unsigned long long &result) {
  LONGLONG_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for unsigned long long");
}

void performOp(UnaryOpType op, unsigned a, unsigned &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for unsigned integer");
}

void performOp(UnaryOpType op, short a, short &result) {
  ONE_OP(BITWISE_NOT, ~a);
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for signed short");
}

void performOp(UnaryOpType op, unsigned short a, unsigned short &result) {
  ONE_OP(BITWISE_NOT, ~a);
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for unsigned short");
}

// Do the operation specified by 'op' on 'a' and 'b' where the 'op' is one of
// the operators that return a boolean. This works for floating point, integer
// and boolean data types.
template <typename T>
void performOp(UnaryOpType op, T a, unsigned char &result) {
  if constexpr (std::is_floating_point<T>::value) {
    ONE_OP(IS_FINITE, std::isfinite(a))
    ONE_OP(IS_INF, std::isinf(a))
    ONE_OP(IS_NAN, a != a)
  }
  ONE_OP(LOGICAL_NOT, !a);
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a boolean operator");
}

template <typename T> void performOp(UnaryOpType op, T a, char &result) {
  // Currently there are no operations that produce a 'char' result.
  throw std::logic_error(
      "performOp(UnaryOpType,T, char) should never be called");
}
template <typename T> void performOp(UnaryOpType op, T a, signed char &result) {
  // Currently there are no operations that produce a 'signed char' result.
  throw std::logic_error(
      "performOp(UnaryOpType,T, signed char) should never be called");
}
#undef COMMON_OPS
#undef ONE_OP

// Execute a cast from devSrcType to devDstType on the host for verification.
// For (float,half) => integer-types the rounding mode should have been set
// appropriately before calling this function.
template <typename HostSrcType, typename HostDstType>
void performCast(const bool isIpuModel, const HostSrcType a,
                 HostDstType &result, const Type devSrcType,
                 const Type devDstType) {
  if (devDstType == BOOL) {
    result = (a != 0);
  } else {
    result = static_cast<HostDstType>(a);
  }
}

/// Check if two values are 'equal enough', for verification. This is the
/// overloaded version for floating point type
bool equalValues(const bool isIpuModel, const Operation op,
                 const Type &dataType, float expected, float actual) {
  double tolerance = 0.000001;

  if (isCast(op)) {
    Type outType = std::get<Type>(op);
    if (dataType == FLOAT && outType == HALF) {
      tolerance = 0.003;
    } else {
      return (actual == expected);
    }
  } else {
    UnaryOpType unaryOp = std::get<UnaryOpType>(op);
    // For floating types there are some operators where we expect the result
    // from the device to be bit exact with the one from the host
    if ((dataType == FLOAT || dataType == HALF) &&
        (unaryOp == UnaryOpType::ABSOLUTE || unaryOp == UnaryOpType::CEIL ||
         unaryOp == UnaryOpType::FLOOR || unaryOp == UnaryOpType::RELU)) {
      return expected == actual;
    } else {

      // Horrible contortions to verify result for halves. We should really
      // have a half bit-exact computation library for the host.
      if (dataType == HALF) {
        tolerance = 0.003;
        float clipTreshHalf =
            (isIpuModel) ? std::numeric_limits<float>::infinity() : 65504.0f;
        float clipValueHalf = 65488.0f;
        if (actual >= clipTreshHalf) {
          return expected >= clipValueHalf;
        } else if (actual <= -clipTreshHalf) {
          return expected <= -clipValueHalf;
        }
      }
    }
  }
  bool isEqual = false;
  double delta = expected - actual;
  if (expected == 0) {
    isEqual = std::abs(delta) < 10e-6;
  } else {
    delta = std::abs(delta / expected);
    isEqual = (delta <= tolerance);
  }
  return isEqual;
}

/// Check if two values are 'equal enough'.
/// For int/unsigned/boolean values, results must be bit exact ...
template <typename T>
bool equalValues(const bool isIpuModel, const Operation op,
                 const Type &dataType, T expected, T actual) {
  // ... or almost bit exact. The square root shows a discrepancy of
  // 1 sometimes.
  if constexpr (std::is_integral<T>::value) {
    if (!isCast(op) && std::get<UnaryOpType>(op) == UnaryOpType::SQRT)
      return std::abs((int)(expected - actual)) <= 1;
  }
  return expected == actual;
}

/// Fills the host buffer with values appropriate to the data type and the
/// operation/cast being performed.
template <typename HostDataType>
void fillHostBuffer(Operation op, const Type &dataType, unsigned randomSeed,
                    std::vector<HostDataType> &buf) {
  bool nonZero = false;

  // Using a specific random generator means that we get the same random values
  // on different platforms.
  RandomEngine rndEng;
  if (randomSeed != 0 || dataType == BOOL)
    rndEng = std::minstd_rand(randomSeed);

  // For integer types we generate values in the closed interval [min, max]
  // For floating point types we generate values in the open interval [min, max)
  HostDataType min = 0, max = 0;

  if (isCast(op)) {
    bool isSrcUnsigned =
        (dataType == UNSIGNED_CHAR || dataType == UNSIGNED_SHORT ||
         dataType == UNSIGNED_INT);
    Type dstType = std::get<Type>(op);
    bool isDstUnsigned = (dstType == UNSIGNED_CHAR ||
                          dstType == UNSIGNED_SHORT || dstType == UNSIGNED_INT);

    min = (isSrcUnsigned || isDstUnsigned) ? 0 : -128;
    max = (isSrcUnsigned || isDstUnsigned) ? 255 : 127;
  } else {
    UnaryOpType unaryOp = std::get<UnaryOpType>(op);
    if constexpr (std::is_floating_point<HostDataType>::value) {
      // For floating point, we limit the range
      HostDataType absMax;
      absMax = (dataType == HALF) ? 254.0 : 32000.0;
      min = -absMax;
      max = absMax;

      if (unaryOp == UnaryOpType::TAN) {
        min = -1.2;
        max = 1.2;
      } else if (unaryOp == UnaryOpType::SIN || unaryOp == UnaryOpType::COS) {
        min = -M_PI;
        max = M_PI;
      } else if (unaryOp == UnaryOpType::TANH) {
        min = (dataType == FLOAT) ? 0.02 : -20.0;
        max = (dataType == FLOAT) ? 8 : 20.0;
      } else if (unaryOp == UnaryOpType::SIGMOID) {
        min = (dataType == HALF) ? -3.0 : 0.55;
        max = (dataType == HALF) ? 3.0 : 17.0;
      } else if (unaryOp == UnaryOpType::LOGARITHM) {
        min = 0.01;
        max = 0.7;
      } else if (unaryOp == UnaryOpType::LOGARITHM_ONE_PLUS) {
        min = -0.9;
      } else if (unaryOp == UnaryOpType::EXPONENT ||
                 unaryOp == UnaryOpType::EXPONENT_MINUS_ONE) {
        // These limits also guarantee that the instructions have the highest
        // latency
        min = (dataType == HALF) ? -10 : -80.0;
        max = 10.0;
        nonZero = true;
      } else if (unaryOp == UnaryOpType::SQRT ||
                 unaryOp == UnaryOpType::RSQRT) {
        min = 0.1;
      } else if (unaryOp == UnaryOpType::ASIN) {
        min = -1;
        max = 1;
      } else if (unaryOp == UnaryOpType::SQUARE) {
        if (dataType == HALF)
          min = 0.01;
      }
    } else {
      // Non floating point case (INT, UNSIGNED, BOOL).
      // Note that for BOOL we ignore max, min.
      min = (unaryOp == UnaryOpType::SQRT)
                ? 0
                : std::numeric_limits<HostDataType>::min();
      max = std::numeric_limits<HostDataType>::max();
    }
  } // not a cast

  fillBuffer(dataType, rndEng, buf, 10, min, max, nonZero);

  // If checking for infinities/nans, make sure we do have some infinities/nans
  if constexpr (std::is_floating_point<HostDataType>::value) {
    if (!isCast(op)) {
      UnaryOpType unaryOp = std::get<UnaryOpType>(op);
      if (unaryOp == UnaryOpType::IS_FINITE || unaryOp == UnaryOpType::IS_INF ||
          unaryOp == UnaryOpType::IS_NAN) {
        // We want to make 1 in 5 elements to be a 'special value' (inf or NaN)
        std::uniform_int_distribution<int> d1(1, 5);
        // and we have 4 different types of 'special values'
        static std::array<float, 4> specials = {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::signaling_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};
        std::uniform_int_distribution<int> d2(1, specials.size());
        for (unsigned i = 0; i < buf.size(); i++) {
          if (d1(*rndEng) == 1) {
            buf[i] = specials[d2(*rndEng)];
          }
        }
      }
    }
  }
}

// Macros to help selecting and calling the appropriate templated function,
// based on the runtime value of two poplar::Type variables (srcType, dstType)
// that specify the type of input and output tensor for UnaryOp and Cast
// vertices.
// The code using this macros must first define another macro like the
// following:
//
// #define SELECT_ONE(IPU_SRC_TYPE, IPU_DST_TYPE, HOST_SRC_TYPE, HOST_DST_TYPE)
//           if (srcType == IPU_SRC_TYPE && dstType == IPU_DST_TYPE)
//              func<HOST_DATA_TYPE, HOST_OUT_TYPE>(srcType, dstType, ...)
//
// and then invoke the SELECT_BY_TYPES macro, followed by the raising of an
// exception to indicate that no match has been found:
//
//   SELECT_BY_TYPES()
//   throw invalid_types(dataType, outputType);
//
// The type pairs defined here must include all pairs valid for UnaryOpType
// and Cast operations
//
#define SELECT_BY_SRC_TYPE(IPU_SRC_TYPE, HOST_SRC_TYPE)                        \
  SELECT_ONE(IPU_SRC_TYPE, FLOAT, HOST_SRC_TYPE, float)                        \
  SELECT_ONE(IPU_SRC_TYPE, HALF, HOST_SRC_TYPE, float)                         \
  SELECT_ONE(IPU_SRC_TYPE, INT, HOST_SRC_TYPE, int)                            \
  SELECT_ONE(IPU_SRC_TYPE, LONGLONG, HOST_SRC_TYPE, long long)                 \
  SELECT_ONE(IPU_SRC_TYPE, UNSIGNED_INT, HOST_SRC_TYPE, unsigned int)          \
  SELECT_ONE(IPU_SRC_TYPE, UNSIGNED_LONGLONG, HOST_SRC_TYPE,                   \
             unsigned long long)                                               \
  SELECT_ONE(IPU_SRC_TYPE, SHORT, HOST_SRC_TYPE, short)                        \
  SELECT_ONE(IPU_SRC_TYPE, UNSIGNED_SHORT, HOST_SRC_TYPE, unsigned short)      \
  SELECT_ONE(IPU_SRC_TYPE, BOOL, HOST_SRC_TYPE, unsigned char)                 \
  SELECT_ONE(IPU_SRC_TYPE, CHAR, HOST_SRC_TYPE, char)                          \
  SELECT_ONE(IPU_SRC_TYPE, SIGNED_CHAR, HOST_SRC_TYPE, signed char)            \
  SELECT_ONE(IPU_SRC_TYPE, UNSIGNED_CHAR, HOST_SRC_TYPE, unsigned char)

#define SELECT_BY_TYPES()                                                      \
  SELECT_BY_SRC_TYPE(FLOAT, float)                                             \
  SELECT_BY_SRC_TYPE(HALF, float)                                              \
  SELECT_BY_SRC_TYPE(INT, int)                                                 \
  SELECT_BY_SRC_TYPE(LONGLONG, long long)                                      \
  SELECT_BY_SRC_TYPE(UNSIGNED_INT, unsigned int)                               \
  SELECT_BY_SRC_TYPE(UNSIGNED_LONGLONG, unsigned long long)                    \
  SELECT_BY_SRC_TYPE(SHORT, short)                                             \
  SELECT_BY_SRC_TYPE(UNSIGNED_SHORT, unsigned short)                           \
  SELECT_BY_SRC_TYPE(BOOL, unsigned char)                                      \
  SELECT_BY_SRC_TYPE(CHAR, char)                                               \
  SELECT_BY_SRC_TYPE(SIGNED_CHAR, signed char)                                 \
  SELECT_BY_SRC_TYPE(UNSIGNED_CHAR, unsigned char)

//*************************************************************************
// Contains information relative to a single test (one specific vertex,
// and one SizeDesc value) for UnaryOp or Cast
template <typename VertexDesc> struct TestRecord {
  SizeDesc size;
  std::unique_ptr<VertexDesc> vertex;

  TestOperand in;
  TestOperand out;

  // Stream names used to transfer the host data and the output. Must be
  // different for each test that is run in the same graph/compute set.
  std::string writeName;
  std::string readName;

  // Is the output buffer padding made up of Nan values?
  bool padOutWithNan = false;

  /// \param[in] v      The vertex (with operation and data type) to test.
  /// \param[in] seq    A sequential index, different for each test
  /// \param[in] tSizes The data sizes to use for the test.
  TestRecord(std::unique_ptr<VertexDesc> v, unsigned seq, const SizeDesc &sz)
      : vertex(std::move(v)) {
    writeName = vertex->inName + "_" + to_string(seq);
    readName = vertex->outName + "_" + to_string(seq);
    size = sz.adjust(vertex->is2D);
  }
  TestRecord(TestRecord &&) = default;

  std::string toString() { return vertex->vClassFmt + size.toString(); }
};

#endif // popops_UnaryCodeletsTest_hpp
