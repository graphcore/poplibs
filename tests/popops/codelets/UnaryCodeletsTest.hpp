// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_UnaryCodeletsTest_hpp
#define popops_UnaryCodeletsTest_hpp

// Definitions/declarations used in elementwise unary operation test code.

#include "CodeletsTestsCommon.hpp"

using popops::expr::UnaryOpType;

// A map that, given a UnaryOpType, returns a string with its name
const std::map<UnaryOpType, const std::string> unaryOpToString = {
#define ONE_OP(opId)                                                           \
  { UnaryOpType::opId, #opId }
    ONE_OP(ABSOLUTE),
    ONE_OP(ASIN),
    ONE_OP(BITWISE_NOT),
    ONE_OP(CEIL),
    ONE_OP(COS),
    ONE_OP(COUNT_LEADING_ZEROS),
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

// Fills a vector with all operations
void setAllOps(std::vector<UnaryOpType> &ops) {
  for (auto &e : unaryOpToString) {
    ops.push_back(e.first);
  }
}

// Returns a string with all operations, comma separated. Used for the help
// message
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

// Definitions for an overloaded 'performOp' function to execute the operation
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

// Do the operation specified by 'op' on 'a' and 'b', where the operands are
// floating point types ('float' on the host, HALF or FLOAT on the device)
void performOp(UnaryOpType op, float a, float &result) {
  COMMON_OPS();
  ONE_OP(ABSOLUTE, std::abs(a));
  ONE_OP(ASIN, std::asin(a));
  ONE_OP(CEIL, std::ceil(a));
  ONE_OP(COS, std::cos(a));
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
template <typename T> void performOp(UnaryOpType op, T a, HostBool &result) {
  ONE_OP(IS_FINITE, std::isfinite(a))
  ONE_OP(IS_INF, std::isinf(a))
  ONE_OP(IS_NAN, a != a)
  ONE_OP(LOGICAL_NOT, !a);
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a boolean operator");
}

#undef COMMON_OPS
#undef ONE_OP

/// Check if two values are 'equal enough', for verification. This is the
/// overloaded for floating point type
bool equalValues(const bool isIpuModel, const UnaryOpType op,
                 const Type &dataType, float expected, float actual) {
  // For floating types there are some operators where we expect the result
  // from the device to be bit exact with the one from the host
  if ((dataType == FLOAT || dataType == HALF) &&
      (op == UnaryOpType::ABSOLUTE || op == UnaryOpType::CEIL ||
       op == UnaryOpType::FLOOR || op == UnaryOpType::RELU)) {
    return expected == actual;
  } else {

    double tolerance = 0.000001;

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

    bool isEqual = false;
    double delta = std::abs(expected - actual);
    if (expected == 0) {
      isEqual = delta < 10e-6;
    } else {
      delta = delta / expected;
      isEqual = (delta <= tolerance);
    }
    return isEqual;
  }
}

/// Check if two values are 'equal enough'.
/// For int/unsigned/boolean values, results must be bit exact
template <typename T>
bool equalValues(const bool isIpuModel, const UnaryOpType op,
                 const Type &dataType, T expected, T actual) {
  return expected == actual;
}

/// Fills the host buffer with values appropriate to the data type and the
/// operation being performed.
template <typename HOST_DATA_TYPE>
void fillHostBuffer(UnaryOpType op, const Type &dataType, unsigned randomSeed,
                    std::vector<HOST_DATA_TYPE> &buf) {
  bool nonZero = false;

  // Using a specific random generator means that we get the same random values
  // on different platforms.
  RandomEngine rndEng;
  if (randomSeed != 0 || dataType == BOOL)
    rndEng = std::minstd_rand(randomSeed);

  // For integer types we generate values in the closed interval [min, max]
  // For floating point types we generate values in the open interval [min, max)
  HOST_DATA_TYPE min = 0, max = 0;

  if (std::is_floating_point<HOST_DATA_TYPE>::value) {

    // For floating point, we limit the range
    HOST_DATA_TYPE absMax;
    absMax = (dataType == HALF) ? 300.0 : 32000.0;
    min = -absMax;
    max = absMax;

    if (op == UnaryOpType::TANH) {
      min = -20.0;
      max = 20.0;
    } else if (op == UnaryOpType::SIGMOID) {
      min = (dataType == HALF) ? -3.0 : -20.0;
      max = 20.0;
    } else if (op == UnaryOpType::LOGARITHM) {
      min = 0.1;
    } else if (op == UnaryOpType::LOGARITHM_ONE_PLUS) {
      min = -0.9;
    } else if (op == UnaryOpType::EXPONENT ||
               op == UnaryOpType::EXPONENT_MINUS_ONE) {
      min = (dataType == HALF) ? -10 : -80.0;
      max = 10.0;
    } else if (op == UnaryOpType::SQRT || op == UnaryOpType::RSQRT) {
      min = 0;
    } else if (op == UnaryOpType::ASIN) {
      min = -1;
      max = 1;
    }
  } else {
    // Non floating point case (INT, UNSIGNED). For BOOL these are ignored
    min = std::numeric_limits<HOST_DATA_TYPE>::min();
    max = std::numeric_limits<HOST_DATA_TYPE>::max();
  }

  fillBuffer(dataType, rndEng, buf, 10, min, max, nonZero);

  // If checking for infinities/nans, make sure we do have some infinities/nans
  if (op == UnaryOpType::IS_FINITE || op == UnaryOpType::IS_INF ||
      op == UnaryOpType::IS_NAN) {
    // We want to make 1 in 5 elements the same
    std::uniform_int_distribution<int> d(1, 5);
    float val = (op == UnaryOpType::IS_NAN)
                    ? std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::signaling_NaN();
    for (unsigned i = 0; i < buf.size(); i++) {
      if (d(*rndEng) == 1) {
        buf[i] = val;
      }
    }
  }
}

#endif // popops_UnaryCodeletsTest_hpp
