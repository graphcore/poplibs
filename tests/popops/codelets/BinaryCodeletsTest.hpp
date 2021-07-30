// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_codelets_BinaryCodeletsTest_hpp
#define popops_codelets_BinaryCodeletsTest_hpp

// Definitions/declarations used in elementwise binary operation test code.

#include "CodeletsTestsCommon.hpp"

using popops::expr::BinaryOpType;

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
bool isBoolOp(BinaryOpType op) {
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

// These operations are common to INT, UNSIGNED_INT, LONGLONG and
// UNSIGNED_LONGLONG types
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
// of various integer types.

void performOp(BinaryOpType op, int a, int b, int &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for signed integer");
}

void performOp(BinaryOpType op, unsigned a, unsigned b, unsigned &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for signed integer");
}

void performOp(BinaryOpType op, long long a, long long b, long long &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for signed integer");
}

void performOp(BinaryOpType op, unsigned long long a, unsigned long long b,
               unsigned long long &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for unsigned integer");
}

void performOp(BinaryOpType op, short a, short b, short &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for signed short");
}

void performOp(BinaryOpType op, unsigned short a, unsigned short b,
               unsigned short &result) {
  INTEGER_OPS();
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a valid operator for unsigned short");
}

// Do the operation specified by 'op' on 'a' and 'b' where the 'op' is one of
// the operators that return a boolean. This works for floating point, integer
// and boolean data types.
template <typename T>
void performOp(BinaryOpType op, T a, T b, unsigned char &result) {
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

#define ONE_OP_OUTX2(opId, expr)                                               \
  if (op == BinaryOpType::opId) {                                              \
    auto val = expr;                                                           \
    return {val, 1.0 - val};                                                   \
  }

// Do the operation specified by 'op' on 'a' and 'b' where the 'op' is one of
// the operators that return a boolean. This works for floating point, integer
// and boolean data types.
template <typename T>
std::pair<float, float> performOp(BinaryOpType op, T a, T b) {
  ONE_OP_OUTX2(EQUAL, a == b);
  ONE_OP_OUTX2(GREATER_THAN_EQUAL, a >= b);
  ONE_OP_OUTX2(GREATER_THAN, a > b);
  ONE_OP_OUTX2(LESS_THAN_EQUAL, a <= b);
  ONE_OP_OUTX2(LESS_THAN, a < b);
  ONE_OP_OUTX2(LOGICAL_AND, a && b);
  ONE_OP_OUTX2(LOGICAL_OR, a || b);
  ONE_OP_OUTX2(NOT_EQUAL, a != b);
  throw std::logic_error(std::to_string(unsigned(op)) +
                         " is not a boolean operator");
}

#undef COMMON_OPS
#undef ONE_OP

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

      tolerance = 0.003;

      if (op == BinaryOpType::DIVIDE || op == BinaryOpType::ATAN2) {
        if (std::abs(expected) < 1e-5) {
          tolerance = 0.02;
        } else if (std::abs(expected) < 1e-4) {
          tolerance = 0.01;
        }
      } else if (op == BinaryOpType::POWER) {
        // POWER in half is not very precise
        tolerance = 0.012;
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
}

/// Check if two values are 'equal enough'.
/// For int/unsigned/boolean values, results must be bit exact
template <typename T>
bool equalValues(const bool isIpuModel, const BinaryOpType op,
                 const Type &dataType, T expected, T actual) {
  return expected == actual;
}

//*************************************************************************
/// Fills the host buffers with values appropriate to the data type and the
/// operation being performed.
template <typename HostDataType>
void fillHostBuffers(BinaryOpType op, const Type &dataType, unsigned randomSeed,
                     std::vector<HostDataType> &buf1,
                     std::vector<HostDataType> &buf2) {
  bool nonZero = false;

  // Using a specific random generator means that we get the same random values
  // on different platforms.
  RandomEngine rndEng;
  if (randomSeed != 0 || dataType == BOOL)
    rndEng = std::minstd_rand(randomSeed);

  // For integer types we generate values in the closed interval [min, max]
  // For floating point types we generate values in the open interval [min, max)
  HostDataType min1 = 0, max1 = 0;
  HostDataType min2 = 0, max2 = 0;

  if constexpr (std::is_floating_point<HostDataType>::value) {

    // For floating point, we limit the range
    HostDataType absMax;
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
      // In case of HALF,ADD we make both value positive, while for HALF,
      // SUBTRACT the first is positive and the second negative, because
      // subtracting two values that are very similar can gives results that
      // are very inaccurate [for instance (1.0) + (-1.00000001)].
      if (op == BinaryOpType::ADD) {
        min1 = min2 = 0;
      } else if (op == BinaryOpType::SUBTRACT) {
        min1 = max2 = 0;
      } else if (op == BinaryOpType::DIVIDE) {
        min1 = 0.01;
        // In case of HALF,DIVIDE we must avoid overflow, so we choose a
        // limited range for the divisor
        min2 = 0.7;
        max2 = 600;
      } else if (op == BinaryOpType::ATAN2) {
        min1 = 0.01;
      }
    }

    // VARIANCE_TO_INV_STD_DEV is: 1/sqrt(a+b), so (a+b) must be non-negative.
    // To simplify, we force both operands to be non-negative
    if (op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) {
      min1 = min2 = 0;
    }
  } else {
    // Non floating point case (INT, UNSIGNED)./ For BOOL these are ignored
    min1 = min2 = std::numeric_limits<HostDataType>::min();
    max1 = max2 = std::numeric_limits<HostDataType>::max();
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
  if (std::is_same<HostDataType, int>::value ||
      std::is_same<HostDataType, unsigned>::value) {
    if (op == BinaryOpType::DIVIDE || op == BinaryOpType::REMAINDER) {
      min2 = (dataType == UNSIGNED_INT || dataType == UNSIGNED_LONGLONG)
                 ? 0
                 : -32767;
      max2 = 32767;
    }
  }

  // These operations must have a second operand != 0
  if (op == BinaryOpType::DIVIDE || op == BinaryOpType::ATAN2 ||
      op == BinaryOpType::REMAINDER) {
    nonZero = true;
  }

  fillBuffer(dataType, rndEng, buf1, 100, min1, max1, nonZero);
  fillBuffer(dataType, rndEng, buf2, 255, min2, max2, nonZero);

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

#endif // popops_codelets_BinaryCodeletsTest_hpp
