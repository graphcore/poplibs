#include "ExprOpUtil.hpp"
#include "poputil/exceptions.hpp"

namespace popops {
namespace expr {

std::string unaryOpTypeToString(UnaryOpType op) {
  switch (op) {
  case UnaryOpType::ABSOLUTE:
    return "ABSOLUTE";
  case UnaryOpType::BITWISE_NOT:
    return "BITWISE_NOT";
  case UnaryOpType::CEIL:
    return "CEIL";
  case UnaryOpType::COS:
    return "COS";
  case UnaryOpType::COUNT_LEADING_ZEROS:
    return "COUNT_LEADING_ZEROS";
  case UnaryOpType::EXPONENT:
    return "EXPONENT";
  case UnaryOpType::EXPONENT_MINUS_ONE:
    return "EXPONENT_MINUS_ONE";
  case UnaryOpType::FLOOR:
    return "FLOOR";
  case UnaryOpType::INVERSE:
    return "INVERSE";
  case UnaryOpType::IS_FINITE:
    return "IS_FINITE";
  case UnaryOpType::LOGARITHM:
    return "LOGARITHM";
  case UnaryOpType::LOGARITHM_ONE_PLUS:
    return "LOGARITHM_ONE_PLUS";
  case UnaryOpType::LOGICAL_NOT:
    return "LOGICAL_NOT";
  case UnaryOpType::NEGATE:
    return "NEGATE";
  case UnaryOpType::POPCOUNT:
    return "POPCOUNT";
  case UnaryOpType::ROUND:
    return "ROUND";
  case UnaryOpType::SIGNUM:
    return "SIGNUM";
  case UnaryOpType::SIN:
    return "SIN";
  case UnaryOpType::TANH:
    return "TANH";
  case UnaryOpType::SQRT:
    return "SQRT";
  case UnaryOpType::SQUARE:
    return "SQUARE";
  case UnaryOpType::SIGMOID:
    return "SIGMOID";
  case UnaryOpType::RSQRT:
    return "RSQRT";
  }
  throw poputil::poplibs_error("Op not supported");
}

std::string binaryOpTypeToString(BinaryOpType op) {
  switch (op) {
  case BinaryOpType::ADD:
    return "ADD";
  case BinaryOpType::ATAN2:
    return "ATAN2";
  case BinaryOpType::BITWISE_AND:
    return "BITWISE_AND";
  case BinaryOpType::BITWISE_OR:
    return "BITWISE_OR";
  case BinaryOpType::BITWISE_XOR:
    return "BITWISE_XOR";
  case BinaryOpType::BITWISE_XNOR:
    return "BITWISE_XNOR";
  case BinaryOpType::DIVIDE:
    return "DIVIDE";
  case BinaryOpType::EQUAL:
    return "EQUAL";
  case BinaryOpType::GREATER_THAN_EQUAL:
    return "GREATER_THAN_EQUAL";
  case BinaryOpType::GREATER_THAN:
    return "GREATER_THAN";
  case BinaryOpType::LESS_THAN_EQUAL:
    return "LESS_THAN_EQUAL";
  case BinaryOpType::LOGICAL_AND:
    return "LOGICAL_AND";
  case BinaryOpType::LOGICAL_OR:
    return "LOGICAL_OR";
  case BinaryOpType::LESS_THAN:
    return "LESS_THAN";
  case BinaryOpType::MAXIMUM:
    return "MAXIMUM";
  case BinaryOpType::MINIMUM:
    return "MINIMUM";
  case BinaryOpType::MULTIPLY:
    return "MULTIPLY";
  case BinaryOpType::NOT_EQUAL:
    return "NOT_EQUAL";
  case BinaryOpType::POWER:
    return "POWER";
  case BinaryOpType::REMAINDER:
    return "REMAINDER";
  case BinaryOpType::SHIFT_LEFT:
    return "SHIFT_LEFT";
  case BinaryOpType::SHIFT_RIGHT:
    return "SHIFT_RIGHT";
  case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
    return "SHIFT_RIGHT_SIGN_EXTEND";
  case BinaryOpType::SUBTRACT:
    return "SUBTRACT";
  }
  throw poputil::poplibs_error("Op not supported");
}

std::string broadcastOpTypeToString(BroadcastOpType op) {
  switch (op) {
  case BroadcastOpType::ADD:
    return "ADD";
  case BroadcastOpType::SCALED_ADD:
    return "SCALED_ADD";
  case BroadcastOpType::INV_STD_DEV_TO_VARIANCE:
    return "INV_STD_DEV_TO_VARIANCE";
  case BroadcastOpType::MULTIPLY:
    return "MULTIPLY";
  case BroadcastOpType::SUBTRACT:
    return "SUBTRACT";
  case BroadcastOpType::VARIANCE_TO_INV_STD_DEV:
    return "VARIANCE_TO_INV_STD_DEV";
  }
  throw poputil::poplibs_error("Op not supported");
}

bool isSpecialCase(BinaryOpType op) {
  return op == BinaryOpType::VARIANCE_TO_INV_STD_DEV ||
         op == BinaryOpType::INV_STD_DEV_TO_VARIANCE ||
         op == BinaryOpType::BITWISE_XNOR || op == BinaryOpType::SHIFT_RIGHT;
}

std::string handleSpecialCase(BinaryOpType op, const std::string &param1,
                              const std::string &param2) {
  assert(isSpecialCase(op) && "Binary operation is not a special case");
  switch (op) {
  case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
    return "internal_rsqrt(" + param1 + "+" + param2 + ")";
  case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
    return "1/(" + param1 + "*" + param1 + ") - " + param2;
  case BinaryOpType::BITWISE_XNOR:
    return "~(" + param1 + "^" + param2 + ")";
  case BinaryOpType::SHIFT_RIGHT:
    return "(unsigned)" + param1 + " >> (unsigned)" + param2;
  default:
    throw poputil::poplibs_error("Binary operation not a special case.");
  }
}

bool hasFunctionSemantics(BinaryOpType op) {
  return op == BinaryOpType::ATAN2 || op == BinaryOpType::MAXIMUM ||
         op == BinaryOpType::MINIMUM || op == BinaryOpType::POWER ||
         op == BinaryOpType::REMAINDER;
}

poplar::Type getReturnType(BinaryOpType op,
                           const std::pair<std::string, poplar::Type> &lhs,
                           const std::pair<std::string, poplar::Type> &rhs) {
  switch (op) {
  case BinaryOpType::EQUAL:
  case BinaryOpType::GREATER_THAN_EQUAL:
  case BinaryOpType::GREATER_THAN:
  case BinaryOpType::LESS_THAN_EQUAL:
  case BinaryOpType::LOGICAL_AND:
  case BinaryOpType::LOGICAL_OR:
  case BinaryOpType::LESS_THAN:
  case BinaryOpType::NOT_EQUAL:
    return poplar::BOOL;
  case BinaryOpType::ADD:
  case BinaryOpType::ATAN2:
  case BinaryOpType::DIVIDE:
  case BinaryOpType::BITWISE_AND:
  case BinaryOpType::BITWISE_OR:
  case BinaryOpType::BITWISE_XOR:
  case BinaryOpType::BITWISE_XNOR:
  case BinaryOpType::MAXIMUM:
  case BinaryOpType::MINIMUM:
  case BinaryOpType::MULTIPLY:
  case BinaryOpType::POWER:
  case BinaryOpType::REMAINDER:
  case BinaryOpType::SUBTRACT:
  case BinaryOpType::SHIFT_LEFT:
  case BinaryOpType::SHIFT_RIGHT:
  case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
  case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
  case BinaryOpType::INV_STD_DEV_TO_VARIANCE: {

    bool lhsIsConst = lhs.first[0] == 'C';
    if (lhsIsConst) {
      return rhs.second;
    }
    return lhs.second;
  }
  default:
    throw poputil::poplibs_error(
        "Binary Op type is unrecognised (getReturnType)");
  }
}

bool isBitwiseOperation(BinaryOpType op) {
  switch (op) {
  case BinaryOpType::BITWISE_AND:
  case BinaryOpType::BITWISE_OR:
  case BinaryOpType::BITWISE_XOR:
  case BinaryOpType::SHIFT_LEFT:
  case BinaryOpType::SHIFT_RIGHT:
  case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
  case BinaryOpType::BITWISE_XNOR:
    return true;
  case BinaryOpType::ADD:
  case BinaryOpType::ATAN2:
  case BinaryOpType::DIVIDE:
  case BinaryOpType::EQUAL:
  case BinaryOpType::GREATER_THAN_EQUAL:
  case BinaryOpType::GREATER_THAN:
  case BinaryOpType::LESS_THAN_EQUAL:
  case BinaryOpType::LOGICAL_AND:
  case BinaryOpType::LOGICAL_OR:
  case BinaryOpType::LESS_THAN:
  case BinaryOpType::MAXIMUM:
  case BinaryOpType::MINIMUM:
  case BinaryOpType::MULTIPLY:
  case BinaryOpType::NOT_EQUAL:
  case BinaryOpType::POWER:
  case BinaryOpType::REMAINDER:
  case BinaryOpType::SUBTRACT:
  case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
  case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
    return true;
  default:
    throw poputil::poplibs_error(
        "Binary Op type is unrecognised (isBitwiseOperation)");
  }
}

poplar::StringRef getBinaryOpAsString(BinaryOpType op, poplar::Type t) {
  bool isFloatingPoint = t == poplar::HALF || t == poplar::FLOAT;
  switch (op) {
  case BinaryOpType::ADD:
    return "+";
  case BinaryOpType::ATAN2:
    return "atan2";
  case BinaryOpType::BITWISE_AND:
    return "&";
  case BinaryOpType::BITWISE_OR:
    return "|";
  case BinaryOpType::BITWISE_XOR:
    return "^";
  case BinaryOpType::DIVIDE:
    return "/";
  case BinaryOpType::EQUAL:
    return "==";
  case BinaryOpType::GREATER_THAN_EQUAL:
    return ">=";
  case BinaryOpType::GREATER_THAN:
    return ">";
  case BinaryOpType::LESS_THAN_EQUAL:
    return "<=";
  case BinaryOpType::LOGICAL_AND:
    return "&&";
  case BinaryOpType::LOGICAL_OR:
    return "||";
  case BinaryOpType::LESS_THAN:
    return "<";
  case BinaryOpType::MAXIMUM:
    return isFloatingPoint ? "NAMESPACE::fmax" : "max";
  case BinaryOpType::MINIMUM:
    return isFloatingPoint ? "NAMESPACE::fmin" : "min";
  case BinaryOpType::MULTIPLY:
    return "*";
  case BinaryOpType::NOT_EQUAL:
    return "!=";
  case BinaryOpType::POWER:
    return "NAMESPACE::pow";
  case BinaryOpType::REMAINDER:
    return "internal_remainder";
  case BinaryOpType::SHIFT_LEFT:
    return "<<";
  case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
    return ">>";
  case BinaryOpType::SUBTRACT:
    return "-";
  case BinaryOpType::SHIFT_RIGHT:
  case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
  case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
  case BinaryOpType::BITWISE_XNOR:
    // Special case.
    return "";
  default:
    throw poputil::poplibs_error(
        "Binary Op type not supported by getBinaryOpAsString");
  }
}

poplar::Type getReturnType(UnaryOpType op, poplar::Type inType) {
  if (UnaryOpType::IS_FINITE == op) {
    return poplar::BOOL;
  }

  return inType;
}

bool isSpecialCase(UnaryOpType op) {
  return op == UnaryOpType::SQUARE || op == UnaryOpType::SIGNUM ||
         op == UnaryOpType::INVERSE || op == UnaryOpType::COUNT_LEADING_ZEROS;
}

std::string handleSpecialCase(UnaryOpType op, const std::string &param) {
  assert(isSpecialCase(op) && "Unary operation is not a special case");
  switch (op) {
  case UnaryOpType::COUNT_LEADING_ZEROS:
    return param + "? __builtin_clz(" + param + ") : 32;";
  case UnaryOpType::SQUARE:
    return param + " * " + param;
  case UnaryOpType::SIGNUM:
    return "(0 < " + param + ") - (" + param + " < 0)";
  case UnaryOpType::INVERSE:
    return "Traits<decltype(" + param + ")>::ONE() /" + param;
  default:
    throw poputil::poplibs_error("Unary operation is not a special case.");
  }
}

poplar::StringRef getUnaryOpAsString(UnaryOpType op) {
  switch (op) {
  case UnaryOpType::ABSOLUTE:
    return "internal_abs";
  case UnaryOpType::BITWISE_NOT:
    return "~";
  case UnaryOpType::CEIL:
    return "NAMESPACE::ceil";
  case UnaryOpType::COS:
    return "NAMESPACE::cos";
  case UnaryOpType::EXPONENT:
    return "NAMESPACE::exp";
  case UnaryOpType::EXPONENT_MINUS_ONE:
    return "NAMESPACE::expm1";
  case UnaryOpType::FLOOR:
    return "NAMESPACE::floor";
  case UnaryOpType::IS_FINITE:
    return "isfinite";
  case UnaryOpType::LOGARITHM:
    return "NAMESPACE::log";
  case UnaryOpType::LOGARITHM_ONE_PLUS:
    return "NAMESPACE::log1p";
  case UnaryOpType::LOGICAL_NOT:
    return "!";
  case UnaryOpType::NEGATE:
    return "-";
  case UnaryOpType::SIN:
    return "NAMESPACE::sin";
  case UnaryOpType::TANH:
    return "NAMESPACE::tanh";
  case UnaryOpType::ROUND:
    return "NAMESPACE::round";
  case UnaryOpType::SQRT:
    return "NAMESPACE::sqrt";
  case UnaryOpType::RSQRT:
    return "internal_rsqrt";
  case UnaryOpType::POPCOUNT:
    return "__builtin_popcount";
  case UnaryOpType::SIGMOID:
    return "internal_sigmoid";
  case UnaryOpType::COUNT_LEADING_ZEROS:
  case UnaryOpType::SQUARE:
  case UnaryOpType::SIGNUM:
  case UnaryOpType::INVERSE:
    // Special cases
    assert(false && "Special cases shouldn't be handled via get string.");
    return "";
  default:
    throw poputil::poplibs_error("Unary operation not recognised.");
  }
}

} // namespace expr
} // namespace popops
