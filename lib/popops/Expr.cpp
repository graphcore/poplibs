// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <iomanip>
#include <popops/Expr.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/exceptions.hpp>
#include <regex>
#include <sstream>
#include <string>

namespace poputil {
template <> poplar::ProfileValue toProfileValue(const popops::expr::Expr &p) {
  return poplar::ProfileValue("<expr::Expr>");
}
} // namespace poputil

namespace popops {
namespace expr {

Expr::~Expr() {}

template <> void ExprType<Const>::loc() {}
template <> void ExprType<Cast>::loc() {}
template <> void ExprType<PlaceHolder>::loc() {}
template <> void ExprType<UnaryOp>::loc() {}
template <> void ExprType<BinaryOp>::loc() {}
template <> void ExprType<TernaryOp>::loc() {}

double Const::getDataAsDouble() const {
  char *rawData = this->getData();
  const auto constType = this->getType();
  if (constType == poplar::BOOL) {
    return static_cast<double>(*reinterpret_cast<bool *>(rawData));
  }
  if (constType == poplar::CHAR) {
    return static_cast<double>(*reinterpret_cast<char *>(rawData));
  }
  if (constType == poplar::UNSIGNED_CHAR) {
    return static_cast<double>(*reinterpret_cast<unsigned char *>(rawData));
  }
  if (constType == poplar::SIGNED_CHAR) {
    return static_cast<double>(*reinterpret_cast<signed char *>(rawData));
  }
  if (constType == poplar::SIGNED_CHAR) {
    return static_cast<double>(*reinterpret_cast<signed char *>(rawData));
  }
  if (constType == poplar::UNSIGNED_SHORT) {
    return static_cast<double>(*reinterpret_cast<unsigned short *>(rawData));
  }
  if (constType == poplar::SHORT) {
    return static_cast<double>(*reinterpret_cast<signed short *>(rawData));
  }
  if (constType == poplar::UNSIGNED_INT) {
    return static_cast<double>(*reinterpret_cast<unsigned int *>(rawData));
  }
  if (constType == poplar::INT) {
    return static_cast<double>(*reinterpret_cast<signed int *>(rawData));
  }
  if (constType == poplar::UNSIGNED_LONG) {
    return static_cast<double>(*reinterpret_cast<unsigned long *>(rawData));
  }
  if (constType == poplar::LONG) {
    return static_cast<double>(*reinterpret_cast<signed long *>(rawData));
  }
  if (constType == poplar::FLOAT) {
    return static_cast<double>(*reinterpret_cast<float *>(rawData));
  }
  if (constType == poplar::HALF) {
    // The actual type behind the half should be a float.
    assert(this->getTypeTraits().isFloat == true &&
           this->getTypeTraits().size == sizeof(float));
    return static_cast<double>(*reinterpret_cast<float *>(rawData));
  }
  if (constType == poplar::UNSIGNED_LONGLONG) {
    auto typeValue = *reinterpret_cast<unsigned long long *>(rawData);
    auto doubleValue = static_cast<double>(typeValue);
    if (static_cast<unsigned long long>(doubleValue) != typeValue) {
      throw poputil::poplibs_error("Error in conversion of value to double");
    }
    return doubleValue;
  }
  if (constType == poplar::LONGLONG) {
    auto typeValue = *reinterpret_cast<signed long long *>(rawData);
    auto doubleValue = static_cast<double>(typeValue);
    if (static_cast<signed long long>(doubleValue) != typeValue) {
      throw poputil::poplibs_error("Error in conversion of value to double");
    }
    return doubleValue;
  }
  throw poputil::poplibs_error("Constant type is not supported: " +
                               this->getType().toString());
}

std::uint64_t Const::getDataForUnsignedIntegral() const {
  char *rawData = this->getData();
  const auto constType = this->getType();
  if (constType == poplar::BOOL) {
    return static_cast<std::uint64_t>(*reinterpret_cast<bool *>(rawData));
  } else if (constType == poplar::UNSIGNED_CHAR) {
    return static_cast<std::uint64_t>(
        *reinterpret_cast<unsigned char *>(rawData));
  } else if (constType == poplar::UNSIGNED_SHORT) {
    return static_cast<std::uint64_t>(
        *reinterpret_cast<unsigned short *>(rawData));
  } else if (constType == poplar::UNSIGNED_INT) {
    return static_cast<std::uint64_t>(
        *reinterpret_cast<unsigned int *>(rawData));
  } else if (constType == poplar::UNSIGNED_LONG) {
    return static_cast<std::uint64_t>(
        *reinterpret_cast<unsigned long *>(rawData));
  } else if (constType == poplar::UNSIGNED_LONGLONG) {
    auto typeValue = *reinterpret_cast<unsigned long long *>(rawData);
    auto uint64Val = static_cast<std::uint64_t>(typeValue);
    if (static_cast<unsigned long long>(uint64Val) != typeValue) {
      throw poputil::poplibs_error("Error in conversion of value to uint64_t");
    }
    return typeValue;
  } else {
    throw poputil::poplibs_error("Error in conversion of value to uint64_t");
  }
}

std::string Const::printValue() const {
  char *rawData = this->getData();

  if (this->getType() == poplar::BOOL) {
    return std::to_string(*reinterpret_cast<bool *>(rawData));
  }
  if (this->getType() == poplar::CHAR) {
    return std::to_string(*reinterpret_cast<char *>(rawData));
  }
  if (this->getType() == poplar::UNSIGNED_CHAR) {
    return std::to_string(*reinterpret_cast<unsigned char *>(rawData));
  }
  if (this->getType() == poplar::SIGNED_CHAR) {
    return std::to_string(*reinterpret_cast<signed char *>(rawData));
  }
  if (this->getType() == poplar::SIGNED_CHAR) {
    return std::to_string(*reinterpret_cast<signed char *>(rawData));
  }
  if (this->getType() == poplar::UNSIGNED_SHORT) {
    return std::to_string(*reinterpret_cast<unsigned short *>(rawData));
  }
  if (this->getType() == poplar::SHORT) {
    return std::to_string(*reinterpret_cast<signed short *>(rawData));
  }
  if (this->getType() == poplar::UNSIGNED_INT) {
    return std::to_string(*reinterpret_cast<unsigned int *>(rawData));
  }
  if (this->getType() == poplar::INT) {
    return std::to_string(*reinterpret_cast<signed int *>(rawData));
  }
  if (this->getType() == poplar::UNSIGNED_LONG) {
    return std::to_string(*reinterpret_cast<unsigned long *>(rawData));
  }
  if (this->getType() == poplar::LONG) {
    return std::to_string(*reinterpret_cast<signed long *>(rawData));
  }
  if (this->getType() == poplar::UNSIGNED_LONGLONG) {
    return std::to_string(*reinterpret_cast<unsigned long long *>(rawData));
  }
  if (this->getType() == poplar::LONGLONG) {
    return std::to_string(*reinterpret_cast<signed long long *>(rawData));
  }
  if (this->getType() == poplar::FLOAT) {
    std::stringstream ss;
    ss << std::defaultfloat << std::setprecision(9)
       << *reinterpret_cast<float *>(rawData);
    if (ss.str().find(".") == std::string::npos) {
      ss << ".f";
    } else {
      ss << "f";
    }
    return ss.str();
  }
  if (this->getType() == poplar::HALF) {
    // The actual type behind the half should be a float.
    assert(this->getTypeTraits().isFloat == true &&
           this->getTypeTraits().size == sizeof(float));
    std::stringstream ss;
    ss << std::defaultfloat << std::setprecision(9)
       << *reinterpret_cast<float *>(rawData);
    return ss.str();
  }
  throw poputil::poplibs_error("Constant type is not supported: " +
                               this->getType().toString());
}

const std::vector<std::string> UnaryOp::UnaryOpNames = {"ABS",
                                                        "ASIN",
                                                        "B_NOT",
                                                        "CBRT",
                                                        "CEIL",
                                                        "COS",
                                                        "COUNT_LEADING_ZEROS",
                                                        "EXP",
                                                        "EXP_M_1",
                                                        "FLOOR",
                                                        "INV",
                                                        "IS_FINITE",
                                                        "IS_INF",
                                                        "IS_NAN",
                                                        "LOG",
                                                        "LOG_ONE_PLUS",
                                                        "NOT",
                                                        "NEG",
                                                        "POPCOUNT",
                                                        "SIGNUM",
                                                        "SIN",
                                                        "TAN",
                                                        "TANH",
                                                        "ROUND",
                                                        "SQRT",
                                                        "SQU",
                                                        "SIGMOID",
                                                        "RSQRT",
                                                        "RELU"};

const std::vector<std::string> BinaryOp::BinaryOpNames = {
    "ADD",
    "ATAN2",
    "B_AND",
    "B_OR",
    "B_XOR",
    "B_XNOR",
    "DIV",
    "EQU",
    "G_T_EQ",
    "G_T",
    "INV_STD_DEV_TO_VARIANCE",
    "L_T_EQ",
    "AND",
    "OR",
    "L_T",
    "MAX",
    "MIN",
    "MUL",
    "N_EQ",
    "POW",
    "REM",
    "SHIFT_LEFT",
    "SHIFT_RIGHT",
    "SHIFT_RIGHT_SIGN_EXTEND",
    "SUB",
    "VARIANCE_TO_INV_STD_DEV"};

const std::vector<std::string> TernaryOp::TernaryOpNames = {"CLAMP", "SELECT"};

std::string Const::name(const std::vector<poplar::Tensor> &) const {
  // can't have . or - in class names need to remove these from the
  // string
  std::regex dotRegex("\\.");
  std::string result = this->printValue();
  result = std::regex_replace(result, dotRegex, "z");
  std::regex minusRegex("-");
  return std::regex_replace(result, minusRegex, "m");
}

static std::string typeShortName(const poplar::Type &type) {
  std::string result;
  // give some types shortened name to try and keep codelet name shorter
  if (type == poplar::UNSIGNED_INT) {
    result = "uint";
  } else if (type == poplar::UNSIGNED_SHORT) {
    result = "ushort";
  } else {
    result = type.toString();
  }
  std::regex spaceRegex(" ");
  return std::regex_replace(result, spaceRegex, "_");
}

std::string Cast::name(const std::vector<poplar::Tensor> &inputs) const {
  return "Cast_" + a->name(inputs) + "_" + typeShortName(bType);
}

std::string PlaceHolder::name(const std::vector<poplar::Tensor> &inputs) const {
  auto type = inputs[index - 1].elementType();
  return typeShortName(type) + "_" + std::to_string(index) + "_";
}

} // namespace expr
} // namespace popops
