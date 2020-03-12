// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/Expr.hpp>
#include <poputil/exceptions.hpp>
#include <regex>

namespace popops {
namespace expr {

Expr::~Expr() {}

template <> void ExprType<Const>::loc() {}
template <> void ExprType<Cast>::loc() {}
template <> void ExprType<PlaceHolder>::loc() {}
template <> void ExprType<UnaryOp>::loc() {}
template <> void ExprType<BinaryOp>::loc() {}
template <> void ExprType<TernaryOp>::loc() {}

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
    return std::to_string(*reinterpret_cast<float *>(rawData)) + "f";
  }
  if (this->getType() == poplar::HALF) {
    // The actual type behind the half should be a float.
    assert(this->getTypeTraits().isFloat == true &&
           this->getTypeTraits().size == sizeof(float));

    return std::to_string(*reinterpret_cast<float *>(rawData));
  }
  throw poputil::poplibs_error("Constant type is not supported: " +
                               this->getType().toString());
}

const std::vector<std::string> UnaryOp::UnaryOpNames = {
    "ABS",    "ASIN",      "B_NOT",
    "CEIL",   "COS",       "COUNT_LEADING_ZEROS",
    "EXP",    "EXP_M_1",   "FLOOR",
    "INV",    "IS_FINITE", "IS_INF",
    "IS_NAN", "LOG",       "LOG_ONE_PLUS",
    "NOT",    "NEG",       "POPCOUNT",
    "SIGNUM", "SIN",       "TAN",
    "TANH",   "ROUND",     "SQRT",
    "SQU",    "SIGMOID",   "RSQRT"};

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
