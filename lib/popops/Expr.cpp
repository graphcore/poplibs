// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "ExprOpUtil.hpp"
#include <boost/range/numeric.hpp>
#include <cstdint>
#include <iomanip>
#include <popops/Expr.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/exceptions.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>

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

bool deepEquals(const Expr &a, const Expr &b) { return a.deepEquals(b); }

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

static std::string
buildName(std::string_view opName,
          std::initializer_list<std::string_view> inputNames) {
  auto plusLength = [](size_t length, const std::string_view &value) {
    return length + value.size();
  };

  std::string_view first = "u_", mid = "_", last = "_d";

  size_t length = opName.size() + first.size() + last.size() +
                  mid.size() * (inputNames.size() - 1);
  length = boost::accumulate(inputNames, length, plusLength);

  std::string result;
  result.reserve(length);
  result.append(opName);

  auto nIt = inputNames.begin(), nEnd = inputNames.end();
  result.append(first).append(*nIt++);
  while (nIt != nEnd) {
    result.append(mid).append(*nIt++);
  }
  result.append(last);
  return result;
}

std::string UnaryOp::name(const std::vector<poplar::Tensor> &inputs) const {
  return buildName(unaryOpTypeToString(type), {a->name(inputs)});
}

std::string BinaryOp::name(const std::vector<poplar::Tensor> &inputs) const {
  return buildName(binaryOpTypeToString(type),
                   {a->name(inputs), b->name(inputs)});
}

std::string TernaryOp::name(const std::vector<poplar::Tensor> &inputs) const {
  return buildName(ternaryOpTypeToString(type),
                   {a->name(inputs), b->name(inputs), c->name(inputs)});
}

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

bool Const::deepEquals(const Expr &other) const {
  const auto *otherConst = other.getAs<Const>();
  return otherConst && type == otherConst->type &&
         std::memcmp(data.get(), otherConst->data.get(), typeTraits.size) == 0;
}

bool Cast::deepEquals(const Expr &other) const {
  const auto *otherCast = other.getAs<Cast>();
  return otherCast && bType == otherCast->bType && a->deepEquals(*otherCast->a);
}

bool PlaceHolder::deepEquals(const Expr &other) const {
  const auto *otherPlaceholder = other.getAs<PlaceHolder>();
  return otherPlaceholder && index == otherPlaceholder->index;
}

bool UnaryOp::deepEquals(const Expr &other) const {
  const auto *otherUnary = other.getAs<UnaryOp>();
  return otherUnary && type == otherUnary->type &&
         a->deepEquals(*otherUnary->a);
}

bool BinaryOp::deepEquals(const Expr &other) const {
  const auto *otherBinary = other.getAs<BinaryOp>();
  return otherBinary && type == otherBinary->type &&
         a->deepEquals(*otherBinary->a) && b->deepEquals(*otherBinary->b);
}

bool TernaryOp::deepEquals(const Expr &other) const {
  const auto *otherTernary = other.getAs<TernaryOp>();
  return otherTernary && type == otherTernary->type &&
         a->deepEquals(*otherTernary->a) && b->deepEquals(*otherTernary->b) &&
         c->deepEquals(*otherTernary->c);
}

void Const::print(std::ostream &os, unsigned indent, bool prettyPrint) const {
  if (prettyPrint)
    os << std::string(indent, ' ');
  os << "Const(" << printValue() << ")";
}

void Cast::print(std::ostream &os, unsigned indent, bool prettyPrint) const {
  if (prettyPrint)
    os << std::string(indent, ' ');
  os << "Cast(" << bType << ",";
  if (prettyPrint)
    os << "\n";
  a->print(os, indent + 2, prettyPrint);
  os << ")";
}

void PlaceHolder::print(std::ostream &os, unsigned indent,
                        bool prettyPrint) const {
  if (prettyPrint)
    os << std::string(indent, ' ');
  os << "PlaceHolder(" << index << ")";
}

void UnaryOp::print(std::ostream &os, unsigned indent, bool prettyPrint) const {
  if (prettyPrint)
    os << std::string(indent, ' ');
  os << debugName(type) << "(";
  if (prettyPrint)
    os << "\n";
  a->print(os, indent + 2, prettyPrint);
  os << ")";
}

void BinaryOp::print(std::ostream &os, unsigned indent,
                     bool prettyPrint) const {
  if (prettyPrint)
    os << std::string(indent, ' ');
  os << debugName(type) << "(";
  if (prettyPrint)
    os << "\n";
  a->print(os, indent + 2, prettyPrint);
  os << ",";
  if (prettyPrint)
    os << "\n";
  b->print(os, indent + 2, prettyPrint);
  os << ")";
}

void TernaryOp::print(std::ostream &os, unsigned indent,
                      bool prettyPrint) const {
  if (prettyPrint)
    os << std::string(indent, ' ');
  os << debugName(type) << "(";
  if (prettyPrint)
    os << "\n";
  a->print(os, indent + 2, prettyPrint);
  os << ",";
  if (prettyPrint)
    os << "\n";
  b->print(os, indent + 2, prettyPrint);
  os << ",";
  if (prettyPrint)
    os << "\n";
  c->print(os, indent + 2, prettyPrint);
  os << ")";
}

std::ostream &operator<<(std::ostream &os, const Expr &expr) {
  expr.print(os, 0, true);
  return os;
}

} // namespace expr
} // namespace popops
