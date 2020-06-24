// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popsparse_FullyConnectedVector_hpp
#define popsparse_FullyConnectedVector_hpp

#include <ostream>
#include <vector>

namespace popsparse {
namespace fullyconnected {

template <typename T> struct Vector {
  T groups;
  T x;
  T y;
  T z;

  Vector() = default;
  // Broadcasting constructor
  Vector(const T &val) : groups(val), x(val), y(val), z(val) {}
  Vector(std::initializer_list<T> &&vals) {
    auto it = vals.begin();
    groups = *it++;
    x = *it++;
    y = *it++;
    z = *it++;
    assert(it == vals.end());
  }
  Vector(const std::vector<T> &vals) {
    assert(vals.size() == 4);
    groups = vals[0];
    x = vals[1];
    y = vals[2];
    z = vals[3];
  }

  template <typename F>
  Vector<T> &binaryOpInPlace(const Vector<T> &b, const F &f) {
    groups = f(groups, b.groups);
    x = f(x, b.x);
    y = f(y, b.y);
    z = f(z, b.z);
    return *this;
  }

  template <typename F>
  Vector<T> binaryOp(const Vector<T> &b, const F &f) const {
    Vector<T> result;
    result.groups = f(groups, b.groups);
    result.x = f(x, b.x);
    result.y = f(y, b.y);
    result.z = f(z, b.z);
    return result;
  }

  template <typename F> static Vector<T> generate(const F &f) {
    Vector<T> result;
    result.groups = f();
    result.x = f();
    result.y = f();
    result.z = f();
    return result;
  }

  template <typename ResultType, typename F>
  Vector<ResultType> transform(const F &f) const {
    Vector<ResultType> result;
    result.groups = f(groups);
    result.x = f(x);
    result.y = f(y);
    result.z = f(z);
    return result;
  }

  template <typename VecT = T> std::vector<VecT> asStdVector() const {
    return {groups, x, y, z};
  }

  friend inline std::ostream &operator<<(std::ostream &os, const Vector<T> &v) {
    os << "{" << v.groups << "," << v.x << "," << v.y << "," << v.z << "}";
    return os;
  }

  constexpr std::size_t size() const { return 4; }
};

// operator+= if element-wise operator exists
template <typename T>
inline auto operator*=(Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(a.x *= T()), Vector<T>()) {
  return a.binaryOpInPlace(b, std::multiplies<T>());
}

// operator+ if element-wise operator exists
template <typename T>
inline auto operator*(const Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(T() * T()), Vector<T>()) {
  return a.binaryOp(b, std::multiplies<T>());
}

// operator+= if element-wise operator exists
template <typename T>
inline auto operator+=(Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(a.x += T()), Vector<T>()) {
  return a.binaryOpInPlace(b, std::plus<T>());
}

// operator+ if element-wise operator exists
template <typename T>
inline auto operator+(const Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(T() + T()), Vector<T>()) {
  return a.binaryOp(b, std::plus<T>());
}

// operator-= if element-wise operator exists
template <typename T>
inline auto operator-=(Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(a.x -= T()), Vector<T>()) {
  return a.binaryOpInPlace(b, std::minus<T>());
}

// operator- if element-wise operator exists
template <typename T>
inline auto operator-(const Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(T() - T()), Vector<T>()) {
  return a.binaryOp(b, std::minus<T>());
}

// operator/= if element-wise operator exists
template <typename T>
inline auto operator/=(Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(a.x /= T()), Vector<T>()) {
  return a.binaryOpInPlace(b, std::divides<T>());
}

// operator/ if element-wise operator exists
template <typename T>
inline auto operator/(const Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(T() / T()), Vector<T>()) {
  return a.binaryOp(b, std::divides<T>());
}

// operator%= if element-wise operator exists
template <typename T>
inline auto operator%=(Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(a.x %= T()), Vector<T>()) {
  return a.binaryOpInPlace(b, std::modulus<T>());
}

// operator% if element-wise operator exists
template <typename T>
inline auto operator%(const Vector<T> &a, const Vector<T> &b)
    -> decltype((void)(T() % T()), Vector<T>()) {
  return a.binaryOp(b, std::modulus<T>());
}

} // end namespace fullyconnected
} // end namespace popsparse

#endif // popsparse_FullyConnectedVector_hpp
