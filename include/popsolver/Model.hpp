// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popsolver_Model_hpp
#define popsolver_Model_hpp

#include <popsolver/PriorityGroupID.hpp>
#include <popsolver/Variable.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/optional.hpp>

#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace popsolver {

class Constraint;
class Scheduler;

struct ConstraintEvaluationSummary {
  using CountType = std::uint64_t;
  CountType call = 0;
  CountType product = 0;
  CountType sum = 0;
  CountType max = 0;
  CountType min = 0;
  CountType less = 0;
  CountType lessOrEqual = 0;

  CountType unknown = 0;

  CountType total() const {
    return call + product + sum + max + min + less + lessOrEqual + unknown;
  }

  void operator+=(const ConstraintEvaluationSummary &other) {
    call += other.call;
    product += other.product;
    sum += other.sum;
    max += other.max;
    min += other.min;
    less += other.less;
    lessOrEqual += other.lessOrEqual;
    unknown += other.unknown;
  }
};

// Extra breakdown information is collected when POPLIBS_LOG_LEVEL is set to
// TRACE or higher
std::ostream &operator<<(std::ostream &os,
                         const ConstraintEvaluationSummary &s);

class DataType {
public:
  using UnderlyingType = std::uint64_t;

private:
  UnderlyingType underlying = 0;

  template <typename T> constexpr bool representable(T x, T delta = 0) {
    assert(x >= 0);
    const auto partial = static_cast<UnderlyingType>(x);
    const auto xPrime = static_cast<T>(partial);
    return std::abs(x - xPrime) <= delta;
  }

public:
  DataType() = default;
  template <typename T> explicit constexpr DataType(T x) : underlying(x) {
    static_assert(std::is_unsigned<T>::value, "Must be unsigned type");
    static_assert(sizeof(T) <= sizeof(UnderlyingType),
                  "Size of type must be less or equal underlying type");
  }

  explicit constexpr DataType(double x) {
    assert(representable(x, 1.0) &&
           "can't represent double with popsolver::DataType");
    underlying = x;
  }
  explicit constexpr DataType(int64_t x) {
    assert(representable(x) &&
           "can't represent int64_t with popsolver::DataType");
    underlying = x;
  }
  explicit constexpr DataType(int x) {
    assert(representable(x) && "can't represent int with popsolver::DataType");
    underlying = x;
  }

  static constexpr DataType min() {
    return DataType{std::numeric_limits<UnderlyingType>::min()};
  }
  static constexpr DataType max() {
    return DataType{std::numeric_limits<UnderlyingType>::max()};
  }

  constexpr UnderlyingType operator*() const { return underlying; }
  explicit operator UnderlyingType() const { return underlying; }
  explicit operator UnderlyingType &() { return underlying; }

  template <typename T> T getAs() const {
    assert(static_cast<UnderlyingType>(static_cast<T>(underlying)) ==
               underlying &&
           "loss of precision for value in popsolver::DataType when converting "
           "to type");
    assert(underlying <= std::numeric_limits<T>::max() &&
           "value in popsolver::DataType is too large to fit in type");
    return static_cast<T>(underlying);
  }

  DataType &operator++() {
    underlying++;
    return *this;
  }
  DataType &operator--() {
    underlying--;
    return *this;
  }

  DataType operator++(int) {
    underlying++;
    return DataType{underlying - 1};
  }
  DataType operator--(int) {
    underlying--;
    return DataType{underlying + 1};
  }

  void operator+=(DataType r) { underlying += *r; }
  void operator*=(DataType r) { underlying *= *r; }

  friend std::istream &operator>>(std::istream &is, DataType &x);
};

inline DataType operator+(DataType l, DataType r) { return DataType{*l + *r}; }
inline DataType operator-(DataType l, DataType r) { return DataType{*l - *r}; }
inline DataType operator*(DataType l, DataType r) { return DataType{*l * *r}; }
inline DataType operator/(DataType l, DataType r) { return DataType{*l / *r}; }
inline DataType operator%(DataType l, DataType r) { return DataType{*l % *r}; }

inline bool operator<(DataType l, DataType r) { return *l < *r; }
inline bool operator>(DataType l, DataType r) { return *l > *r; }
inline bool operator<=(DataType l, DataType r) { return *l <= *r; }
inline bool operator>=(DataType l, DataType r) { return *l >= *r; }
inline bool operator!=(DataType l, DataType r) { return *l != *r; }
inline bool operator==(DataType l, DataType r) { return *l == *r; }

inline std::ostream &operator<<(std::ostream &os, DataType x) {
  os << *x;
  return os;
}
inline std::istream &operator>>(std::istream &is, DataType &x) {
  is >> x.underlying;
  return is;
}
} // namespace popsolver

namespace std {
template <> struct hash<popsolver::DataType> {
  std::size_t operator()(const popsolver::DataType &k) const {
    return std::hash<popsolver::DataType::UnderlyingType>()(*k);
  }
};
} // namespace std

namespace popsolver {

class Domain {
public:
  DataType min_;
  DataType max_;
  Domain(DataType min_, DataType max_) : min_(min_), max_(max_) {}
  DataType min() const { return min_; }
  DataType max() const { return max_; }
  DataType val() const {
    assert(min_ == max_);
    return min_;
  }
  // When min == max, size() returns 0.
  DataType size() const { return max_ - min_; }
};

class Domains {
public:
  friend class Scheduler;
  std::vector<Domain> domains;
  using iterator = std::vector<Domain>::iterator;
  using const_iterator = std::vector<Domain>::const_iterator;
  Domain &operator[](Variable v) { return domains[v]; }
  const Domain &operator[](Variable v) const { return domains[v]; }
  iterator begin() { return domains.begin(); }
  iterator end() { return domains.end(); }
  const_iterator begin() const { return domains.begin(); }
  const_iterator end() const { return domains.end(); }
  std::size_t size() const { return domains.size(); }
  void push_back(const Domain &d) { domains.push_back(d); }
  template <typename... Args> void emplace_back(Args &&...args) {
    domains.emplace_back(std::forward<Args>(args)...);
  }
};

class Model;

class Solution {
  std::vector<DataType> values;
  ConstraintEvaluationSummary constraintEvalSummary;

public:
  Solution() = default;
  Solution(std::vector<DataType> values) : values(values) {}
  DataType &operator[](Variable v) { return values[v]; }
  DataType operator[](Variable v) const { return values[v]; }
  bool validSolution() const { return values.size() > 0; }
  ConstraintEvaluationSummary constraintsEvaluated() {
    return constraintEvalSummary;
  }

  friend class Model;
};

class Model {
  void addConstraint(std::unique_ptr<Constraint> c);
  std::pair<bool, ConstraintEvaluationSummary>
  minimize(Scheduler &scheduler, const std::vector<Variable> &objectives,
           const boost::adjacency_matrix<> &prioGroupReachability,
           bool &foundSolution, Solution &solution);
  Variable product(const Variable *begin, const Variable *end,
                   const std::string &debugName);
  std::string makeBinaryOpDebugName(const Variable *begin, const Variable *end,
                                    const std::string &opStr) const;
  std::string makeBinaryOpDebugName(const std::vector<Variable> &vars,
                                    const std::string &opStr) const;
  std::string makeParameterisedOpDebugName(const Variable *begin,
                                           const Variable *end,
                                           const std::string &opStr) const;
  std::string makeParameterisedOpDebugName(const std::vector<Variable> &vars,
                                           const std::string &opStr) const;
  std::string makeProductDebugName(const Variable *begin,
                                   const Variable *end) const;
  std::string makeProductDebugName(const std::vector<Variable> &v) const;
  std::string makeSumDebugName(const Variable *begin,
                               const Variable *end) const;
  std::string makeSumDebugName(const std::vector<Variable> &v) const;
  const std::string &getDebugName(Variable v) const;

public:
  Model();
  ~Model();
  std::vector<std::string> debugNames;
  std::unordered_map<DataType, Variable> constants;
  /// Per-variable, the priority group that variable belongs to.
  std::vector<PriorityGroupID> priorityGroup;
  /// A directed graph of relative priorities of priority groups.
  /// Here an edge from group a to group b indicates priority of
  /// a over b.
  boost::adjacency_list<> priorityGraph;
  PriorityGroupID defaultPriorityGroup;
  std::vector<std::unique_ptr<Constraint>> constraints;
  Domains initialDomains;

  /// Add a new variable.
  Variable addVariable(const std::string &debugName = "");
  /// Add a new variable with a domain of [min,max].
  Variable addVariable(DataType min, DataType max,
                       const std::string &debugName = "");
  Variable addVariable(DataType::UnderlyingType min,
                       DataType::UnderlyingType max,
                       const std::string &debugName = "");
  /// Add a constant with the specified value.
  Variable addConstant(DataType value, const std::string &debugName = "");
  Variable addConstant(DataType::UnderlyingType value,
                       const std::string &debugName = "");

  /// Set the priority group to which a variable belongs.
  /// A variable may belong to only one priority group, and multiple calls
  /// to setPriorityGroup overwrite the priority group of that variable.
  void setPriorityGroup(Variable v, PriorityGroupID prioGroup);
  /// Prioritise group \p a over group \p b when minimising the model.
  /// This means we search the domain of variables in group \p a prior
  /// to those in group \p b.
  void prioritiseOver(PriorityGroupID a, PriorityGroupID b);

  /// Add a constant with value 0 (for convenience).
  Variable zero();
  /// Add a constant with value 1 (for convenience).
  Variable one();
  /// Add a new variable that is the product of the specified variables.
  Variable product(const std::vector<Variable> &vars,
                   const std::string &debugName = "");
  /// Add a new variable that is the sum of the specified variables.
  Variable sum(std::vector<Variable> vars, const std::string &debugName = "");
  /// Add a new variable that is the max of the specified variables.
  Variable max(std::vector<Variable> vars, const std::string &debugName = "");
  /// Add a new variable that is the min of the specified variables.
  Variable min(std::vector<Variable> vars, const std::string &debugName = "");
  /// Add a new variable that is \p left divided by \p right, rounded down to
  /// the nearest integer.
  Variable floordiv(Variable left, Variable right,
                    const std::string &debugName = "");
  /// Add a new variable that is \p left divided by \p right, rounded up to the
  /// nearest integer.
  Variable ceildiv(Variable left, Variable right,
                   const std::string &debugName = "");
  // Return m.ceildiv(dividend, divisor) and constrain the divisor so it is
  // the smallest divisor that gives us that result.
  Variable ceildivConstrainDivisor(const Variable dividend,
                                   const Variable divisor,
                                   const std::string &debugName = "");
  /// Add a new variable that is \p left mod \p right.
  Variable mod(Variable left, Variable right,
               const std::string &debugName = "");
  /// Add a new variable that is \p right subtracted from \p left.
  Variable sub(Variable left, Variable right,
               const std::string &debugName = "");
  /// Constrain the left variable to be less than the right variable.
  void less(Variable left, Variable right);
  /// Constrain the left variable to be less than the right constant.
  void less(Variable left, DataType right);
  /// Constrain the left constant to be less than the right variable.
  void less(DataType left, Variable right);
  /// Reified version of < constraint such that the returned variable
  /// has value (left < right) ? 1 : 0.
  Variable reifiedLess(Variable left, Variable right);
  /// Constrain the left variable to be less than or equal to the right
  /// variable.
  void lessOrEqual(Variable left, Variable right);
  /// Constrain the left variable to be less than or equal to the right
  /// constant.
  void lessOrEqual(Variable left, DataType right);
  /// Constrain the left constant to be less than or equal to the right
  /// variable.
  void lessOrEqual(DataType left, Variable right);
  /// Reified version of <= constraint such that the returned variable
  /// has value (left <= right) ? 1 : 0.
  Variable reifiedLessOrEqual(Variable left, Variable right);
  /// Constrain the left variable to be equal to the right variable.
  void equal(Variable left, Variable right);
  /// Constrain the left variable to be equal to the right constant.
  void equal(Variable left, DataType right);
  /// Constrain the left constant to be equal to the right variable.
  void equal(DataType left, Variable right);
  /// Reified version of == constraint such that the returned variable
  /// has value (left == right) ? 1 : 0.
  Variable reifiedEqual(Variable left, Variable right);
  /// Constrain the right variable to be a factor of the left constant.
  void factorOf(DataType left, Variable right);
  /// Constrain the right variable to be a factor of the left variable.
  void factorOf(Variable left, Variable right);
  /// Returns a variable with value given by the boolean or of the two
  /// given variables.
  Variable booleanOr(Variable left, Variable right);
  /// Returns a variable with value given by the boolean and of the two
  /// given variables.
  Variable booleanAnd(Variable left, Variable right);
  /// Returns a variable with value given by the boolean negation of the
  /// given variable.
  Variable booleanNot(Variable v);

  /// Add a new variable that is the result of applying the specified function
  /// to the specified variables. Input variable domains are restricted to be
  /// at most the range of T.
  /// Supported specialisations are DataType, unsigned.
  template <typename T>
  Variable
  call(std::vector<Variable> vars,
       std::function<boost::optional<DataType>(const std::vector<T> &values)> f,
       const std::string &debugName = "");

  PriorityGroupID getDefaultPriorityGroup() const;
  PriorityGroupID addPriorityGroup();

  /// Find a solution that minimizes the value of the specified variables.
  /// Lexicographical comparison is used to compare the values of the variables.
  /// \returns The solution
  Solution minimize(const std::vector<Variable> &v);
  /// Find a solution that minimizes the specified variable.
  /// \returns The solution
  Solution minimize(Variable v) { return minimize(std::vector<Variable>({v})); }
};

} // End namespace popsolver.

#endif // popsolver_Model_hpp
