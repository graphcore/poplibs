// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popsolver_Model_hpp
#define popsolver_Model_hpp

#include <boost/optional.hpp>
#include <cassert>
#include <functional>
#include <memory>
#include <popsolver/Variable.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace popsolver {

class Constraint;
class Scheduler;

class Domain {
public:
  unsigned min_;
  unsigned max_;
  Domain(unsigned min_, unsigned max_) : min_(min_), max_(max_) {}
  unsigned min() const { return min_; }
  unsigned max() const { return max_; }
  unsigned val() const {
    assert(min_ == max_);
    return min_;
  }
  unsigned size() const { return max_ - min_ + 1; }
};

class Domains {
public:
  friend class Scheduler;
  std::vector<Domain> domains;
  using iterator = std::vector<Domain>::iterator;
  using const_iterator = std::vector<Domain>::const_iterator;
  Domain &operator[](Variable v) { return domains[v.id]; }
  const Domain &operator[](Variable v) const { return domains[v.id]; }
  iterator begin() { return domains.begin(); }
  iterator end() { return domains.end(); }
  const_iterator begin() const { return domains.begin(); }
  const_iterator end() const { return domains.end(); }
  std::size_t size() const { return domains.size(); }
  void push_back(const Domain &d) { domains.push_back(d); }
  template <typename... Args> void emplace_back(Args &&... args) {
    domains.emplace_back(std::forward<Args>(args)...);
  }
};

class Solution {
  std::vector<unsigned> values;

public:
  Solution() = default;
  Solution(std::vector<unsigned> values) : values(values) {}
  unsigned &operator[](Variable v) { return values[v.id]; }
  unsigned operator[](Variable v) const { return values[v.id]; }
  bool validSolution() const { return values.size() > 0; }
};

class Model {
  void addConstraint(std::unique_ptr<Constraint> c);
  bool minimize(Scheduler &scheduler, const std::vector<Variable> &objectives,
                bool &foundSolution, Solution &solution);
  Variable product(const Variable *begin, const Variable *end,
                   const std::string &debugName);
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
  std::unordered_map<unsigned, Variable> constants;
  std::vector<bool> isCallOperand;
  std::vector<std::unique_ptr<Constraint>> constraints;
  Domains initialDomains;

  /// Add a new variable.
  Variable addVariable(const std::string &debugName = "");
  /// Add a new variable with a domain of [min,max].
  Variable addVariable(unsigned min, unsigned max,
                       const std::string &debugName = "");
  /// Add a constant with the specified value.
  Variable addConstant(unsigned value, const std::string &debugName = "");
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
  void less(Variable left, unsigned right);
  /// Constrain the left constant to be less than the right variable.
  void less(unsigned left, Variable right);
  /// Constrain the left variable to be less than or equal to the right
  /// variable.
  void lessOrEqual(Variable left, Variable right);
  /// Constrain the left variable to be less than or equal to the right
  /// constant.
  void lessOrEqual(Variable left, unsigned right);
  /// Constrain the left constant to be less than or equal to the right
  /// variable.
  void lessOrEqual(unsigned left, Variable right);
  /// Constrain the left variable to be equal to the right variable.
  void equal(Variable left, Variable right);
  /// Constrain the left variable to be equal to the right constant.
  void equal(Variable left, unsigned right);
  /// Constrain the left constant to be equal to the right variable.
  void equal(unsigned left, Variable right);
  /// Constrain the right variable to be a factor of the left constant.
  void factorOf(unsigned left, Variable right);
  /// Add a new variable that is the result of applying the specified function
  /// to the specified variables.
  Variable call(std::vector<Variable> vars,
                std::function<boost::optional<unsigned>(
                    const std::vector<unsigned> &values)>
                    f,
                const std::string &debugName = "");
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
