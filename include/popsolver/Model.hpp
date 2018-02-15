#ifndef popsolver_Model_hpp
#define popsolver_Model_hpp

#include <cassert>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <popsolver/Variable.hpp>

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
  unsigned val() const { assert(min_ == max_); return min_; }
  unsigned size() const { return max_ - min_ + 1; }
};

class Domains {
public:
  friend class Scheduler;
  std::vector<Domain> domains;
  using iterator = std::vector<Domain>::iterator;
  using const_iterator = std::vector<Domain>::const_iterator;
  Domain &operator[](Variable v) {
    return domains[v.id];
  }
  const Domain &operator[](Variable v) const {
    return domains[v.id];
  }
  iterator begin() { return domains.begin(); }
  iterator end() { return domains.end(); }
  const_iterator begin() const { return domains.begin(); }
  const_iterator end() const { return domains.end(); }
  std::size_t size() const { return domains.size(); }
  void push_back(const Domain &d) {
    domains.push_back(d);
  }
};

class Solution {
  std::vector<unsigned> values;
public:
  Solution() = default;
  Solution(std::vector<unsigned> values) : values(values) {}
  unsigned &operator[](Variable v) {
    return values[v.id];
  }
  unsigned operator[](Variable v) const {
    return values[v.id];
  }
};

/// Type of object that is thrown if no solution exists.
class NoSolution : std::runtime_error {
public:
  NoSolution() : std::runtime_error("No solution exists") {}
};

class Model {
  void addConstraint(std::unique_ptr<Constraint> c);
  bool minimize(Scheduler &scheduler,
                const std::vector<Variable> &objectives,
                bool &foundSolution, Solution &solution);
  Variable product(const Variable *begin, const Variable *end);
public:
  Model();
  ~Model();
  std::unordered_map<unsigned, Variable> constants;
  std::vector<bool> isCallOperand;
  std::vector<std::unique_ptr<Constraint>> constraints;
  Domains initialDomains;

  /// Add a new variable.
  Variable addVariable();
  /// Add a new variable with a domain of [min,max].
  Variable addVariable(unsigned min, unsigned max);
  /// Add a constant with the specified value.
  Variable addConstant(unsigned value);
  /// Add a new variable that is the product of the specified variables.
  Variable product(const std::vector<Variable> &vars);
  /// Add a new variable that is the sum of the specified variables.
  Variable sum(std::vector<Variable> vars);
  /// Add a new variable that is \a left divided by \a right, rounded down to
  /// the nearest integer.
  Variable floordiv(Variable left, Variable right);
  /// Add a new variable that is \a left divided by \a right, rounded up to the
  /// nearest integer.
  Variable ceildiv(Variable left, Variable right);
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
  /// Add a new variable that is the result of applying the specified function
  /// to the specified variables.
  Variable call(
      std::vector<Variable> vars,
      std::function<unsigned (const std::vector<unsigned> &values)> f);
  /// Find a solution that minimizes the value of the specified variables.
  /// Lexicographical comparison is used to compare the values of the variables.
  /// \returns The solution
  /// \throws NoSolution if no solution exists.
  Solution minimize(const std::vector<Variable> &v);
  /// Find a solution that minimizes the specified variable.
  /// \returns The solution
  /// \throws NoSolution if no solution exists.
  Solution minimize(Variable v) {
    return minimize(std::vector<Variable>({v}));
  }
};

} // End namespace popsolver.

#endif // popsolver_Model_hpp
