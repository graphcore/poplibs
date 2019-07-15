#ifndef _popsolver_Constraint_hpp_
#define _popsolver_Constraint_hpp_

#include <functional>
#include <popsolver/Variable.hpp>
#include <vector>

namespace popsolver {

class Scheduler;

class Constraint {
public:
  virtual ~Constraint();
  /// Update the domains of variable referenced by this constraint.
  /// \returns false if the constraint cannot be met, true otherwise.
  virtual bool propagate(Scheduler &scheduler) = 0;
  /// Return a vector of variable that the constraint references.
  virtual std::vector<Variable> getVariables() = 0;
};

class GenericAssignment : public Constraint {
  Variable result;
  std::vector<Variable> vars;
  std::function<unsigned (const std::vector<unsigned> &values)> f;
  // Vector for storing variable values, used in the propagate() method. This
  // is a class member instead of a local variable to reduce the number of
  // allocations needed.
  std::vector<unsigned> values;
public:
  GenericAssignment(
      Variable result, std::vector<Variable> vars_,
      std::function<unsigned (const std::vector<unsigned> &values)> f) :
    result(result), vars(std::move(vars_)), f(f), values(vars.size()) {}
  bool propagate(Scheduler &scheduler) override;
  std::vector<Variable> getVariables() override {
    std::vector<Variable> allVars = vars;
    allVars.push_back(result);
    return allVars;
  }
};

class Product : public Constraint {
  Variable result;
  Variable left;
  Variable right;
public:
  Product(Variable result, Variable left, Variable right) :
    result(result), left(left), right(right) {}
  bool propagate(Scheduler &scheduler) override;
  std::vector<Variable> getVariables() override {
    return {result, left, right};
  }
};

class Sum : public Constraint {
  Variable result;
  std::vector<Variable> vars;
public:
  Sum(Variable result, std::vector<Variable> vars) :
    result(result), vars(std::move(vars)) {}
  bool propagate(Scheduler &scheduler) override;
  std::vector<Variable> getVariables() override {
    std::vector<Variable> allVars = vars;
    allVars.push_back(result);
    return allVars;
  }
};

class Max : public Constraint {
  Variable result;
  std::vector<Variable> vars;
public:
  Max(Variable result, std::vector<Variable> vars) :
    result(result), vars(std::move(vars)) {}
  bool propagate(Scheduler &scheduler) override;
  std::vector<Variable> getVariables() override {
    std::vector<Variable> allVars = vars;
    allVars.push_back(result);
    return allVars;
  }
};

class Less : public Constraint {
  Variable left;
  Variable right;
public:
  Less(Variable left, Variable right) : left(left), right(right) {}
  bool propagate(Scheduler &scheduler) override;
  std::vector<Variable> getVariables() override {
    return {left, right};
  }
};

class LessOrEqual : public Constraint {
  Variable left;
  Variable right;
public:
  LessOrEqual(Variable left, Variable right) :
    left(left), right(right) {}
  bool propagate(Scheduler &scheduler) override;
  std::vector<Variable> getVariables() override {
    return {left, right};
  }
};

} // End namespace popsolver.

#endif // _popsolver_Constraint_hpp_
