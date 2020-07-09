// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#ifndef _popsolver_Constraint_hpp_
#define _popsolver_Constraint_hpp_

#include <poplar/ArrayRef.hpp>
#include <popsolver/Model.hpp>
#include <popsolver/Variable.hpp>

#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <functional>
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
  virtual poplar::ArrayRef<Variable> getVariables() = 0;
};

template <typename T> class GenericAssignment : public Constraint {
  // first variable is the result, remaining variables are the arguments
  std::vector<Variable> vars;
  std::function<boost::optional<DataType>(const std::vector<T> &)> f;
  // Vector for storing variable values, used in the propagate() method. This
  // is a class member instead of a local variable to reduce the number of
  // allocations needed.
  std::vector<DataType> values;

public:
  GenericAssignment(
      Variable result, std::vector<Variable> vars_,
      std::function<boost::optional<DataType>(const std::vector<T> &)> f)
      : vars(), f(f), values(vars_.size()) {
    vars.reserve(vars_.size() + 1);
    vars.push_back(result);
    vars.insert(std::end(vars), std::begin(vars_), std::end(vars_));
  }

  bool propagate(Scheduler &scheduler) override;
  poplar::ArrayRef<Variable> getVariables() override { return vars; }
};

class Product : public Constraint {
  // vars is {result, left, right}
  std::array<Variable, 3> vars;

public:
  Product(Variable result, Variable left, Variable right)
      : vars{result, left, right} {}
  bool propagate(Scheduler &scheduler) override;
  poplar::ArrayRef<Variable> getVariables() override { return vars; }
};

class Sum : public Constraint {
  // first variable is the result, remaining variables are the arguments
  std::vector<Variable> vars;

public:
  Sum(Variable result, std::vector<Variable> vars_) : vars() {
    vars.reserve(vars_.size() + 1);
    vars.push_back(result);
    vars.insert(std::end(vars), std::begin(vars_), std::end(vars_));
  }
  bool propagate(Scheduler &scheduler) override;
  poplar::ArrayRef<Variable> getVariables() override { return vars; }
};

class Max : public Constraint {
  // first variable is the result, remaining variables are the arguments
  std::vector<Variable> vars;

public:
  Max(Variable result, std::vector<Variable> vars_) : vars() {
    vars.reserve(vars_.size() + 1);
    vars.push_back(result);
    vars.insert(std::end(vars), std::begin(vars_), std::end(vars_));
  }
  bool propagate(Scheduler &scheduler) override;
  poplar::ArrayRef<Variable> getVariables() override { return vars; }
};

class Min : public Constraint {
  // first variable is the result, remaining variables are the arguments
  std::vector<Variable> vars;

public:
  Min(Variable result, std::vector<Variable> vars_) : vars() {
    vars.reserve(vars_.size() + 1);
    vars.push_back(result);
    vars.insert(std::end(vars), std::begin(vars_), std::end(vars_));
  }
  bool propagate(Scheduler &scheduler) override;
  poplar::ArrayRef<Variable> getVariables() override { return vars; }
};

class Less : public Constraint {
  // vars is {left, right}
  std::array<Variable, 2> vars;

public:
  Less(Variable left, Variable right) : vars{left, right} {}
  bool propagate(Scheduler &scheduler) override;
  poplar::ArrayRef<Variable> getVariables() override { return vars; }
};

class LessOrEqual : public Constraint {
  // vars is {left, right}
  std::array<Variable, 2> vars;

public:
  LessOrEqual(Variable left, Variable right) : vars{left, right} {}
  bool propagate(Scheduler &scheduler) override;
  poplar::ArrayRef<Variable> getVariables() override { return vars; }
};

} // End namespace popsolver.

#endif // _popsolver_Constraint_hpp_
