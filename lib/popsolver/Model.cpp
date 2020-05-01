// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <popsolver/Model.hpp>

#include "Constraint.hpp"
#include "Scheduler.hpp"
#include <algorithm>
#include <boost/optional.hpp>
#include <cassert>
#include <limits>

using namespace popsolver;

Model::Model() = default;

Model::~Model() = default;

void Model::addConstraint(std::unique_ptr<Constraint> c) {
  constraints.push_back(std::move(c));
}

std::string Model::makeProductDebugName(const Variable *begin,
                                        const Variable *end) const {
  auto name = getDebugName(*begin);
  for (auto it = std::next(begin); it != end; ++it) {
    name += "*" + getDebugName(*it);
  }
  return name;
}

std::string Model::makeProductDebugName(const std::vector<Variable> &v) const {
  return makeProductDebugName(&v[0], &v[v.size() - 1]);
}

std::string Model::makeSumDebugName(const Variable *begin,
                                    const Variable *end) const {
  auto name = getDebugName(*begin);
  for (auto it = std::next(begin); it != end; ++it) {
    name += "+" + getDebugName(*it);
  }
  return name;
}

std::string Model::makeSumDebugName(const std::vector<Variable> &v) const {
  return makeSumDebugName(&v[0], &v[v.size() - 1]);
}

const std::string &Model::getDebugName(Variable v) const {
  return debugNames[v.id];
}

Variable Model::addVariable(unsigned min, unsigned max,
                            const std::string &debugName) {
  assert(min <= max);
  Variable v(initialDomains.size());
  if (min == max) {
    auto result = constants.emplace(min, v);
    if (!result.second)
      return result.first->second;
  }
  initialDomains.push_back({min, max});
  isCallOperand.push_back(false);
  if (debugName.empty()) {
    debugNames.emplace_back("var#" + std::to_string(v.id));
  } else {
    debugNames.emplace_back(debugName);
  }
  return v;
}

Variable Model::addVariable(const std::string &debugName) {
  return addVariable(0, std::numeric_limits<unsigned>::max() - 1, debugName);
}

Variable Model::addConstant(unsigned value, const std::string &debugName) {
  return addVariable(value, value,
                     debugName.empty() ? std::to_string(value) : debugName);
}

Variable Model::product(const Variable *begin, const Variable *end,
                        const std::string &debugName) {
  const auto numVariables = end - begin;
  assert(numVariables > 0);
  if (numVariables == 1)
    return *begin;
  const auto mid = begin + numVariables / 2;
  const auto left = product(begin, mid, makeProductDebugName(begin, mid));
  const auto right = product(mid, end, makeProductDebugName(mid, end));
  auto result = addVariable(debugName);
  auto p = std::unique_ptr<Constraint>(new Product(result, left, right));
  addConstraint(std::move(p));
  return result;
}

Variable Model::product(const std::vector<Variable> &vars,
                        const std::string &debugName) {
  if (vars.empty())
    return addConstant(1, debugName);
  return product(vars.data(), vars.data() + vars.size(), debugName);
}

Variable Model::sum(std::vector<Variable> vars, const std::string &debugName) {
  auto result = addVariable(debugName);
  auto p = std::unique_ptr<Constraint>(new Sum(result, std::move(vars)));
  addConstraint(std::move(p));
  return result;
}

Variable Model::max(std::vector<Variable> vars, const std::string &debugName) {
  auto result = addVariable(debugName);
  auto p = std::unique_ptr<Constraint>(new Max(result, std::move(vars)));
  addConstraint(std::move(p));
  return result;
}

Variable Model::min(std::vector<Variable> vars, const std::string &debugName) {
  auto result = addVariable(debugName);
  auto p = std::unique_ptr<Constraint>(new Min(result, std::move(vars)));
  addConstraint(std::move(p));
  return result;
}

Variable Model::sub(Variable left, Variable right,
                    const std::string &debugName) {
  auto result = addVariable(debugName);
  auto p = std::unique_ptr<Constraint>(new Sum(left, {result, right}));
  addConstraint(std::move(p));
  return result;
}

Variable Model::floordiv(Variable left, Variable right,
                         const std::string &debugName) {
  auto result = addVariable(debugName);
  auto resultTimesRight =
      product({result, right}, makeProductDebugName({result, right}));
  // result * right <= left
  lessOrEqual(resultTimesRight, left);
  // result * right + right > left
  less(left, sum({resultTimesRight, right},
                 makeSumDebugName({resultTimesRight, right})));
  return result;
}

Variable Model::ceildiv(Variable left, Variable right,
                        const std::string &debugName) {
  auto result = addVariable(debugName);
  auto resultTimesRight =
      product({result, right}, makeProductDebugName({result, right}));
  // result * right >= left
  lessOrEqual(left, resultTimesRight);
  // result * right < left + right
  less(resultTimesRight, sum({left, right}, makeSumDebugName({left, right})));
  return result;
}

Variable Model::ceildivConstrainDivisor(const Variable left,
                                        const Variable right,
                                        const std::string &debugName) {
  const auto result = ceildiv(left, right, debugName);
  // The "remainder" from the division is < right for the minimal divisor
  // result * right < left + result
  less(product({result, right}, makeProductDebugName({result, right})),
       sum({left, result}, makeSumDebugName({left, result})));
  return result;
}

Variable Model::mod(Variable left, Variable right,
                    const std::string &debugName) {
  // modulo A % B calculated by: X = A - (floordiv(A, B) * B)
  return sub(left, product({floordiv(left, right), right}), debugName);
}

void Model::less(Variable left, Variable right) {
  auto p = std::unique_ptr<Constraint>(new Less(left, right));
  addConstraint(std::move(p));
}

void Model::less(Variable left, unsigned right) {
  auto rightVar = addConstant(right);
  less(left, rightVar);
}

void Model::less(unsigned left, Variable right) {
  auto leftVar = addConstant(left);
  less(leftVar, right);
}

void Model::lessOrEqual(Variable left, Variable right) {
  auto p = std::unique_ptr<Constraint>(new LessOrEqual(left, right));
  addConstraint(std::move(p));
}

void Model::lessOrEqual(Variable left, unsigned right) {
  auto rightVar = addConstant(right);
  lessOrEqual(left, rightVar);
}

void Model::lessOrEqual(unsigned left, Variable right) {
  auto leftVar = addConstant(left);
  lessOrEqual(leftVar, right);
}

void Model::equal(Variable left, Variable right) {
  auto p = std::unique_ptr<Constraint>(new Sum(left, {right}));
  addConstraint(std::move(p));
}

void Model::equal(unsigned left, Variable right) {
  equal(addConstant(left), right);
}

void Model::equal(Variable left, unsigned right) {
  equal(left, addConstant(right));
}

void Model::factorOf(unsigned left, Variable right) {
  const auto result = addVariable();
  equal(left, product({result, right}));
}

Variable Model::call(std::vector<Variable> vars,
                     std::function<boost::optional<unsigned>(
                         const std::vector<unsigned> &values)>
                         f,
                     const std::string &debugName) {
  for (auto var : vars) {
    isCallOperand[var.id] = true;
  }
  auto result = addVariable(debugName);
  auto p = std::unique_ptr<Constraint>(
      new GenericAssignment(result, std::move(vars), f));
  addConstraint(std::move(p));
  return result;
}

static bool foundLowerCostSolution(const Domains domains,
                                   const std::vector<Variable> &objectives,
                                   Solution &previousSolution) {
  for (auto v : objectives) {
    if (domains[v].val() < previousSolution[v])
      return true;
    if (domains[v].val() > previousSolution[v])
      return false;
  }
  return false;
}

bool Model::minimize(Scheduler &scheduler,
                     const std::vector<Variable> &objectives,
                     bool &foundSolution, Solution &solution) {
  // Find an unassigned variable.
  const auto &domains = scheduler.getDomains();
  // Call constraints cannot be used to cut down the search space until all of
  // their operands are set. Prefer to set variables that are operands of calls
  // first. Use the size of the domain to break ties.
  auto varLess = [&](Variable i, Variable j) {
    if (isCallOperand[i.id] != isCallOperand[j.id])
      return static_cast<bool>(isCallOperand[i.id]);
    return domains[i].size() < domains[j].size();
  };
  boost::optional<Variable> v;
  const auto numVars = domains.size();
  for (unsigned i = 0; i != numVars; ++i) {
    if (domains[Variable(i)].size() > 1 && (!v || varLess(Variable(i), *v))) {
      v = Variable(i);
    }
  }
  if (!v) {
    // All variables are assigned.
    if (!foundSolution) {
      std::vector<unsigned> values;
      values.reserve(domains.size());
      for (const auto &d : domains) {
        values.push_back(d.val());
      }
      solution = Solution(std::move(values));
      foundSolution = true;
      return true;
    }
    if (foundLowerCostSolution(domains, objectives, solution)) {
      for (std::size_t i = 0; i != domains.size(); ++i) {
        solution[Variable(i)] = domains[Variable(i)].val();
      }
      return true;
    }
    return false;
  }
  // Evaluate the cost for every possible value of this variable.
  bool improvedSolution = false;
  for (unsigned value = scheduler.getDomains()[*v].min();
       value <= scheduler.getDomains()[*v].max(); ++value) {
    auto savedDomains = scheduler.getDomains();
    scheduler.set(*v, value);
    bool valueImprovedSolution = false;
    if (scheduler.propagate() &&
        minimize(scheduler, objectives, foundSolution, solution)) {
      valueImprovedSolution = true;
    }
    scheduler.setDomains(std::move(savedDomains));
    if (valueImprovedSolution) {
      improvedSolution = true;
      scheduler.setMax(objectives.front(), solution[objectives.front()]);
      bool succeeded = scheduler.propagate();
      assert(succeeded);
      (void)succeeded;
    }
  }
  return improvedSolution;
}

Solution Model::minimize(const std::vector<Variable> &v) {
  bool foundSolution = false;
  Solution solution;

  std::vector<Constraint *> constraintPtrs;
  constraintPtrs.reserve(constraints.size());
  for (const auto &c : constraints) {
    constraintPtrs.push_back(c.get());
  }
  // Perform initial constraint propagation.
  Scheduler scheduler(initialDomains, std::move(constraintPtrs));
  if (scheduler.initialPropagate() &&
      minimize(scheduler, v, foundSolution, solution))
    return solution;
  return Solution();
}
