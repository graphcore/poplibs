#include <popsolver/Model.hpp>

#include <algorithm>
#include <boost/optional.hpp>
#include <limits>
#include "Constraint.hpp"
#include "Scheduler.hpp"

using namespace popsolver;

Model::Model() = default;

Model::~Model() = default;

void Model::addConstraint(std::unique_ptr<Constraint> c) {
  constraints.push_back(std::move(c));
}

Variable Model::addVariable(unsigned min, unsigned max) {
  assert(min <= max);
  Variable v(initialDomains.size());
  if (min == max) {
    auto result = constants.emplace(min, v);
    if (!result.second)
      return result.first->second;
  }
  initialDomains.push_back({min, max});
  isCallOperand.push_back(false);
  return v;
}

Variable Model::addVariable() {
  return addVariable(0, std::numeric_limits<unsigned>::max() - 1);
}

Variable Model::addConstant(unsigned value) {
  return addVariable(value, value);
}

Variable Model::product(const Variable *begin, const Variable *end) {
  const auto numVariables = end - begin;
  assert(numVariables > 0);
  if (numVariables == 1)
    return *begin;
  const auto mid = begin + numVariables / 2;
  const auto left = product(begin, mid);
  const auto right = product(mid, end);
  auto result = addVariable();
  auto p = std::unique_ptr<Constraint>(
             new Product(result, left, right)
           );
  addConstraint(std::move(p));
  return result;
}

Variable Model::product(const std::vector<Variable> &vars) {
  if (vars.empty())
    return addConstant(1);
  return product(vars.data(), vars.data() + vars.size());
}

Variable Model::sum(std::vector<Variable> vars) {
  auto result = addVariable();
  auto p = std::unique_ptr<Constraint>(
             new Sum(result, std::move(vars))
           );
  addConstraint(std::move(p));
  return result;
}

Variable Model::floordiv(Variable left, Variable right) {
  auto result = addVariable();
  auto resultTimesRight = product({result, right});
  // result * right <= left
  lessOrEqual(resultTimesRight, left);
  // result * right + right > left
  less(left, sum({resultTimesRight, right}));
  return result;
}

Variable Model::ceildiv(Variable left, Variable right) {
  auto result = addVariable();
  auto resultTimesRight = product({result, right});
  // result * right >= left
  lessOrEqual(left, resultTimesRight);
  // result * right < left + right
  less(resultTimesRight, sum({left, right}));
  return result;
}

void Model::less(Variable left, Variable right) {
  auto p = std::unique_ptr<Constraint>(
             new Less(left, right)
           );
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
  auto p = std::unique_ptr<Constraint>(
             new LessOrEqual(left, right)
           );
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

Variable Model::call(
    std::vector<Variable> vars,
    std::function<unsigned (const std::vector<unsigned> &values)> f) {
  for (auto var : vars) {
    isCallOperand[var.id] = true;
  }
  auto result = addVariable();
  auto p = std::unique_ptr<Constraint>(
             new GenericAssignment(result, std::move(vars), f)
           );
  addConstraint(std::move(p));
  return result;
}

static bool
foundLowerCostSolution(const Domains domains,
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
    if (domains[Variable(i)].size() > 1 &&
        (!v || varLess(Variable(i), *v))) {
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
  // Perform initial constraint propogation.
  Scheduler scheduler(initialDomains, std::move(constraintPtrs));
  if (scheduler.initialPropagate() &&
      minimize(scheduler, v, foundSolution, solution))
    return solution;
  return Solution();
}
