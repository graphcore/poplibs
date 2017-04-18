#include <popsolver/Model.hpp>

#include <algorithm>
#include <limits>
#include "Constraint.hpp"
#include "Scheduler.hpp"

using namespace popsolver;

Model::Model() = default;

Model::~Model() = default;

void Model::addConstraint(std::unique_ptr<Constraint> c) {
  for (auto v : c->getVariables()) {
    variableConstraints[v.id].push_back(c.get());
  }
  constraints.push_back(std::move(c));
}

Variable Model::addVariable(unsigned min, unsigned max) {
  assert(min <= max);
  Variable v(initialDomains.size());
  initialDomains.push_back({min, max});
  variableConstraints.emplace_back();
  return v;
}

Variable Model::addVariable() {
  return addVariable(0, std::numeric_limits<unsigned>::max() - 1);
}

Variable Model::product(std::vector<Variable> vars) {
  auto result = addVariable();
  auto p = std::unique_ptr<Constraint>(
             new Product(result, std::move(vars))
           );
  addConstraint(std::move(p));
  return result;
}

Variable Model::sum(std::vector<Variable> vars) {
  auto result = addVariable();
  auto p = std::unique_ptr<Constraint>(
             new Sum(result, std::move(vars))
           );
  addConstraint(std::move(p));
  return result;
}

void Model::lessOrEqual(Variable left, Variable right) {
  auto p = std::unique_ptr<Constraint>(
             new LessOrEqual(left, right)
           );
  addConstraint(std::move(p));
}

void Model::lessOrEqual(Variable left, unsigned right) {
  auto rightVar = addVariable(right, right);
  lessOrEqual(left, rightVar);
}

void Model::lessOrEqual(unsigned left, Variable right) {
  auto leftVar = addVariable(left, left);
  lessOrEqual(leftVar, right);
}

Variable Model::call(
    std::vector<Variable> vars,
    std::function<unsigned (const std::vector<unsigned> &values)> f) {
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
  // Find the unassigned variable with the smallest domain.
  const auto &domains = scheduler.getDomains();
  auto match = domains.end();
  for (auto it = domains.begin(); it != domains.end(); ++it) {
    if (it->size() > 1 &&
        (match == domains.end() || it->size() < match->size()))
      match = it;
  }
  if (match == domains.end()) {
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
  Variable variable(match - domains.begin());
  bool improvedSolution = false;
  for (unsigned value = scheduler.getDomains()[variable].min();
       value <= scheduler.getDomains()[variable].max(); ++value) {
    auto savedDomains = scheduler.getDomains();
    scheduler.set(variable, value);
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

  // Perform initial constraint propogation.
  Scheduler scheduler(initialDomains, variableConstraints);
  if (scheduler.initialPropagate() &&
      minimize(scheduler, v, foundSolution, solution))
    return solution;
  throw NoSolution();
}
