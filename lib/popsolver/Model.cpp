// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <popsolver/Model.hpp>

#include "Constraint.hpp"
#include "GraphUtil.hpp"
#include "Scheduler.hpp"

#include <boost/graph/depth_first_search.hpp>
#include <boost/optional.hpp>

#include <algorithm>
#include <cassert>
#include <limits>
#include <ostream>

using namespace popsolver;

Model::Model() {
  defaultPriorityGroup = PriorityGroupID(boost::add_vertex(priorityGraph));
}

Model::~Model() = default;

void Model::addConstraint(std::unique_ptr<Constraint> c) {
  constraints.push_back(std::move(c));
}

std::string Model::makeBinaryOpDebugName(const Variable *begin,
                                         const Variable *end,
                                         const std::string &opStr) const {
  assert(begin != end);
  auto name = getDebugName(*begin);
  for (auto it = std::next(begin); it != end; ++it) {
    name += opStr + getDebugName(*it);
  }
  return name;
}

std::string Model::makeBinaryOpDebugName(const std::vector<Variable> &v,
                                         const std::string &opStr) const {
  return makeBinaryOpDebugName(&v[0], &v[v.size()], opStr);
}

std::string
Model::makeParameterisedOpDebugName(const Variable *begin, const Variable *end,
                                    const std::string &opStr) const {
  assert(begin != end);
  auto name = opStr + "(" + getDebugName(*begin);
  for (auto it = std::next(begin); it != end; ++it) {
    name += "," + getDebugName(*it);
  }
  name += ")";
  return name;
}

std::string
Model::makeParameterisedOpDebugName(const std::vector<Variable> &v,
                                    const std::string &opStr) const {
  return makeParameterisedOpDebugName(&v[0], &v[v.size()], opStr);
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
  return debugNames[v];
}

Variable Model::addVariable(DataType min, DataType max,
                            const std::string &debugName) {
  assert(min <= max);
  Variable v(initialDomains.size());
  if (min == max) {
    auto result = constants.emplace(min, v);
    if (!result.second)
      return result.first->second;
  }
  initialDomains.push_back({min, max});
  priorityGroup.push_back(getDefaultPriorityGroup());
  if (debugName.empty()) {
    debugNames.emplace_back("var#" + std::to_string(v.value));
  } else {
    debugNames.emplace_back(debugName);
  }
  return v;
}

Variable Model::addVariable(const std::string &debugName) {
  return addVariable(popsolver::DataType{0}, DataType::max(), debugName);
}

Variable Model::addVariable(DataType::UnderlyingType min,
                            DataType::UnderlyingType max,
                            const std::string &debugName) {
  return addVariable(DataType{min}, DataType{max}, debugName);
}

Variable Model::addConstant(DataType value, const std::string &debugName) {
  return addVariable(value, value,
                     debugName.empty() ? std::to_string(*value) : debugName);
}

Variable Model::addConstant(DataType::UnderlyingType value,
                            const std::string &debugName) {
  return addConstant(DataType{value}, debugName);
}

Variable Model::zero() { return addConstant(0u); }

Variable Model::one() { return addConstant(1u); }

Variable Model::product(const Variable *begin, const Variable *end,
                        const std::string &debugName = "") {
  const auto numVariables = end - begin;
  assert(numVariables > 0);
  if (numVariables == 1)
    return *begin;
  const auto mid = begin + numVariables / 2;
  const auto left = product(begin, mid);
  const auto right = product(mid, end);
  auto result = addVariable(
      debugName.empty() ? makeBinaryOpDebugName(begin, end, "*") : debugName);
  auto p = std::unique_ptr<Constraint>(new Product(result, left, right));
  addConstraint(std::move(p));
  return result;
}

Variable Model::product(const std::vector<Variable> &vars,
                        const std::string &debugName) {
  if (vars.empty())
    return addConstant(1);
  return product(vars.data(), vars.data() + vars.size(), debugName);
}

Variable Model::sum(std::vector<Variable> vars, const std::string &debugName) {
  if (vars.empty()) {
    return addConstant(0);
  }
  if (vars.size() == 1) {
    return vars[0];
  }
  auto result = addVariable(debugName.empty() ? makeBinaryOpDebugName(vars, "+")
                                              : debugName);
  auto p = std::unique_ptr<Constraint>(new Sum(result, std::move(vars)));
  addConstraint(std::move(p));
  return result;
}

Variable Model::max(std::vector<Variable> vars, const std::string &debugName) {
  if (vars.size() == 1) {
    return vars[0];
  }
  auto result =
      addVariable(debugName.empty() ? makeParameterisedOpDebugName(vars, "max")
                                    : debugName);
  auto p = std::unique_ptr<Constraint>(new Max(result, std::move(vars)));
  addConstraint(std::move(p));
  return result;
}

Variable Model::min(std::vector<Variable> vars, const std::string &debugName) {
  if (vars.size() == 1) {
    return vars[0];
  }
  auto result =
      addVariable(debugName.empty() ? makeParameterisedOpDebugName(vars, "min")
                                    : debugName);
  auto p = std::unique_ptr<Constraint>(new Min(result, std::move(vars)));
  addConstraint(std::move(p));
  return result;
}

Variable Model::sub(Variable left, Variable right,
                    const std::string &debugName) {
  auto result =
      addVariable(debugName.empty() ? makeBinaryOpDebugName({left, right}, "-")
                                    : debugName);
  auto p = std::unique_ptr<Constraint>(new Sum(left, {result, right}));
  addConstraint(std::move(p));
  return result;
}

Variable Model::floordiv(Variable left, Variable right,
                         const std::string &debugName) {
  auto result =
      addVariable(debugName.empty()
                      ? makeParameterisedOpDebugName({left, right}, "floordiv")
                      : debugName);
  auto resultTimesRight = product({result, right});
  // result * right <= left
  lessOrEqual(resultTimesRight, left);
  // result * right + right > left
  less(left, sum({resultTimesRight, right}));
  return result;
}

Variable Model::ceildiv(Variable left, Variable right,
                        const std::string &debugName) {
  auto result = addVariable(
      debugName.empty() ? makeParameterisedOpDebugName({left, right}, "ceildiv")
                        : debugName);
  auto resultTimesRight = product({result, right});
  // result * right >= left
  lessOrEqual(left, resultTimesRight);
  // result * right < left + right
  less(resultTimesRight, sum({left, right}));
  return result;
}

Variable Model::ceildivConstrainDivisor(const Variable left,
                                        const Variable right,
                                        const std::string &debugName) {
  const auto result = ceildiv(left, right, debugName);
  // The "remainder" from the division is < right for the minimal divisor
  // result * right < left + result
  less(product({result, right}), sum({left, result}));
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

void Model::less(Variable left, DataType right) {
  auto rightVar = addConstant(right);
  less(left, rightVar);
}

void Model::less(DataType left, Variable right) {
  auto leftVar = addConstant(left);
  less(leftVar, right);
}

Variable Model::reifiedLess(Variable left, Variable right) {
  // For the condition to be true, (right - min(left, right)) > 0.
  // We can therefore calculate (right - min(left, right)), and
  // max out the value at 1 i.e.:
  // min(1, right - min(left, right))
  return min({one(), sub(right, min({left, right}))});
}

void Model::lessOrEqual(Variable left, Variable right) {
  auto p = std::unique_ptr<Constraint>(new LessOrEqual(left, right));
  addConstraint(std::move(p));
}

void Model::lessOrEqual(Variable left, DataType right) {
  auto rightVar = addConstant(right);
  lessOrEqual(left, rightVar);
}

void Model::lessOrEqual(DataType left, Variable right) {
  auto leftVar = addConstant(left);
  lessOrEqual(leftVar, right);
}

Variable Model::reifiedLessOrEqual(Variable left, Variable right) {
  // For the condition to be false, left - min(left, right) > 0.
  // We can therefore calculate left - min(left, right),
  // max out the value at 1, and invert it i.e.:
  // 1 - min(1, left - min(left, right)).
  return sub(one(), min({one(), sub(left, min({left, right}))}));
}

void Model::equal(Variable left, Variable right) {
  auto p = std::unique_ptr<Constraint>(new Sum(left, {right}));
  addConstraint(std::move(p));
}

void Model::equal(DataType left, Variable right) {
  equal(addConstant(left), right);
}

void Model::equal(Variable left, DataType right) {
  equal(left, addConstant(right));
}

Variable Model::reifiedEqual(Variable left, Variable right) {
  // For the condition to be false max(left, right) - min(left, right) > 0.
  // We can therefore calculate max(left, right) - min(left, right),
  // max out the value at 1, and invert it i.e.:
  // 1 - min(1, (max(left, right) - min(left, right))).
  return sub(one(), min({one(), sub(max({left, right}), min({left, right}))}));
}

void Model::factorOf(DataType left_, Variable right) {
  const auto left = addConstant(left_);
  const auto result =
      addVariable(makeParameterisedOpDebugName({left, right}, "factorOf"));
  equal(left, product({result, right}));
}

void Model::factorOf(Variable left, Variable right) {
  const auto result =
      addVariable(makeParameterisedOpDebugName({left, right}, "factorOf"));
  equal(left, product({result, right}));
}

Variable Model::booleanOr(Variable left, Variable right) {
  return min({one(), sum({min({left, one()}), min({right, one()})})});
}

Variable Model::booleanAnd(Variable left, Variable right) {
  return min({one(), product({min({left, one()}), min({right, one()})})});
}

Variable Model::booleanNot(Variable v) { return sub(one(), min({v, one()})); }

template <typename T>
Variable Model::call(
    std::vector<Variable> vars,
    std::function<boost::optional<DataType>(const std::vector<T> &values)> f,
    const std::string &debugName) {
  auto result = addVariable(debugName);
  auto p = std::unique_ptr<Constraint>(
      new GenericAssignment<T>(result, std::move(vars), f));
  addConstraint(std::move(p));
  return result;
}

template Variable
Model::call<DataType>(std::vector<Variable>,
                      std::function<boost::optional<DataType>(
                          const std::vector<DataType> &values)>,
                      const std::string &);

template Variable
Model::call<unsigned>(std::vector<Variable>,
                      std::function<boost::optional<DataType>(
                          const std::vector<unsigned> &values)>,
                      const std::string &);

template Variable
Model::call<uint64_t>(std::vector<Variable>,
                      std::function<boost::optional<DataType>(
                          const std::vector<uint64_t> &values)>,
                      const std::string &);

PriorityGroupID Model::getDefaultPriorityGroup() const {
  return PriorityGroupID(0u);
}

PriorityGroupID Model::addPriorityGroup() {
  auto newID = boost::add_vertex(priorityGraph);
  return PriorityGroupID(newID);
}

void Model::setPriorityGroup(Variable v, PriorityGroupID prioGroup) {
  priorityGroup[v] = prioGroup;
}

void Model::prioritiseOver(PriorityGroupID a, PriorityGroupID b) {
  boost::add_edge(a.value, b.value, priorityGraph);
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

std::pair<bool, ConstraintEvaluationSummary>
Model::minimize(Scheduler &scheduler, const std::vector<Variable> &objectives,
                const boost::adjacency_matrix<> &prioGroupReachability,
                bool &foundSolution, Solution &solution) {
  ConstraintEvaluationSummary summary{};
  // Find an unassigned variable.
  const auto &domains = scheduler.getDomains();

  const auto lhsIsHigherPriority = [&](Variable i, Variable j) {
    const auto iPriorityGroup = priorityGroup[i];
    const auto jPriorityGroup = priorityGroup[j];
    if (boost::edge(iPriorityGroup, jPriorityGroup, prioGroupReachability)
            .second) {
      return true;
    }
    if (boost::edge(jPriorityGroup, iPriorityGroup, prioGroupReachability)
            .second) {
      return false;
    }
    // Use the size of the domain to break ties.
    return domains[i].size() < domains[j].size();
  };
  boost::optional<Variable> v;
  const auto numVars = domains.size();
  for (std::size_t i = 0; i != numVars; ++i) {
    if (domains[Variable(i)].size() != popsolver::DataType{0} &&
        (!v || lhsIsHigherPriority(Variable(i), *v))) {
      v = Variable(i);
    }
  }
  if (!v) {
    // All variables are assigned.
    if (!foundSolution) {
      std::vector<DataType> values;
      values.reserve(domains.size());
      for (const auto &d : domains) {
        values.push_back(d.val());
      }
      solution = Solution(std::move(values));
      foundSolution = true;
      return {true, summary};
    }
    if (foundLowerCostSolution(domains, objectives, solution)) {
      for (std::size_t i = 0; i != domains.size(); ++i) {
        solution[Variable(i)] = domains[Variable(i)].val();
      }
      return {true, summary};
    }
    return {false, summary};
  }
  // Evaluate the cost for every possible value of this variable.
  bool improvedSolution = false;
  for (DataType value = scheduler.getDomains()[*v].min();
       value <= scheduler.getDomains()[*v].max(); ++value) {
    auto savedDomains = scheduler.getDomains();
    scheduler.set(*v, value);

    const auto valueImprovedSolution = [&]() {
      const auto x = scheduler.propagate();
      summary += x.second;
      if (x.first) {
        const auto y = minimize(scheduler, objectives, prioGroupReachability,
                                foundSolution, solution);
        summary += y.second;
        if (y.first) {
          return true;
        }
      }
      return false;
    }();

    scheduler.setDomains(std::move(savedDomains));
    if (valueImprovedSolution) {
      improvedSolution = true;
      scheduler.setMax(objectives.front(), solution[objectives.front()]);
      const auto succeeded = scheduler.propagate();
      assert(succeeded.first);
      summary += succeeded.second;
    }
  }
  return {improvedSolution, summary};
}

template <typename Tag>
struct DAGCheckerVisitor : public boost::base_visitor<DAGCheckerVisitor<Tag>> {
  typedef Tag event_filter;
  template <typename Edge, typename Graph> void operator()(Edge e, Graph &g) {
    throw std::logic_error("Graph formed by prioritisation of priority groups "
                           "over one another does not form a DAG");
  }
};

template <class Tag> inline DAGCheckerVisitor<Tag> checkIsDAG(Tag) {
  return DAGCheckerVisitor<Tag>{};
}

Solution Model::minimize(const std::vector<Variable> &v) {
  bool foundSolution = false;
  Solution solution;

  // Check the graph formed by prioritisation of groups
  // over one another is a DAG.
  boost::depth_first_search(priorityGraph,
                            boost::visitor(boost::make_dfs_visitor(
                                checkIsDAG(boost::on_back_edge()))));

  const auto prioGroupReachability = pairwise_reachability(priorityGraph);

  std::vector<Constraint *> constraintPtrs;
  constraintPtrs.reserve(constraints.size());
  for (const auto &c : constraints) {
    constraintPtrs.push_back(c.get());
  }
  ConstraintEvaluationSummary summary{};
  // Perform initial constraint propagation.
  Scheduler scheduler(initialDomains, std::move(constraintPtrs));
  const auto success = [&]() {
    const auto x = scheduler.initialPropagate();
    summary += x.second;
    if (x.first) {
      const auto y = minimize(scheduler, v, prioGroupReachability,
                              foundSolution, solution);
      summary += y.second;
      if (y.first) {
        return true;
      }
    }
    return false;
  }();
  if (success) {
    solution.constraintEvalSummary = summary;
    return solution;
  } else {
    Solution invalidSolution{};
    invalidSolution.constraintEvalSummary = summary;
    return invalidSolution;
  }
}
