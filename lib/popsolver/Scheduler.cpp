// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "Scheduler.hpp"

#include "Constraint.hpp"

#include <poplibs_support/logging.hpp>

using namespace popsolver;

Scheduler::Scheduler(Domains domains_, std::vector<Constraint *> constraints_)
    : domains(std::move(domains_)), constraints(std::move(constraints_)) {
  const auto numConstraints = constraints.size();
  queued.resize(numConstraints);
  for (std::size_t c = 0; c != numConstraints; ++c) {
    for (auto v : constraints[c]->getVariables()) {
      if (variableConstraints.size() <= v) {
        variableConstraints.resize(v + 1);
      }
      variableConstraints[v].push_back(c);
    }
  }
}

std::pair<bool, ConstraintEvaluationSummary> Scheduler::propagate() {
  ConstraintEvaluationSummary constraintEvalCount{};
  while (!worklist.empty()) {
    const auto cid = worklist.front();
    const auto constraint = constraints[cid];
    worklist.pop();
    constraintEvalCount++;

    // Note we don't remove the constaint from the queued set until after the
    // call to propagate(). The propagate() method is responsible for computing
    // the fixed point such that a second call to propagate() immediately after
    // would make no further changes.
    bool succeeded = constraint->propagate(*this);
    queued[cid] = 0;
    if (!succeeded) {
      return {false, constraintEvalCount};
    }
  }
  return {true, constraintEvalCount};
}

std::pair<bool, ConstraintEvaluationSummary> Scheduler::initialPropagate() {
  for (std::size_t i = 0; i != variableConstraints.size(); ++i) {
    queueConstraints(Variable(i));
  }
  return propagate();
}
