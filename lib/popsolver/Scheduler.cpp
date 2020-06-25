// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "Scheduler.hpp"

#include "Constraint.hpp"

#include <poplibs_support/logging.hpp>

using namespace popsolver;

Scheduler::Scheduler(Domains domains_, std::vector<Constraint *> constraints_)
    : domains(std::move(domains_)), constraints(std::move(constraints_)) {
  unsigned numConstraints = constraints.size();
  queued.resize(numConstraints);
  for (unsigned c = 0; c != numConstraints; ++c) {
    for (auto v : constraints[c]->getVariables()) {
      if (variableConstraints.size() <= v.id) {
        variableConstraints.resize(v.id + 1);
      }
      variableConstraints[v.id].push_back(c);
    }
  }
}

std::pair<bool, ConstraintEvaluationSummary> Scheduler::propagate() {
  ConstraintEvaluationSummary constraintEvalCount{};
  while (!worklist.empty()) {
    const auto cid = worklist.front();
    const auto constraint = constraints[cid];
    worklist.pop();

    if (poplibs_support::logging::shouldLog(
            poplibs_support::logging::Level::Trace)) {
      // Only provide a breakdown when we are trace log level because this is
      // too slow to be on by default.
      if (dynamic_cast<GenericAssignment *>(constraint) != nullptr) {
        constraintEvalCount.call++;
      } else if (dynamic_cast<Product *>(constraint) != nullptr) {
        constraintEvalCount.product++;
      } else if (dynamic_cast<Sum *>(constraint) != nullptr) {
        constraintEvalCount.sum++;
      } else if (dynamic_cast<Max *>(constraint) != nullptr) {
        constraintEvalCount.max++;
      } else if (dynamic_cast<Min *>(constraint) != nullptr) {
        constraintEvalCount.min++;
      } else if (dynamic_cast<Less *>(constraint) != nullptr) {
        constraintEvalCount.less++;
      } else if (dynamic_cast<LessOrEqual *>(constraint) != nullptr) {
        constraintEvalCount.lessOrEqual++;
      } else {
        constraintEvalCount.unknown++;
      }
    } else {
      constraintEvalCount.unknown++;
    }

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
  for (unsigned i = 0; i != variableConstraints.size(); ++i) {
    queueConstraints(Variable(i));
  }
  return propagate();
}
