#include "Scheduler.hpp"

#include "Constraint.hpp"

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

bool Scheduler::propagate() {
  while (!worklist.empty()) {
    auto c = worklist.front();
    worklist.pop();
    // Note we don't remove the constaint from the queued set until after the
    // call to propagate(). The propagate() method is responsible for computing
    // the fixed point such that a second call to propagate() immediately after
    // would make no further changes.
    bool succeeded = constraints[c]->propagate(*this);
    queued[c] = 0;
    if (!succeeded)
      return false;
  }
  return true;
}

bool Scheduler::initialPropagate() {
  for (unsigned i = 0; i != variableConstraints.size(); ++i) {
    queueConstraints(Variable(i));
  }
  return propagate();
}
