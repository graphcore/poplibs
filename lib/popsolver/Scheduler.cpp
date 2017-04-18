#include "Scheduler.hpp"

#include "Constraint.hpp"

using namespace popsolver;

bool Scheduler::propagate() {
  while (!worklist.empty()) {
    auto c = worklist.front();
    worklist.pop();
    // Note we don't remove the constaint from the queued set until after the
    // call to propagate(). The propagate() method is responsible for computing
    // the fixed point such that a second call to propagate() immediately after
    // would make no further changes.
    bool succeeded = c->propagate(*this);
    queued.erase(c);
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
