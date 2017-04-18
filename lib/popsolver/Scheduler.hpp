#ifndef _popsolver_Scheduler_hpp_
#define _popsolver_Scheduler_hpp_

#include <popsolver/Model.hpp>
#include <queue>
#include <unordered_set>
#include <vector>

namespace popsolver {

/// The Scheduler class schedules propagation of constraints. All modifications
/// to domains of variables are made via method on this class to ensure that
/// the relevent constraints are propagated whenever a variable's domain
/// changes.
class Scheduler {
  Domains domains;
  std::queue<Constraint*> worklist;
  std::unordered_set<Constraint*> queued;
  /// Map from each variable to the set of constraints to propagte when the
  /// domain of the variable changes.
  std::vector<std::vector<Constraint*>> variableConstraints;
  void queueConstraints(Variable v) {
    for (auto c : variableConstraints[v.id]) {
      if (queued.insert(c).second)
        worklist.push(c);
    }
  }
public:
  Scheduler(Domains &domains,
            const std::vector<std::vector<Constraint*>> &variableConstraints) :
    domains(domains), variableConstraints(variableConstraints) {}
  const Domains &getDomains() { return domains; }
  void setDomains(Domains value) { domains = value; }
  void set(Variable v, unsigned value) {
    assert(value >= domains[v].min_);
    assert(value <= domains[v].max_);
    domains[v].min_ = domains[v].max_ = value;
    queueConstraints(v);
  }
  void setMin(Variable v, unsigned value) {
    assert(value >= domains[v].min_);
    assert(value <= domains[v].max_);
    domains[v].min_ = value;
    queueConstraints(v);
  }
  void setMax(Variable v, unsigned value) {
    assert(value >= domains[v].min_);
    assert(value <= domains[v].max_);
    domains[v].max_ = value;
    queueConstraints(v);
  }
  bool propagate();
  bool initialPropagate();
};

} // End namespace popsolver.

#endif // _popsolver_Scheduler_hpp_
