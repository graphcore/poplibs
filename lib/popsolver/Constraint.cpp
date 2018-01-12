#include "Constraint.hpp"

#include <limits>
#include <popsolver/Model.hpp>
#include "Scheduler.hpp"

using namespace popsolver;

Constraint::~Constraint() = default;

bool Product::
propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  bool madeChange;
  do {
    madeChange = false;
    static_assert(sizeof(unsigned long long) >= sizeof(unsigned) * 2, "");
    auto maxProduct = static_cast<unsigned long long>(domains[left].max()) *
                      domains[right].max();
    auto minProduct = static_cast<unsigned long long>(domains[left].min()) *
                      domains[right].min();
    if (minProduct > domains[result].max() ||
        maxProduct < domains[result].min()) {
      return false;
    }
    if (minProduct > domains[result].min()) {
      scheduler.setMin(result, minProduct);
      madeChange = true;
    }
    if (maxProduct < domains[result].max()) {
      scheduler.setMax(result, maxProduct);
      madeChange = true;
    }
    if (domains[right].min() != 0) {
      auto newLeftMax = domains[result].max() / domains[right].min();
      if (newLeftMax < domains[left].max()) {
        if (newLeftMax < domains[left].min())
          return false;
        scheduler.setMax(left, newLeftMax);
        madeChange = true;
      }
    }
    if (domains[left].min() != 0) {
      auto newRightMax = domains[result].max() / domains[left].min();
      if (newRightMax < domains[right].max()) {
        if (newRightMax < domains[right].min())
          return false;
        scheduler.setMax(right, newRightMax);
        madeChange = true;
      }
    }
    if (domains[right].max() != 0) {
      auto newLeftMin = domains[result].min() / domains[right].max() +
                        (domains[result].min() % domains[right].max() != 0);
      if (newLeftMin > domains[left].min()) {
        if (newLeftMin > domains[left].max())
          return false;
        scheduler.setMin(left, newLeftMin);
        madeChange = true;
      }
    }
    if (domains[left].max() != 0) {
      auto newRightMin = domains[result].min() / domains[left].max() +
                         (domains[result].min() % domains[left].max() != 0);
      if (newRightMin > domains[right].min()) {
        if (newRightMin > domains[right].max())
          return false;
        scheduler.setMin(right, newRightMin);
        madeChange = true;
      }
    }
  } while (madeChange);
  return true;
}

bool Sum::
propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  // The data type used to store the max sum must be large enough to store
  // the maximum value of the result plus the maximum value of any operand so
  // that we can compute the max sum of a subset containing all but one variable
  // by subtracting the the maximum value of the variable from the max sum.
  static_assert(sizeof(unsigned long long) >= sizeof(unsigned) * 2, "");
  unsigned long long minSum = 0;
  unsigned long long maxSum = 0;
  for (const auto &v : vars) {
    minSum = minSum + domains[v].min();
    maxSum = maxSum + domains[v].max();
  }
  if (minSum > domains[result].max() ||
      maxSum < domains[result].min()) {
    return false;
  }
  if (minSum > domains[result].min()) {
    scheduler.setMin(result, minSum);
  }
  if (maxSum < domains[result].max()) {
    scheduler.setMax(result, maxSum);
  }
  for (const auto &v : vars) {
    auto &domain = domains[v];
    auto minOtherVarsSum = minSum - domain.min();
    if (minOtherVarsSum > domains[result].max())
      return false;
    auto newMax = domains[result].max() - minOtherVarsSum;
    if (newMax < domain.min())
      return false;
    if (newMax < domain.max())
      scheduler.setMax(v, newMax);
    auto maxOtherVarsSum = maxSum - domain.max();
    if (maxOtherVarsSum < domains[result].min()) {
      auto newMin = domains[result].min() - maxOtherVarsSum;
      if (newMin > domain.max())
        return false;
      if (newMin > domain.min())
        scheduler.setMin(v, newMin);
    }
  }
  return true;
}

bool Less::
propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  if (domains[left].min() >= domains[right].max()) {
    return false;
  }
  if (domains[left].min() >= domains[right].min()) {
    scheduler.setMin(right, domains[left].min() + 1);
  }
  if (domains[right].max() <= domains[left].max()) {
    scheduler.setMax(left, domains[right].max() - 1);
  }
  return true;
}

bool LessOrEqual::
propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  if (domains[left].min() > domains[right].max()) {
    return false;
  }
  if (domains[left].min() > domains[right].min()) {
    scheduler.setMin(right, domains[left].min());
  }
  if (domains[right].max() < domains[left].max()) {
    scheduler.setMax(left, domains[right].max());
  }
  return true;
}

bool GenericAssignment::
propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  for (std::size_t i = 0; i != vars.size(); ++i) {
    if (domains[vars[i]].size() > 1) {
      return true;
    }
    values[i] = domains[vars[i]].val();
  }
  auto x = f(values);
  if (x < domains[result].min() || x > domains[result].max())
    return false;
  if (domains[result].min() != x || domains[result].max() != x) {
    scheduler.set(result, x);
  }
  return true;
}
