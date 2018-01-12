#include "Constraint.hpp"

#include <limits>
#include <popsolver/Model.hpp>
#include "Scheduler.hpp"

using namespace popsolver;

/// Saturated multipliciation.
static unsigned long long satMul(unsigned long long a, unsigned long long b) {
#if __GNUC__ >= 5 || __has_builtin(__builtin_umulll_overflow)
  unsigned long long result;
  if (__builtin_umulll_overflow(a, b, &result))
    return std::numeric_limits<unsigned long long>::max();
#else
  unsigned long long result = a * b;
  bool overflow = a != 0 && result / a != b;
  if (overflow)
    result = std::numeric_limits<unsigned long long>::max();
#endif
  return result;
}

Constraint::~Constraint() = default;

bool Product::
propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  bool madeChange;
  do {
    madeChange = false;
    // The data type used to store the max product must be large enough to store
    // the maximum value of the result times the maximum value of any operand so
    // that we can compute the max product of a subset containing all but on
    // variable by dividing max product by the maximum value of the variable.
    static_assert(sizeof(unsigned long long) >= sizeof(unsigned) * 2, "");
    unsigned long long minProduct = 1;
    unsigned long long maxProduct = 1;
    for (const auto &v : vars) {
      minProduct = satMul(minProduct, domains[v].min());
      maxProduct = satMul(maxProduct, domains[v].max());
    }
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
    for (const auto &v : vars) {
      auto &domain = domains[v];
      if (minProduct != 0) {
        auto minOtherVarsProduct = minProduct / domain.min();
        auto newMax = domains[result].max() / minOtherVarsProduct;
        if (newMax < domain.min())
          return false;
        if (newMax < domain.max()) {
          scheduler.setMax(v, newMax);
          madeChange = true;
        }
      }
      if (maxProduct != 0) {
        auto maxOtherVarsProduct = maxProduct / domain.max();
        auto newMin = domains[result].min() / maxOtherVarsProduct +
                      (domains[result].min() % maxOtherVarsProduct != 0);
        if (newMin > domain.max())
          return false;
        auto oldMin = domain.min();
        if (newMin > oldMin) {
          scheduler.setMin(v, newMin);
          madeChange = true;
        }
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
