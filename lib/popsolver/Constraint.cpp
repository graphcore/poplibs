#include "Constraint.hpp"

#include <limits>
#include <popsolver/Model.hpp>
#include "Scheduler.hpp"

using namespace popsolver;

/// Saturated multipliciation.
static unsigned satMul(unsigned a, unsigned b) {
  static_assert(sizeof(unsigned long long) >= sizeof(unsigned) * 2, "");
  auto result = (unsigned long long)a * b;
  auto max = std::numeric_limits<unsigned>::max();
  return std::min<unsigned long long>(result, max);
}

/// Saturated addition.
static unsigned satAdd(unsigned a, unsigned b) {
  auto result = a + b;
  if (result < a)
    return std::numeric_limits<unsigned>::max();
  return result;
}

Constraint::~Constraint() = default;

bool Product::
propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  unsigned minProduct = 1;
  for (const auto &v : vars) {
    minProduct = satMul(minProduct, domains[v].min());
  }
  if (minProduct > domains[result].max()) {
    return false;
  }
  if (minProduct > domains[result].min()) {
    scheduler.setMin(result, minProduct);
  }
  if (minProduct != 0) {
    for (const auto &v : vars) {
      auto &domain = domains[v];
      auto newMax = domains[result].max() / (minProduct / domain.min());
      if (newMax < domain.min())
        return false;
      auto oldMax = domain.max();
      if (newMax >= oldMax)
        continue;
      scheduler.setMax(v, newMax);
    }
  }
  unsigned maxProduct = 1;
  for (const auto &v : vars) {
    maxProduct = satMul(maxProduct, domains[v].max());
  }
  if (maxProduct < domains[result].min()) {
    return false;
  }
  if (maxProduct < domains[result].max()) {
    scheduler.setMax(result, maxProduct);
  }
  return true;
}

bool Sum::
propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  unsigned minSum = 0;
  for (const auto &v : vars) {
    minSum = satAdd(minSum, domains[v].min());
  }
  if (minSum > domains[result].max()) {
    return false;
  }
  if (minSum > domains[result].min()) {
    scheduler.setMin(result, minSum);
  }
  for (const auto &v : vars) {
    auto &domain = domains[v];
    auto newMax = domains[result].max() - minSum + domain.min();
    if (newMax < domain.min())
      return false;
    if (newMax >= domain.max())
      continue;
    scheduler.setMax(v, newMax);
  }
  unsigned maxSum = 0;
  for (const auto &v : vars) {
    maxSum = satAdd(maxSum, domains[v].max());
  }
  if (maxSum < domains[result].min()) {
    return false;
  }
  if (maxSum < domains[result].max()) {
    scheduler.setMax(result, maxSum);
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
    scheduler.setMax(left, domains[right].min());
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
