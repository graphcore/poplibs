// Copyright (c) Graphcore Ltd, All rights reserved.
#include "Constraint.hpp"

#include "Scheduler.hpp"
#include <boost/range/iterator_range.hpp>
#include <limits>
#include <popsolver/Model.hpp>

using namespace popsolver;

Constraint::~Constraint() = default;

bool Product::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  const auto result = vars[0];
  const auto left = vars[1];
  const auto right = vars[2];

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

bool Sum::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  const auto result = vars[0];
  const auto args =
      boost::make_iterator_range(std::begin(vars) + 1, std::end(vars));

  // The data type used to store the max sum must be large enough to store
  // the maximum value of the result plus the maximum value of any operand so
  // that we can compute the max sum of a subset containing all but one variable
  // by subtracting the the maximum value of the variable from the max sum.
  static_assert(sizeof(unsigned long long) >= sizeof(unsigned) * 2, "");
  unsigned long long minSum = 0;
  unsigned long long maxSum = 0;
  for (const auto &v : args) {
    minSum = minSum + domains[v].min();
    maxSum = maxSum + domains[v].max();
  }
  if (minSum > domains[result].max() || maxSum < domains[result].min()) {
    return false;
  }
  if (minSum > domains[result].min()) {
    scheduler.setMin(result, minSum);
  }
  if (maxSum < domains[result].max()) {
    scheduler.setMax(result, maxSum);
  }
  for (const auto &v : args) {
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

bool Max::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  const auto result = vars[0];
  const auto args =
      boost::make_iterator_range(std::begin(vars) + 1, std::end(vars));

  // give A = max(B, C), we can deduce:
  //  - upperbound(A) = min(upperbound(A), max(upperbound(B), upperbound(C))),
  //  - lowerbound(A) = max(lowerbound(A), max(lowerbound(B), lowerbound(C))),
  //  - upperbound(B) = min(upperbound(A), upperbound(B)),
  //  - upperbound(C) = min(upperbound(A), upperbound(C))
  // propagation will fail if:
  //  - upperbound(A) < max(lowerbound(B), lowerbound(C)),
  //  - lowerbound(A) > max(upperbound(B), upperbound(C))

  const auto resultLower = domains[result].min();
  const auto resultUpper = domains[result].max();

  auto maxLowerBound = std::numeric_limits<unsigned>::min();
  auto minLowerBound = std::numeric_limits<unsigned>::max();
  auto maxUpperBound = std::numeric_limits<unsigned>::min();
  auto minUpperBound = std::numeric_limits<unsigned>::max();

  for (const auto &var : args) {
    if (domains[var].min() > resultUpper) {
      return false;
    } else if (domains[var].max() > resultUpper) {
      scheduler.setMax(var, resultUpper);
    }

    maxLowerBound = std::max(maxLowerBound, domains[var].min());
    minLowerBound = std::min(minLowerBound, domains[var].min());
    maxUpperBound = std::max(maxUpperBound, domains[var].max());
    minUpperBound = std::min(minUpperBound, domains[var].max());
  }

  assert(resultUpper >= maxLowerBound);
  if (resultLower > maxUpperBound) {
    return false;
  }

  if (resultUpper > maxUpperBound) {
    scheduler.setMax(result, maxUpperBound);
  }

  if (resultLower < maxLowerBound) {
    scheduler.setMin(result, maxLowerBound);
  }

  return true;
}

bool Less::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  const auto left = vars[0];
  const auto right = vars[1];

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

bool LessOrEqual::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  const auto left = vars[0];
  const auto right = vars[1];

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

bool GenericAssignment::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  for (std::size_t i = 1; i != vars.size(); ++i) {
    if (domains[vars[i]].size() > 1) {
      return true;
    }
    values[i - 1] = domains[vars[i]].val();
  }
  const auto result = vars[0];
  const auto x = f(values);
  if (x < domains[result].min() || x > domains[result].max())
    return false;
  if (domains[result].min() != x || domains[result].max() != x) {
    scheduler.set(result, x);
  }
  return true;
}
