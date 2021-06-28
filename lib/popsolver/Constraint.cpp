// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "Constraint.hpp"

#include "Scheduler.hpp"

#include <popsolver/Model.hpp>

#include <poplibs_support/Visitor.hpp>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/range/iterator_range.hpp>

#include <limits>

using namespace popsolver;

using BigInteger = boost::multiprecision::uint128_t;
static_assert(
    sizeof(BigInteger) >= sizeof(DataType::UnderlyingType) * 2,
    "BigInteger isn't large enough to perform operations on DataType");

Constraint::~Constraint() = default;

bool Product::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  const auto result = vars[0];
  const auto left = vars[1];
  const auto right = vars[2];

  bool madeChange;
  do {
    madeChange = false;

    // Check if the result doesn't match the inputs
    // left[max] * right[max] = result[max]
    // left[min] * right[min] = result[min]
    const BigInteger maxProduct =
        BigInteger{*domains[left].max()} * BigInteger{*domains[right].max()};
    const BigInteger minProduct =
        BigInteger{*domains[left].min()} * BigInteger{*domains[right].min()};
    if (minProduct > *domains[result].max() ||
        maxProduct < *domains[result].min()) {
      return false;
    }
    if (minProduct > *domains[result].min()) {
      // This is fine to do unchecked because minProduct is less than the
      // value in *domains[result].max() which is limited to DataType. Otherwise
      // We would have already bailed out.
      scheduler.setMin(result,
                       popsolver::DataType{
                           minProduct.convert_to<DataType::UnderlyingType>()});
      madeChange = true;
    }
    if (maxProduct < *domains[result].max()) {
      // This is fine to do unchecked because we are less than the value of
      // *domains[result].max(), which is of type DataType.
      scheduler.setMax(result,
                       popsolver::DataType{
                           maxProduct.convert_to<DataType::UnderlyingType>()});
      madeChange = true;
    }

    // Check if the inputs do not match the result
    // We shortcut some calculations avoiding a costly divide by noting:
    // result[min] <= left[max] * right[min] <= result[max]
    // result[min] <= left[min] * right[max] <= result[max]

    // If we want to reduce the value of left[max] we can check if it
    // is valid by comparing against result[max].
    //
    // If (left[max] * right[min] <= result[max]) is not true,
    // we can reduce the value of left[max] to be result[max] / right[min]. And
    // likewise for all other permutations.

    // left[max] * right[min] <= result[max]
    if (domains[right].min() != popsolver::DataType{0} &&
        BigInteger{*domains[right].min()} * BigInteger{*domains[left].max()} >
            BigInteger{*domains[result].max()}) {
      auto newLeftMax = domains[result].max() / domains[right].min();
      assert(newLeftMax < domains[left].max());
      if (newLeftMax < domains[left].min())
        return false;
      scheduler.setMax(left, newLeftMax);
      madeChange = true;
    }
    // left[min] * right[max] <= result[max]
    if (domains[left].min() != popsolver::DataType{0} &&
        BigInteger{*domains[left].min()} * BigInteger{*domains[right].max()} >
            BigInteger{*domains[result].max()}) {
      auto newRightMax = domains[result].max() / domains[left].min();
      assert(newRightMax < domains[right].max());
      if (newRightMax < domains[right].min())
        return false;
      scheduler.setMax(right, newRightMax);
      madeChange = true;
    }

    // left[min] * right[max] >= result[min]
    if (domains[right].max() != popsolver::DataType{0} &&
        BigInteger{*domains[right].max()} * BigInteger{*domains[left].min()} <
            BigInteger{*domains[result].min()}) {
      auto newLeftMin =
          domains[result].min() / domains[right].max() +
          popsolver::DataType{(domains[result].min() % domains[right].max() !=
                               popsolver::DataType{0})};
      if (newLeftMin > domains[left].min()) {
        if (newLeftMin > domains[left].max())
          return false;
        scheduler.setMin(left, newLeftMin);
        madeChange = true;
      }
    }
    // left[max] * right[min] >= result[min]
    if (domains[left].max() != popsolver::DataType{0} &&
        BigInteger{*domains[left].max()} * BigInteger{*domains[right].min()} <
            BigInteger{*domains[result].min()}) {
      auto newRightMin =
          domains[result].min() / domains[left].max() +
          popsolver::DataType{(domains[result].min() % domains[left].max() !=
                               popsolver::DataType{0})};
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
  BigInteger minSum = 0;
  BigInteger maxSum = 0;
  for (const auto &v : args) {
    minSum = minSum + *domains[v].min();
    maxSum = maxSum + *domains[v].max();
  }
  if (minSum > *domains[result].max() || maxSum < *domains[result].min()) {
    return false;
  }
  if (minSum > *domains[result].min()) {
    scheduler.setMin(
        result,
        popsolver::DataType{minSum.convert_to<DataType::UnderlyingType>()});
  }
  if (maxSum < *domains[result].max()) {
    scheduler.setMax(
        result,
        popsolver::DataType{maxSum.convert_to<DataType::UnderlyingType>()});
  }
  for (const auto &v : args) {
    auto &domain = domains[v];
    auto minOtherVarsSum = minSum - *domain.min();
    if (minOtherVarsSum > *domains[result].max())
      return false;
    const auto newMax = *domains[result].max() - minOtherVarsSum;
    if (newMax < *domain.min())
      return false;
    const auto oldMax = *domain.max();
    if (newMax < oldMax)
      scheduler.setMax(v, popsolver::DataType{
                              newMax.convert_to<DataType::UnderlyingType>()});
    auto maxOtherVarsSum = maxSum - oldMax;
    if (maxOtherVarsSum < *domains[result].min()) {
      auto newMin = *domains[result].min() - maxOtherVarsSum;
      if (newMin > *domain.max())
        return false;
      if (newMin > *domain.min())
        scheduler.setMin(v, popsolver::DataType{
                                newMin.convert_to<DataType::UnderlyingType>()});
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

  auto maxLowerBound = DataType::min();
  auto minLowerBound = DataType::max();
  auto maxUpperBound = DataType::min();
  auto minUpperBound = DataType::max();

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

bool Min::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  const auto result = vars[0];
  const auto args =
      boost::make_iterator_range(std::begin(vars) + 1, std::end(vars));

  // give A = min(B, C), we can deduce:
  //  - upperbound(A) = min(upperbound(A), min(upperbound(B), upperbound(C))),
  //  - lowerbound(A) = max(lowerbound(A), min(lowerbound(B), lowerbound(C))),
  //  - lowerbound(B) = max(lowerbound(A), lowerbound(B)),
  //  - lowerbound(C) = max(lowerbound(A), lowerbound(C))
  // propagation will fail if:
  //  - upperbound(A) < max(lowerbound(B), lowerbound(C)),
  //  - lowerbound(A) > max(upperbound(B), upperbound(C))

  const auto resultLower = domains[result].min();
  const auto resultUpper = domains[result].max();

  auto maxLowerBound = DataType::min();
  auto minLowerBound = DataType::max();
  auto maxUpperBound = DataType::min();
  auto minUpperBound = DataType::max();

  for (const auto &var : args) {
    if (domains[var].max() < resultLower) {
      return false;
    } else if (domains[var].min() < resultLower) {
      scheduler.setMin(var, resultLower);
    }

    maxLowerBound = std::max(maxLowerBound, domains[var].min());
    minLowerBound = std::min(minLowerBound, domains[var].min());
    maxUpperBound = std::max(maxUpperBound, domains[var].max());
    minUpperBound = std::min(minUpperBound, domains[var].max());
  }

  assert(resultLower <= minUpperBound);
  if (resultUpper < minLowerBound) {
    return false;
  }

  if (resultUpper > minUpperBound) {
    scheduler.setMax(result, minUpperBound);
  }

  if (resultLower < minLowerBound) {
    scheduler.setMin(result, minLowerBound);
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
    scheduler.setMin(right, domains[left].min() + popsolver::DataType{1});
  }
  if (domains[right].max() <= domains[left].max()) {
    scheduler.setMax(left, domains[right].max() - popsolver::DataType{1});
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

template <> bool GenericAssignment<DataType>::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  for (std::size_t i = 1; i != vars.size(); ++i) {
    const auto domain = domains[vars[i]];
    if (domain.size() != DataType{0}) {
      return true;
    }
    values[i - 1] = domain.val();
  }

  const auto x = f(values);

  if (!x) {
    return false;
  }
  const auto result = vars[0];
  if (x.get() < domains[result].min() || x.get() > domains[result].max()) {
    return false;
  }
  if (domains[result].min() != x.get() || domains[result].max() != x.get()) {
    scheduler.set(result, x.get());
  }
  return true;
}

template <typename T>
bool GenericAssignment<T>::propagate(Scheduler &scheduler) {
  const Domains &domains = scheduler.getDomains();
  for (std::size_t i = 1; i != vars.size(); ++i) {
    // Input operands must be with the range allowed for the data type
    const auto &domain = domains[vars[i]];
    if (domain.min() > popsolver::DataType{std::numeric_limits<T>::max()} ||
        domain.max() < popsolver::DataType{std::numeric_limits<T>::min()}) {
      return false;
    }
    if (domain.min() < popsolver::DataType{std::numeric_limits<T>::min()}) {
      scheduler.setMin(vars[i],
                       popsolver::DataType{std::numeric_limits<T>::min()});
    }
    if (domain.max() > popsolver::DataType{std::numeric_limits<T>::max()}) {
      scheduler.setMax(vars[i],
                       popsolver::DataType{std::numeric_limits<T>::max()});
    }
  }

  for (std::size_t i = 1; i != vars.size(); ++i) {
    const auto domain = domains[vars[i]];
    if (domain.size() != popsolver::DataType{0}) {
      return true;
    }
    values[i - 1] = domain.val();
  }

  std::vector<T> castedValues{};
  castedValues.reserve(values.size());
  for (auto v : values) {
    castedValues.push_back(v.template getAs<T>());
  }

  const auto x = f(castedValues);

  if (!x) {
    return false;
  }
  const auto result = vars[0];
  if (x.get() < domains[result].min() || x.get() > domains[result].max()) {
    return false;
  }
  if (domains[result].min() != x.get() || domains[result].max() != x.get()) {
    scheduler.set(result, x.get());
  }
  return true;
}

template class popsolver::GenericAssignment<DataType>;
template class popsolver::GenericAssignment<unsigned>;
template class popsolver::GenericAssignment<uint64_t>;
