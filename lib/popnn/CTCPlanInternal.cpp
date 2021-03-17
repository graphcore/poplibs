// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCPlanInternal.hpp"

#include "CTCInferencePlan.hpp"
#include "CTCLossPlan.hpp"

#include <poplibs_support/Visitor.hpp>

#include <iostream>

namespace popnn {
namespace ctc {

const LossPlan &Plan::Impl::getAsLossPlan() const {
  try {
    return boost::get<const LossPlan &>(plan);
  } catch (...) {
    throw poputil::poplibs_error(
        "Invalid CTC Loss plan provided (must be created "
        "by planning for loss and not inference).");
  }
}

const InferencePlan &Plan::Impl::getAsInferencePlan() const {
  try {
    return boost::get<const InferencePlan &>(plan);
  } catch (...) {
    throw poputil::poplibs_error(
        "Invalid CTC Inference plan provided (must be created "
        "by planning for inference and not loss).");
  }
}

std::unique_ptr<Plan::Impl> Plan::Impl::clone() const {
  return std::make_unique<Plan::Impl>(*this);
}

bool operator<(const Plan::Impl &a, const Plan::Impl &b) {
  return a.plan < b.plan;
}
bool operator==(const Plan::Impl &a, const Plan::Impl &b) {
  return a.plan == b.plan;
}

std::ostream &operator<<(std::ostream &o, const Plan::Impl &p) {
  boost::apply_visitor(
      poplibs_support::make_visitor<void>([&](const auto &plan) { o << plan; }),
      p.plan);
  return o;
}

// Plan definition
Plan::Plan() : impl(std::make_unique<Plan::Impl>()) {}
Plan::Plan(std::unique_ptr<Plan::Impl> impl) : impl(std::move(impl)) {}
Plan::~Plan() = default;

Plan::Plan(const Plan &other) { impl = other.impl->clone(); }
Plan::Plan(Plan &&other) = default;
Plan &Plan::operator=(const Plan &other) {
  impl = other.impl->clone();
  return *this;
}
Plan &Plan::operator=(Plan &&other) = default;

bool operator<(const Plan &a, const Plan &b) { return *a.impl < *b.impl; }
bool operator==(const Plan &a, const Plan &b) { return *a.impl == *b.impl; }
bool operator!=(const Plan &a, const Plan &b) { return !(a == b); }

std::ostream &operator<<(std::ostream &o, const Plan &p) {
  o << *p.impl;
  return o;
}

} // namespace ctc
} // namespace popnn

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const popnn::ctc::Plan::Impl &p) {
  poplar::ProfileValue::Map v;

  return boost::apply_visitor(
      poplibs_support::make_visitor<poplar::ProfileValue>(
          [](const auto &plan) { return toProfileValue(plan); }),
      p.plan);
}

template <> poplar::ProfileValue toProfileValue(const popnn::ctc::Plan &p) {
  return toProfileValue(p.getImpl());
}
} // namespace poputil
