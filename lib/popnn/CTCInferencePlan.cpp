// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCInferencePlan.hpp"
#include "CTCPlanInternal.hpp"

namespace popnn {
namespace ctc {

bool operator<(const CtcInferencePlannerParams &a,
               const CtcInferencePlannerParams &b) {
  // TODO
  return false;
}

bool operator==(const CtcInferencePlannerParams &a,
                const CtcInferencePlannerParams &b) {
  // TODO
  return true;
}

static auto getTupleOfMembers(const InferencePlan &p) {
  return std::tie(p.params);
}
bool operator<(const InferencePlan &a, const InferencePlan &b) noexcept {
  return getTupleOfMembers(a) < getTupleOfMembers(b);
}
bool operator==(const InferencePlan &a, const InferencePlan &b) noexcept {
  return getTupleOfMembers(a) == getTupleOfMembers(b);
}

std::ostream &operator<<(std::ostream &o, const InferencePlan &p) {
  o << "Not yet implemented\n";
  return o;
}

} // namespace ctc

namespace ctc_infer {
ctc::Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
               unsigned batchSize, unsigned maxTime, unsigned numClasses,
               unsigned beamwidth, const poplar::OptionFlags &options) {

  // Some simple parameters based on splitting by numClasses alone
  ctc::InferencePlan plan;
  plan.params = {inType,  poplar::FLOAT, inType,     batchSize,
                 maxTime, maxTime,       numClasses, beamwidth};
  plan.parallel.batch = batchSize;
  plan.parallel.time = 1;
  plan.parallel.label = 1;
  plan.parallel.beam = 1;
  plan.parallel.classes = numClasses;

  return std::make_unique<ctc::Plan::Impl>(ctc::Plan::Impl{std::move(plan)});
}
} // namespace ctc_infer
} // namespace popnn

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const popnn::ctc::InferencePlan &p) {
  poplar::ProfileValue::Map v;
  return v;
}
} // namespace poputil
