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
  return std::make_unique<ctc::Plan::Impl>(
      ctc::Plan::Impl{ctc::InferencePlan{}});
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
