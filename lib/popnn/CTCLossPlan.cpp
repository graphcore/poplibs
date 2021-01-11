// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"
#include <popnn/CTCLoss.hpp>

namespace popnn {
namespace ctc_loss {
Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
          const poplar::Type &outType, const std::size_t batchSize,
          const std::size_t maxTime, const std::size_t maxLabels,
          const std::size_t numClasses) {
  Plan::Impl plan;
  plan.parallel.batch = batchSize;
  return std::make_unique<Plan::Impl>(std::move(plan));
}

std::ostream &operator<<(std::ostream &o, const Plan::Impl &p) {
  o << "CTCLoss plan:\n";
  o << "  Serial Partition:\n";
  o << "    batchSplit=" << p.serial.batch << "\n";
  o << "    timeSplit=" << p.serial.time << "\n";
  o << "    labelSplit=" << p.serial.label << "\n";
  o << "  Parallel Partition:\n";
  o << "    batchSplit=" << p.parallel.batch << "\n";
  o << "    timeSplit=" << p.parallel.time << "\n";
  o << "    labelSplit=" << p.parallel.label << "\n";
  o << "    sliceIntoOutput=" << p.parallel.sliceIntoOutput << "\n";
  o << "    alphabetSplit=" << p.parallel.alphabet << "\n";
  o << "    sliceFromInput=" << p.parallel.sliceFromInput << "\n";
  return o;
}

// Complete the definition of the Plan class
Plan::~Plan() = default;
Plan &Plan::operator=(Plan &&) = default;
Plan::Plan(std::unique_ptr<Plan::Impl> impl) : impl(std::move(impl)) {}

std::ostream &operator<<(std::ostream &o, const Plan &p) {
  o << *p.impl;
  return o;
}
} // namespace ctc_loss
} // namespace popnn

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const popnn::ctc_loss::Plan &p) {
  poplar::ProfileValue::Map v;
  v.insert({"serial.batch", toProfileValue(p.impl->serial.batch)});
  v.insert({"serial.time", toProfileValue(p.impl->serial.time)});
  v.insert({"serial.label", toProfileValue(p.impl->serial.label)});
  v.insert({"parallel.batch", toProfileValue(p.impl->parallel.batch)});
  v.insert({"parallel.time", toProfileValue(p.impl->parallel.time)});
  v.insert({"parallel.label", toProfileValue(p.impl->parallel.label)});
  v.insert({"parallel.sliceIntoOutput",
            toProfileValue(p.impl->parallel.sliceIntoOutput)});
  v.insert({"parallel.alphabet", toProfileValue(p.impl->parallel.alphabet)});
  v.insert({"parallel.sliceFromInput",
            toProfileValue(p.impl->parallel.sliceFromInput)});

  return v;
}
} // namespace poputil