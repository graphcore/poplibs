// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCInferencePlan.hpp"
#include "CTCPlanInternal.hpp"

#include <poplibs_support/logging.hpp>

namespace popnn {
namespace ctc {

static auto getTupleOfMembers(const CtcInferencePlannerParams &p) {
  return std::tie(p.inType, p.partialsType, p.outType, p.batchSize, p.maxTime,
                  p.maxLabelLength, p.numClasses, p.beamWidth);
}

bool operator<(const CtcInferencePlannerParams &a,
               const CtcInferencePlannerParams &b) {
  return getTupleOfMembers(a) < getTupleOfMembers(b);
}

bool operator==(const CtcInferencePlannerParams &a,
                const CtcInferencePlannerParams &b) {
  return getTupleOfMembers(a) == getTupleOfMembers(b);
}

std::ostream &operator<<(std::ostream &o, const CtcInferencePlannerParams &p) {
  o << "CTCInference params:\n";
  o << "  inType                       " << p.inType << "\n";
  o << "  partialsType                 " << p.partialsType << "\n";
  o << "  outType                      " << p.outType << "\n";
  o << "  batchSize                    " << p.batchSize << "\n";
  o << "  maxTime                      " << p.maxTime << "\n";
  o << "  maxLabelLength               " << p.maxLabelLength << "\n";
  o << "  numClasses                   " << p.numClasses << "\n";
  o << "  beamWidth                    " << p.beamWidth << "\n";
  return o;
}

static auto getTupleOfMembers(const InferencePlan &p) {
  return std::tie(p.params, p.parallel.batch, p.parallel.time, p.parallel.copy,
                  p.parallel.extend, p.parallel.extendVerticesPerPartition,
                  p.parallel.merge, p.parallel.preSelectCopy,
                  p.parallel.preSelectExtend, p.parallel.select,
                  p.parallel.output);
}
bool operator<(const InferencePlan &a, const InferencePlan &b) noexcept {
  return getTupleOfMembers(a) < getTupleOfMembers(b);
}
bool operator==(const InferencePlan &a, const InferencePlan &b) noexcept {
  return getTupleOfMembers(a) == getTupleOfMembers(b);
}

std::ostream &operator<<(std::ostream &o, const InferencePlan &p) {
  o << "CTCInference plan:\n";
  o << "  Parallel Partition:\n";
  o << "    batch                      " << p.parallel.batch << "\n";
  o << "    time                       " << p.parallel.time << "\n";
  o << "    extendPartitions           " << p.parallel.extend << "\n";
  o << "    extendVerticesPerPartition "
    << p.parallel.extendVerticesPerPartition << "\n";
  o << "    copyPartitions             " << p.parallel.copy << "\n";
  o << "    mergePartitions            " << p.parallel.merge << "\n";
  o << "    preSelectCopy              " << p.parallel.preSelectCopy << "\n";
  o << "    preSelectExtend            " << p.parallel.preSelectExtend << "\n";
  o << "    select                     " << p.parallel.select << "\n";
  o << "    outputPartitions           " << p.parallel.output << "\n";
  o << "    (Tiles per batch entry)    " << p.batchEntryPartitions() << "\n";
  o << "    (Tiles)                    " << p.numTiles() << "\n";
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

  poplibs_support::logging::popnn::debug("Planning CTCInference with:\n{}",
                                         plan.params);

  // Cannot split by time at the moment
  plan.parallel.time = 1;
  // Each batch occupies a separate set of tiles
  plan.parallel.batch = batchSize;

  // Extend candidate generation is partitioned by class. The blank class is
  // not part of an extend operation so use 1 class per partition.
  // 1 to `beamwidth` extend candidates are generated per partition
  plan.parallel.extend = numClasses - 1;
  // Within the extend partition we can choose how many vertices to use,
  // beamwidth is the most fragmented this can be.
  // For test, code the rule that we can use up to 5 workers, which is
  // efficient as we have used 1 worker to generate a copy candidate
  plan.parallel.extendVerticesPerPartition = std::min(beamwidth, 5u);
  // Copy candidate generation is partitioned by beam.  One copy candidate is
  // generated per beam output
  plan.parallel.copy = beamwidth;

  // Merge candidate generation is partitioned by beam
  // TODO - could be beam - 1 ?
  plan.parallel.merge = beamwidth;

  // Selection of copy and extend beams spread over this many tiles for the
  // extend beam dimension
  plan.parallel.preSelectExtend = beamwidth;
  // Selection of copy and extend beams spread over this many vertices for the
  // copy beam dimension
  plan.parallel.preSelectCopy = beamwidth;

  // Select - by most probable candidate
  plan.parallel.select = 1;
  // For output generation
  plan.parallel.output = beamwidth;

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
