// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCInferencePlan.hpp"
#include "CTCPlanInternal.hpp"

#include <poplibs_support/PlanConstraints.hpp>
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

  using VisitorOutType = std::tuple<unsigned, unsigned, unsigned>;
  const auto [select, rank, reduce] =
      boost::apply_visitor(poplibs_support::make_visitor<VisitorOutType>(
                               [&](const SelectPartitions<unsigned> &s) {
                                 return VisitorOutType({s.select, 0u, 0u});
                               },
                               [&](const RankPartitions<unsigned> &s) {
                                 return VisitorOutType({0u, s.rank, s.reduce});
                               }),
                           p.parallel.sort);

  return std::tie(p.params, p.parallel.batch, p.parallel.time, p.parallel.copy,
                  p.parallel.extend, p.parallel.extendVerticesPerPartition,
                  p.parallel.merge, p.parallel.preSelectCopy,
                  p.parallel.preSelectExtend, select, rank, reduce,
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

  boost::apply_visitor(
      poplibs_support::make_visitor<void>(
          [&](const SelectPartitions<unsigned> &s) {
            o << "    sortSelect                 " << s.select << "\n";
          },
          [&](const RankPartitions<unsigned> &s) {
            o << "    sortRank                   " << s.rank << "\n";
            o << "    sortReduce                 " << s.reduce << "\n";
          }),
      p.parallel.sort);

  o << "    outputPartitions           " << p.parallel.output << "\n";
  o << "    (Tiles per batch entry)    " << p.batchEntryPartitions() << "\n";
  o << "    (Tiles)                    " << p.numTiles() << "\n";
  return o;
}

struct CtcInferenceOpts {
  poplar::Type partialsType = poplar::FLOAT;
  SortMethod sortMethod = SortMethod::RANK;
};

static CtcInferenceOpts
parseInferenceOptions(const poplar::OptionFlags &options) {
  CtcInferenceOpts opts;
  std::map<std::string, poplar::Type> partialsTypeMap{{"half", poplar::HALF},
                                                      {"float", poplar::FLOAT}};

  std::map<std::string, SortMethod> sortMethodMap{
      {"select", SortMethod::SELECT}, {"rank", SortMethod::RANK}};

  const poplibs::OptionSpec spec{
      {"sortMethod",
       poplibs::OptionHandler::createWithEnum(opts.sortMethod, sortMethodMap)},
      {"partialsType", poplibs::OptionHandler::createWithEnum(opts.partialsType,
                                                              partialsTypeMap)},
  };

  for (const auto &entry : options) {
    spec.parse(entry.first, entry.second);
  }
  return opts;
}

std::ostream &operator<<(std::ostream &o, const SortMethod &m) {
  switch (m) {
  case SortMethod::SELECT:
    o << "SELECT";
    break;
  case SortMethod::RANK:
    o << "RANK";
    break;
  };
  return o;
}

std::ostream &operator<<(std::ostream &o, const CtcInferenceOpts &opt) {
  o << "CTCInference options:\n";
  o << "  sortMethod                   " << opt.sortMethod << "\n";
  o << "  partialsType                 " << opt.partialsType << "\n";
  return o;
}
} // namespace ctc

namespace ctc_infer {
ctc::Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
               unsigned batchSize, unsigned maxTime, unsigned numClasses,
               unsigned beamwidth, const poplar::OptionFlags &options) {

  ctc::CtcInferenceOpts opts = ctc::parseInferenceOptions(options);

  // Some simple parameters based on splitting by numClasses alone
  ctc::InferencePlan plan;
  plan.params = {inType,  opts.partialsType, inType,     batchSize,
                 maxTime, maxTime,           numClasses, beamwidth};

  poplibs_support::logging::popnn::debug("Planning CTCInference with:\n{}\n{}",
                                         plan.params, opts);

  const auto target = graph.getTarget();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto tiles = target.getTilesPerIPU();

  // Cannot split by time at the moment
  plan.parallel.time = 1;

  // Plan using the following functions which aid with scaling the number of
  // partitions. Ideally spread as much as possible for speed, but otherwise
  // fewer partitions so that things will fit at the cost of speed.
  auto findMaxPartitions = [](unsigned size, unsigned divisor) {
    auto perPartition = poplibs_support::ceildiv(size, divisor);
    return poplibs_support::ceildiv(size, perPartition);
  };

  auto findMaxBatchPartitions = [](unsigned size, unsigned divisor) {
    auto perPartition = std::max(poplibs_support::ceildiv(size, divisor), 1u);
    return poplibs_support::ceildiv(size, perPartition);
  };

  // Each batch entry occupies a separate set of tiles if possible but does
  // not have to when the number of tiles is a limiting factor
  plan.parallel.batch = findMaxBatchPartitions(batchSize, tiles);
  const auto tilesPerBatchEntry = tiles / plan.parallel.batch;

  // Extend candidate generation is partitioned by class. The blank class is
  // not part of an extend operation so use 1 class per partition.
  // 1 to `beamwidth` extend candidates are generated per partition
  plan.parallel.extend = findMaxPartitions(numClasses - 1, tilesPerBatchEntry);

  // Within the extend partition we can choose how many vertices to use,
  // beamwidth is the most fragmented this can be.
  // For test, code the rule that we can use up to 5 workers, which is
  // efficient as we have used 1 worker to generate a copy candidate
  plan.parallel.extendVerticesPerPartition =
      std::min(beamwidth, numWorkers - 1);

  // Copy candidate generation is partitioned by beam.  One copy candidate is
  // generated per beam output
  plan.parallel.copy = findMaxPartitions(beamwidth, tilesPerBatchEntry);

  // Merge candidate generation is partitioned by beam
  // TODO - could be beam - 1 ?
  plan.parallel.merge = findMaxPartitions(beamwidth, tilesPerBatchEntry);

  // Selection of copy and extend beams spread over this many tiles for the
  // extend beam dimension
  plan.parallel.preSelectExtend =
      findMaxPartitions(beamwidth, tilesPerBatchEntry);
  // Selection of copy and extend beams spread over this many vertices for the
  // copy beam dimension
  plan.parallel.preSelectCopy =
      findMaxPartitions(beamwidth, tilesPerBatchEntry);

  // Sort - by most probable candidate
  if (opts.sortMethod == popnn::ctc::SortMethod::RANK) {
    const auto candidatesToRank = beamwidth * numClasses;
    const auto rankingsPerPartition =
        std::max(numWorkers, poplibs_support::ceildiv(candidatesToRank,
                                                      tilesPerBatchEntry));

    plan.parallel.sort = popnn::ctc::RankPartitions<unsigned>(
        {poplibs_support::ceildiv(candidatesToRank, rankingsPerPartition),
         findMaxPartitions(beamwidth, tilesPerBatchEntry)});
  } else {
    plan.parallel.sort = popnn::ctc::SelectPartitions<unsigned>({1});
  }

  // For output generation
  plan.parallel.output = findMaxPartitions(beamwidth, tilesPerBatchEntry);

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
