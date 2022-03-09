// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCInferencePlan.hpp"
#include "CTCPlanInternal.hpp"

#include <poplibs_support/PlanConstraints.hpp>
#include <poplibs_support/logging.hpp>

using namespace poplibs_support;

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
  std::vector<unsigned> groups(p.parallel.sort.size());
  std::vector<unsigned> tilesPerGroup(p.parallel.sort.size());
  std::vector<unsigned> rankPartitions(p.parallel.sort.size());
  std::vector<unsigned> reducePartitions(p.parallel.sort.size());
  for (unsigned i = 0; i < p.parallel.sort.size(); i++) {
    groups[i] = p.parallel.sort[i].groups;
    tilesPerGroup[i] = p.parallel.sort[i].groupsPerTile;
    rankPartitions[i] = p.parallel.sort[i].rankPartitions;
    reducePartitions[i] = p.parallel.sort[i].reducePartitions;
  }
  return std::tie(p.params, p.parallel.batch, p.parallel.time, p.parallel.copy,
                  p.parallel.extend, p.parallel.extendVerticesPerPartition,
                  p.parallel.merge, p.parallel.selectCopy,
                  p.parallel.selectExtend, groups, tilesPerGroup,
                  rankPartitions, reducePartitions, p.parallel.output);
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
  o << "    selectCopy                 " << p.parallel.selectCopy << "\n";
  o << "    selectExtend               " << p.parallel.selectExtend << "\n";

  for (unsigned stage = 0; stage < p.parallel.sort.size(); stage++) {
    o << "    Sort stage:" << stage << ", " << p.parallel.sort[stage].groups
      << " group(s) with " << p.parallel.sort[stage].groupsPerTile
      << " groups per tile\n";
    o << "    Partitions:\n";
    o << "        sortRank               "
      << p.parallel.sort[stage].rankPartitions << "\n";
    o << "        sortReduce             "
      << p.parallel.sort[stage].reducePartitions << "\n";
  }
  o << "    outputPartitions           " << p.parallel.output << "\n";
  o << "    (Tiles per batch entry)    " << p.batchEntryPartitions() << "\n";
  o << "    (Tiles)                    " << p.numTiles() << "\n";
  return o;
}

struct CtcInferenceOpts {
  poplar::Type partialsType = poplar::FLOAT;
  std::vector<unsigned> sortStageGroups;
};

static CtcInferenceOpts
parseInferenceOptions(const poplar::OptionFlags &options) {
  CtcInferenceOpts opts;
  std::map<std::string, poplar::Type> partialsTypeMap{{"half", poplar::HALF},
                                                      {"float", poplar::FLOAT}};

  const poplibs::OptionSpec spec{
      {"partialsType", poplibs::OptionHandler::createWithEnum(opts.partialsType,
                                                              partialsTypeMap)},
      {"sortStageGroups",
       poplibs::OptionHandler::createWithList(opts.sortStageGroups)},
  };

  for (const auto &entry : options) {
    spec.parse(entry.first, entry.second);
  }
  return opts;
}

std::ostream &operator<<(std::ostream &o, const CtcInferenceOpts &opt) {
  o << "CTCInference options:\n";
  o << "  partialsType                 " << opt.partialsType << "\n";
  o << "  sortStageGroups              {";
  for (unsigned i = 0; i < opt.sortStageGroups.size(); i++) {
    o << opt.sortStageGroups[i];
    if (i != opt.sortStageGroups.size() - 1) {
      o << ",";
    }
  }
  o << "}\n";
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

  logging::popnn::debug("Planning CTCInference with:\n{}\n{}", plan.params,
                        opts);

  const auto target = graph.getTarget();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto tiles = target.getTilesPerIPU();

  // Cannot split by time at the moment
  plan.parallel.time = 1;

  // Plan using the following functions which aid with scaling the number of
  // partitions. Ideally spread as much as possible for speed, but otherwise
  // fewer partitions so that things will fit at the cost of speed.
  auto findMaxPartitions = [](unsigned size, unsigned divisor) {
    auto perPartition = gccs::ceildiv(size, divisor);
    return gccs::ceildiv(size, perPartition);
  };

  auto findMaxBatchPartitions = [](unsigned size, unsigned divisor) {
    auto perPartition = std::max(gccs::ceildiv(size, divisor), 1u);
    return gccs::ceildiv(size, perPartition);
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
  plan.parallel.selectExtend = findMaxPartitions(beamwidth, tilesPerBatchEntry);
  // Selection of copy and extend beams spread over this many vertices for the
  // copy beam dimension
  plan.parallel.selectCopy = findMaxPartitions(beamwidth, tilesPerBatchEntry);

  // Sort - by most probable candidate
  //
  // Plan in stages
  // Each stage divides the input candidates into `groups` which are sorted
  // independently. and will result in `sortStageGroups[stage] * beamwidth`
  // results to then be sorted again.
  // The last stage will have sortStageGroups = 1 and so create a
  // single result.
  // Results each contain the beamwidth most probable sort results

  // TODO - A planner may be necessary to get the best out of the choice of
  // multiple stages, and the number of groups in those stages.  Presently we
  // apply a heuristc that picks out cases where 2 stages are better
  // that 1.

  // Use 2 stages if the number of classes is such that 2 stages will be of
  // benefit.  Observations showed that `singleStageClassLimit` classes was the
  // point at which a consistent speed benefit was seen by having 2 stages,
  // more than 2 stages didn't produce an obvious benefit. This isn't a function
  // of beamwidth as a stage will have to produce groups * beamwidth candidates
  // so we use numClasses instead of the total number of candidates.
  const auto singleStageClassLimit = 20;
  if (opts.sortStageGroups.size() > 0) {
    // The option defines the stages and the number of groups per stage. It is
    // possible to select more than 2 stages, and as it is an undocumented test
    // option it is the users responsibility to make sure it makes sense.
    plan.parallel.sort.reserve(opts.sortStageGroups.size());
    for (unsigned i = 0; i < opts.sortStageGroups.size(); i++) {
      plan.parallel.sort.push_back({opts.sortStageGroups[i]});
    }
  } else {
    // Automatically select - the logic here will only ever result in 1 or 2
    // stages.  If 3 stages are useful then most of this needs to be revisited.
    if (numClasses >= singleStageClassLimit) {
      // Sort in 2 stages
      plan.parallel.sort.reserve(2);

      const auto toSort = beamwidth * numClasses;
      // Form 2 equally balanced stages
      unsigned numGroups = std::sqrt(numClasses);
      auto candidatesPerGroup = gccs::ceildiv(toSort, numGroups);

      // We need to ensure that every group contains at least beamwidth
      // candidates otherwise the method of rank, reduce will fail (Reduce will
      // receive a group of ranked outputs where some weren't written by rank).
      // Having few items in a group isn't going to be very efficient anyhow,
      // but this constraint is what the algorithm needs in order to be correct.

      // Check the size of the last group - and make sure it is >= beamwidth
      while (toSort - (numGroups - 1) * candidatesPerGroup < beamwidth) {
        numGroups--;
        candidatesPerGroup = gccs::ceildiv(toSort, numGroups);
      }

      plan.parallel.sort.push_back({numGroups});
      plan.parallel.sort.push_back({1});
    } else {
      // Sort in 1 stage
      plan.parallel.sort.push_back({1});
    }
  }

  // Given the number of stages, and groups within each stage, work out the
  // number of partitions IN EACH GROUP to use when we do the sort.
  // Each stage will result in candidates = beamwidth * groupsInTheStage
  // which is then the number to sort in the next stage.
  const auto stages = plan.parallel.sort.size();

  auto candidatesToRankPerGroup =
      gccs::ceildiv(beamwidth * numClasses, plan.parallel.sort[0].groups);
  for (unsigned stage = 0; stage < stages; stage++) {
    // A group has this many tiles available to divide work over
    const auto tilesPerGroup =
        std::max(1u, tilesPerBatchEntry / plan.parallel.sort[stage].groups);

    // There is no speed cost in having upto numWorkers candidates ranked
    // in any partition so choose at least that many to reduce the complexity
    const auto rankingsPerPartition = std::max(
        numWorkers, gccs::ceildiv(candidatesToRankPerGroup, tilesPerGroup));
    plan.parallel.sort[stage].rankPartitions =
        gccs::ceildiv(candidatesToRankPerGroup, rankingsPerPartition);
    plan.parallel.sort[stage].reducePartitions =
        findMaxPartitions(beamwidth, tilesPerGroup);

    plan.parallel.sort[stage].groupsPerTile =
        gccs::ceildiv(plan.parallel.sort[stage].groups, tilesPerBatchEntry);

    if (stage != stages - 1) {
      // For the next loop, how many candidates will there be after this
      // stage?
      candidatesToRankPerGroup =
          gccs::ceildiv(beamwidth * plan.parallel.sort[stage].groups,
                        plan.parallel.sort[stage + 1].groups);
    }
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
