// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "conv_analysis.hpp"

#include <poplibs_test/ProgressBar.hpp>
#include <poputil/exceptions.hpp>

#include <poplar/StringRef.hpp>

#include <spdlog/fmt/bundled/printf.h>
#include <spdlog/fmt/fmt.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

using namespace poplar;
using namespace poplin;
using namespace poplin::internal;

static bool contains(StringRef string, StringRef needle) {
  if (string.size() >= needle.size())
    for (size_t i = 0; i < string.size() - needle.size(); ++i)
      if (StringRef{string.data() + i, needle.size()} == needle)
        return true;
  return false;
}

static bool r_contains(StringRef string, StringRef needle) {
  return string.end() != std::find_end(string.begin(), string.end(),
                                       needle.begin(), needle.end());
}

static bool starts_with(StringRef string, StringRef prefix) {
  return string.size() >= prefix.size() &&
         StringRef{string.data(), prefix.size()} == prefix;
}

static StringRef trim(StringRef string) {
  if (string.empty())
    return string;
  const char *first = string.data();
  const char *last = string.data() + string.size() - 1;
  while (*first == ' ')
    ++first;
  while (*last == ' ')
    --last;
  return StringRef(first, std::distance(first, last + 1));
}

// Analysis specific to the ConvPartialnx1 vertices.
namespace amp {

void updateDetailedPlanCosts(bool reportPerTile, bool reportPerSerialSplit,
                             DetailedPlanCosts &costs) {
  using poplin::PlanCosts;
  if (!reportPerTile) {
    costs.apply([&costs](PlanCosts &c) {
      if (c.cycles != PlanCosts::unknown)
        c.cycles *= costs.parallelSplit;
      if (c.memory != PlanCosts::unknown)
        c.memory *= costs.parallelSplit;
    });
  }

  if (reportPerSerialSplit) {
    // The profiler reports totals across all splits and itemised numbers
    // as per-serial split.
    costs.total.cycles /= costs.serialSplit;
  } else {
    costs.apply(
        [&costs](PlanCosts &c) {
          if (c.cycles != PlanCosts::unknown)
            c.cycles *= costs.serialSplit;
        },
        false);
  }
}

size_t MeasuredPlanCosts::totalCycles() const noexcept {
  size_t total = 0;
  apply([&total](const poplin::PlanCosts &c) { total += c.cycles; }, false);
  return total;
}

size_t MeasuredPlanCosts::totalMemory() const noexcept {
  // Memory inside the serial splits is maxed rather than summed
  // because we assume the temporary memory will be reused.
  return broadcast.memory + tileLevelTransform.memory + rearrangement.memory +
         std::max({compute.memory, exchange.memory, transform.memory,
                   reduction.memory, dynamicSlice.memory, dynamicUpdate.memory,
                   unknown.memory});
}

poplin::PlanCosts MeasuredPlanCosts::toPlanCosts() const noexcept {
  return {totalCycles(), totalMemory()};
}

poplin::PlanCosts &
MeasuredPlanCosts::selectCosts(const std::string &name,
                               bool isInsideSerialSplit) noexcept {
  if (r_contains(name, "PreArrange") || r_contains(name, "preRegroup"))
    return isInsideSerialSplit ? transform : broadcast;
  else if (r_contains(name, "weightsRearranged"))
    return rearrangement;
  else if (r_contains(name, "tileLevelActs"))
    return broadcast;
  else if (r_contains(name, "ExchangePre"))
    return exchange;
  else if (r_contains(name, "dynamicSlice"))
    return dynamicSlice;
  else if (r_contains(name, "dynamicUpdate"))
    return dynamicUpdate;
  else if (r_contains(name, "Reduce"))
    return reduction;
  else if (r_contains(name, "Cast"))
    return outputCast;
  else if (r_contains(name, "Convolve"))
    return compute;
  else if (r_contains(name, "Padding"))
    return transform;
  else if (r_contains(name, "Transpose"))
    return isInsideSerialSplit ? transform : broadcast;
  else {
    fmt::print(std::cerr, "Warning: unrecognised program: '{}'\n", name);
    return unknown;
  }
}

MemoryCostProgramVisitor::MemoryCostProgramVisitor(MeasuredPlanCosts &planCosts,
                                                   bool reportPerTile_,
                                                   size_t activeTiles_,
                                                   bool isInsideSerialSplit_,
                                                   bool printVars_)
    : costs(planCosts), reportPerTile(reportPerTile_),
      activeTiles(activeTiles_), isInsideSerialSplit(isInsideSerialSplit_),
      printVars(printVars_) {}

void MemoryCostProgramVisitor::visitDoExchange(
    const pva::DoExchangeProgram &doExchange) {
  // Note that this doesn't include control/code bytes per tile to
  // better match the planner.
  size_t memory = 0;
  size_t transformBytes = 0;
  size_t tileLevelTransformBytes = 0;

  for (auto &var : doExchange.vars()) {
    auto name = var.name();
    if (printVars)
      fmt::print(std::cout, "  var={}, {} B\n", name, var.bytes());
    // Don't include alwaysLive variables as they exist in all stages
    // so attributing them to a single stage is a bit unfair, and this
    // is closer to how the planner views things.
    if (r_contains(name, "partialReduceOut") || contains(name, "mergedVars"))
      transformBytes += var.bytes();
    else if (r_contains(name, "zeroPadding#")) // padding to kernel height
      tileLevelTransformBytes += var.bytes();
    else if (contains(name, "message") || !var.alwaysLive()) {
      memory += var.bytes();
    }
  }

  if (reportPerTile) {
    memory /= activeTiles;
    transformBytes /= activeTiles;
    tileLevelTransformBytes /= activeTiles;
  }

  // This is closer to how the planner reports memory.
  poplin::PlanCosts *c = nullptr;
  if (r_contains(doExchange.name(), "tileLevelActs"))
    c = &costs.broadcast;
  else
    c = &costs.exchange;

  // We assume the temporary memory is reused across serial splits.
  if (isInsideSerialSplit) {
    c->memory = std::max(c->memory, memory);
  } else {
    c->memory += memory;
  }
  costs.transform.memory = std::max(costs.transform.memory, transformBytes);
  costs.tileLevelTransform.memory =
      std::max(costs.tileLevelTransform.memory, tileLevelTransformBytes);
}

void MemoryCostProgramVisitor::visitOnTileExecute(
    const pva::OnTileExecuteProgram &onTileExecute) {
  size_t memory = 0;
  size_t workListBytes = 0;

  // This can be slow because each lowered var is reading from a "file"
  // on the first attribute access and allocating a new unique_ptr. This
  // doesn't parallelise nicely with threads due to being disk limited.
  for (auto &v : onTileExecute.vars()) {
    auto name = v.name();
    if (printVars)
      fmt::print(std::cout, "  var={}, {}B\n", name, v.bytes());
    if (r_contains(name, "worklist"))
      workListBytes += v.bytes();
    else if (r_contains(name, "tileLevelActs"))
      ; // These variables have already been accounted for in broadcast.
    else if (r_contains(name, "zeroPadding#") ||
             r_contains(name, "mergedVars") ||
             r_contains(name, "partialReduceOut"))
      ; // These variables have already been accounted for in visitDoExchange.
    else
      memory += v.bytes();
  }

  if (reportPerTile) {
    memory /= activeTiles;
    workListBytes /= activeTiles;
  }

  poplin::PlanCosts &c =
      costs.selectCosts(onTileExecute.name(), isInsideSerialSplit);

  // We assume the temporary memory is reused across serial splits.
  if (isInsideSerialSplit) {
    c.memory = std::max(c.memory, memory);
  } else {
    c.memory += memory;
  }

  costs.totalWorkListBytes += workListBytes;
}

MeasuredPlanCosts getMeasuredCosts(const std::string &pass,
                                   const std::string &profileDir,
                                   size_t parallelSplit, size_t serialSplit,
                                   bool printVars, bool reportPerTile,
                                   bool reportPerSerialSplit) {
  using poplin::PlanCosts;

  MeasuredPlanCosts costs;
  const auto &report = pva::openReport(profileDir + "/profile.pop");

  // Find in the compilation report the Repeat/Sequence that starts the
  // serial slice (if any).
  std::unordered_set<size_t> progsInRepeatLoop;
  progsInRepeatLoop.reserve(32); // inexact but cheap
  for (auto &prog : report.compilation().programs()) {
    auto name = prog->name();
    if (!contains(name, pass)) {
      continue;
    }

    if (prog->type() == pva::Program::Type::Repeat) {
      auto tmp = prog->children();
      if (tmp.size() == 1 && tmp[0]->type() == pva::Program::Type::Sequence) {
        auto seq = tmp[0];
        for (auto &c : seq->children())
          progsInRepeatLoop.emplace(c->_id());
      }
    }
  }

  auto steps = report.execution().steps();

  // Track the progress of the analysis.
  ProgressBar progressBar("Analysing: ", steps.size(), !printVars);

  for (const auto &b : steps) {
    auto prog = b.program();
    auto name = prog->name();
    if (printVars)
      fmt::print(std::cout, "program={}\n", name);

    // Skip any programs not containing the debug names we use for the
    // various convolutions this tool can perform.
    if (!contains(name, pass)) {
      ++progressBar;
      continue;
    }

    // The planner doesn't include these so neither do we.
    if (r_contains(name, "UpdateWeights") ||
        r_contains(name, "loopIncrement")) {
      ++progressBar;
      continue;
    }

    // Track when we enter the serial splits.
    bool isInsideSerialSplit =
        progsInRepeatLoop.find(prog->_id()) != progsInRepeatLoop.end();

    PlanCosts &c = costs.selectCosts(name, isInsideSerialSplit);

    // Compute the number of cycles used. When reporting per tile use the
    // cycles for the slowest tile as it's likely all other tiles will end
    // up waiting for it if there is an internal sync after this program.
    // Note that this may not always be true and could lead to some
    // inaccuracies.
    uint64_t activeTiles = 0;
    if (reportPerTile) {
      uint64_t maxCycles = 0;
      for (auto cycles : b.cyclesByTile()) {
        if (cycles) {
          maxCycles = std::max(cycles, maxCycles);
          ++activeTiles;
        }
      }
      c.cycles += maxCycles;
    } else {
      uint64_t totalCycles = 0;
      for (auto cycles : b.cyclesByTile()) {
        if (cycles) {
          totalCycles += cycles;
        }
        ++activeTiles;
      }
      c.cycles += totalCycles;
    }

    // Record the memory used by the program.
    MemoryCostProgramVisitor visitor(costs, reportPerTile, activeTiles,
                                     isInsideSerialSplit, printVars);
    prog->accept(visitor);

    ++progressBar;
  }

  // The above sums all serial splits so divide back into the serial splits.
  // Note that by doing this at the end we avoid some precision loss when
  // working with small numbers of cycles. Only applicable to stages inside
  // the serial splits.
  if (reportPerSerialSplit && serialSplit > 1) {
    costs.dynamicSlice.cycles /= serialSplit;
    costs.transform.cycles /= serialSplit;
    costs.tileLevelTransform.cycles /= serialSplit;
    costs.exchange.cycles /= serialSplit;
    costs.compute.cycles /= serialSplit;
    costs.reduction.cycles /= serialSplit;
    costs.dynamicUpdate.cycles /= serialSplit;
    costs.addInPlace.cycles /= serialSplit;
    costs.totalWorkListBytes /= serialSplit;
  }

  // Compute the totals.
  costs.total.cycles = costs.totalCycles();
  costs.total.memory = costs.totalMemory();

  return costs;
}

void compareCosts(const std::string &title, DetailedPlanCosts const &estimates,
                  MeasuredPlanCosts const &actual, bool reportVerbose,
                  bool reportPerTile, bool reportPerSerialSplit) {
  using fmt::print;
  using poplin::PlanCosts;
  using std::cout;

  // Print out a summary.
  if (reportPerTile)
    print(cout, "\n{}: Summary for the slowest tile (of {} tiles):\n\n", title,
          estimates.parallelSplit);
  else
    print(cout, "\n{}: Summary for {} tiles:\n\n", title,
          estimates.parallelSplit);
  auto reportSummary = [](auto name, auto estimate, auto actual) -> bool {
    print(cout, "{}:\n", name);
    print(cout, "    measured: {}\n", actual);
    print(cout, "    estimate: {} ({:.3}% different)\n", estimate,
          100 * (((double)estimate - actual) / actual));
    return actual != estimate;
  };
  bool reportDetailedCycles =
      reportSummary("  Cycles", estimates.total.cycles, actual.total.cycles);
  bool reportDetailedMemory = reportSummary(
      "  Memory (B)", estimates.total.memory, actual.total.memory);
  print(cout, "\n");

  // Print out a detailed report.
  auto printRow = [&](bool cycles, auto category, auto measured_,
                      auto estimate_) {
    auto measured = cycles ? measured_.cycles : measured_.memory;
    auto estimate = cycles ? estimate_.cycles : estimate_.memory;
    // Skip stages for which the values are identical or uknown.
    if (!reportVerbose && (measured == 0 || measured == PlanCosts::unknown) &&
        (estimate == 0 || estimate == PlanCosts::unknown))
      return;
    // Compute the percentage difference of the impact on the total. This
    // tends to be more useful than the difference between the measurement
    // and the estimate.
    auto measured_total = cycles ? actual.total.cycles : actual.total.memory;
    auto estimate_total =
        cycles ? estimates.total.cycles : estimates.total.memory;
    auto total_diff = (estimate_total <= measured_total)
                          ? measured_total - estimate_total
                          : estimate_total - measured_total;
    double percentage_diff = NAN;
    if (measured == 0 && estimate == 0)
      percentage_diff = 0.0;
    else if (measured != PlanCosts::unknown && estimate != PlanCosts::unknown)
      percentage_diff = 100.0 * ((double)estimate - measured) / total_diff;
    // Print a row of the table.
    print(cout, " {: <20} | {: >12} | {: >12} | {: > 8.5}\n", category,
          measured != PlanCosts::unknown ? std::to_string(measured) : "n/a",
          estimate != PlanCosts::unknown ? std::to_string(estimate) : "n/a",
          percentage_diff);
  };

  constexpr PlanCosts unknown_costs = {PlanCosts::unknown, PlanCosts::unknown};
  for (bool cycles : {true, false}) {
    bool shouldReport = cycles ? reportDetailedCycles : reportDetailedMemory;
    if (!shouldReport && !reportVerbose)
      continue;
    print(cout, "{} breakdown:\n\n", cycles ? "Cycles" : "Memory");
    // clang-format off
    print(cout, " Category             |     Measured |     Estimate | % Impact \n");
    print(cout, "----------------------+--------------+--------------+----------\n");
    // clang-format on
    printRow(cycles, "broadcast", actual.broadcast, estimates.broadcast);
    printRow(cycles, "rearrangement", actual.rearrangement,
             estimates.rearrangement);
    printRow(cycles, "dynamicSlice", actual.dynamicSlice,
             estimates.dynamicSlice);
    printRow(cycles, "transform", actual.transform, estimates.transform);
    printRow(cycles, "exchange", actual.exchange, estimates.exchange);
    printRow(cycles, "tileLevelTransform", actual.tileLevelTransform,
             estimates.tileLevelTransform);
    printRow(cycles, "inputsCast", actual.inputsCast, estimates.inputsCast);
    printRow(cycles, "compute", actual.compute, estimates.compute);
    printRow(cycles, "reduction", actual.reduction, estimates.reduction);
    printRow(cycles, "dynamicUpdate", actual.dynamicUpdate,
             estimates.dynamicUpdate);
    printRow(cycles, "addInPlace", actual.addInPlace, estimates.addInPlace);
    printRow(cycles, "outputCast", actual.outputCast, estimates.outputCast);
    printRow(cycles, "unknown", actual.unknown, unknown_costs);
    // clang-format off
    print(cout, "----------------------+--------------+--------------+----------\n");
    // clang-format on
    printRow(cycles, "total", actual.total, estimates.total);
    print(cout, "\n");
  }

  if (reportVerbose) {
    print(cout, "Note 1: Work-lists are excluded from totals.\n");
    print(cout, "Note 2: This does not handle program overlap.\n");
  }
}

void getSimulatedCosts(const std::string &simulatorTraceFilePath,
                       unsigned numTiles, bool keepFullLabelNames,
                       bool reportPerTile, bool reportPerSerialSplit) {
  std::ifstream o(simulatorTraceFilePath);
  if (!o)
    throw poputil::poplibs_error("File does not exist: " +
                                 simulatorTraceFilePath);

  // Parse a branch instruction: brnzdec      $m4 [ 0x00000000 ], 0x0004c894
  const std::regex branchRegex(
      R"(^(br\w*)\s+\$\w(\d\d?)\s*\[ (0x[0-9a-f]+) \], (0x[0-9a-f]+)$)");

  // Parse a rpt instruction:    rpt          $m3 [ 0x0000000a ], 0x00000008
  const std::regex rptRegex(
      R"(rpt\s+\$m\d\d?\s+\[ (0x[0-9a-f]+) \], (0x[0-9a-f]+)$)");

  // Parse a ld instruction:: ld32         $m5 [ 0x00000001 ], ...
  const std::regex ld32Regex(
      R"(^ld[\w\d]*\s+\$(m|a)(\d\d?)\s?\[ (0x[0-9a-f]+) \], .*)");

  const auto isLoop = [](StringRef labelName) -> bool {
    return (r_contains(labelName, "Loop") ||
            r_contains(labelName, "ZeroConvGroup") ||
            r_contains(labelName, "ZeroOutChanGroup")) &&
           !r_contains(labelName, "end");
  };

  // Record the conv partial vertex being used as soon as we see it.
  std::string vertexName;
  const auto isVertexStart = [&vertexName](StringRef labelName,
                                           unsigned offset) -> bool {
    if (offset != 0)
      return false;
    if (!vertexName.empty())
      return labelName == vertexName;
    if (!starts_with(labelName, "__runCodelet_poplin__ConvPartial"))
      return false;
    vertexName = labelName;
    return true;
  };

  struct CyclesInfo {
    uint64_t cycles = 0;
    uint64_t stalls = 0;
    uint64_t syncs = 0;
    uint64_t instructions = 0;
    uint64_t bundles = 0;
    uint64_t overlapped = 0;
    CyclesInfo &operator+=(const CyclesInfo &o) {
      cycles += o.cycles;
      stalls += o.stalls;
      syncs += o.syncs;
      instructions += o.instructions;
      bundles += o.bundles;
      overlapped += o.overlapped;
      return *this;
    }
  };
  struct Label : public CyclesInfo {
    std::string name;
    uint64_t startAddress = 0;
    uint64_t execCount = 0;
    bool isBrLoop = false;
    bool isRptLoop = false;
  };
  struct State {
    Label *prevLabel = nullptr;
    Label *currentLabel = nullptr;
    uint64_t rptLoopCount = 0;
    uint64_t rptStart = 0;
    uint64_t rptNumInstructions = 0;
    std::array<unsigned, 16> registers = {};
    bool wasLastInstructionASync = false;
  };
  struct ContextInfo {
    CyclesInfo totals;
    std::vector<Label> labels;
    State state;
  };
  struct TileInfo {
    std::array<ContextInfo, 7> contexts;
    size_t previousCycles = 0;
    unsigned previousContextId = 0;
    unsigned numSerialSplits = 0;
  };
  std::vector<TileInfo> contextsPerTile(numTiles);

  std::string line;
  line.reserve(256);
  while (std::getline(o, line)) {
    if (!(line.size() > 6 && starts_with(line, "t[")))
      continue;

    // Parse a line of CISS trace. The parsing is done manually as it's
    // about 4x quicker than regex, which makes a big difference due to
    // the large amount of output CISS can dump. A single line looks like:
    //
    // t[0.0]: 0x0004d960 (0x4180000f): label +   8:   sync 0x0000000f m @ 12
    //   ^ ^   ^            ^ instruction op code
    //   | |   + program counter
    //   | + execution context
    //   + tile number
    //
    // where "m @ 12" means *m*ain pipeline and a total cycle count of 12.
    char *end = nullptr;
    unsigned tile = std::strtoul(line.data() + 2, &end, 10);

    // Skip tiles we're not interested in.
    if (tile >= contextsPerTile.size())
      continue;

    unsigned contextId = std::strtoul(end + 1, &end, 10);
    unsigned address = std::strtoul(end + 2, &end, 16);
    end = std::find(end + 1, line.data() + line.size(), ')');
    end += 2; // skip ):
    char *plus = std::find(end, line.data() + line.size(), '+');
    auto labelName = trim(StringRef(end, std::distance(end, plus)));
    unsigned offsetIntoLabel = std::strtoul(plus + 1, &end, 10);
    end += 1; // skip :
    char *at_ = std::find(end, line.data() + line.size(), '@');
    auto instruction = trim(StringRef(end, std::distance(end, at_ - 2)));
    uint64_t cycles = std::strtoull(at_ + 1, &end, 10);

    // Unpack loop state.
    auto &tileInfo = contextsPerTile[tile];
    auto &previousCycles = tileInfo.previousCycles;
    auto &previousContextId = tileInfo.previousContextId;
    auto &numSerialSplits = tileInfo.numSerialSplits;

    auto &contextInfo = tileInfo.contexts[contextId];
    auto &totals = contextInfo.totals;
    auto &labels = contextInfo.labels;
    auto &state = contextInfo.state;

    // Stalls are delays caused by instruction dependencies.
    bool isStall = instruction.size() == 1 && instruction[0] == '-';

    if (isVertexStart(labelName, offsetIntoLabel)) {
      // Discount any cycles from before the execution of the vertex. Note that
      // also on first execution CISS doesn't always start at 0 which can lead
      // to misleading cycle counts, so also discount that here. Force the
      // first instruction to count as 1 cycle even if it's not correct because
      // it's still more accurate than starting at a random number of cycles.
      // Also note that this discounts cycles from between serial splits.
      if (cycles != 0)
        previousCycles = cycles - 1;
      if (numSerialSplits == 1 && reportPerSerialSplit)
        break;
      if (!isStall)
        numSerialSplits += 1;
    }

    // Categorize cycles.

    uint64_t cyclesTaken = cycles - previousCycles;
    // Instructions can be interleaved with instructions issued by other
    // contexts, which is useful to hide the pipeline latency. Note that
    // this does not track interleaving from the same context.
    bool isOverlapped = contextId != previousContextId;
    // Instruction bundles are simulated as an m and an a instruction in at
    // the same cycle step. Count the second instruction as the bundle.
    bool isBundle = cyclesTaken == 0;

    // Track stalls due to syncs separately to just stalls.
    if (!isStall)
      state.wasLastInstructionASync = starts_with(instruction, "sync");

    CyclesInfo cyclesInfo;
    cyclesInfo.cycles += cyclesTaken;
    cyclesInfo.overlapped += isOverlapped;
    if (isStall && state.wasLastInstructionASync)
      cyclesInfo.syncs += cyclesTaken;
    else if (isStall)
      cyclesInfo.stalls += cyclesTaken;
    else if (isBundle)
      cyclesInfo.bundles += 1;
    else
      cyclesInfo.instructions += 1;

    totals += cyclesInfo;

    // Attribute the instruction to a label; specifically loops.

    // Try to use an existing label with the same name if possible,
    // otherwise create a new label.
    auto getOrMakeLabel = [&](StringRef labelName) -> Label * {
      auto it = std::find_if(
          labels.rbegin(), labels.rend(),
          [labelName](const auto &label) { return label.name == labelName; });
      if (it != labels.rend())
        return &*it;

      auto &new_label = labels.emplace_back();
      new_label.name = labelName;
      new_label.isBrLoop = false;
      new_label.isRptLoop = false;
      new_label.startAddress = address;
      return &new_label;
    };

    // Only use new labels for loops and when we don't have a parent label.
    if (state.currentLabel == nullptr || isLoop(labelName))
      state.currentLabel = getOrMakeLabel(labelName);

    // Stalls should be attributed to the prior instruction for syncs, branches,
    // put and get instructions but to the next instruction for register
    // bubbles. CISS, however, always attributes stalls to the next instruction
    // by making the address and opcode match the next instruction. This is
    // problematic when the next instruction is under a different label, e.g:
    // for
    //
    //    label A:
    //      put
    //    label B:
    //      add
    //
    // CISS will print:
    //
    //    address A, label A: put
    //    address B, label B: -
    //    ^ this 5 more times ^
    //    address B, label B: add
    //
    if (!isStall) {
      if (starts_with(instruction, "sync") || starts_with(instruction, "br") ||
          starts_with(instruction, "put") || starts_with(instruction, "uput") ||
          starts_with(instruction, "get"))
        state.prevLabel = state.currentLabel;
      else
        state.prevLabel = nullptr;
    }

    if (isStall && state.prevLabel)
      *state.prevLabel += cyclesInfo;
    else
      *state.currentLabel += cyclesInfo;

    if (isStall) {
      // no control flow changes.
    } else if (state.rptNumInstructions == state.rptStart && isBundle) {
      // co-issued with rpt; don't want to decrement rpt count for this.
    } else if (state.rptNumInstructions) {
      // inside the body of a rpt loop
      if (state.rptNumInstructions == state.rptStart) {
        state.currentLabel = getOrMakeLabel(labelName);
        state.currentLabel->isRptLoop = true;
        state.currentLabel->execCount += state.rptLoopCount;
      }
      state.rptNumInstructions -= 1;
      if (state.rptNumInstructions == 0) {
        // exiting the rpt loop
        state.currentLabel -= 1;
      }
    } else if (starts_with(instruction, "rpt")) {
      std::cmatch rptMatch;
      if (!std::regex_match(instruction.begin(), instruction.end(), rptMatch,
                            rptRegex))
        throw poputil::poplibs_error(
            fmt::format("Unknown repeat instruction: {}", instruction));
      auto rptLoopCount = std::strtoul(rptMatch[1].first, nullptr, 16);
      auto rptBodySize = std::strtoul(rptMatch[2].first, nullptr, 16);
      // Convert to units of instructions.
      state.rptStart = rptLoopCount * (rptBodySize / 8 + 1) * 2;
      state.rptNumInstructions = state.rptStart;
      state.rptLoopCount = rptLoopCount;
    } else if (starts_with(instruction, "br") && instruction[2] != ' ' &&
               instruction[2] != 'i') {
      std::cmatch branchMatch;
      if (!std::regex_match(instruction.begin(), instruction.end(), branchMatch,
                            branchRegex))
        throw poputil::poplibs_error(
            fmt::format("Unknown branch instruction: {}", instruction));
      poplar::StringRef branchInstruction(branchMatch[1].first,
                                          branchMatch[1].length());
      auto dstReg = std::strtoul(branchMatch[2].first, nullptr, 10);
      auto dstRegValue = std::strtoul(branchMatch[3].first, nullptr, 16);
      auto targetAddress = std::strtoul(branchMatch[4].first, nullptr, 16);

      // See if the branch is taken or not.
      const bool isBranchTaken = [&]() {
        if (branchInstruction == "brnzdec") {
          // CISS prints the register value after the effects of the branch,
          // which for brnzdec means the dst register got decremented if the
          // branch was taken. This makes a loop counter of 0 indistinguishable
          // from 1. To work-around this we keep track of the values in the
          // registers and use that instead.
          if (dstRegValue > 1)
            return true;
          auto &loopCount = state.registers.at(dstReg);
          if (loopCount > 0) {
            loopCount -= 1;
            return true;
          } else {
            return false;
          }
        }
        if (branchInstruction == "brnz")
          return dstRegValue != 0;
        if (branchInstruction == "brz")
          return dstRegValue == 0;
        if (branchInstruction == "brneg")
          return dstRegValue < 0;
        if (branchInstruction == "brpos")
          return dstRegValue >= 0;
        if (branchInstruction == "br" || branchInstruction == "bri")
          return true;
        throw poputil::poplibs_error(
            fmt::format("Unknown branch instruction: {}", branchInstruction));
      }();
      // Might be we've already seen the label. If we haven't seen the label
      // yet then it can't be a loop.
      Label *targetLabel = &labels.back();
      for (; targetLabel >= labels.data(); --targetLabel)
        if (targetLabel->startAddress == targetAddress)
          break;
      if (targetLabel >= labels.data()) {
        assert(!targetLabel->isRptLoop);
        // Could just be indirect control flow (not a loop).
        if (!targetLabel->isBrLoop &&
            (targetAddress >= address || // skip forward
             !isLoop(targetLabel->name))) {
          if (isBranchTaken)
            state.currentLabel = targetLabel;
        } else { // likely loop
          targetLabel->isBrLoop = true;
          if (isBranchTaken) {
            state.currentLabel = targetLabel;
          } else { // exiting br loop
            if (targetLabel == labels.data()) {
              state.currentLabel = nullptr;
            } else {
              state.currentLabel = targetLabel - 1;
            }
          }
        }
      }
    } else {
      if (offsetIntoLabel == 0 && labelName == state.currentLabel->name)
        state.currentLabel->execCount += 1;
    }

    previousContextId = contextId;
    previousCycles = cycles;

    if ((starts_with(instruction, "ld32") && instruction[4] == ' ') ||
        (starts_with(instruction, "ldz16") && instruction[5] == ' ')) {
      // Use the fact that the conv loops always load their loop counts before
      // branching (and thankfully don't mutate them in-between). Parse a line
      std::cmatch ld32Match;
      if (!std::regex_match(instruction.begin(), instruction.end(), ld32Match,
                            ld32Regex))
        throw poputil::poplibs_error(
            fmt::format("Unknown load instruction: {}", instruction));

      auto dstReg = std::strtoul(ld32Match[1].first, nullptr, 10);
      char pipeline = *ld32Match[2].first - 'a';
      auto dstRegValue = std::strtoul(ld32Match[3].first, nullptr, 16);
      if (pipeline == 'm')
        state.registers.at(dstReg) = dstRegValue;
      // ignore aux registers as they don't affect branching.
    }
  }

  using fmt::print;
  using std::cout;

  TileInfo *tileToReport = nullptr;
  TileInfo totalTile;
  uint64_t totalCyclesAllTiles = 0;
  if (reportPerTile) {
    // Report the slowest tile.
    uint64_t maxCyclesPerTile = 0;
    for (auto &tileInfo : contextsPerTile) {
      uint64_t totalCycles = 0;
      for (const auto &context : tileInfo.contexts)
        totalCycles += context.totals.cycles;
      if (totalCycles > maxCyclesPerTile) {
        tileToReport = &tileInfo;
        maxCyclesPerTile = totalCycles;
      }
      totalCyclesAllTiles += totalCycles;
    }
    print(cout,
          "Reporting {} cycles for tile {} ({} total cycles for {} tiles):\n\n",
          maxCyclesPerTile, std::distance(contextsPerTile.data(), tileToReport),
          totalCyclesAllTiles, contextsPerTile.size());
  } else {
    // Report the total across all tiles.
    tileToReport = &totalTile;
    for (auto &tileInfo : contextsPerTile) {
      for (size_t i = 0; i < tileInfo.contexts.size(); ++i) {
        totalCyclesAllTiles += tileInfo.contexts[i].totals.cycles;
        totalTile.contexts[i].totals += tileInfo.contexts[i].totals;
        for (const auto &label : tileInfo.contexts[i].labels) {
          auto it = std::find_if(totalTile.contexts[i].labels.begin(),
                                 totalTile.contexts[i].labels.end(),
                                 [&label](const auto &totalLabel) {
                                   return totalLabel.name == label.name;
                                 });
          if (it == totalTile.contexts[i].labels.end())
            totalTile.contexts[i].labels.push_back(label);
          else
            *it += label;
        }
      }
    }
    print(cout, "Reporting {} total cycles for {} tiles:\n\n",
          totalCyclesAllTiles, contextsPerTile.size());
  }

  // Print a report comparing each execution context.
  {
    constexpr char format[] = "{1: <{0}} | {2: >8} | {3: >8} | {4: >8} | {5: "
                              ">8} | {6: >8} | {7: >8}\n";
    constexpr char divider[] = "{1:-<{0}}-+-{2:->8}-+-{3:->8}-+-{4:->8}-+-{5:->"
                               "8}-+-{6:->8}-+-{7:->8}\n";
    constexpr size_t width = 10;
    print(cout, "Context breakdown:\n\n");
    print(cout, format, width, "Context", "Cycles", "Stalls", "Syncs", "Instns",
          "Bundles", "Overlap");
    print(cout, divider, width, "-", "-", "-", "-", "-", "-", "-");

    CyclesInfo total = {};
    for (size_t i = 0; i < 7; ++i) {
      const auto &info = tileToReport->contexts[i].totals;
      const std::string name =
          (i == 0) ? "supervisor" : "worker   " + std::to_string(i);
      print(cout, format, width, name, info.cycles, info.stalls, info.syncs,
            info.instructions, info.bundles, info.overlapped);
      total += info;
    }
    print(cout, divider, width, "-", "-", "-", "-", "-", "-", "-");
    print(cout, format, width, "total", total.cycles, total.stalls, total.syncs,
          total.instructions, total.bundles, total.overlapped);
  }

  // Prettify the label names before we report them.
  if (!keepFullLabelNames) {
    constexpr std::array<const char *, 9> supervisor = {
        "ConvGroupLoop", "InChanLoop",       "OutChanLoop",
        "KyLoop",        "KxLoop",           "AmpOutGroupLoop",
        "ZeroConvGroup", "ZeroOutChanGroup", "ConvPartialnx1"};
    constexpr std::array<const char *, 3> worker = {
        "PartitionLoop", "Loop_start_Amp", "Loop_start_zero"};
    for (size_t i = 0; i < 7; ++i) {
      const auto &names = (i == 0)
                              ? ArrayRef{supervisor.data(), supervisor.size()}
                              : ArrayRef{worker.data(), worker.size()};
      for (auto &label : tileToReport->contexts[i].labels)
        for (auto name : names)
          if (r_contains(label.name, name))
            label.name = name;
    }
  }

  // Print a report of each label we recorded per context.
  for (size_t i = 0; i < 7; ++i) {
    const auto &labels = tileToReport->contexts[i].labels;
    const std::string name =
        (i == 0) ? "supervisor" : "worker   " + std::to_string(i);
    // The names can be very long so align the table dynamically.
    size_t width = 0;
    for (const auto &li : labels) {
      width = std::max(li.name.size(), width);
    }
    constexpr char format[] = "{1: <{0}} | {2: >8} | {3: >8} | {4: >8} | {5: "
                              ">8} | {6: >8} | {7: >8} | {8: >8} | {9: >8}\n";
    constexpr char divider[] = "{1:-<{0}}-+-{2:->8}-+-{3:->8}-+-{4:->8}-+-{5:->"
                               "8}-+-{6:->8}-+-{7:->8}-+-{8:->8}-+-{9:->8}\n";
    print(cout, "\nLabel breakdown for {} ({} labels):\n\n", name,
          labels.size());
    print(cout, format, width, "Loop Name", "Cycles", "Stalls", "Syncs",
          "Instns", "Bundles", "Overlap", "Count", "Cyc/Cnt");
    print(cout, divider, width, "-", "-", "-", "-", "-", "-", "-", "-", "-");
    Label totals;
    for (const auto &li : labels) {
      print(cout, format, width, li.name, li.cycles, li.stalls, li.syncs,
            li.instructions, li.bundles, li.overlapped, li.execCount,
            li.cycles / li.execCount);
      totals += li;
      totals.execCount += li.execCount;
    }
    print(cout, divider, width, "-", "-", "-", "-", "-", "-", "-", "-", "-");
    print(cout, format, width, "total", totals.cycles, totals.stalls,
          totals.syncs, totals.instructions, totals.bundles, totals.overlapped,
          totals.execCount, totals.cycles / totals.execCount);
  }

  // Try to map cycles to the loop counts.
  {
    print(cout, "\nSimulated dim size and overhead:\n\n");
    const auto &supervisor = tileToReport->contexts[0].labels;
    const auto g1 = supervisor[3].execCount;
    const auto ic1 = supervisor[4].execCount / supervisor[3].execCount;
    const auto oc1 = supervisor[5].execCount / supervisor[4].execCount;
    const auto ky = supervisor[6].execCount / supervisor[5].execCount;
    const auto kx = supervisor[7].execCount / supervisor[6].execCount;
    const auto numReportedTiles = reportPerTile ? 1 : numTiles;
    print(cout, "  g1 ={}, overhead={}\n", g1,
          supervisor[3].cycles / supervisor[3].execCount / numReportedTiles);
    print(cout, "  ic1={}, overhead={}\n", ic1,
          supervisor[4].cycles / supervisor[4].execCount / numReportedTiles);
    print(cout, "  oc1={}, overhead={}\n", oc1,
          supervisor[5].cycles / supervisor[5].execCount / numReportedTiles);
    print(cout, "  ky ={}, overhead={}\n", ky,
          supervisor[6].cycles / supervisor[6].execCount / numReportedTiles);
    print(cout, "  kx ={}, overhead={}\n", kx,
          supervisor[7].cycles / supervisor[7].execCount / numReportedTiles);
    print(cout, "\n");
  }
}

} // namespace amp
