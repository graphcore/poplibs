// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/print.hpp"
#include <poplibs_support/TestDevice.hpp>

#include "poplibs_test/TempDir.hpp"
#include "popops/Cast.hpp"
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Quarter.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/ProgressBar.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/ConvPreplan.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/GraphFunction.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <pva/pva.hpp>

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <algorithm>
#include <cassert>
#include <exception>
#include <fstream>
#include <istream>
#include <ostream>
#include <random>

// Default tolerances used in tests with uniform distributions of random data
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poputil;
using namespace poplibs_support;
using namespace poplin::internal;

using poplibs_test::Pass;

const OptionFlags defaultEngineOptions;

static bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

static bool r_contains(const std::string &haystack, const std::string &needle) {
  return haystack.rfind(needle) != std::string::npos;
}

static void updateDetailedPlanCosts(bool reportPerTile,
                                    bool reportPerSerialSplit,
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

// Like DetailedPlanCosts but with a couple of helper methods.
struct MeasuredPlanCosts : public DetailedPlanCosts {
  poplin::PlanCosts unknown = {};
  size_t totalWorkListBytes = 0;

  size_t totalCycles() const noexcept {
    size_t total = 0;
    apply([&total](const poplin::PlanCosts &c) { total += c.cycles; }, false);
    return total;
  }

  size_t totalMemory() const noexcept {
    // Memory inside the serial splits is maxed rather than summed
    // because we assume the temporary memory will be reused.
    return broadcast.memory + tileLevelTransform.memory + rearrangement.memory +
           std::max({compute.memory, exchange.memory, transform.memory,
                     reduction.memory, dynamicSlice.memory,
                     dynamicUpdate.memory, unknown.memory});
  }

  template <typename Function>
  void apply(Function fn, bool includeTotal = true) {
    DetailedPlanCosts::apply(fn, includeTotal);
    fn(unknown);
  }
  template <typename Function>
  void apply(Function fn, bool includeTotal = true) const {
    DetailedPlanCosts::apply(fn, includeTotal);
    fn(unknown);
  }

  poplin::PlanCosts toPlanCosts() const noexcept {
    return {totalCycles(), totalMemory()};
  }

  poplin::PlanCosts &selectCosts(const std::string &name,
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
};

struct MemoryCostProgramVisitor : public pva::ProgramVisitor {
  MemoryCostProgramVisitor(MeasuredPlanCosts &planCosts,
                           bool reportPerTile_ = false, size_t activeTiles_ = 1,
                           bool isInsideSerialSplit_ = false,
                           bool printVars_ = false)
      : costs(planCosts), reportPerTile(reportPerTile_),
        activeTiles(activeTiles_), isInsideSerialSplit(isInsideSerialSplit_),
        printVars(printVars_) {}

  void visitDoExchange(const pva::DoExchangeProgram &doExchange) override {
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

  void
  visitOnTileExecute(const pva::OnTileExecuteProgram &onTileExecute) override {
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

private:
  MeasuredPlanCosts &costs;
  bool reportPerTile;
  size_t activeTiles = 1;
  bool isInsideSerialSplit = false;
  bool printVars;
};

static MeasuredPlanCosts
getActualCosts(const std::string &pass, const std::string &profileDir,
               size_t parallelSplit, size_t serialSplit, bool printVars = false,
               bool reportPerTile = false, bool reportPerSerialSplit = false) {
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

static void compareCosts(const std::string &title,
                         DetailedPlanCosts const &estimates,
                         MeasuredPlanCosts const &actual,
                         bool reportVerbose = false, bool reportPerTile = false,
                         bool reportPerSerialSplit = false) {
  using fmt::print;
  using poplin::PlanCosts;
  using std::cout;

  // Print out a summary.
  if (reportPerTile)
    print(cout, "\n{}: Summary for the average tile (of {} tiles):\n\n", title,
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
    printRow(cycles, "total", actual.total, estimates.total);
    print(cout, "\n");
  }

  if (reportVerbose) {
    print(cout, "Note 1: Work-lists are excluded from totals.\n");
    print(cout, "Note 2: This does not handle program overlap.\n");
  }
}

static void overloadConstraintsFromFile(const std::string &path,
                                        std::string &s) {
  if (!path.empty()) {
    std::ifstream is(path, std::ios_base::in);
    if (!is.good()) {
      throw poputil::poplibs_error("Constraints file " + path +
                                   " could not be opened");
    }
    is.seekg(0, std::ios::end);
    const auto bytes = is.tellg();
    s = std::string(bytes, '\0');
    is.seekg(0);
    is.read(&s[0], bytes);
  }
}

static Tensor createGenericConvInput(Graph &graph,
                                     const poplin::ConvParams &params,
                                     const std::string &name = "") {
  return poplibs_test::util::createGenericConvInput(
      graph, params.inputType, params.getBatchSize(), params.getNumConvGroups(),
      params.getNumInputChansPerConvGroup(), params.getInputFieldShape(), name);
}

static Tensor
convolve(bool useCreateOutput, poplar::Graph &graph, const poplar::Tensor &in,
         const poplar::Tensor &weights, const poplin::ConvParams &params,
         bool transposeAndFlipWeights, poplar::program::Sequence &prog,
         const poplar::DebugContext &debugContext,
         const poplar::OptionFlags &options, poplin::PlanningCache *cache) {
  if (useCreateOutput) {
    auto out =
        poplin::createConvOutput(graph, params, debugContext, options, cache);
    poplin::convolutionWithOutput(graph, in, weights, out, params,
                                  transposeAndFlipWeights, prog, debugContext,
                                  options, cache);
    return out;
  } else {
    return poplin::convolution(graph, in, weights, params,
                               transposeAndFlipWeights, prog, debugContext,
                               options, cache);
  }
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  unsigned fwdInChansPerConvGroup;
  unsigned fwdOutChansPerConvGroup;
  ShapeOption<std::size_t> inputFieldSizeOption;
  ShapeOption<std::size_t> kernelSizeOption;
  unsigned numConvGroups = 1;
  ShapeOption<unsigned> truncationLowerOption, truncationUpperOption,
      truncationOption;
  ShapeOption<unsigned> inDilationOption;
  ShapeOption<unsigned> paddingLowerOption, paddingUpperOption, paddingOption;
  ShapeOption<bool> flipInputOption;
  ShapeOption<unsigned> kernelTruncationLowerOption,
      kernelTruncationUpperOption, kernelTruncationOption;
  ShapeOption<unsigned> kernelDilationOption;
  ShapeOption<unsigned> kernelPaddingLowerOption, kernelPaddingUpperOption,
      kernelPaddingOption;
  ShapeOption<bool> flipKernelOption;
  ShapeOption<unsigned> outputTruncationOption, outputTruncationLowerOption,
      outputTruncationUpperOption;
  ShapeOption<unsigned> strideOption;
  ShapeOption<unsigned> outputPaddingOption, outputPaddingLowerOption,
      outputPaddingUpperOption;
  unsigned batchSize;
  bool bias;
  Type inputType;
  Type outputType;
  double absoluteTolerance, relativeTolerance;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  boost::optional<unsigned> inputLoadWidth;
  bool reportPlan;
  bool reportVarStorage;
  unsigned numDeterminismChecks;
  bool enableConvolutionReuse;
  bool remapOutputTensor;
  bool useCreateInput;
  bool useCreateOutput;
  bool preplan;
  QuarterMetadata::Format fp8FormatFwdIn = QuarterMetadata::Format::F143;
  QuarterMetadata::Format fp8FormatWeights = QuarterMetadata::Format::F143;
  QuarterMetadata::Format fp8FormatBwdIn = QuarterMetadata::Format::F143;
  int fp8ScaleFwdIn = 1, fp8ScaleWeights = 2, fp8scaleBwdIn = 1;

  Pass pass = Pass::ALL;
  std::string fwdPlanConstraints, fwdPlanConstraintsFile, bwdPlanConstraints,
      bwdPlanConstraintsFile, wuPlanConstraints, wuPlanConstraintsFile,
      convOptionsString;
  boost::optional<std::string> fwdOutFile, fwdInFile, bwdOutFile, bwdInFile;
  poplin::PlanningCache cache;

  boost::optional<std::string> profileDir;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling report")
    ("profile-dir",
     po::value<decltype(profileDir)>(&profileDir)
      ->default_value(boost::none),
     "Write profile files to the specified directory.")
    ("test-planner",
     "Compare the profiler's measurements to the planner's estimates. "
     "Useful for testing the planner's estimates.")
    ("test-planner-existing-profile", "Use an existing profile. "
     "The profile should be specified using the 'profile-dir' option")
    ("test-planner-report-all-tiles",
     "Report the total cycle/memory usage for all tiles.")
    ("test-planner-report-all-serial-splits",
     "Report the cycle/memory usage for all serial splits.")
    ("test-planner-report-verbose",
     "Always report with maximum detail")
    ("test-planner-print-vars",
     "Print every program and vars in the profile that's related to the conv.")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("input-channels", po::value<unsigned>(&fwdInChansPerConvGroup)->required(),
     "Number of input channels per grouped convolution")
    ("output-channels",
     po::value<unsigned>(&fwdOutChansPerConvGroup)->required(),
     "Number of output channels per grouped convolution")
    ("field",
     po::value<ShapeOption<std::size_t>>(&inputFieldSizeOption),
      "Field size")
    ("kernel-size",
      po::value<ShapeOption<std::size_t>>(&kernelSizeOption)->default_value(1),
     "Size of square kernel. If set, it is an error to also set either "
     "kernel-height and/or kernel-width")
    ("bias", po::value<bool>(&bias)->default_value(true),
     "Add a bias to each channel")
    ("data-type",
     po::value<Type>(&inputType)->default_value(HALF),
     "Type of the input and output data")
    ("input-type",
     po::value<Type>(&inputType),
     "Type of the input data")
    ("output-type",
     po::value<Type>(&outputType),
     "Type of the output data and the parameters")
    ("fp8-scale-fwd",
      po::value<int>(&fp8ScaleFwdIn)->default_value(fp8ScaleFwdIn),
     "Scaling to apply to the fwd input if its type is quarter")
    ("fp8-scale-weights",
      po::value<int>(&fp8ScaleWeights)->default_value(fp8ScaleWeights),
     "Scaling to apply to the weights input if its type is quarter")
    ("fp8-scale-bwd",
      po::value<int>(&fp8scaleBwdIn)->default_value(fp8scaleBwdIn),
     "Scaling to apply to the bwd input if its type is quarter")
    ("fp8-format-fwd",
      po::value<QuarterMetadata::Format>(&fp8FormatFwdIn)->
      default_value(fp8FormatFwdIn),
     "The data format of the fwd input if its type is quarter")
    ("fp8-format-weights",
      po::value<QuarterMetadata::Format>(&fp8FormatWeights)->
      default_value(fp8FormatWeights),
     "The data format of the weights input if its type is quarter")
    ("fp8-format-bwd",
      po::value<QuarterMetadata::Format>(&fp8FormatBwdIn)->
      default_value(fp8FormatBwdIn),
     "The data format of the bwd input if its type is quarter")
    ("truncation",
     po::value<ShapeOption<unsigned>>(&truncationOption)->default_value(0),
     "Amount to truncate the start and end of each dimension of the input")
    ("truncation-upper",
     po::value<ShapeOption<unsigned>>(&truncationUpperOption)->default_value(0),
     "Amount to truncate the end of each dimension of the input")
    ("truncation-lower",
     po::value<ShapeOption<unsigned>>(&truncationLowerOption)->default_value(0),
     "Amount to truncate the start of each dimension of the input")
    ("in-dilation",
     po::value<ShapeOption<unsigned>>(&inDilationOption)->default_value(1),
     "Input dilation")
    ("padding",
     po::value<ShapeOption<unsigned>>(&paddingOption)->default_value(0),
     "Amount of zero padding to add to the start and end of each dimension")
    ("padding-upper",
     po::value<ShapeOption<unsigned>>(&paddingUpperOption)->default_value(0),
     "Amount of zero padding to add at the end of each dimension")
    ("padding-lower",
     po::value<ShapeOption<unsigned>>(&paddingLowerOption)->default_value(0),
     "Amount of zero padding to add at the start of each dimension")
    ("flip-input",
     po::value<ShapeOption<bool>>(&flipInputOption)->default_value(false),
     "Whether to flip each input spatial field")
    ("kernel-truncation",
     po::value<ShapeOption<unsigned>>(&kernelTruncationOption)
         ->default_value(0),
     "Amount to truncate the start and end of each dimension of the kernel")
    ("kernel-truncation-upper",
     po::value<ShapeOption<unsigned>>(&kernelTruncationUpperOption)
         ->default_value(0),
     "Amount to truncate the end of each dimension of the kernel")
    ("kernel-truncation-lower",
     po::value<ShapeOption<unsigned>>(&kernelTruncationLowerOption)
         ->default_value(0),
     "Amount to truncate the start of each dimension of the kernel")
    ("kernel-dilation",
     po::value<ShapeOption<unsigned>>(&kernelDilationOption)
         ->default_value(1),
     "Kernel dilation")
    ("kernel-padding",
     po::value<ShapeOption<unsigned>>(&kernelPaddingOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the start and end of each "
     "dimension")
    ("kernel-padding-upper",
     po::value<ShapeOption<unsigned>>(&kernelPaddingUpperOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the start of each dimension")
    ("kernel-padding-lower",
     po::value<ShapeOption<unsigned>>(&kernelPaddingLowerOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the end of each dimension")
    ("flip-kernel",
     po::value<ShapeOption<bool>>(&flipKernelOption)->default_value(false),
     "Whether to flip each kernel spatial field")
    ("output-truncation",
     po::value<ShapeOption<unsigned>>(&outputTruncationOption)
         ->default_value(0),
     "Number of output elements to truncate")
    ("output-truncation-upper",
     po::value<ShapeOption<unsigned>>(&outputTruncationUpperOption)
         ->default_value(0),
     "Number of output elements to truncate at the end of each dimension")
    ("output-truncation-lower",
     po::value<ShapeOption<unsigned>>(&outputTruncationLowerOption)
         ->default_value(0),
     "Number of output elements to truncate at the start of each dimension")
    ("stride",
     po::value<ShapeOption<unsigned>>(&strideOption)->default_value(1),
     "Stride")
    ("output-padding",
     po::value<ShapeOption<unsigned>>(&outputPaddingOption)->default_value(0),
     "Number of output elements to truncate")
    ("output-padding-upper",
     po::value<ShapeOption<unsigned>>(&outputPaddingUpperOption)
         ->default_value(0),
     "Number of output elements to truncate at the end of each dimension")
    ("output-padding-lower",
     po::value<ShapeOption<unsigned>>(&outputPaddingLowerOption)
         ->default_value(0),
     "Number of output elements to truncate at the start of each dimension")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("plan-only", "Only plan the requested passes, don't build or run a graph")
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model. Upon failure, the error code relates to what multiple of this tolerance failed. "
     "Error code 11 -> 1-2x; 12 -> 2-3x; 13 -> 3-4x, e.g. tolerance of 2 and failure of 5, returns error code 12.")
    ("absolute-tolerance",
     po::value<double>(&absoluteTolerance),
     "Absolute tolerance to use when validating results against the reference "
     "model. Upon failure, the error code relates to what multiple of this tolerance failed. "
     "Error code 11 -> 1-2x; 12 -> 2-3x; 13 -> 3-4x, e.g. tolerance of 2 and failure of 5, returns error code 12.")
    ("ipus",
     po::value<unsigned>(&numIPUs)->default_value(numIPUs),
     "Number of IPUs")
    ("tiles-per-ipu", po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("input-load-width", po::value(&inputLoadWidth),
     "Load width for AMP/tile memory when the device is an IPUModel")
    ("workers-per-tile",
     po::value<unsigned>(),
     "Number of worker contexts per tile")
    ("batch-size",
     po::value<unsigned>(&batchSize),
     "Batch size")
    ("conv-groups",
     po::value<unsigned>(&numConvGroups)->default_value(1),
     "Number of convolution groups in grouped convolution")
    ("fwd-plan-constraints",
     po::value<std::string>(&fwdPlanConstraints)
        ->default_value(fwdPlanConstraints),
     "Constraints on the chosen convolution plan for the forward pass "
     "as a JSON string")
    ("fwd-plan-constraints-file",
     po::value<std::string>(&fwdPlanConstraintsFile)
        ->default_value(fwdPlanConstraintsFile),
     "Constraints on the chosen convolution plan for the forward pass "
     "as a path to a JSON file")
    ("bwd-plan-constraints",
     po::value<std::string>(&bwdPlanConstraints)
        ->default_value(bwdPlanConstraints),
     "Constraints on the chosen convolution plan for the backward pass "
     "as a JSON string")
    ("bwd-plan-constraints-file",
     po::value<std::string>(&bwdPlanConstraintsFile)
        ->default_value(bwdPlanConstraintsFile),
     "Constraints on the chosen convolution plan for the backward pass "
     "as a path to a JSON file")
    ("wu-plan-constraints",
     po::value<std::string>(&wuPlanConstraints)
        ->default_value(wuPlanConstraints),
     "Constraints on the chosen convolution plan for the weight update pass "
     "as a JSON string")
    ("wu-plan-constraints-file",
     po::value<std::string>(&wuPlanConstraintsFile)
        ->default_value(wuPlanConstraintsFile),
     "Constraints on the chosen convolution plan for the weight update pass "
     "as a path to a JSON file")
    ("report-plan", po::value<bool>(&reportPlan)->default_value(false),
     "Display plan")
    ("report-var-storage",
     po::value<bool>(&reportVarStorage)->default_value(false),
     "Report variable storage information")
    ("remap-output-tensor",
     po::value<bool>(&remapOutputTensor)->default_value(false),
     "Remap output tensor if layout is detected to be poor")
    ("enable-convolution-reuse",
     po::value<bool>(&enableConvolutionReuse)->default_value(true),
     "Apply optimization to reuse the forward convolution in the backward pass")
    ("convolution-options", po::value<std::string>(&convOptionsString),
    "Options to use for the convolution, specified as a JSON string, "
    "e.g. {\"key\":\"value\"}")
    ("use-create-input",
     po::value<bool>(&useCreateInput)->default_value(true),
     "Use the input allocation function to create an input tensor with a "
     "layout optimised for the plan. If set to false use a generic layout that "
     "is independent of the plan and representative of a typical layout in a "
     "neural network")
    ("use-create-output",
     po::value<bool>(&useCreateOutput)->default_value(false),
     "Use the output allocation function to create an output tensor "
     "before convolving")
    ("num-determinism-checks",
     po::value<unsigned>(&numDeterminismChecks)->default_value(0),
     "The amount of additional identical executions (results are compared to check determinism)."
     "This option is required to be 0 if ignore-data is set or single-phase is not 'all' or device-type is not Hw.")
    ("preplan",
     po::value<bool>(&preplan)->default_value(true),
     "Whether or not to preplan the convolutions")
    ("fwd-in-file",
     po::value<decltype(fwdInFile)>(&fwdInFile)->default_value(boost::none),
      "If specified the file to load the FWD pass input tensor from")
    ("fwd-out-file",
     po::value<decltype(fwdOutFile)>(&fwdOutFile)->default_value(boost::none),
      "If specified the file to write the FWD pass output tensor to")
    ("bwd-in-file",
     po::value<decltype(bwdInFile)>(&bwdInFile)->default_value(boost::none),
      "If specified the file to load the BWD pass input tensor from")
    ("bwd-out-file",
     po::value<decltype(bwdOutFile)>(&bwdOutFile)->default_value(boost::none),
      "If specified the file to write the BWD pass output tensor to")
  ;
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      std::cout << "A multi-dimensional shape can be specified using a brace "
                   "enclosed comma\n"
                   "separated list, for example --stride={1,2}. You may also "
                   "specify a single\n"
                   "number without braces in which case that value is used for "
                   "each dimension,\n"
                   "for example --stride=2\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (fwdInFile || bwdInFile) {
    if (vm.count("field") || vm.count("input-type") || vm.count("batch-size")) {
      std::cerr << "Cannot specify --field, --batch-size or --input-type with "
                   "an input file\n";
      return 1;
    }
  } else {
    if (vm.count("field") == 0) {
      std::cerr << "Must use an input file or specify --field\n";
      return 1;
    }
    if (vm.count("batch-size") == 0) {
      batchSize = 1;
    }
  }

  bool testPlannerReportPerTile =
      vm.count("test-planner-report-all-tiles") == 0;
  bool testPlannerReportPerSerialSplit =
      vm.count("test-planner-report-all-serial-splits") == 0;
  bool testPlannerPrintVars = vm.count("test-planner-print-vars") != 0;
  bool testPlannerReportVerbose = vm.count("test-planner-report-verbose") != 0;
  if (vm.count("test-planner-existing-profile")) {
    if (vm.count("test-planner")) {
      std::cerr << "The 'test-planner' and 'test-planner-existing-"
                   "profile' options cannot be used together.\n";
      return 1;
    }
    if (vm.count("profile-dir") == 0) {
      std::cerr << "The 'profile-dir' option must be specified when using the "
                   "'test-planner-existing-profile' option.\n";
      return 1;
    }

    auto phases = [pass]() -> std::vector<std::string> {
      switch (pass) {
      case Pass::FWD:
        return {"fwd"};
      case Pass::BWD:
        return {"bwd"};
      case Pass::WU:
        return {"wu"};
      case Pass::ALL:
        return {"fwd", "bwd", "wu"};
      }
      POPLIB_UNREACHABLE();
    }();

    // Just look at the measurements from the profile and skip running a conv.
    DetailedPlanCosts estimates = {};
    for (const auto &phase : phases) {
      std::string name = *profileDir + "/" + phase + "-plan-estimates.txt";
      std::ifstream in(name);
      if (in) {
        in >> estimates;
        updateDetailedPlanCosts(testPlannerReportPerTile,
                                testPlannerReportPerSerialSplit, estimates);
      } else {
        fmt::print(std::cerr, "Warning: could not find plan estimates at: {}\n",
                   name);
      }
      // Read the profile back and compare the costs.
      MeasuredPlanCosts actual = getActualCosts(
          phase, *profileDir, estimates.parallelSplit, estimates.serialSplit,
          testPlannerPrintVars, testPlannerReportPerTile,
          testPlannerReportPerSerialSplit);
      compareCosts(phase, estimates, actual, testPlannerReportVerbose,
                   testPlannerReportPerTile, testPlannerReportPerSerialSplit);
    }

    return 0;
  }

  auto &inputFieldSize = inputFieldSizeOption.val;

  if (inputLoadWidth && !isIpuModel(deviceType)) {
    std::cerr << "Cannot model input load width on non-IPUModel device\n";
    return 1;
  }

  auto dev = [&]() -> TestDevice {
    if (isIpuModel(deviceType)) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      IPUModel ipuModel(deviceTypeToIPUName(deviceType));
      ipuModel.numIPUs = numIPUs;
      if (inputLoadWidth) {
        // dataPathWidth is specified in bits
        ipuModel.dataPathWidth = *inputLoadWidth * 8;
        assert(*inputLoadWidth % 4 == 0);
        ipuModel.fp32ConvUnitInputLoadElemsPerCycle = *inputLoadWidth / 4;
        ipuModel.fp16ConvUnitInputLoadElemsPerCycle = *inputLoadWidth / 2;
        ipuModel.fp8ConvUnitInputLoadElemsPerCycle = *inputLoadWidth / 1;
        // adjust 16.32 units to remove restriction based on store width.
        const auto pipelineDepth = ipuModel.fp16ConvUnitMaxPipelineDepth;
        const auto fp32StoredPerCycle = ipuModel.dataPathWidth / 8 / 4;
        ipuModel.fp16InFp32OutConvUnitsPerTile =
            std::min(ipuModel.fp16InFp16OutConvUnitsPerTile,
                     pipelineDepth * fp32StoredPerCycle);
      }
      if (vm.count("profile") || profileDir) {
        ipuModel.compileIPUCode = true;
      }
      if (vm.count("workers-per-tile"))
        ipuModel.numWorkerContexts = vm["workers-per-tile"].as<unsigned>();
      if (tilesPerIPU)
        ipuModel.tilesPerIPU = *tilesPerIPU;
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      if (tilesPerIPU)
        return createTestDevice(deviceType, numIPUs, *tilesPerIPU);
      else
        return createTestDeviceFullSize(deviceType, numIPUs);
    }
  }();
  Graph graph(dev.getTarget());
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  // Create input tensors from a file if specified.
  Tensor prevAct;
  Tensor zDeltas;

  if (fwdInFile) {
    std::ifstream in(fwdInFile.get());
    if (!in.good()) {
      throw poputil::poplibs_error("<fwd-in-file> file " + fwdInFile.get() +
                                   " could not be opened");
    }
    auto inFileTensors =
        graph.deserializeTensors(in, SerializationFormat::Binary);
    if (inFileTensors.size()) {
      prevAct = inFileTensors[0];
      std::cout << "Importing FWD input from file, shape:" << prevAct.shape()
                << " Element type:" << prevAct.elementType() << "\n";
      inputType = prevAct.elementType();
      batchSize = prevAct.dim(0);
      if (prevAct.dim(1) != fwdInChansPerConvGroup * numConvGroups) {
        std::cerr
            << "The product of conv-groups and input-channels "
               " must be equal to dimension 1 of the FWD input file tensor,"
               " which is "
            << prevAct.dim(1) << "\n";
        return 1;
      }
      inputFieldSize.resize(prevAct.rank() - 2);
      for (unsigned i = 0; i < prevAct.rank() - 2; i++) {
        inputFieldSize[i] = prevAct.dim(2 + i);
      }
    } else {
      std::cerr << "No Tensors in fwd-in-file\n";
      return 1;
    }
  }

  if (bwdInFile) {
    std::ifstream in(bwdInFile.get());
    if (!in.good()) {
      throw poputil::poplibs_error("<bwd-in-file> file " + bwdInFile.get() +
                                   " could not be opened");
    }
    auto inFileTensors =
        graph.deserializeTensors(in, SerializationFormat::Binary);
    if (inFileTensors.size()) {
      zDeltas = inFileTensors[0];
      std::cout << "Importing BWD input from file, shape:" << zDeltas.shape()
                << " Element type:" << zDeltas.elementType() << "\n";
      outputType = zDeltas.elementType();
      batchSize = zDeltas.dim(0);
      if (zDeltas.dim(1) != fwdOutChansPerConvGroup * numConvGroups) {
        std::cerr
            << "The product of conv-groups and output-channels "
               " must be equal to dimension 1 of the BWD input file tensor,"
               " which is"
            << zDeltas.dim(1) << "\n";
        return 1;
      }
    } else {
      std::cerr << "No Tensors in bwd-in-file\n";
      return 1;
    }
  }

  const auto numFieldDims = inputFieldSize.size();

  kernelSizeOption.broadcast(numFieldDims);
  auto &kernelSize = kernelSizeOption.val;

  struct UpperLowerOption {
    ShapeOption<unsigned> &lowerOption;
    ShapeOption<unsigned> &upperOption;
    std::string name;
  } upperLowerOptionTriples[] = {
      {paddingLowerOption, paddingUpperOption, "padding"},
      {truncationLowerOption, truncationUpperOption, "truncation"},
      {kernelTruncationLowerOption, kernelTruncationUpperOption,
       "kernel-truncation"},
      {kernelPaddingLowerOption, kernelPaddingUpperOption, "kernel-padding"},
      {outputTruncationLowerOption, outputTruncationUpperOption,
       "output-truncation"},
      {outputPaddingLowerOption, outputPaddingUpperOption, "output-padding"}};
  for (const auto &entry : upperLowerOptionTriples) {
    if (!vm[entry.name].defaulted()) {
      std::string conflictingOptions[] = {entry.name + "-lower",
                                          entry.name + "-upper"};
      for (auto option : conflictingOptions) {
        if (!vm[option].defaulted()) {
          std::cerr << "--" << entry.name << " as well as --";
          std::cerr << option << " set\n";
          return 1;
        }
      }
      entry.lowerOption = vm[entry.name].as<ShapeOption<unsigned>>();
      entry.upperOption = vm[entry.name].as<ShapeOption<unsigned>>();
    }
    entry.lowerOption.broadcast(numFieldDims);
    entry.upperOption.broadcast(numFieldDims);
  }
  auto &truncationLower = truncationLowerOption.val;
  auto &truncationUpper = truncationUpperOption.val;
  auto &paddingLower = paddingLowerOption.val;
  auto &paddingUpper = paddingUpperOption.val;

  auto &outputTruncationLower = outputTruncationLowerOption.val;
  auto &outputTruncationUpper = outputTruncationUpperOption.val;
  auto &outputPaddingLower = outputPaddingLowerOption.val;
  auto &outputPaddingUpper = outputPaddingUpperOption.val;

  auto &kernelTruncationLower = kernelTruncationLowerOption.val;
  auto &kernelTruncationUpper = kernelTruncationUpperOption.val;
  auto &kernelPaddingLower = kernelPaddingLowerOption.val;
  auto &kernelPaddingUpper = kernelPaddingUpperOption.val;

  inDilationOption.broadcast(numFieldDims);
  auto &inDilation = inDilationOption.val;
  flipInputOption.broadcast(numFieldDims);
  auto &flipInput = flipInputOption.val;

  kernelDilationOption.broadcast(numFieldDims);
  auto &kernelDilation = kernelDilationOption.val;
  flipKernelOption.broadcast(numFieldDims);
  auto &flipKernel = flipKernelOption.val;

  strideOption.broadcast(numFieldDims);
  auto &stride = strideOption.val;
  const auto fwdInChans = fwdInChansPerConvGroup * numConvGroups;
  const auto fwdOutChans = fwdOutChansPerConvGroup * numConvGroups;

  const bool planOnly = vm.count("plan-only");
  const bool inferenceOnly = vm.count("inference-only");
  const bool ignoreData = vm.count("ignore-data");

  bool testingQuarter = inputType == QUARTER;
  bool doFwdPass = pass == Pass::ALL || pass == Pass::FWD;
  bool doBwdPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::BWD);
  auto doWuPass = [=]() {
    if (!inferenceOnly && (pass == Pass::ALL || pass == Pass::WU)) {
      if (testingQuarter) {
        std::cerr << "Weight update pass with quarter data type will implement "
                     " operations on half, so is excluded from this test\n";
        return false;
      }
      return true;
    }
    return false;
  }();

  if ((vm["output-type"].empty() != vm["input-type"].empty()) ||
      (!vm["data-type"].defaulted() && !vm["output-type"].empty())) {
    throw poputil::poplibs_error("Please specify either --data-type OR "
                                 "(--input-type AND --output-type), not both.");
  }
  if (vm["output-type"].empty()) {
    outputType = inputType;
  }

  if (vm["tolerance"].empty()) {
    if (outputType == FLOAT) {
      relativeTolerance = FLOAT_REL_TOL;
    } else {
      relativeTolerance = HALF_REL_TOL;
    }
  }
  if (vm["absolute-tolerance"].empty()) {
    if (outputType == FLOAT) {
      absoluteTolerance = FLOAT_ABS_TOL;
    } else {
      absoluteTolerance = HALF_ABS_TOL;
    }
  }

  if (numDeterminismChecks && (ignoreData || !doWuPass)) {
    throw poputil::poplibs_error(
        "Determinism checks cannot ignore data or avoid weight upload pass.");
  }
  if (numDeterminismChecks && deviceType != DeviceType::Hw) {
    throw poputil::poplibs_error(
        "Determinism checks only work on Hardware device");
  }

  const poplin::ConvParams::InputTransform inputTransform{
      truncationLower, truncationUpper, inDilation,
      paddingLower,    paddingUpper,    flipInput};
  const poplin::ConvParams::InputTransform kernelTransform{
      kernelTruncationLower, kernelTruncationUpper, kernelDilation,
      kernelPaddingLower,    kernelPaddingUpper,    flipKernel,
  };
  const poplin::ConvParams::OutputTransform outputTransform{
      outputTruncationLower, outputTruncationUpper, stride, outputPaddingLower,
      outputPaddingUpper};
  const auto params = poplin::ConvParams{inputType,
                                         outputType,
                                         batchSize,
                                         inputFieldSize,
                                         kernelSize,
                                         fwdInChansPerConvGroup,
                                         fwdOutChansPerConvGroup,
                                         numConvGroups,
                                         inputTransform,
                                         kernelTransform,
                                         outputTransform};

  const auto outFieldSize = params.getOutputFieldShape();
  const auto bwdParams = getGradientParams(params);
  if (bwdInFile) {
    auto shape = zDeltas.shape();
    shape.erase(shape.begin(), shape.begin() + 2);
    if (outFieldSize != shape) {
      std::cerr << "Calculated output field size does not match the dimensions"
                   " of tensor in the BWD pass input file\n";
      std::cerr << "File:" << shape << " Calculated:" << outFieldSize << "\n";
      return 1;
    }
  }

  OptionFlags convOptions;
  convOptions.set(
      {{"remapOutputTensor", remapOutputTensor ? "true" : "false"}});

  if (!convOptionsString.empty()) {
    poplar::readJSON(convOptionsString, convOptions);
  }

  auto fwdOptions = convOptions;
  fwdOptions.set("pass", inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  overloadConstraintsFromFile(fwdPlanConstraintsFile, fwdPlanConstraints);
  fwdOptions.set("planConstraints", fwdPlanConstraints);
  auto bwdOptions = convOptions;
  bwdOptions.set("pass", "TRAINING_BWD");
  overloadConstraintsFromFile(bwdPlanConstraintsFile, bwdPlanConstraints);
  bwdOptions.set("planConstraints", bwdPlanConstraints);
  auto wuOptions = convOptions;
  wuOptions.set("pass", "TRAINING_WU");
  overloadConstraintsFromFile(wuPlanConstraintsFile, wuPlanConstraints);
  wuOptions.set("planConstraints", wuPlanConstraints);

  // Validate all passes options
  poplin::convolutionValidateOptions(fwdOptions);
  poplin::convolutionValidateOptions(bwdOptions);
  poplin::convolutionValidateOptions(wuOptions);

  if (preplan) {
    const auto &replicatedTarget = graph.getTarget();
    std::set<poplin::ConvPlanParams> convs;

    if (doFwdPass) {
      convs.insert(std::make_tuple(&replicatedTarget, params, &fwdOptions));
    }

    if (doBwdPass) {
      convs.insert(std::make_tuple(&replicatedTarget, bwdParams, &bwdOptions));
    }

    if (doWuPass) {
      auto wuParams = getWeightUpdateParams(params);
      convs.insert(
          std::make_tuple(&replicatedTarget, std::move(wuParams), &wuOptions));
    }

    poplin::preplan(convs, {}, cache);
  }

  // Record the plan estimates for each phase being run, if testing the planner.
  std::vector<std::string> phases;
  std::vector<DetailedPlanCosts> all_estimates;
  phases.reserve(pass == Pass::ALL ? 3 : 1);
  all_estimates.reserve(pass == Pass::ALL ? 3 : 1);
  if (vm.count("test-planner")) {
    if (doFwdPass) {
      phases.push_back("fwd");
      all_estimates.push_back(
          reportDetailedPlanEstimatedCosts(graph, params, fwdOptions, &cache));
    }
    if (doBwdPass) {
      phases.push_back("bwd");
      all_estimates.push_back(reportDetailedPlanEstimatedCosts(
          graph, bwdParams, bwdOptions, &cache));
    }
    if (doWuPass) {
      phases.push_back("wu");
      auto wuParams = getWeightUpdateParams(params);
      all_estimates.push_back(
          reportDetailedPlanEstimatedCosts(graph, wuParams, wuOptions, &cache));
    }

    for (size_t i = 0; i < phases.size(); ++i) {
      const auto &phase = phases[i];
      auto &estimates = all_estimates[i];

      // Dump the estimates to a file for the existing profile option.
      if (profileDir) {
        std::ofstream out(*profileDir + "/" + phase + "-plan-estimates.txt");
        out << estimates;
      }

      // Canonicalise the estimates based on the options.
      updateDetailedPlanCosts(testPlannerReportPerTile,
                              testPlannerReportPerSerialSplit, estimates);
    }
  }

  if (reportPlan || planOnly) {
    std::cout << "Convolution parameters:\n"
                 " Batch size: "
              << params.batchSize
              << "\n"
                 " Kernel:"
              << params.kernelShape
              << "\n"
                 " Stride:"
              << params.outputTransform.stride
              << "\n"
                 " Padding Lower: "
              << params.inputTransform.paddingLower
              << "\n"
                 " Padding Upper: "
              << params.inputTransform.paddingUpper
              << "\n"
                 " Group size: "
              << params.numConvGroups
              << "\n"
                 " Input: "
              << params.inputChannelsPerConvGroup << "x"
              << params.inputFieldShape
              << "\n"
                 " Output: "
              << params.outputChannelsPerConvGroup << "x" << outFieldSize
              << "\n";

    if (doFwdPass) {
      std::cout << "Forward plan:\n";
      poplin::reportPlanInfo(std::cout, graph, params, fwdOptions, &cache);
      std::cout << "Forward FLOPs: " << getFwdFlops(params) << "\n";
    }

    if (doBwdPass) {
      std::cout << "Backward plan:\n";
      poplin::reportPlanInfo(std::cout, graph, bwdParams, bwdOptions, &cache);
      std::cout << "Backward FLOPs: " << getBwdFlops(bwdParams) << "\n";
    }

    if (doWuPass) {
      std::cout << "WU plan:\n";
      poplin::reportWeightUpdatePlanInfo(std::cout, graph, params, wuOptions,
                                         &cache);
      std::cout << "WU FLOPs: " << getWuFlops(params) << "\n";
    }

    if (planOnly) {
      return 0;
    }
  }

  const auto &target = graph.getTarget();
  std::size_t maxAccsPerOutputElement = 0;
  if (doFwdPass) {
    const std::size_t numOutElems = product(params.getOutputFieldShape()) *
                                    params.getNumOutputChans() *
                                    params.getBatchSize();
    if (numOutElems) {
      const std::size_t fwdMaxAccsPerOutElem =
          (poplin::getFwdFlops(params) / 2) / numOutElems;
      maxAccsPerOutputElement =
          std::max(maxAccsPerOutputElement, fwdMaxAccsPerOutElem);
    }
  }
  if (doBwdPass) {
    const std::size_t numOutElems = product(params.getInputFieldShape()) *
                                    params.getNumInputChans() *
                                    params.getBatchSize();
    if (numOutElems) {
      const std::size_t bwdMaxAccsPerOutElem =
          (poplin::getBwdFlops(params) / 2) / numOutElems;
      maxAccsPerOutputElement =
          std::max(maxAccsPerOutputElement, bwdMaxAccsPerOutElem);
    }
  }
  if (doWuPass) {
    const std::size_t numOutElems = product(params.getKernelShape()) *
                                    params.getNumInputChans() *
                                    params.getNumOutputChans();
    if (numOutElems) {
      const std::size_t wuMaxAccsPerOutElem =
          (poplin::getWuFlops(params) / 2) / numOutElems;
      maxAccsPerOutputElement =
          std::max(maxAccsPerOutputElement, wuMaxAccsPerOutElem);
    }
  }

  // To avoid destructive addition (a + b) + c != a + (b + c), which is
  // particularly poor with halves, we look to only using values which we can
  // represent exactly, such that (a + b) + c == a + (b + c).
  // Accumulating random binary distribution of {-1, 1}, with a mean of 0
  // provides us this most of the time. Such that accumulating many items from
  // this distribution is unlikely to be not exactly representable as it should
  // generally be a small number. It's less likely the larger the convolution
  // however.

  // We also disable this for determinism checks, which is testing stochastic
  // rounding specifically.
  const bool useUniformRandomData =
      numDeterminismChecks ||
      isLikelyToHaveNumericalErrorsUsingBernoulli(maxAccsPerOutputElement,
                                                  inputType, outputType);

  if (useUniformRandomData) {
    std::cout << "Using uniform random data\n";
  } else {
    std::cout << "Using random binary {-1,1} data with no error tolerance\n";
    if (vm["tolerance"].empty()) {
      relativeTolerance = 0;
    }
    if (vm["absolute-tolerance"].empty()) {
      absoluteTolerance = 0;
    }
  }
  // Always generate the fwd program as it maps the weights and biases. Only
  // actually create the engine if the fwd pass is to be run
  auto fwdProg = Sequence();
  auto revProg = Sequence();

  // Create tensors if not loaded from a file
  if (!fwdInFile) {
    if (useCreateInput) {
      prevAct =
          poplin::createInput(graph, params, "prevAct", fwdOptions, &cache);
    } else {
      prevAct = createGenericConvInput(graph, params, "prevAct");
    }
  }
  Tensor weights =
      poplin::createWeights(graph, params, "weights", fwdOptions, &cache);

  if (!bwdInFile && (doBwdPass || doWuPass)) {
    if (useCreateInput) {
      zDeltas =
          poplin::createInput(graph, bwdParams, "zDeltas", bwdOptions, &cache);
    } else {
      zDeltas = createGenericConvInput(graph, bwdParams, "zDeltas");
    }
  }

  // create the forward convolution as a tensor function as we may be able to
  // reuse it for the backwards pass.
  auto fwdConv = [&]() -> graphfn::TensorFunction {
    using graphfn::input;
    const auto conv = [&](std::vector<Tensor> &args, Sequence &prog) {
      return convolve(useCreateOutput, graph, args[0], args[1], params, false,
                      prog, "fwd", fwdOptions, &cache);
    };
    return {graph, {input(prevAct, "in"), input(weights, "weights")}, conv};
  }();

  std::vector<Tensor> fwdArgs{prevAct, weights};
  Tensor nextAct = fwdConv(fwdArgs, fwdProg);

  Tensor biases;
  if (bias) {
    biases = poplin::createBiases(graph, nextAct, params, "biases", fwdOptions,
                                  &cache);
    poplin::addBias(graph, nextAct, biases, fwdProg, "bias");
  }
  if (!doFwdPass) {
    fwdProg = Sequence();
  }

  const auto learningRate = [&]() {
    if (useUniformRandomData) {
      return 0.05;
    } else {
      return 1.0;
    }
  }();

  Tensor prevDeltas;
  if (doBwdPass) {
    // we may be able to reuse the forward pass convolution if the convolution
    // is symmetrical.
    if (enableConvolutionReuse &&
        params.canonicalize() == bwdParams.canonicalize()) {
      // transform the weights prior to the convolution so we can reuse the
      // existing sub-graph.
      auto bwdWeights = poplin::createWeights(graph, bwdParams, "bwdWeights",
                                              fwdOptions, &cache);
      poplin::weightsTransposeChansFlipXY(graph, weights, bwdWeights, revProg,
                                          "bwd");

      std::vector<Tensor> bwdArgs{zDeltas, bwdWeights};
      prevDeltas = fwdConv(bwdArgs, revProg);
    } else {
      prevDeltas = convolve(useCreateOutput, graph, zDeltas, weights, bwdParams,
                            true, revProg, "bwd", bwdOptions, &cache);
    }
  }
  if (doWuPass) {
    auto scale = graph.addConstant(weights.elementType(), {}, -learningRate);
    graph.setTileMapping(scale, 0);
    poplin::convolutionWeightUpdate(graph, zDeltas, weights, prevAct, params,
                                    scale, revProg, "wu", wuOptions, &cache);
    if (bias) {
      auto scale = graph.addConstant(FLOAT, {}, -learningRate);
      graph.setTileMapping(scale, 0);
      poplin::convolutionBiasUpdate(graph, zDeltas, biases, scale, convOptions,
                                    revProg);
    }
  }
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  auto rawHostPrevAct = allocateHostMemoryForTensor(
      prevAct, "prevAct", graph, uploadProg, downloadProg, tmap);
  auto rawHostWeights = allocateHostMemoryForTensor(
      weights, "weights", graph, uploadProg, downloadProg, tmap);

  std::unique_ptr<char[]> rawPrevActsMetadata, rawWeightsMetadata;
  if (testingQuarter) {
    rawPrevActsMetadata =
        allocateHostMemoryForTensor(prevAct.getMetadata(), "prevActMetadata",
                                    graph, boost::none, downloadProg, tmap);
    rawWeightsMetadata =
        allocateHostMemoryForTensor(weights.getMetadata(), "weightsMetadata",
                                    graph, boost::none, downloadProg, tmap);
  }
  Tensor parentBiases;
  std::unique_ptr<char[]> rawHostBiases;
  if (bias) {
    rawHostBiases = allocateHostMemoryForTensor(biases, "biases", graph,
                                                uploadProg, downloadProg, tmap);
  }
  auto rawHostNextAct = allocateHostMemoryForTensor(
      nextAct, "nextAct", graph, uploadProg, downloadProg, tmap);
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  std::unique_ptr<char[]> rawHostZDeltasMetadata;
  if (doBwdPass || doWuPass) {
    rawHostZDeltas = allocateHostMemoryForTensor(
        zDeltas, "zDeltas", graph, uploadProg, downloadProg, tmap);
  }
  if (doBwdPass) {
    rawHostPrevDeltas = allocateHostMemoryForTensor(
        prevDeltas, "prevDeltas", graph, uploadProg, downloadProg, tmap);
    if (testingQuarter) {
      rawHostZDeltasMetadata =
          allocateHostMemoryForTensor(zDeltas.getMetadata(), "zDeltasMetadata",
                                      graph, boost::none, downloadProg, tmap);
    }
  }
  std::vector<Program> programs;
  const auto fwdProgIndex = programs.size(); // 0
  programs.push_back(std::move(fwdProg));
  const auto revProgIndex = programs.size(); // 1
  programs.push_back(std::move(revProg));
  const auto uploadProgIndex = programs.size(); // 2
  programs.push_back(std::move(uploadProg));
  const auto downloadProgIndex = programs.size(); // 3
  programs.push_back(std::move(downloadProg));

  std::optional<TempDir> tempDir;
  OptionFlags engineOptions = defaultEngineOptions;
  if (vm.count("test-planner") || vm.count("profile") || profileDir) {
    if (vm.count("test-planner")) {
      engineOptions.set("autoReport.all", "true");
    } else /* profile or profileDir */ {
      engineOptions.set("autoReport.outputExecutionProfile", "true");
    }
    if (profileDir) {
      engineOptions.set("autoReport.directory", *profileDir);
    } else {
      tempDir.emplace(TempDir::create());
      engineOptions.set("autoReport.directory", tempDir->getPath());
    }
  }

  int rc = EXIT_SUCCESS;
  std::stringstream errs;

  // Put the engine in a new scope so that we can force the profile to be
  // written on scope exit.
  {
    Engine engine(graph, std::move(programs), engineOptions);

    if (vm.count("compile-only"))
      return 0;

    attachStreams(engine, tmap);
    boost::multi_array<double, 3> hostPrevAct(
        boost::extents[batchSize][fwdInChans][product(inputFieldSize)]);
    boost::multi_array<double, 4> hostWeights(
        boost::extents[numConvGroups][fwdOutChansPerConvGroup]
                      [fwdInChansPerConvGroup][product(kernelSize)]);
    boost::multi_array<double, 1> hostBiases(boost::extents[fwdOutChans]);
    boost::multi_array<double, 3> hostNextAct(
        boost::extents[batchSize][fwdOutChans][product(outFieldSize)]);
    std::mt19937 randomEngine;
    if (useUniformRandomData) {
      writeRandomValues(target, inputType, hostPrevAct, -2.0, 2.0,
                        randomEngine);
      writeRandomValues(target, inputType, hostWeights, -1.0, +1.0,
                        randomEngine);
    } else {
      writeRandomBinaryValues(target, inputType, hostPrevAct, -1.0, 1.0,
                              randomEngine);
      writeRandomBinaryValues(target, inputType, hostWeights, -1.0, 1.0,
                              randomEngine);
    }
    if (bias) {
      if (useUniformRandomData) {
        writeRandomValues(target, outputType, hostBiases, -2.0, +6.0,
                          randomEngine);
      } else {
        writeRandomBinaryValues(target, outputType, hostBiases, -1.0, 1.0,
                                randomEngine);
      }
    } else {
      std::fill(hostBiases.data(),
                hostBiases.data() + hostBiases.num_elements(), 0.0);
    }
    if (testingQuarter) {
      copy(target, hostPrevAct, inputType,
           QuarterMetadata(fp8FormatFwdIn, fp8ScaleFwdIn),
           rawHostPrevAct.get());
    } else {
      copy(target, hostPrevAct, inputType, rawHostPrevAct.get());
    }

    boost::multi_array<double, 4> duplicatedHostWeights(
        boost::extents[numConvGroups][fwdOutChansPerConvGroup]
                      [fwdInChansPerConvGroup][product(kernelSize)]);
    boost::multi_array<double, 1> duplicatedHostBiases(
        boost::extents[fwdOutChans]);

    // Used for determinism checking
    boost::multi_array<double, 4> prevExecutionWeights(
        boost::extents[numConvGroups][fwdOutChansPerConvGroup]
                      [fwdInChansPerConvGroup][product(kernelSize)]);
    boost::multi_array<double, 1> prevExecutionBiases(
        boost::extents[fwdOutChans]);

    boost::multi_array<double, 3> hostZDeltas(
        boost::extents[batchSize][bwdParams.getNumInputChans()]
                      [product(outFieldSize)]);
    if (useUniformRandomData) {
      writeRandomValues(target, inputType, hostZDeltas, -3.0, 3.0,
                        randomEngine);
    } else {
      writeRandomBinaryValues(target, inputType, hostZDeltas, -1.0, 1.0,
                              randomEngine);
    }

    const auto fwdModel = [&](const auto &hostPrevAct, const auto &hostWeights,
                              const auto &hostBiases) {
      boost::multi_array<double, 3> modelNextAct(
          boost::extents[batchSize][fwdOutChans][product(outFieldSize)]);
      poplibs_test::conv::convolution(
          vectorConvert<unsigned>(inputFieldSize), truncationLower,
          truncationUpper, inDilation, paddingLower, paddingUpper, flipInput,
          vectorConvert<unsigned>(kernelSize), kernelTruncationLower,
          kernelTruncationUpper, kernelDilation, kernelPaddingLower,
          kernelPaddingUpper, flipKernel, outputTruncationLower,
          outputTruncationUpper, stride, outputPaddingLower, outputPaddingUpper,
          hostPrevAct, hostWeights, hostBiases, modelNextAct);
      return modelNextAct;
    };

    const auto bwdModel = [&](const auto &hostZDeltas,
                              const auto &modelWeights) {
      boost::multi_array<double, 3> modelPrevDeltas(
          boost::extents[batchSize][fwdInChans][product(inputFieldSize)]);
      poplibs_test::conv::convolutionBackward(
          vectorConvert<unsigned>(inputFieldSize), truncationLower,
          truncationUpper, inDilation, paddingLower, paddingUpper, flipInput,
          vectorConvert<unsigned>(kernelSize), kernelTruncationLower,
          kernelTruncationUpper, kernelDilation, kernelPaddingLower,
          kernelPaddingUpper, flipKernel, outputTruncationLower,
          outputTruncationUpper, stride, outputPaddingLower, outputPaddingUpper,
          hostZDeltas, modelWeights, modelPrevDeltas);
      return modelPrevDeltas;
    };

    const auto wuModel = [&](const auto &hostPrevAct, const auto &hostZDeltas,
                             const auto &hostWeights, const auto &hostBiases) {
      auto modelWeights = hostWeights;
      auto modelBiases = hostBiases;
      poplibs_test::conv::weightUpdate(
          vectorConvert<unsigned>(inputFieldSize), truncationLower,
          truncationUpper, inDilation, paddingLower, paddingUpper, flipInput,
          vectorConvert<unsigned>(kernelSize), kernelTruncationLower,
          kernelTruncationUpper, kernelDilation, kernelPaddingLower,
          kernelPaddingUpper, flipKernel, outputTruncationLower,
          outputTruncationUpper, stride, outputPaddingLower, outputPaddingUpper,
          learningRate, hostPrevAct, hostZDeltas, modelWeights, modelBiases);
      return std::make_pair(modelWeights, modelBiases);
    };

    enum class DataValidation { AgainstModel, AgainstPreviousRuns };
    const auto validationMethod = [&]() -> boost::optional<DataValidation> {
      if (ignoreData) {
        return boost::none;
      } else {
        if (numDeterminismChecks > 0) {
          return DataValidation::AgainstPreviousRuns;
        } else {
          return DataValidation::AgainstModel;
        }
      }
    }();

    for (unsigned determinismCheckIdx = 0;
         determinismCheckIdx < numDeterminismChecks + 1;
         ++determinismCheckIdx) {

      duplicatedHostWeights = hostWeights;
      if (bias) {
        duplicatedHostBiases = hostBiases;
      }
      if (testingQuarter) {
        copy(target, duplicatedHostWeights, inputType,
             QuarterMetadata(fp8FormatWeights, fp8ScaleWeights),
             rawHostWeights.get());
      } else {
        copy(target, duplicatedHostWeights, inputType, rawHostWeights.get());
      }
      if (bias) {
        copy(target, duplicatedHostBiases, outputType, rawHostBiases.get());
      }

      if (doBwdPass || doWuPass) {
        if (testingQuarter) {
          copy(target, hostZDeltas, inputType,
               QuarterMetadata(fp8FormatBwdIn, fp8scaleBwdIn),
               rawHostZDeltas.get());
        } else {
          copy(target, hostZDeltas, inputType, rawHostZDeltas.get());
        }
      }

      dev.bind([&](const Device &d) {
        engine.load(d);
        if (validationMethod) {
          engine.run(uploadProgIndex);
        }
        // Run the forward pass.
        engine.run(fwdProgIndex);
        if (doBwdPass || doWuPass) {
          // Run the backwards and/or weight update passes.
          engine.run(revProgIndex);
        }
        if (validationMethod) {
          engine.run(downloadProgIndex);
        }
      });

      bool fwdFailed = false;
      bool bwdFailed = false;
      bool weightsFailed = false;
      bool biasesFailed = false;

      auto checkMetadata = [&](const std::unique_ptr<char[]> &src,
                               QuarterMetadata::Format format, int scale,
                               const std::string &message) {
        boost::multi_array<double, 1> hostMetadata(boost::extents[1]);
        copy(target, UNSIGNED_CHAR, src.get(), hostMetadata);
        auto expectedMetadata = QuarterMetadata(format, scale).getBinary();
        if (static_cast<unsigned>(hostMetadata[0]) != expectedMetadata) {
          std::cerr << message << " metadata incorrect: " << hostMetadata[0]
                    << " expected " << expectedMetadata << "\n";
          return true;
        }
        return false;
      };

      if (doFwdPass) {
        copy(target, outputType, rawHostNextAct.get(), hostNextAct);
        if (validationMethod == DataValidation::AgainstModel) {
          fwdFailed =
              !checkIsClose("fwd", hostNextAct,
                            fwdModel(hostPrevAct, hostWeights, hostBiases),
                            relativeTolerance, absoluteTolerance);
        }
        if (testingQuarter) {
          checkMetadata(rawPrevActsMetadata, fp8FormatFwdIn, fp8ScaleFwdIn,
                        "fwdActs");
          checkMetadata(rawWeightsMetadata, fp8FormatWeights, fp8ScaleWeights,
                        "weights");
          // Note: No metadata check for output as it is of type half
        }
        if (fwdOutFile) {
          std::ofstream out(fwdOutFile.get());
          std::cout << "Saving FWD tensor with shape:" << nextAct.shape()
                    << " Element type:" << nextAct.elementType() << "\n";
          graph.serializeTensors(out, {nextAct}, SerializationFormat::Binary);
        }
      }

      if (doBwdPass || doWuPass) {
        boost::multi_array<double, 3> hostPrevDeltas(
            boost::extents[batchSize][params.getNumInputChans()]
                          [product(inputFieldSize)]);

        if (doBwdPass) {
          copy(target, outputType, rawHostPrevDeltas.get(), hostPrevDeltas);
          if (validationMethod == DataValidation::AgainstModel) {
            bwdFailed = !checkIsClose("bwd", hostPrevDeltas,
                                      bwdModel(hostZDeltas, hostWeights),
                                      relativeTolerance, absoluteTolerance);
          }
          if (testingQuarter) {
            checkMetadata(rawHostZDeltasMetadata, fp8FormatBwdIn, fp8scaleBwdIn,
                          "zDeltas");
            checkMetadata(rawWeightsMetadata, fp8FormatWeights, fp8ScaleWeights,
                          "weights");
          }
          if (bwdOutFile) {
            std::ofstream out(bwdOutFile.get());
            std::cout << "Saving BWD tensor with shape:" << prevDeltas.shape()
                      << " Element type:" << prevDeltas.elementType() << "\n";
            graph.serializeTensors(out, {prevDeltas},
                                   SerializationFormat::Binary);
          }
        }
        if (doWuPass) {
          copy(target, inputType, rawHostWeights.get(), duplicatedHostWeights);
          if (bias) {
            copy(target, outputType, rawHostBiases.get(), duplicatedHostBiases);
          }

          if (validationMethod == DataValidation::AgainstModel) {
            // Take dimensions and shape from host tensors.
            auto modelWeights = hostWeights;
            auto modelBiases = hostBiases;
            std::tie(modelWeights, modelBiases) =
                wuModel(hostPrevAct, hostZDeltas, hostWeights, hostBiases);

            boost::multi_array<double, 4> hostWeights = duplicatedHostWeights;
            auto failed = !checkIsClose("weights", hostWeights, modelWeights,
                                        relativeTolerance, absoluteTolerance);
            if (failed) {
              weightsFailed = true;
            }

            if (bias) {
              boost::multi_array<double, 1> hostBiases = duplicatedHostBiases;
              failed = !checkIsClose("biases", hostBiases, modelBiases,
                                     relativeTolerance, absoluteTolerance);
              if (failed) {
                biasesFailed = true;
              }
            }
          }

          if (validationMethod == DataValidation::AgainstPreviousRuns) {
            prevExecutionWeights = duplicatedHostWeights;
            if (bias) {
              prevExecutionBiases = duplicatedHostBiases;
            }

            if (determinismCheckIdx > 0) {
              if (duplicatedHostWeights != prevExecutionWeights) {
                weightsFailed = true;
              }
              if (bias) {
                if (duplicatedHostBiases != prevExecutionBiases) {
                  biasesFailed = true;
                }
              }
            }
          }
        }
      }

      if (validationMethod) {
        const std::vector<std::pair<std::string, int>> results{
            {"fwd", fwdFailed},
            {"bwd", bwdFailed},
            {"weights", weightsFailed},
            {"biases", biasesFailed},
        };
        for (const auto &result : results) { // Report all failures
          if (result.second) {
            errs << result.first << " validation failed\n";
          }
        }
        for (const auto &result : results) { // Abort if any failed
          if (result.second) {
            rc = EXIT_FAILURE;
          }
        }
      }

    } // for num_determinism_checks

    if (vm.count("profile") && !vm.count("test-planner")) {
      auto reportOptions = OptionFlags{{"showExecutionSteps", "true"}};
      if (reportVarStorage) {
        reportOptions.set("showVarStorage", "true");
      }
      engine.printProfileSummary(std::cout, reportOptions);
    }
  }

  if (!all_estimates.empty()) {
    // Save the constraints file(s) (if any) into the auto report
    // directory for posterity. Don't bother saving it if the profile
    // directory is a temporary directory.
    if (!tempDir) {
      using namespace boost::filesystem;
      if (!fwdPlanConstraintsFile.empty())
        copy_file(fwdPlanConstraintsFile, *profileDir + "/fwd-constraints.json",
                  copy_option::overwrite_if_exists);
      if (!bwdPlanConstraintsFile.empty())
        copy_file(bwdPlanConstraintsFile, *profileDir + "/bwd-constraints.json",
                  copy_option::overwrite_if_exists);
      if (!wuPlanConstraintsFile.empty())
        copy_file(wuPlanConstraintsFile, *profileDir + "/wu-constraints.json",
                  copy_option::overwrite_if_exists);
    }
    // Loop through all the estimates, grab the corresponding measurements
    // and dump a textual report to the command line.
    for (size_t i = 0; i < phases.size(); ++i) {
      const auto &phase = phases[i];
      const auto &estimates = all_estimates[i];
      // Tally up the cycles and memory used according to the profile.
      std::string actualProfileDir = tempDir ? tempDir->getPath() : *profileDir;
      // Read the profile back and compare the costs.
      MeasuredPlanCosts actual = getActualCosts(
          phase, actualProfileDir, estimates.parallelSplit,
          estimates.serialSplit, testPlannerPrintVars, testPlannerReportPerTile,
          testPlannerReportPerSerialSplit);
      compareCosts(phase, estimates, actual, testPlannerReportVerbose,
                   testPlannerReportPerTile, testPlannerReportPerSerialSplit);
    }
  }

  std::cerr << errs.str();
  return rc;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
