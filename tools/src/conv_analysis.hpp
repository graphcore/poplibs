// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poplin_tools_src_conv_analysis_hpp
#define poplin_tools_src_conv_analysis_hpp

#include "conv_analysis.hpp"

#include <poplin/Convolution.hpp>

#include <poputil/exceptions.hpp>

#include <poplar/StringRef.hpp>

#include <pva/pva.hpp>

#include <spdlog/fmt/fmt.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <istream>
#include <ostream>
#include <regex>
#include <string>
#include <vector>

// Analysis specific to the ConvPartialnx1 vertices.
namespace amp {

void updateDetailedPlanCosts(bool reportPerTile, bool reportPerSerialSplit,
                             poplin::internal::DetailedPlanCosts &costs);

// Like DetailedPlanCosts but with a couple of helper methods.
struct MeasuredPlanCosts : public poplin::internal::DetailedPlanCosts {
  poplin::PlanCosts unknown = {};
  size_t totalWorkListBytes = 0;

  size_t totalCycles() const noexcept;
  size_t totalMemory() const noexcept;

  template <typename Function>
  void apply(Function fn, bool includeTotal = true) {
    poplin::internal::DetailedPlanCosts::apply(fn, includeTotal);
    fn(unknown);
  }
  template <typename Function>
  void apply(Function fn, bool includeTotal = true) const {
    poplin::internal::DetailedPlanCosts::apply(fn, includeTotal);
    fn(unknown);
  }

  poplin::PlanCosts toPlanCosts() const noexcept;
  poplin::PlanCosts &selectCosts(const std::string &name,
                                 bool isInsideSerialSplit) noexcept;
};

struct MemoryCostProgramVisitor : public pva::ProgramVisitor {
  MemoryCostProgramVisitor(MeasuredPlanCosts &planCosts,
                           bool reportPerTile_ = false, size_t activeTiles_ = 1,
                           bool isInsideSerialSplit_ = false,
                           bool printVars_ = false);

  void visitDoExchange(const pva::DoExchangeProgram &doExchange) override;

  void
  visitOnTileExecute(const pva::OnTileExecuteProgram &onTileExecute) override;

private:
  MeasuredPlanCosts &costs;
  bool reportPerTile;
  size_t activeTiles = 1;
  bool isInsideSerialSplit = false;
  bool printVars;
};

MeasuredPlanCosts getMeasuredCosts(const std::string &pass,
                                   const std::string &profileDir,
                                   size_t parallelSplit, size_t serialSplit,
                                   bool printVars = false,
                                   bool reportPerTile = false,
                                   bool reportPerSerialSplit = false);

void compareCosts(const std::string &title,
                  poplin::internal::DetailedPlanCosts const &estimates,
                  MeasuredPlanCosts const &actual, bool reportVerbose = false,
                  bool reportPerTile = false,
                  bool reportPerSerialSplit = false);

void getSimulatedCosts(const std::string &simulatorTraceFilePath,
                       unsigned numTiles = 1, bool keepFullLabelNames = false,
                       bool reportPerTile = true,
                       bool reportPerSerialSplit = false);

} // namespace amp

#endif // poplin_tools_src_conv_analysis_hpp
