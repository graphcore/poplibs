// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplin_PlanningObjective_hpp
#define poplin_PlanningObjective_hpp

#include "ConvPlanTypes.hpp"
#include <poplibs_support/Compiler.hpp>
#include <popsolver/Model.hpp>

namespace poplin {

class PlanningObjective {
public:
  enum Type {
    MINIMIZE_CYCLES,
    MINIMIZE_COST_DIFF,
    MINIMIZE_TILE_TEMP_MEMORY,
    MINIMIZE_TILES
  };

private:
  Type type;
  popsolver::DataType cyclesBound = popsolver::DataType::max();
  popsolver::DataType tileTempMemoryBound = popsolver::DataType::max();

  // when minimising for cost difference you have the option to either minimise
  // for temp memory or tiles once a plan that fits has been found.
  bool minimizeForTiles;

  PlanningObjective(Type type, bool minimizeForTiles)
      : type(type), minimizeForTiles(minimizeForTiles) {}

public:
  PlanningObjective() {}
  static PlanningObjective minimizeCycles() {
    return PlanningObjective(MINIMIZE_CYCLES, false);
  }
  static PlanningObjective minimizeCostDiff(const bool minimizeForTiles) {
    return PlanningObjective(MINIMIZE_COST_DIFF, minimizeForTiles);
  }
  static PlanningObjective minimizeTileTempMemory() {
    return PlanningObjective(MINIMIZE_TILE_TEMP_MEMORY, false);
  }
  static PlanningObjective minimizeTiles() {
    return PlanningObjective(MINIMIZE_TILES, false);
  }

  friend std::ostream &operator<<(std::ostream &os, const PlanningObjective &);

  PlanningObjective &setCyclesBound(popsolver::DataType bound) {
    assert(type != MINIMIZE_CYCLES);
    assert(*bound > 0);
    cyclesBound = bound;
    return *this;
  }
  PlanningObjective &setTileTempMemoryBound(popsolver::DataType bound) {
    assert(type != MINIMIZE_TILE_TEMP_MEMORY);
    assert(*bound > 0);
    tileTempMemoryBound = bound;
    return *this;
  }

  popsolver::DataType getCyclesBound() const { return cyclesBound; }
  popsolver::DataType getTileTempMemoryBound() const {
    return tileTempMemoryBound;
  }
  bool getMinimizeForTiles() const { return minimizeForTiles; }

  Type getType() const { return type; }

  // this function should mirror the variables we pass into `s.minimize`.
  bool lowerCost(Cost a, Cost b) const {
    switch (type) {
    case MINIMIZE_CYCLES:
      return std::tie(a.totalCycles, a.totalTempBytes) <
             std::tie(b.totalCycles, b.totalTempBytes);
    case MINIMIZE_COST_DIFF: {
      const auto aSecondary =
          minimizeForTiles ? a.totalTiles : a.totalTempBytes;
      const auto bSecondary =
          minimizeForTiles ? b.totalTiles : b.totalTempBytes;
      return std::tie(a.totalPerStepCycleDiff, aSecondary) <
             std::tie(b.totalPerStepCycleDiff, bSecondary);
    }
    case MINIMIZE_TILE_TEMP_MEMORY:
      return std::tie(a.totalTempBytes, a.totalCycles) <
             std::tie(b.totalTempBytes, b.totalCycles);
    case MINIMIZE_TILES:
      return std::tie(a.totalTiles, a.totalCycles) <
             std::tie(b.totalTiles, b.totalCycles);
    }
    POPLIB_UNREACHABLE();
  }
};

inline std::ostream &operator<<(std::ostream &os, const PlanningObjective &po) {
  switch (po.type) {
  case PlanningObjective::MINIMIZE_CYCLES: {
    os << "{ minimise cycles";
    break;
  }
  case PlanningObjective::MINIMIZE_COST_DIFF: {
    os << "{ minimise cost diff";
    if (po.minimizeForTiles) {
      os << " - tiles";
    } else { // temp memory
      os << " - temp memory";
    }
    break;
  }
  case PlanningObjective::MINIMIZE_TILE_TEMP_MEMORY: {
    os << "{ minimise tile temp memory";
    break;
  }
  case PlanningObjective::MINIMIZE_TILES: {
    os << "{ minimise tiles";
    break;
  }
  }
  const auto hasCycleBound = po.cyclesBound != popsolver::DataType::max();
  const auto hasTileTempMemoryBound =
      po.tileTempMemoryBound != popsolver::DataType::max();
  const auto hasBoundSet = hasCycleBound || hasTileTempMemoryBound;
  if (hasBoundSet) {
    os << " : ";
    if (hasCycleBound) {
      os << "cycle bound = " << po.cyclesBound;
    }
    if (hasCycleBound && hasTileTempMemoryBound) {
      os << ", ";
    }
    if (hasTileTempMemoryBound) {
      os << "tile temp memory bound = " << po.tileTempMemoryBound << "B";
    }
  }
  os << " }";
  return os;
}

} // namespace poplin

#endif // poplin_PlanningObjective_hpp
