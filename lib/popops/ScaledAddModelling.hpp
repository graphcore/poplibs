// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_ScaledAddModelling_hpp
#define popops_ScaledAddModelling_hpp

#include <gccs/popsolver/Model.hpp>

// Forward declarations
namespace poplar {
class Target;
class Type;
} // namespace poplar

namespace popops {
namespace modelling {

struct ScaledAddEstimates {
  ScaledAddEstimates(const gccs::popsolver::Variable &init) : cycles(init) {}
  gccs::popsolver::Variable cycles;
};

ScaledAddEstimates modelContiguousScaledAdd(
    const poplar::Target &target, const poplar::Type &dataType,
    const poplar::Type &dataBType, const bool isMemConstrained,
    gccs::popsolver::Model &m, const gccs::popsolver::Variable &mNumElems,
    const std::string &debugPrefix);

} // end namespace modelling
} // end namespace popops

#endif // popops_ScaledAddModelling_hpp
