// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_FillModelling_hpp
#define popops_FillModelling_hpp

#include <gccs/popsolver/Model.hpp>

// Forward declarations
namespace poplar {
class Target;
class Type;
} // namespace poplar

namespace popops {
namespace modelling {

struct FillEstimates {
  FillEstimates(const gccs::popsolver::Variable &init) : cycles(init) {}
  gccs::popsolver::Variable cycles;
};

FillEstimates modelContiguousFill(const poplar::Target &target,
                                  const poplar::Type &type,
                                  gccs::popsolver::Model &m,
                                  const gccs::popsolver::Variable &numElems,
                                  const std::string &debugPrefix = "");

} // end namespace modelling
} // end namespace popops

#endif // popops_FillModelling_hpp
