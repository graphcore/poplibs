// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_FillModelling_hpp
#define popops_FillModelling_hpp

#include <popsolver/Model.hpp>

// Forward declarations
namespace poplar {
class Target;
class Type;
} // namespace poplar

namespace popops {
namespace modelling {

struct FillEstimates {
  FillEstimates(const popsolver::Variable &init) : cycles(init) {}
  popsolver::Variable cycles;
};

FillEstimates modelContiguousFill(const poplar::Target &target,
                                  const poplar::Type &type, popsolver::Model &m,
                                  const popsolver::Variable &numElems,
                                  const std::string &debugPrefix = "");

} // end namespace modelling
} // end namespace popops

#endif // popops_FillModelling_hpp
