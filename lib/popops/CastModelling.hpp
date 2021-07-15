// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_CastModelling_hpp
#define popops_CastModelling_hpp

#include <popsolver/Model.hpp>

// Forward declarations
namespace poplar {
class Target;
class Type;
} // namespace poplar

namespace popops {
namespace modelling {

struct CastEstimates {
  CastEstimates(const popsolver::Variable &init) : cycles(init) {}
  popsolver::Variable cycles;
};

CastEstimates modelContiguousCast(const poplar::Target &target,
                                  const poplar::Type &inType,
                                  const poplar::Type &outType,
                                  popsolver::Model &m,
                                  const popsolver::Variable &mNumElems,
                                  const std::string &debugPrefix = "");

} // end namespace modelling
} // end namespace popops

#endif // popops_CastModelling_hpp
