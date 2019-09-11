#ifndef popops_reduction_ReductionVertex_hpp_
#define popops_reduction_ReductionVertex_hpp_
#include <popops/Reduce.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poplibs_support/Compiler.hpp>
#include "ReductionConnection.hpp"

namespace popops {

std::string inline getReductionVertexOpName(popops::Operation op) {
  switch (op) {
  case popops::Operation::ADD: return "ReduceAdd";
  case popops::Operation::SQUARE_ADD: return "ReduceSquareAdd";
  case popops::Operation::MUL: return "ReduceMul";
  case popops::Operation::MIN: return "ReduceMin";
  case popops::Operation::MAX: return "ReduceMax";
  case popops::Operation::LOGICAL_AND: return "ReduceAnd";
  case popops::Operation::LOGICAL_OR: return "ReduceOr";
  }
  POPLIB_UNREACHABLE();
}

std::string inline getReductionVertexName(const std::string &opName,
                   const poplar::Type &partialType,
                   const poplar::Type &outputType,
                   bool isUpdate,
                   ReductionSpecialisation specialisation,
                   bool scaling = false) {
  if (specialisation == ReductionSpecialisation::ALL_REGIONS_CONTINUOUS) {
    if (scaling) {
      return poputil::templateVertex("popops::ScaledContinuousReduce",
                                     "popops::" + opName,
                                     partialType, outputType, isUpdate);
    } else {
      return poputil::templateVertex("popops::ContinuousReduce",
                                     "popops::" + opName,
                                     partialType, outputType, isUpdate);
    }
  }
  if (scaling) {
    return poputil::templateVertex("popops::ScaledReduce", "popops::" + opName,
                                   partialType, outputType, isUpdate,
                                   static_cast<unsigned>(specialisation));
  }
  else {
    return poputil::templateVertex("popops::Reduce", "popops::" + opName,
                                   partialType, outputType, isUpdate,
                                   static_cast<unsigned>(specialisation));
  }
}

std::string inline getReductionVertexName(
    const ReduceParams &params,
    const poplar::Type &partialType,
    const poplar::Type &outputType,
    ReductionSpecialisation specialisation,
    bool scaling = false) {
  std::string opName = getReductionVertexOpName(params.op);
  bool isUpdate = params.update;
  return getReductionVertexName(opName, partialType, outputType, isUpdate,
                                specialisation, scaling);
}

std::string inline getPartialsEqualSizeReductionVertexName(
                   const std::string &opName,
                   const poplar::Type &partialType,
                   const poplar::Type &outputType,
                   bool isUpdate,
                   bool isScale) {
  std::string vertex;

  vertex = isScale ? "popops::ScaledReducePartialsEqualSize" :
                     "popops::ReducePartialsEqualSize";
  return poputil::templateVertex(vertex, "popops::" + opName,
                                 partialType, outputType, isUpdate);
}

std::string inline getPartialsEqualSizeReductionVertexName(
                   const ReduceParams &params,
                   const poplar::Type &partialType,
                   const poplar::Type &outputType) {
  std::string opName = getReductionVertexOpName(params.op);
  return getPartialsEqualSizeReductionVertexName(opName,
         partialType,
         outputType,
         params.update,
         params.useScale);
}

} // namespace popops

#endif // popops_reduction_ReductionVertex_hpp_
