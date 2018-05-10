#ifndef popops_reduction_ReductionVertex_hpp_
#define popops_reduction_ReductionVertex_hpp_
#include <popops/Reduce.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poplibs_support/Compiler.hpp>

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

std::string inline getReductionVertexName(const std::string opName,
                                   const poplar::Type &partialType,
                                   const poplar::Type &outputType,
                                   bool partialsAreOutputSize,
                                   bool isScale, bool isUpdate) {
  return poputil::templateVertex("popops::Reduce",
                                 "popops::" + opName,
                                 partialType, outputType, partialsAreOutputSize,
                                 isScale, isUpdate);
}

std::string inline getReductionVertexName(const ReduceParams &params,
                                          const poplar::Type &partialType,
                                          const poplar::Type &outputType,
                                          bool partialsAreOutputSize) {
  std::string opName = getReductionVertexOpName(params.op);
  bool isScale = false;
  if ((params.op == popops::Operation::ADD ||
       params.op == popops::Operation::SQUARE_ADD) &&
      params.scale != 1.0f)
    isScale = true;
  bool isUpdate = params.update;
  return getReductionVertexName(opName, partialType, outputType,
                                partialsAreOutputSize, isScale, isUpdate);
}

}

#endif // popops_reduction_ReductionVertex_hpp_
