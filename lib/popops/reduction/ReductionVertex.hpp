// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef popops_reduction_ReductionVertex_hpp_
#define popops_reduction_ReductionVertex_hpp_
#include "ReductionConnection.hpp"
#include <poplibs_support/Compiler.hpp>
#include <popops/Reduce.hpp>
#include <poputil/VertexTemplates.hpp>

namespace poputil {

template <> struct VertexTemplateToString<popops::ReductionSpecialisation> {
  static std::string to_string(const popops::ReductionSpecialisation &redType) {
    switch (redType) {
    case popops::ReductionSpecialisation::DEFAULT:
      return "popops::ReductionSpecialisation::DEFAULT";
    case popops::ReductionSpecialisation::SCALAR_OUTPUT_REGIONS:
      return "popops::ReductionSpecialisation::SCALAR_OUTPUT_REGIONS";
    case popops::ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT:
      return "popops::ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT";
    case popops::ReductionSpecialisation::SINGLE_OUTPUT_REGION:
      return "popops::ReductionSpecialisation::SINGLE_OUTPUT_REGION";
    case popops::ReductionSpecialisation::ALL_REGIONS_CONTINUOUS:
    case popops::ReductionSpecialisation::PARTIALS_EQUAL_SIZE:
      throw poputil::poplibs_error("Reduction specialisation ");
    default:
      throw poputil::poplibs_error("Unsupported reduction specialisation");
    }
  }
};

} // end namespace poputil

namespace popops {

std::string inline getReductionVertexOpName(popops::Operation op) {
  switch (op) {
  case popops::Operation::ADD:
    return "ReduceAdd";
  case popops::Operation::SQUARE_ADD:
    return "ReduceSquareAdd";
  case popops::Operation::MUL:
    return "ReduceMul";
  case popops::Operation::MIN:
    return "ReduceMin";
  case popops::Operation::MAX:
    return "ReduceMax";
  case popops::Operation::LOGICAL_AND:
    return "ReduceAnd";
  case popops::Operation::LOGICAL_OR:
    return "ReduceOr";
  }
  POPLIB_UNREACHABLE();
}

std::string inline getReductionVertexName(
    const std::string &opName, const poplar::Type &partialType,
    const poplar::Type &outputType, bool isUpdate,
    ReductionSpecialisation specialisation, bool scaling = false) {
  if (specialisation == ReductionSpecialisation::ALL_REGIONS_CONTINUOUS) {
    if (scaling) {
      return poputil::templateVertex("popops::ScaledContinuousReduce",
                                     "popops::" + opName, partialType,
                                     outputType, isUpdate);
    } else {
      return poputil::templateVertex("popops::ContinuousReduce",
                                     "popops::" + opName, partialType,
                                     outputType, isUpdate);
    }
  }
  if (specialisation == ReductionSpecialisation::PARTIALS_EQUAL_SIZE) {
    if (scaling) {
      return poputil::templateVertex("popops::ScaledReducePartialsEqualSize",
                                     "popops::" + opName, partialType,
                                     outputType, isUpdate);
    } else {
      return poputil::templateVertex("popops::ReducePartialsEqualSize",
                                     "popops::" + opName, partialType,
                                     outputType, isUpdate);
    }
  }
  if (scaling) {
    return poputil::templateVertex("popops::ScaledReduce", "popops::" + opName,
                                   partialType, outputType, isUpdate,
                                   specialisation);
  } else {
    return poputil::templateVertex("popops::Reduce", "popops::" + opName,
                                   partialType, outputType, isUpdate,
                                   specialisation);
  }
}

std::string inline getReductionVertexName(
    const ReduceParams &params, const poplar::Type &partialType,
    const poplar::Type &outputType, ReductionSpecialisation specialisation,
    bool scaling = false) {
  std::string opName = getReductionVertexOpName(params.op);
  bool isUpdate = params.update;
  return getReductionVertexName(opName, partialType, outputType, isUpdate,
                                specialisation, scaling);
}

} // namespace popops

#endif // popops_reduction_ReductionVertex_hpp_
