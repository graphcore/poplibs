// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/MatMul.hpp"

#include "FullyConnectedTensorMetaData.hpp"
#include "MatMulOptions.hpp"
#include "MatMulTensorMetaData.hpp"
#include "MatMulUtils.hpp"
#include "TensorMetaDataBase.hpp"

#include "poplibs_support/logging.hpp"
#include "poplibs_support/print.hpp"

#include "popsparse/FullyConnected.hpp"

#include <sstream>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

namespace popsparse {
namespace dynamic {

SparseTensor createSparseDenseMatMulLHS(Graph &graph, const Type &inputType,
                                        const MatMulParams &params,
                                        const std::string &debugName,
                                        const OptionFlags &optionFlags,
                                        PlanningCache *cache) {
  const auto options = parseMatMulOptionFlags(optionFlags);
  logging::popsparse::debug(
      "popsparse::createSparseDenseMatMulLHS: '{}' params={}, options={}",
      debugName, params, options);

  const auto fcParams = getFullyConnectedParams(params);
  const auto fcOptions = getFullyConnectedOptions(options);

  const auto weights = createFullyConnectedWeights(graph, inputType, fcParams,
                                                   debugName, fcOptions, cache);

  // Create new meta-data for the sparse tensor referencing the fully-connected
  // meta-data
  assert(dynamic_cast<const FullyConnectedTensorMetaData *>(
      weights.getOpMetaData().getData()));
  const auto &fcOpMetaData = *static_cast<const FullyConnectedTensorMetaData *>(
      weights.getOpMetaData().getData());
  std::unique_ptr<TensorMetaDataBase> opMetaData =
      std::make_unique<MatMulTensorMetaData>(fcOpMetaData, params, options);

  return SparseTensor(weights.getMetaInfoTensor(), weights.getNzValuesTensor(),
                      std::move(opMetaData));
}

Tensor createSparseDenseMatMulRHS(Graph &graph, const Type &inputType,
                                  const MatMulParams &params,
                                  const std::string &debugName,
                                  const OptionFlags &optionFlags,
                                  PlanningCache *cache) {
  const auto options = parseMatMulOptionFlags(optionFlags);
  logging::popsparse::debug(
      "popsparse::createSparseDenseMatMulRHS: '{}' params={}, options={}",
      debugName, params, options);
  const auto fcParams = getFullyConnectedParams(params);
  const auto fcOptions = getFullyConnectedOptions(options);

  const auto input = createFullyConnectedInput(graph, inputType, fcParams,
                                               debugName, fcOptions, cache);
  return fcActsToMatrix(input, params.getNumGroups()).dimRoll(1, 2);
}

// Validate all the parameters are valid together for a given sparseDenseMatMul
// call
static void validateParameters(const SparseTensor &lhs, const Tensor &rhs,
                               bool transposeLHS, bool transposeRHS,
                               const MatMulOptions &options) {
  const auto &opMetaData = lhs.getOpMetaData();
  if (opMetaData.getData() == nullptr) {
    throw poplibs_error("sparseDenseMatMul left-hand operand has no meta-data "
                        "associated with it");
  }

  const auto *mmMetaData =
      dynamic_cast<const MatMulTensorMetaData *>(opMetaData.getData());
  if (!mmMetaData) {
    throw poplibs_error("sparseDenseMatMul left-hand operand has meta-data "
                        "for a different type of operation");
  }

  const auto &params = mmMetaData->mmParams;
  std::vector<std::size_t> expectedRHSShape;
  if (transposeLHS) {
    expectedRHSShape = {params.getNumGroups(), params.getM(), params.getN()};
  } else {
    expectedRHSShape = {params.getNumGroups(), params.getK(), params.getN()};
  }
  if (transposeRHS) {
    std::swap(expectedRHSShape.at(1), expectedRHSShape.at(2));
  }
  if (rhs.shape() != expectedRHSShape) {
    std::stringstream ss;
    ss << "sparseDenseMatMul right-hand operand's shape ";
    printContainer(rhs.shape(), ss);
    ss << " did not match expected shape";
    printContainer(expectedRHSShape, ss);
    throw poplibs_error(ss.str());
  }

  if (mmMetaData->mmOptions != options) {
    throw poplibs_error("sparseDenseMatMul left-hand operand was created with "
                        "different options to those passed for this operation");
  }
}

poplar::Tensor sparseDenseMatMul(poplar::Graph &graph, const SparseTensor &lhs_,
                                 const Tensor &rhs_, Sequence &prog,
                                 bool transposeLHS, bool transposeRHS,
                                 const std::string &debugPrefix,
                                 const OptionFlags &optionFlags,
                                 PlanningCache *cache) {
  auto lhs = lhs_;
  auto rhs = rhs_;
  const auto options = parseMatMulOptionFlags(optionFlags);
  validateParameters(lhs, rhs, transposeLHS, transposeRHS, options);

  const auto &opMetaData =
      *static_cast<const MatMulTensorMetaData *>(lhs.getOpMetaData().getData());

  logging::popsparse::debug("popsparse::sparseDenseMatMul: '{}' params={}, "
                            "transposeLHS={}, transposeRHS={}, options={}",
                            debugPrefix, opMetaData.mmParams, transposeLHS,
                            transposeRHS, options);

  if (transposeRHS) {
    rhs = rhs.dimRoll(1, 2);
  }

  const auto fcParams = getFullyConnectedParams(opMetaData.mmParams);
  const auto fcOptions = getFullyConnectedOptions(options);

  const std::size_t numGroups = 1;
  lhs = sparseMatrixToFCWeights(lhs);
  rhs = matrixToFCActs(rhs.dimRoll(1, 2), numGroups);

  Tensor out;
  if (transposeLHS) {
    out = fullyConnectedGradA(graph, lhs, rhs, fcParams, prog, debugPrefix,
                              fcOptions, cache);
  } else {
    out = fullyConnectedFwd(graph, lhs, rhs, fcParams, prog, debugPrefix,
                            fcOptions, cache);
  }
  return fcActsToMatrix(out, numGroups).dimRoll(1, 2);
}

} // end namespace dynamic
} // end namespace popsparse
