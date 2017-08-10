#include "poplin/MatMul.hpp"

#include "popconv/Convolution.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/Add.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;

namespace poplin {

class PlanningCacheImpl : public popconv::PlanningCache {};

PlanningCache::PlanningCache() :
  impl(new PlanningCacheImpl) {

}

PlanningCache::~PlanningCache() = default;

static popconv::ConvOptions getConvOptions(
    const MatMulOptions &options) {
  popconv::ConvOptions convOptions;
  if (options.cache) {
    convOptions.cache = options.cache->impl.get();
  }
  convOptions.partialsType = options.partialsType;
  switch (options.fullyConnectedPass) {
  case FullyConnectedPass::NONE:
    convOptions.fullyConnectedPass = popconv::FullyConnectedPass::NONE;
    break;
  case FullyConnectedPass::FWD:
    convOptions.fullyConnectedPass = popconv::FullyConnectedPass::FWD;
    break;
  case FullyConnectedPass::BWD:
    convOptions.fullyConnectedPass = popconv::FullyConnectedPass::BWD;
    break;
  case FullyConnectedPass::WU:
    convOptions.fullyConnectedPass = popconv::FullyConnectedPass::WU;
    break;
  }
  return convOptions;
}

static popconv::ConvParams getConvParams(
    const std::string &dType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const MatMulOptions &options) {
  if (aShape.size() != 2 || bShape.size() != 2) {
    throw popstd::poplib_error("Operand to matrix multiplication is not a "
                               "matrix.");
  }
  if (aShape[1] != bShape[0]) {
    throw popstd::poplib_error("Second dimension of first operand to matrix "
                               "multiplication does not match first dimension "
                               "of second operand.");
  }
  switch (options.fullyConnectedPass) {
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::FWD:
    // A fully connected fwd pass is equivalent to a convolution with
    // input channels = inputSize
    // width = outputSize
    // height = 1
    // output channels = batchSize.
    {
      auto inputSize = aShape[1];
      auto outputSize = aShape[0];
      auto batchSize = bShape[1];
      return
          popconv::ConvParams(dType,
                              {1, 1, outputSize, inputSize} /* inputShape */,
                              {1, 1, batchSize, inputSize} /* kernelShape */,
                              {1, 1},
                              {0, 0}, {0, 0}, {1, 1},
                              {0, 0}, {0, 0}, {1, 1});
    }
  case FullyConnectedPass::BWD:
    // A fully connected bwd pass is equivalent to a convolution with
    // input channels = outputSize
    // width = inputSize
    // height = 1
    // output channels = batchSize.
    {
      auto inputSize = aShape[0];
      auto outputSize = aShape[1];
      auto batchSize = bShape[1];
      return
          popconv::ConvParams(dType,
                              {1, 1, inputSize, outputSize}, /* inputShape */
                              {1, 1, batchSize, outputSize}, /* kernelShape */
                              {1, 1}, /* stride */
                              {0, 0}, {0, 0}, {1, 1},
                              {0, 0}, {0, 0}, {1, 1});
    }
  case FullyConnectedPass::WU:
    // Implement the weight update as a convolutional layer with
    // input channels = batch size
    // width = outputSize
    // height = 1
    // output channels = inputSize
    {
      auto inputSize = bShape[1];
      auto outputSize = aShape[0];
      auto batchSize = bShape[0];
      return
          popconv::ConvParams(dType,
                              {1, 1, outputSize, batchSize}, /* inputShape */
                              {1, 1, inputSize, batchSize}, /* kernelShape */
                              {1, 1}, /* stride */
                              {0, 0}, {0, 0}, {1, 1},
                              {0, 0}, {0, 0}, {1, 1});
    }
  }
}

poplar::Tensor
matMul(poplar::Graph &graph,
       const poplar::Tensor &A, const poplar::Tensor &B,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix,
       const MatMulOptions &options) {
  if (A.rank() != 2 || B.rank() != 2) {
    throw popstd::poplib_error("Operand to matrix multiplication is not a "
                               "matrix.");
  }
  if (A.dim(1) != B.dim(0)) {
    throw popstd::poplib_error("Second dimension of first operand to matrix "
                               "multiplication does not match first dimension "
                               "of second operand.");
  }
  const auto dType = A.elementType();
  // TODO cache.
  popconv::ConvOptions convOptions = getConvOptions(options);
  auto convParams = getConvParams(dType, A.shape(), B.shape(), options);
  Tensor out;
  switch (options.fullyConnectedPass) {
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::FWD:
    // A fully connected fwd pass is equivalent to a convolution with
    // input channels = inputSize
    // width = outputSize
    // height = 1
    // output channels = batchSize.
    {
      auto weights = A;
      auto acts = B.transpose();
      auto inputSize = weights.dim(1);
      auto outputSize = weights.dim(0);
      auto batchSize = acts.dim(0);
      auto weightsView = weights.reshape({1, 1, outputSize, inputSize});
      auto actsView = acts.reshape({1, 1, batchSize, inputSize});
      out = popconv::convolution(graph, weightsView, actsView, convParams,
                                 false, prog, debugPrefix, convOptions);
      out = out[0][0];
      break;
    }
  case FullyConnectedPass::BWD:
    // A fully connected bwd pass is equivalent to a convolution with
    // input channels = outputSize
    // width = inputSize
    // height = 1
    // output channels = batchSize.
    {
      auto weights = A.transpose();
      auto deltas = B.transpose();
      auto inputSize = weights.dim(1);
      auto outputSize = weights.dim(0);
      auto batchSize = deltas.dim(0);
      auto weightsView = weights.reshape({1, 1, outputSize, inputSize});
      auto deltasView =
          deltas.reshape({1, 1, batchSize, outputSize});
      auto weightsTransposed =
          popconv::fullyConnectedWeightTranspose(graph, weightsView,
                                                 convParams, prog, "",
                                                 convOptions);
      out = popconv::convolution(graph, weightsTransposed, deltasView,
                                 convParams, false, prog, debugPrefix,
                                 convOptions);
      out = out[0][0];
      break;
    }
  case FullyConnectedPass::WU:
    // Implement the weight update as a convolutional layer with
    // input channels = batch size
    // width = outputSize
    // height = 1
    // output channels = inputSize
    {
      auto deltas = A.transpose();
      auto acts = B;
      auto inputSize = acts.dim(1);
      auto outputSize = deltas.dim(1);
      auto batchSize = acts.dim(0);
      auto deltasView = deltas.transpose().reshape({1, 1, outputSize,
                                                    batchSize});
      auto actsView = acts.reshape({1, 1, batchSize, inputSize});
      out = popconv::convolution(graph, deltasView, actsView, convParams, true,
                                 prog, debugPrefix, convOptions);
      out = out[0][0];
      break;
    }
  }
  assert(out.rank() == 2);
  assert(out.dim(0) == A.dim(0));
  assert(out.dim(1) == B.dim(1));
  return out;
}

void
matMulAcc(poplar::Graph &graph, const poplar::Tensor &C, float k,
          const poplar::Tensor &A, const poplar::Tensor &B,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix,
          const MatMulOptions &options) {
  auto product = matMul(graph, A, B, prog, debugPrefix, options);
  popstd::addTo(graph, C, product, k, prog, debugPrefix);
}

poplar::Tensor
createMatMulInputLHS(poplar::Graph &graph,
                     const std::string &dType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const MatMulOptions &options) {
  if (options.fullyConnectedPass == FullyConnectedPass::BWD) {
    auto fwdOptions = options;
    fwdOptions.fullyConnectedPass = FullyConnectedPass::FWD;
    auto fwdLHS = createMatMulInputLHS(graph, dType, {aShape[1], aShape[0]},
                                       {aShape[0], bShape[1]}, name,
                                       fwdOptions);
    return fwdLHS.transpose();
  }
  auto convParams = getConvParams(dType, aShape, bShape, options);
  auto convOptions = getConvOptions(options);
  switch (options.fullyConnectedPass) {
  default: assert(0 && "Unexpected pass");
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::FWD:
    {
      auto convInput = popconv::createInput(graph, convParams, name,
                                            convOptions);
      return convInput[0][0];
    }
  case FullyConnectedPass::WU:
    {
      auto convInput = popconv::createInput(graph, convParams, name,
                                            convOptions);
      return convInput[0][0];
    }
  }
}

poplar::Tensor
createMatMulInputRHS(poplar::Graph &graph,
                     const std::string &dType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const MatMulOptions &options) {
  if (options.fullyConnectedPass == FullyConnectedPass::WU) {
    auto fwdOptions = options;
    fwdOptions.fullyConnectedPass = FullyConnectedPass::FWD;
    auto fwdRHS = createMatMulInputRHS(graph, dType, {aShape[0], bShape[1]},
                                       {bShape[1], bShape[0]}, name,
                                       fwdOptions);
    return fwdRHS.transpose();
  }
  auto convParams = getConvParams(dType, aShape, bShape, options);
  auto convOptions = getConvOptions(options);
  switch (options.fullyConnectedPass) {
  default: assert(0 && "Unexpected pass");
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::FWD:
    {
      auto convWeights = popconv::createWeights(graph, convParams, name,
                                                convOptions);
      return convWeights[0][0].transpose();
    }
  case FullyConnectedPass::BWD:
    {
      auto convWeights = popconv::createWeights(graph, convParams, name,
                                                convOptions);
      return convWeights[0][0].transpose();
    }
  }
}

}
