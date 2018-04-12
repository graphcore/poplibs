#include "poplin/MatMul.hpp"
#include "popconv/Convolution.hpp"
#include "poputil/exceptions.hpp"
#include "popops/Add.hpp"
#include "poplibs_support/OptionParsing.hpp"
#include "poplibs_support/Compiler.hpp"
#include <boost/optional.hpp>
#include <cassert>
#include <ostream>
using namespace poplar;
using namespace poplar::program;

namespace poplin {

class PlanningCacheImpl : public popconv::PlanningCache {};

PlanningCache::PlanningCache() :
  impl(new PlanningCacheImpl) {

}

PlanningCache::~PlanningCache() = default;

enum class FullyConnectedPass {
  NONE,
  INFERENCE_FWD,
  TRAINING_FWD,
  TRAINING_BWD,
  TRAINING_WU,
};

/** Options to control the implementation of matrix multiplication */
struct MatMulOptions {
  /** Type used for partial sum calculation */
  poplar::Type partialsType = poplar::FLOAT;
  /// The fully connected pass this multiplication corresponds to. If this
  /// variable is not set to NONE look for a joint plan that avoids the need to
  /// exchange weights. In the forward and backward passes the weight matrix is
  /// assumed to be the right hand side operand of the multiplication. In the
  /// weight update pass we arrange for the result to have the same layout as
  /// the weights so it can be added to the weights without any exchange.
  FullyConnectedPass fullyConnectedPass = FullyConnectedPass::NONE;
  bool operator<(const MatMulOptions &other) const {
    return std::tie(partialsType, fullyConnectedPass) <
             std::tie(other.partialsType, other.fullyConnectedPass);
  }
};

static MatMulOptions parseMatMulOptions(const poplar::OptionFlags &options) {
  MatMulOptions matMulOptions;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec matMulSpec{
    { "partialsType", OptionHandler::createWithEnum(
      matMulOptions.partialsType,
      {
        { "half", poplar::HALF },
        { "float", poplar::FLOAT }
      }) },
    { "fullyConnectedPass", OptionHandler::createWithEnum(
      matMulOptions.fullyConnectedPass,
      {
        { "NONE", FullyConnectedPass::NONE },
        { "INFERENCE_FWD", FullyConnectedPass::INFERENCE_FWD },
        { "TRAINING_FWD", FullyConnectedPass::TRAINING_FWD },
        { "TRAINING_BWD", FullyConnectedPass::TRAINING_BWD },
        { "TRAINING_WU", FullyConnectedPass::TRAINING_WU }
      }) }
  };
  options.list([&](const std::string &option, const std::string &value) {
    matMulSpec.parse(option, value);
  });
  return matMulOptions;
}

static poplar::OptionFlags getConvOptionFlags(const MatMulOptions &options) {
  poplar::OptionFlags convOptions;
  convOptions.set("partialsType", options.partialsType.toString());
  switch (options.fullyConnectedPass) {
  case FullyConnectedPass::NONE:
    convOptions.set("pass", "NONE");
    break;
  case FullyConnectedPass::INFERENCE_FWD:
    convOptions.set("pass", "FC_INFERENCE_FWD");
    break;
  case FullyConnectedPass::TRAINING_FWD:
    convOptions.set("pass", "FC_TRAINING_FWD");
    break;
  case FullyConnectedPass::TRAINING_BWD:
    convOptions.set("pass", "FC_TRAINING_BWD");
    break;
  case FullyConnectedPass::TRAINING_WU:
    convOptions.set("pass", "FC_TRAINING_WU");
    break;
  }
  return convOptions;
}

static popconv::PlanningCache *getConvCache(poplin::PlanningCache *cache) {
  popconv::PlanningCache *convCache = nullptr;
  if (cache) {
    convCache = cache->impl.get();
  }
  return convCache;
}

// Transform a conv activations tensor to a  grouped matrix tensor view
static Tensor matrixFromConvActivations(const Tensor &A, unsigned numGroups) {
  assert(A.rank() == 3);
  assert(A.dim(0) == 1);
  assert(A.dim(1) % numGroups == 0);
  return A.reshape({numGroups, A.dim(1) / numGroups, A.dim(2)})
          .dimShuffle({0, 2, 1});
}

// Transpose a grouped matrix
static Tensor transpose(const Tensor &A) {
  if (A.rank() != 3) {
    throw poputil::poplib_error("Tensor is not a grouped matrix tensor");
  }
  assert(A.rank() == 3);
  return A.dimShuffle({0, 2, 1});
}

// Transfom a conv weights tensor to a grouped matix tensor view
static Tensor matrixFromConvWeights(const Tensor &A) {
  assert(A.rank() == 4);
  assert(A.dim(3) == 1);
  return A.squeeze({3});
}

// Transform a grouped matrix tensor to an activations tensor view with given
//  3D shape containing {numGroups, inputWidth, inputChannels/group}
static Tensor convActivationsFromMatrix(const Tensor &A,
                                        const std::vector<std::size_t> &shape) {
  assert(shape.size() == 3);
  return A.dimShuffle({0, 2, 1}).reshape({1, shape[0] * shape[2], shape[1]});
}

// Transform a grouped matrix tensor to a weights tensor view with given
// 3D shape containing {numGroups, outputChannels/group, inputChannels/group}
static Tensor convWeightsFromMatrix(const Tensor &A,
                                    const std::vector<std::size_t> &shape) {
  assert(shape.size() == 3);
  return A.expand({3});
}

enum class SpecialOpHandling {
  MATMUL_RESULT,
  CREATE_LHS,
  CREATE_RHS
};

// Special handling is required to avoid a convolution being called with zero
// field size. This function returns the result tensor if convolution cannot be
// called to produce results
static boost::optional<Tensor>
specialMatrixOpHandling(Graph &graph,
                        poplar::Type dType,
                        const std::vector<std::size_t> &aShape,
                        const std::vector<std::size_t> &bShape,
                        SpecialOpHandling op) {
  boost::optional<Tensor> resultTensor;
  if (!bShape[2]) {
    Tensor out;
    if (op == SpecialOpHandling::MATMUL_RESULT) {
      out = graph.addVariable(dType, {aShape[0], aShape[1], bShape[2]},
                              VariableMappingMethod::LINEAR);
    } else if (op == SpecialOpHandling::CREATE_LHS) {
      out = graph.addVariable(dType, {aShape[0], aShape[1], aShape[2]},
                              VariableMappingMethod::LINEAR);
    } else if (op == SpecialOpHandling::CREATE_RHS) {
      out = graph.addVariable(dType, {bShape[0], bShape[1], bShape[2]},
                              VariableMappingMethod::LINEAR);
    }
    resultTensor = out;
  }
  return resultTensor;
}

static popconv::ConvParams getConvParams(
    const Type &dType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const MatMulOptions &options) {
  if (aShape.size() != 3 || bShape.size() != 3) {
    throw poputil::poplib_error("Operand to matrix multiplication is not a "
                               "grouped matrix ");
  }
  if (aShape[0] != bShape[0]) {
    throw poputil::poplib_error("Number of matrix multiplication groups must "
                               "be the same for both operands");
  }

  if (aShape[2] != bShape[1]) {
    throw poputil::poplib_error("Third dimension of first operand to matrix "
                               "multiplication does not match second dimension "
                               "of second operand.");
  }

  switch (options.fullyConnectedPass) {
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::INFERENCE_FWD:
  case FullyConnectedPass::TRAINING_FWD:
    // A fully connected fwd pass is equivalent to a 1-d convolution with
    // input channels = inputSize
    // width = outputSize
    // output channels = batchSize.
    {
      const auto inputSize = bShape[1];
      const auto outputSize = bShape[2];
      const auto batchSize = aShape[1];
      const auto numGroups = aShape[0];
      return
          popconv::ConvParams(dType,
                              // batch size
                              1,
                              // input field shape for each channel and batch
                              {outputSize},
                              // kernel shape for each input and output channel
                              {1},
                              // input channels
                              inputSize,
                              // output channels
                              batchSize,
                              // conv groups
                              numGroups,
                              // lower input truncation
                              {0},
                              // upper input truncation
                              {0},
                              // input dilation
                              {1},
                              // lower input padding
                              {0},
                              // upper input padding
                              {0},
                              // flip input
                              {false},
                              // lower kernal truncation
                              {0},
                              // upper kernel truncation
                              {0},
                              // kernel dilation
                              {1},
                              // lower kernal padding
                              {0},
                              // upper kernel padding
                              {0},
                              // flip kernel
                              {false},
                              // lower output truncation
                              {0},
                              // upper output truncation
                              {0},
                              // stride
                              {1},
                              // lower output padding
                              {0},
                              // upper output padding
                              {0});
    }
  case FullyConnectedPass::TRAINING_BWD:
    // A fully connected bwd pass is equivalent to a 1-d convolution with
    // input channels = outputSize
    // width = inputSize
    // output channels = batchSize.
    {
      const auto inputSize = bShape[2];
      const auto outputSize = bShape[1];
      const auto batchSize = aShape[1];
      const auto numGroups = aShape[0];
      return
          popconv::ConvParams(dType,
                              // batch size
                              1,
                              // input field shape for each channel and batch
                              {inputSize},
                              // kernel shape for each input and output channel
                              {1},
                              // input channels
                              outputSize,
                              // output channels
                              batchSize,
                              // conv groups
                              numGroups,
                              // lower input truncation
                              {0},
                              // upper input truncation
                              {0},
                              // input dilation
                              {1},
                              // lower input padding
                              {0},
                              // upper input padding
                              {0},
                              // flip input
                              {false},
                              // lower kernel truncation
                              {0},
                              // upper kernel truncation
                              {0},
                              // kernel dilation
                              {1},
                              // lower kernel padding
                              {0},
                              // upper kernel padding
                              {0},
                              // flip kernel
                              {false},
                              // lower output truncation
                              {0},
                              // upper output truncation
                              {0},
                              // stride
                              {1},
                              // lower output padding
                              {0},
                              // upper output padding
                              {0});
    }
  case FullyConnectedPass::TRAINING_WU:
    // Implement the weight update as a convolutional layer with
    // input channels = batch size
    // width = outputSize
    // output channels = inputSize
    {
      const auto inputSize = aShape[1];
      const auto outputSize = bShape[2];
      const auto batchSize = aShape[2];
      const auto numGroups = aShape[0];
      return
          popconv::ConvParams(dType,
                              // batch size
                              1,
                              // input field shape for each channel and batch
                              {outputSize},
                              // kernel shape for each input and output channel
                              {1},
                              // input channels
                              batchSize,
                              // output channels
                              inputSize,
                              // conv groups
                              numGroups,
                              // lower input truncation
                              {0},
                              // upper input truncation
                              {0},
                              // input dilation
                              {1},
                              // lower input padding
                              {0},
                              // upper input padding
                              {0},
                              // flip input
                              {false},
                              // lower kernel truncation
                              {0},
                              // upper kernel truncation
                              {0},
                              // kernel dilation
                              {1},
                              // lower kernel padding
                              {0},
                              // upper kernel padding
                              {0},
                              // flip kernel
                              {false},
                              // lower output truncation
                              {0},
                              // upper output truncation
                              {0},
                              // stride
                              {1},
                              // lower output padding
                              {0},
                              // upper output padding
                              {0});
    }
  }
  POPLIB_UNREACHABLE();
}


static poplar::Tensor
matMulImpl(poplar::Graph &graph,
       const poplar::Tensor &A, const poplar::Tensor &B,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix,
       const MatMulOptions &options,
       PlanningCache *cache) {
  assert(A.rank() == 3 && B.rank() == 3);
  const auto dType = A.elementType();
  // TODO cache.
  const auto convOptions = getConvOptionFlags(options);
  popconv::PlanningCache *convCache = getConvCache(cache);
  const auto spOut = specialMatrixOpHandling(graph, dType, A.shape(), B.shape(),
                                             SpecialOpHandling::MATMUL_RESULT);
  if (spOut)
    return *spOut;
  auto convParams = getConvParams(dType, A.shape(), B.shape(), options);
  Tensor out;
  switch (options.fullyConnectedPass) {
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::INFERENCE_FWD:
  case FullyConnectedPass::TRAINING_FWD:
    // A fully connected fwd pass is equivalent to a convolution with
    // input channels = inputSize
    // width = outputSize
    // height = 1
    // output channels = batchSize.
    {
      auto weights = transpose(B);
      auto acts = A;
      const auto inputSize = weights.dim(2);
      const auto outputSize = weights.dim(1);
      const auto batchSize = acts.dim(1);
      const auto numGroups = acts.dim(0);
      auto weightsView =
          convActivationsFromMatrix(weights,
                                    {numGroups, outputSize, inputSize});
      auto actsView =
          convWeightsFromMatrix(acts, {numGroups, batchSize, inputSize});
      out = popconv::convolution(graph, weightsView, actsView, convParams,
                                 false, prog, debugPrefix,
                                 convOptions, convCache);
      out = transpose(matrixFromConvActivations(out, numGroups));
      break;
    }
  case FullyConnectedPass::TRAINING_BWD:
    // A fully connected bwd pass is equivalent to a convolution with
    // input channels = outputSize
    // width = inputSize
    // height = 1
    // output channels = batchSize.
    {
      auto weights = B;
      auto deltas = A;
      const auto inputSize = weights.dim(2);
      const auto outputSize = weights.dim(1);
      const auto batchSize = deltas.dim(1);
      const auto numGroups = weights.dim(0);
      auto weightsView =
          convActivationsFromMatrix(weights,
                                   {numGroups, outputSize, inputSize});
      auto deltasView =
          convWeightsFromMatrix(deltas, {numGroups, batchSize, outputSize});
      auto weightsTransposed =
          popconv::fullyConnectedWeightTranspose(graph, weightsView,
                                                 convParams, prog, "",
                                                 convOptions, convCache);
      out = popconv::convolution(graph, weightsTransposed, deltasView,
                                 convParams, false, prog, debugPrefix,
                                 convOptions, convCache);
      out = transpose(matrixFromConvActivations(out, numGroups));
      break;
    }
  case FullyConnectedPass::TRAINING_WU:
    // Implement the weight update as a convolutional layer with
    // input channels = batch size
    // width = outputSize
    // height = 1
    // output channels = inputSize
    {
      auto deltas = B;
      auto acts = transpose(A);
      const auto inputSize = acts.dim(2);
      const auto outputSize = deltas.dim(2);
      const auto batchSize = acts.dim(1);
      const auto numGroups = acts.dim(0);
      auto deltasView =
          convActivationsFromMatrix(transpose(deltas),
                                    {numGroups, outputSize, batchSize});
      auto actsView =
          convWeightsFromMatrix(acts, {numGroups, batchSize, inputSize});
      out = popconv::convolution(graph, deltasView, actsView, convParams, true,
                                 prog, debugPrefix, convOptions, convCache);
      out = transpose(matrixFromConvActivations(out, numGroups));
      break;
    }
  }
  assert(out.rank() == 3);
  assert(out.dim(0) == A.dim(0));
  assert(out.dim(1) == A.dim(1));
  assert(out.dim(2) == B.dim(2));
  return out;
}


static void
matMulDimChecks(const Tensor &A_, const Tensor &B_) {
  if (A_.rank() != 2 || B_.rank() != 2) {
    throw poputil::poplib_error("Operand to matrix multiplication is not a "
                               "matrix.");
  }
  if (A_.dim(1) != B_.dim(0)) {
    throw poputil::poplib_error("Second dimension of first operand to matrix "
                               "multiplication does not match first dimension "
                               "of second operand.");
  }
}

static void
matMulGroupedDimChecks(const Tensor &A, const Tensor &B) {
  if (A.dim(0) != B.dim(0)) {
    throw poputil::poplib_error("Group dimensions for the two operands in the "
                               "grouped multiplication must be the same");
  }
  if (A.rank() != 3 || B.rank() != 3) {
    throw poputil::poplib_error("Operand to grouped matrix multiplication is "
                               "not a matrix.");
  }
  matMulDimChecks(A[0], B[0]);
}

Tensor
transposeGroupedMatrix(const Tensor &A) {
  return transpose(A);
}

void
matMulAcc(poplar::Graph &graph, const poplar::Tensor &C_, float k,
          const poplar::Tensor &A_, const poplar::Tensor &B_,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix,
          const poplar::OptionFlags &options_,
          PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  matMulDimChecks(A_, B_);
  const auto A = A_.expand({0});
  const auto B = B_.expand({0});
  auto product = matMulImpl(graph, A, B, prog, debugPrefix, options, cache)[0];
  popops::addTo(graph, C_, product, k, prog, debugPrefix);
}

void
matMulGroupedAcc(poplar::Graph &graph, const poplar::Tensor &C, float k,
                 const poplar::Tensor &A, const poplar::Tensor &B,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix,
                 const poplar::OptionFlags &options_,
                 PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  matMulGroupedDimChecks(A, B);
  auto product = matMulImpl(graph, A, B, prog, debugPrefix, options, cache);
  popops::addTo(graph, C, product, k, prog, debugPrefix);
}


static poplar::Tensor
createMatMulInputLHSImpl(poplar::Graph &graph,
                     const Type &dType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const MatMulOptions &options,
                     PlanningCache *cache) {
  if (options.fullyConnectedPass == FullyConnectedPass::TRAINING_WU) {
    auto fwdOptions = options;
    fwdOptions.fullyConnectedPass = FullyConnectedPass::TRAINING_FWD;
    auto fwdLHS = createMatMulInputLHSImpl(graph, dType,
                                          {aShape[0], aShape[2], aShape[1]},
                                          {aShape[0], aShape[1], bShape[2]},
                                          name, fwdOptions, cache);
    return transpose(fwdLHS);
  }
  const auto spOut = specialMatrixOpHandling(graph, dType, aShape, bShape,
                                             SpecialOpHandling::CREATE_LHS);
  if (spOut)
    return *spOut;
  auto convParams = getConvParams(dType, aShape, bShape, options);
  auto convOptions = getConvOptionFlags(options);
  auto convCache = getConvCache(cache);
  switch (options.fullyConnectedPass) {
  default: assert(0 && "Unexpected pass");
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::INFERENCE_FWD:
  case FullyConnectedPass::TRAINING_FWD:
    {
      auto convWeights = popconv::createWeights(graph, convParams, name,
                                                convOptions, convCache);
      return matrixFromConvWeights(convWeights);
    }
  case FullyConnectedPass::TRAINING_BWD:
    {
      auto convWeights = popconv::createWeights(graph, convParams, name,
                                                convOptions, convCache);
      return matrixFromConvWeights(convWeights);
    }
  }
}

poplar::Tensor
createMatMulInputRHSImpl(poplar::Graph &graph,
                     const Type &dType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const MatMulOptions &options,
                     PlanningCache *cache) {
  if (options.fullyConnectedPass == FullyConnectedPass::TRAINING_BWD) {
    auto fwdOptions = options;
    fwdOptions.fullyConnectedPass = FullyConnectedPass::TRAINING_FWD;
    auto fwdRHS = createMatMulInputRHSImpl(graph, dType,
                                          {aShape[0], aShape[1], bShape[2]},
                                          {bShape[0], bShape[2], bShape[1]},
                                          name, fwdOptions, cache);
    return transpose(fwdRHS);
  }
  const auto spOut = specialMatrixOpHandling(graph, dType, aShape, bShape,
                                             SpecialOpHandling::CREATE_RHS);
  if (spOut)
    return *spOut;
  auto convParams = getConvParams(dType, aShape, bShape, options);
  const auto convOptions = getConvOptionFlags(options);
  const auto convCache = getConvCache(cache);
  const auto numGroups = convParams.getNumConvGroups();
  switch (options.fullyConnectedPass) {
  default: assert(0 && "Unexpected pass");
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::INFERENCE_FWD:
  case FullyConnectedPass::TRAINING_FWD:
    {
      auto convInput = popconv::createInput(graph, convParams, name,
                                            convOptions, convCache);
      return transpose(matrixFromConvActivations(convInput, numGroups));
    }
  case FullyConnectedPass::TRAINING_WU:
    {
      auto convInput = popconv::createInput(graph, convParams, name,
                                            convOptions, convCache);
      return transpose(matrixFromConvActivations(convInput, numGroups));
    }
  }
}

poplar::Tensor
createMatMulInputRHS(poplar::Graph &graph,
                     const Type &dType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const poplar::OptionFlags &options_,
                     PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  return createMatMulInputRHSImpl(graph, dType,
                                  {1, aShape[0], aShape[1]},
                                  {1, bShape[0], bShape[1]},
                                  name, options, cache)[0];
}

poplar::Tensor
createMatMulGroupedInputRHS(poplar::Graph &graph,
                            const Type &dType,
                            const std::vector<std::size_t> &aShape,
                            const std::vector<std::size_t> &bShape,
                            const std::string &name,
                            const poplar::OptionFlags &options_,
                            PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  return createMatMulInputRHSImpl(graph, dType, aShape, bShape, name,
                                  options, cache);
}

poplar::Tensor
matMul(poplar::Graph &graph,
       const poplar::Tensor &A_, const poplar::Tensor &B_,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix,
       const poplar::OptionFlags &options_,
       PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  matMulDimChecks(A_, B_);
  const auto A = A_.expand({0});
  const auto B = B_.expand({0});
  return matMulImpl(graph, A, B, prog, debugPrefix, options, cache)[0];
}

void matMulReportPlan(std::ostream &out,
                      const poplar::Graph &graph,
                      const poplar::Type &dType,
                      const std::vector<std::size_t> &aShape_,
                      const std::vector<std::size_t> &bShape_,
                      const OptionFlags &options,
                      PlanningCache *cache) {
  auto aShape = aShape_;
  aShape.insert(aShape.begin(), 1);
  auto bShape = bShape_;
  bShape.insert(bShape.begin(), 1);
  return matMulGroupedReportPlan(out, graph, dType, aShape, bShape,
                                 options, cache);
}

poplar::Tensor
matMulGrouped(poplar::Graph &graph,
              const poplar::Tensor &A, const poplar::Tensor &B,
              poplar::program::Sequence &prog,
              const std::string &debugPrefix,
              const poplar::OptionFlags &options_,
              PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  matMulGroupedDimChecks(A, B);
  return matMulImpl(graph, A, B, prog, debugPrefix, options, cache);
}

void matMulGroupedReportPlan(std::ostream &out,
                             const poplar::Graph &graph,
                             const Type &dType,
                             const std::vector<std::size_t> &aShape,
                             const std::vector<std::size_t> &bShape,
                             const poplar::OptionFlags &options_,
                             PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  auto convOptions = getConvOptionFlags(options);
  auto convParams = getConvParams(dType, aShape, bShape, options);
  auto convCache = getConvCache(cache);
  if (!bShape[2]) {
    out << "Matrix multiplication result produced via special handling\n";
    return;
  }
  return popconv::reportPlanInfo(out, graph, convParams,
                                 convOptions, convCache);
}

poplar::Tensor
createMatMulInputLHS(poplar::Graph &graph,
                     const Type &dType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const poplar::OptionFlags &options_,
                     PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  return
    createMatMulInputLHSImpl(graph, dType,
                             {1, aShape[0], aShape[1]},
                             {1, bShape[0], bShape[1]},
                             name, options, cache)[0];
}

poplar::Tensor
createMatMulGroupedInputLHS(poplar::Graph &graph,
                            const Type &dType,
                            const std::vector<std::size_t> &aShape,
                            const std::vector<std::size_t> &bShape,
                            const std::string &name,
                            const poplar::OptionFlags &options_,
                            PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  return createMatMulInputLHSImpl(graph, dType, aShape, bShape, name,
                                  options, cache);
}

}
