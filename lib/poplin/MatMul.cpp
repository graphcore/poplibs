#include "poplin/MatMul.hpp"
#include "poplin/Convolution.hpp"
#include "poplin/ConvUtil.hpp"
#include "poputil/exceptions.hpp"
#include "popops/ScaledAdd.hpp"
#include "poplibs_support/OptionParsing.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_support/StructHelper.hpp"
#include <boost/optional.hpp>
#include <cassert>
#include <ostream>
using namespace poplar;
using namespace poplar::program;

namespace poplin {

namespace matmul {

class PlanningCacheImpl : public poplin::PlanningCache {};

PlanningCache::PlanningCache() :
  impl(new PlanningCacheImpl) {

}

PlanningCache::~PlanningCache() = default;

} // namespace matmul

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
  /// Optional convolution planner constraints. These will be parsed by
  /// the convolution options parsing so just pass these down.
  std::string planConstraints;
  // TODO T9375: these two options have been deprecated and so have no effect.
  // any non-zero value will emit a warning during compilation.
  unsigned tempMemoryBudget = 0;
  unsigned cycleBackoffPercent = 0;
  // proportion of tile memory available for this convolution.
  double availableMemoryProportion = .6;
  bool inputRHSIsPreArranged = false;
  // If set, attempts to regroup left and right matrices to improve
  // rearrangements
  bool useAggressiveRegrouping = false;
  bool operator<(const MatMulOptions &other) const {
    using poplibs_support::makeStructHelper;

    auto helper = makeStructHelper(&MatMulOptions::partialsType,
                                   &MatMulOptions::fullyConnectedPass,
                                   &MatMulOptions::planConstraints,
                                   &MatMulOptions::tempMemoryBudget,
                                   &MatMulOptions::cycleBackoffPercent,
                                   &MatMulOptions::availableMemoryProportion,
                                   &MatMulOptions::inputRHSIsPreArranged,
                                   &MatMulOptions::useAggressiveRegrouping);

    return helper.lt(*this, other);
  }
};

static MatMulOptions parseMatMulOptions(const poplar::OptionFlags &options) {
  MatMulOptions matMulOptions;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec matMulSpec {
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
      }) },
    { "inputRHSIsPreArranged", OptionHandler::createWithBool(
      matMulOptions.inputRHSIsPreArranged)},
    { "tempMemoryBudget", OptionHandler::createWithInteger(
      matMulOptions.tempMemoryBudget
    )},
    { "cycleBackoffPercent", OptionHandler::createWithInteger(
      matMulOptions.cycleBackoffPercent
    )},
    { "availableMemoryProportion", OptionHandler::createWithDouble(
      matMulOptions.availableMemoryProportion
    )},
    { "useAggressiveRegrouping",
      OptionHandler::createWithBool(matMulOptions.useAggressiveRegrouping)
    },
    { "planConstraints", OptionHandler::createWithString(
      matMulOptions.planConstraints)
    },
  };
  for (const auto &entry : options) {
    matMulSpec.parse(entry.first, entry.second);
  }
  return matMulOptions;
}

static poplar::OptionFlags getConvOptionFlags(const MatMulOptions &options) {
  poplar::OptionFlags convOptions;
  convOptions.set("partialsType", options.partialsType.toString());
  convOptions.set("tempMemoryBudget", std::to_string(options.tempMemoryBudget));
  convOptions.set("cycleBackoffPercent",
                  std::to_string(options.cycleBackoffPercent));
  convOptions.set("availableMemoryProportion",
                  std::to_string(options.availableMemoryProportion));
  convOptions.set("useAggressiveRegrouping",
                   options.useAggressiveRegrouping ? "true" : "false");
  convOptions.set("planConstraints", options.planConstraints);
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

static poplin::PlanningCache *getLinCache(matmul::PlanningCache *cache) {
  poplin::PlanningCache *linCache = nullptr;
  if (cache) {
    linCache = cache->impl.get();
  }
  return linCache;
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
    throw poputil::poplibs_error("Tensor is not a grouped matrix tensor");
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

static poplin::ConvParams getConvParams(
    const Type &inputType,
    const Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const MatMulOptions &options) {
  if (aShape.size() != 3 || bShape.size() != 3) {
    throw poputil::poplibs_error("Operand to matrix multiplication is not a "
                               "grouped matrix ");
  }
  if (aShape[0] != bShape[0]) {
    throw poputil::poplibs_error("Number of matrix multiplication groups must "
                               "be the same for both operands");
  }

  if (aShape[2] != bShape[1]) {
    throw poputil::poplibs_error("Third dimension of first operand to matrix "
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
      return poplin::ConvParams{
        inputType,
        outputType,
        1,            // batch size
        {outputSize}, // input field shape for each channel and batch
        {1},          // kernel shape for each input and output channel
        inputSize,    // input channels
        batchSize,    // output channels
        numGroups     // conv groups
      };
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
      return poplin::ConvParams{
        inputType,
        outputType,
        1,           // batch size
        {inputSize}, // input field shape for each channel and batch
        {1},         // kernel shape for each input and output channel
        outputSize,  // input channels
        batchSize,   // output channels
        numGroups    // conv groups
      };
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
      return poplin::ConvParams{
        inputType,
        outputType,
        1,            // batch size
        {outputSize}, // input field shape for each channel and batch
        {1},          // kernel shape for each input and output channel
        batchSize,    // input channels
        inputSize,    // output channels
        numGroups     // conv groups
      };
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
       matmul::PlanningCache *cache,
       const Type &outputType) {
  assert(A.rank() == 3 && B.rank() == 3);
  const auto inputType = A.elementType();
  // TODO cache.
  const auto convOptions = getConvOptionFlags(options);
  poplin::PlanningCache *linCache = getLinCache(cache);
  const auto spOut = specialMatrixOpHandling(
    graph,
    outputType,
    A.shape(),
    B.shape(),
    SpecialOpHandling::MATMUL_RESULT);
  if (spOut)
    return *spOut;
  auto convParams = getConvParams(
    inputType, outputType, A.shape(), B.shape(), options);
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
      out = poplin::convolution(graph, weightsView, actsView, convParams,
                                 false, prog, debugPrefix,
                                 convOptions, linCache);
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
      auto weightsTransposed = weights;
      if (!options.inputRHSIsPreArranged) {
        weightsTransposed =
          poplin::fullyConnectedWeightTranspose(graph, weightsView,
                                                convParams, prog, "",
                                                convOptions, linCache);
      }
      out = poplin::convolution(graph, weightsTransposed, deltasView,
                                 convParams, false, prog, debugPrefix,
                                 convOptions, linCache);
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
      out = poplin::convolution(graph, deltasView, actsView, convParams, true,
                                 prog, debugPrefix, convOptions, linCache);
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
matMulDimChecks(const std::vector<std::size_t> &aShape,
                const std::vector<std::size_t> &bShape) {
  if (aShape.size() != 2 || bShape.size() != 2) {
    throw poputil::poplibs_error("Operand to matrix multiplication is not a "
                               "matrix.");
  }
  if (aShape[1] != bShape[0]) {
    throw poputil::poplibs_error("Second dimension of first operand to matrix "
                               "multiplication does not match first dimension "
                               "of second operand.");
  }
}

static void
matMulGroupedDimChecks(const std::vector<std::size_t> &aShape,
                       const std::vector<std::size_t> &bShape) {
  if (aShape[0] != bShape[0]) {
    throw poputil::poplibs_error("Group dimensions for the two operands in the "
                               "grouped multiplication must be the same");
  }
  if (aShape.size() != 3 || bShape.size() != 3) {
    throw poputil::poplibs_error("Operand to grouped matrix multiplication is "
                               "not a matrix.");
  }
  auto a0Shape = aShape;
  auto b0Shape = bShape;
  a0Shape.erase(a0Shape.begin());
  b0Shape.erase(b0Shape.begin());
  matMulDimChecks(a0Shape, b0Shape);
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
          matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  matMulDimChecks(A_.shape(), B_.shape());
  const auto A = A_.expand({0});
  const auto B = B_.expand({0});
  auto product = matMulImpl(
    graph, A, B, prog, debugPrefix, options, cache, C_.elementType())[0];
  // TODO: scaledAdd should do this but we should do T7132 first.
  // Check if we should regroup the output of the matmul so that it better
  // matches C_ during the add.
  product = regroupIfBeneficial(graph, product, C_, prog, debugPrefix);
  popops::scaledAddTo(graph, C_, product, k, prog, debugPrefix);
}


static void scaleTensorChecks(const poplar::Tensor &scale,
                              const poplar::Type &leftTensorType) {
  if (scale.numElements() != 1) {
    throw
      poputil::poplibs_error("scale k must be a tensor of a single element");
  }
  if (scale.elementType() != leftTensorType) {
    throw poputil::poplibs_error("type for scale (k) tensor should be the "
                                 "same as the type of left hand operand");
  }
}

void
matMulAcc(poplar::Graph &graph, const poplar::Tensor &C_,
          const poplar::Tensor &k,
          const poplar::Tensor &A_, const poplar::Tensor &B_,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix,
          const poplar::OptionFlags &options_,
          matmul::PlanningCache *cache) {
  scaleTensorChecks(k, A_.elementType());
  const auto options = parseMatMulOptions(options_);
  matMulDimChecks(A_.shape(), B_.shape());
  const auto A = A_.expand({0});
  const auto B = B_.expand({0});
  auto product = matMulImpl(
    graph, A, B, prog, debugPrefix, options, cache, C_.elementType())[0];
  // TODO: scaledAdd should do this but we should do T7132 first.
  // Check if we should regroup the output of the matmul so that it better
  // matches C_ during the add.
  product = regroupIfBeneficial(graph, product, C_, prog, debugPrefix);
  popops::scaledAddTo(graph, C_, product, k, prog, debugPrefix);
}

void
matMulGroupedAcc(poplar::Graph &graph, const poplar::Tensor &C,
                 const poplar::Tensor &k,
                 const poplar::Tensor &A, const poplar::Tensor &B,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix,
                 const poplar::OptionFlags &options_,
                 matmul::PlanningCache *cache) {
  scaleTensorChecks(k, A.elementType());
  const auto options = parseMatMulOptions(options_);
  matMulGroupedDimChecks(A.shape(), B.shape());
  auto product = matMulImpl(
    graph, A, B, prog, debugPrefix, options, cache, C.elementType());
  popops::scaledAddTo(graph, C, product, k, prog, debugPrefix);
}


void
matMulGroupedAcc(poplar::Graph &graph, const poplar::Tensor &C, float k,
                 const poplar::Tensor &A, const poplar::Tensor &B,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix,
                 const poplar::OptionFlags &options_,
                 matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  matMulGroupedDimChecks(A.shape(), B.shape());
  auto product = matMulImpl(
    graph, A, B, prog, debugPrefix, options, cache, C.elementType());
  popops::scaledAddTo(graph, C, product, k, prog, debugPrefix);
}


static poplar::Tensor
createMatMulInputLHSImpl(poplar::Graph &graph,
                     const Type &inputType,
                     const Type &outputType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const MatMulOptions &options,
                     matmul::PlanningCache *cache) {
  if (options.fullyConnectedPass == FullyConnectedPass::TRAINING_WU) {
    auto fwdOptions = options;
    fwdOptions.fullyConnectedPass = FullyConnectedPass::TRAINING_FWD;
    auto fwdLHS = createMatMulInputLHSImpl(graph, inputType, outputType,
                                          {aShape[0], aShape[2], aShape[1]},
                                          {aShape[0], aShape[1], bShape[2]},
                                          name, fwdOptions, cache);
    return transpose(fwdLHS);
  }
  const auto spOut = specialMatrixOpHandling(graph, inputType, aShape, bShape,
                                             SpecialOpHandling::CREATE_LHS);
  if (spOut)
    return *spOut;
  auto convParams = getConvParams(
    inputType, outputType, aShape, bShape, options);
  auto convOptions = getConvOptionFlags(options);
  auto linCache = getLinCache(cache);
  switch (options.fullyConnectedPass) {
  default: assert(0 && "Unexpected pass");
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::INFERENCE_FWD:
  case FullyConnectedPass::TRAINING_FWD:
    {
      auto convWeights = poplin::createWeights(graph, convParams, name,
                                                convOptions, linCache);
      return matrixFromConvWeights(convWeights);
    }
  case FullyConnectedPass::TRAINING_BWD:
    {
      auto convWeights = poplin::createWeights(graph, convParams, name,
                                                convOptions, linCache);
      return matrixFromConvWeights(convWeights);
    }
  }
}

poplar::Tensor
createMatMulInputRHSImpl(poplar::Graph &graph,
                     const Type &inputType,
                     const Type &outputType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const MatMulOptions &options,
                     matmul::PlanningCache *cache) {
  if (options.fullyConnectedPass == FullyConnectedPass::TRAINING_BWD) {
    auto fwdOptions = options;
    fwdOptions.fullyConnectedPass = FullyConnectedPass::TRAINING_FWD;
    auto fwdRHS = createMatMulInputRHSImpl(graph, inputType, outputType,
                                          {aShape[0], aShape[1], bShape[2]},
                                          {bShape[0], bShape[2], bShape[1]},
                                          name, fwdOptions, cache);
    return transpose(fwdRHS);
  }
  const auto spOut = specialMatrixOpHandling(graph, inputType, aShape, bShape,
                                             SpecialOpHandling::CREATE_RHS);
  if (spOut)
    return *spOut;
  auto convParams = getConvParams(
    inputType, outputType, aShape, bShape, options);
  const auto convOptions = getConvOptionFlags(options);
  const auto linCache = getLinCache(cache);
  const auto numGroups = convParams.getNumConvGroups();
  switch (options.fullyConnectedPass) {
  default: assert(0 && "Unexpected pass");
  case FullyConnectedPass::NONE:
  case FullyConnectedPass::INFERENCE_FWD:
  case FullyConnectedPass::TRAINING_FWD:
    {
      auto convInput = poplin::createInput(graph, convParams, name,
                                            convOptions, linCache);
      return transpose(matrixFromConvActivations(convInput, numGroups));
    }
  case FullyConnectedPass::TRAINING_WU:
    {
      auto convInput = poplin::createInput(graph, convParams, name,
                                            convOptions, linCache);
      return transpose(matrixFromConvActivations(convInput, numGroups));
    }
  }
}

poplar::Tensor
createMatMulInputRHS(poplar::Graph &graph,
                     const Type &inputType,
                     const Type &outputType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const poplar::OptionFlags &options_,
                     matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  return createMatMulInputRHSImpl(graph, inputType, outputType,
                                  {1, aShape[0], aShape[1]},
                                  {1, bShape[0], bShape[1]},
                                  name, options, cache)[0];
}

poplar::Tensor
createMatMulInputRHS(poplar::Graph &graph,
                     const Type &dataType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const poplar::OptionFlags &options_,
                     matmul::PlanningCache *cache) {
  return createMatMulInputRHS(
    graph, dataType, dataType, aShape, bShape, name, options_, cache);
}

poplar::Tensor
createMatMulGroupedInputRHS(poplar::Graph &graph,
                            const Type &inputType,
                            const Type &outputType,
                            const std::vector<std::size_t> &aShape,
                            const std::vector<std::size_t> &bShape,
                            const std::string &name,
                            const poplar::OptionFlags &options_,
                            matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  return createMatMulInputRHSImpl(
    graph, inputType, outputType, aShape, bShape, name, options, cache);
}

poplar::Tensor
matMul(poplar::Graph &graph,
       const poplar::Tensor &A_, const poplar::Tensor &B_,
       poplar::program::Sequence &prog,
       const Type &outputType,
       const std::string &debugPrefix,
       const poplar::OptionFlags &options_,
       matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  poplibs_support::logging::info("matMul {} x {}, pass={}, name={}",
                                 A_.shape(), B_.shape(),
                                 int(options.fullyConnectedPass),
                                 debugPrefix);

  matMulDimChecks(A_.shape(), B_.shape());
  const auto A = A_.expand({0});
  const auto B = B_.expand({0});
  return matMulImpl(
    graph, A, B, prog, debugPrefix, options, cache, outputType)[0];
}

poplar::Tensor
matMul(poplar::Graph &graph,
       const poplar::Tensor &A_, const poplar::Tensor &B_,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix,
       const poplar::OptionFlags &options_,
       matmul::PlanningCache *cache) {
  return matMul(
    graph, A_, B_, prog, A_.elementType(), debugPrefix, options_, cache);
}

void matMulReportPlan(std::ostream &out,
                      const poplar::Graph &graph,
                      const poplar::Type &inputType,
                      const poplar::Type &outputType,
                      const std::vector<std::size_t> &aShape_,
                      const std::vector<std::size_t> &bShape_,
                      const OptionFlags &options,
                      matmul::PlanningCache *cache) {
  auto aShape = aShape_;
  aShape.insert(aShape.begin(), 1);
  auto bShape = bShape_;
  bShape.insert(bShape.begin(), 1);
  return matMulGroupedReportPlan(
    out, graph, inputType, outputType, aShape, bShape, options, cache);
}

poplar::Tensor
matMulGrouped(poplar::Graph &graph,
              const poplar::Tensor &A, const poplar::Tensor &B,
              poplar::program::Sequence &prog,
              const Type &outputType,
              const std::string &debugPrefix,
              const poplar::OptionFlags &options_,
              matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  matMulGroupedDimChecks(A.shape(), B.shape());
  return matMulImpl(graph, A, B, prog, debugPrefix, options, cache, outputType);
}

void matMulGroupedReportPlan(std::ostream &out,
                             const poplar::Graph &graph,
                             const Type &inputType,
                             const Type &outputType,
                             const std::vector<std::size_t> &aShape,
                             const std::vector<std::size_t> &bShape,
                             const poplar::OptionFlags &options_,
                             matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  auto convOptions = getConvOptionFlags(options);
  auto convParams = getConvParams(
    inputType, outputType, aShape, bShape, options);
  auto linCache = getLinCache(cache);
  if (!bShape[2]) {
    out << "Matrix multiplication result produced via special handling\n";
    return;
  }
  return poplin::reportPlanInfo(out, graph, convParams,
                                 convOptions, linCache);
}

poplar::Tensor
createMatMulInputLHS(poplar::Graph &graph,
                     const Type &inputType,
                     const Type &outputType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const poplar::OptionFlags &options_,
                     matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  return
    createMatMulInputLHSImpl(graph, inputType, outputType,
                             {1, aShape[0], aShape[1]},
                             {1, bShape[0], bShape[1]},
                             name, options, cache)[0];
}

poplar::Tensor
createMatMulInputLHS(poplar::Graph &graph,
                     const Type &dataType,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const poplar::OptionFlags &options_,
                     matmul::PlanningCache *cache) {
  return createMatMulInputLHS(
    graph, dataType, dataType, aShape, bShape, name, options_, cache);
}

poplar::Tensor
createMatMulGroupedInputLHS(poplar::Graph &graph,
                            const Type &inputType,
                            const Type &outputType,
                            const std::vector<std::size_t> &aShape,
                            const std::vector<std::size_t> &bShape,
                            const std::string &name,
                            const poplar::OptionFlags &options_,
                            matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  return createMatMulInputLHSImpl(
    graph, inputType, outputType, aShape, bShape, name, options, cache);
}

static poplar::Tensor
preArrangeMatMulInputRHSImpl(poplar::Graph &graph,
                             const std::vector<std::size_t> &aShape,
                             const poplar::Tensor &B,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix,
                             const MatMulOptions &options,
                             matmul::PlanningCache *cache,
                             const Type &outputType) {
  assert(aShape.size() == 3 && B.rank() == 3);
  const auto fPrefix = debugPrefix + "/PreArrangeMatMulInputRHS";
  const auto inputType = B.elementType();
  const auto convOptions = getConvOptionFlags(options);
  poplin::PlanningCache *linCache = getLinCache(cache);
  auto convParams = getConvParams(
    inputType, outputType, aShape, B.shape(), options);
  Tensor arranged;
  switch (options.fullyConnectedPass) {
  case FullyConnectedPass::TRAINING_BWD:
    if (options.inputRHSIsPreArranged) {
      auto weights = B;
      const auto inputSize = weights.dim(2);
      const auto outputSize = weights.dim(1);
      const auto numGroups = weights.dim(0);
      auto weightsView =
          convActivationsFromMatrix(weights,
                                   {numGroups, outputSize, inputSize});
      auto weightsTransposed =
        poplin::fullyConnectedWeightTranspose(graph, weightsView, convParams,
                                              prog, fPrefix, convOptions,
                                              linCache);
      arranged =
        transpose(matrixFromConvActivations(weightsTransposed, numGroups));
      break;
    }
    // fallthrough
  case FullyConnectedPass::INFERENCE_FWD:
  case FullyConnectedPass::TRAINING_FWD:
  case FullyConnectedPass::TRAINING_WU:
    // No pre-arrangement
    arranged = B;
    break;
  case FullyConnectedPass::NONE:
    throw poputil::poplibs_error("preArrangeMatMulRHS only valid for fully "
                                "connected layers");
  }
  assert(arranged.rank() == 3);
  assert(arranged.dim(0) == B.dim(0));
  assert(arranged.dim(1) == B.dim(1));
  assert(arranged.dim(2) == B.dim(2));
  return arranged;
}

poplar::Tensor
preArrangeMatMulInputRHS(poplar::Graph &graph,
                         const std::vector<std::size_t> &aShape_,
                         const poplar::Tensor &B_,
                         poplar::program::Sequence &prog,
                         const Type &outputType,
                         const std::string &debugPrefix,
                         const poplar::OptionFlags &options_,
                         matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  matMulDimChecks(aShape_, B_.shape());
  auto aShape = aShape_;
  aShape.insert(aShape.begin(), 1);
  const auto B = B_.expand({0});
  return preArrangeMatMulInputRHSImpl(
    graph, aShape, B, prog, debugPrefix, options, cache, outputType)[0];
}

poplar::Tensor
preArrangeMatMulInputRHS(poplar::Graph &graph,
                         const std::vector<std::size_t> &aShape_,
                         const poplar::Tensor &B_,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix,
                         const poplar::OptionFlags &options_,
                         matmul::PlanningCache *cache) {
  return preArrangeMatMulInputRHS(
    graph, aShape_, B_, prog, B_.elementType(), debugPrefix, options_, cache);
}

poplar::Tensor
preArrangeMatMulGroupedInputRHS(poplar::Graph &graph,
                                const std::vector<std::size_t> &aShape,
                                const poplar::Tensor &B,
                                poplar::program::Sequence &prog,
                                const std::string &debugPrefix,
                                const poplar::OptionFlags &options_,
                                matmul::PlanningCache *cache,
                                const Type &outputType) {
  const auto options = parseMatMulOptions(options_);
  matMulGroupedDimChecks(aShape, B.shape());
  return preArrangeMatMulInputRHSImpl(
    graph, aShape, B, prog, debugPrefix, options, cache, outputType);
}

}
