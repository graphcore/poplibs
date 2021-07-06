// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplin/MatMul.hpp"
#include "ConvOptions.hpp"
#include "ConvPlan.hpp"
#include "MatMulInternal.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/StructHelper.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/ConvPreplan.hpp"
#include "poplin/Convolution.hpp"
#include "popops/Rearrange.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/exceptions.hpp"
#include <boost/optional.hpp>
#include <cassert>
#include <ostream>
#include <unordered_map>
using namespace poplar;
using namespace poplar::program;

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const poplin::matmul::PlanningCache &t) {
  return poplar::ProfileValue("<matmul::PlanningCache>");
}
} // namespace poputil

namespace poplin {

bool operator<(const MatMulParams &a, const MatMulParams &b) {
  const auto helper = poplibs_support::makeStructHelper(
      &MatMulParams::inputType, &MatMulParams::outputType,
      &MatMulParams::aShape, &MatMulParams::bShape);
  return helper.lt(a, b);
}

namespace matmul {

PlanningCache::PlanningCache() {}
PlanningCache::~PlanningCache() = default;

std::size_t PlanningCache::size() const { return impl.size(); }

poplin::PlanningCache &PlanningCache::getImpl() { return impl; }

} // namespace matmul

namespace logging = poplibs_support::logging;

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
  // proportion of tile memory available for this matmul.
  double availableMemoryProportion = .6;
  bool inputRHSIsPreArranged = false;
  bool use128BitConvUnitLoad = false;
  bool enableMultiStageReduce = true;
  bool enableFastReduce = false;
  bool remapOutputTensor = true;
  bool gatherOutput = false;
  bool operator<(const MatMulOptions &other) const {
    using poplibs_support::makeStructHelper;

    auto helper = makeStructHelper(
        &MatMulOptions::partialsType, &MatMulOptions::fullyConnectedPass,
        &MatMulOptions::planConstraints,
        &MatMulOptions::availableMemoryProportion,
        &MatMulOptions::inputRHSIsPreArranged,
        &MatMulOptions::use128BitConvUnitLoad,
        &MatMulOptions::enableMultiStageReduce,
        &MatMulOptions::enableFastReduce, &MatMulOptions::remapOutputTensor,
        &MatMulOptions::gatherOutput);

    return helper.lt(*this, other);
  }
};

static std::ostream &operator<<(std::ostream &os, const FullyConnectedPass p) {
  switch (p) {
  case FullyConnectedPass::NONE:
    return os << "NONE";
  case FullyConnectedPass::INFERENCE_FWD:
    return os << "INFERENCE_FWD";
  case FullyConnectedPass::TRAINING_FWD:
    return os << "TRAINING_FWD";
  case FullyConnectedPass::TRAINING_BWD:
    return os << "TRAINING_BWD";
  case FullyConnectedPass::TRAINING_WU:
    return os << "TRAINING_WU";
  }

  const auto id = static_cast<std::underlying_type_t<FullyConnectedPass>>(p);
  throw poputil::poplibs_error("Unknown fully connected pass <" +
                               std::to_string(id) + ">");
}

static MatMulOptions parseMatMulOptions(const poplar::OptionFlags &options) {
  MatMulOptions matMulOptions;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  /*
   * Any changes to matMulSpec must be reflected in the documentation comment in
   * the header.
   */
  const OptionSpec matMulSpec{
      {"partialsType", OptionHandler::createWithEnum(
                           matMulOptions.partialsType,
                           {{"half", poplar::HALF}, {"float", poplar::FLOAT}})},
      {"fullyConnectedPass",
       OptionHandler::createWithEnum(
           matMulOptions.fullyConnectedPass,
           {{"NONE", FullyConnectedPass::NONE},
            {"INFERENCE_FWD", FullyConnectedPass::INFERENCE_FWD},
            {"TRAINING_FWD", FullyConnectedPass::TRAINING_FWD},
            {"TRAINING_BWD", FullyConnectedPass::TRAINING_BWD},
            {"TRAINING_WU", FullyConnectedPass::TRAINING_WU}})},
      {"inputRHSIsPreArranged",
       OptionHandler::createWithBool(matMulOptions.inputRHSIsPreArranged)},
      {"use128BitConvUnitLoad",
       OptionHandler::createWithBool(matMulOptions.use128BitConvUnitLoad)},
      {"enableMultiStageReduce",
       OptionHandler::createWithBool(matMulOptions.enableMultiStageReduce)},
      {"enableFastReduce",
       OptionHandler::createWithBool(matMulOptions.enableFastReduce)},
      {"remapOutputTensor",
       OptionHandler::createWithBool(matMulOptions.remapOutputTensor)},
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(
           matMulOptions.availableMemoryProportion)},
      {"planConstraints",
       OptionHandler::createWithString(matMulOptions.planConstraints)},
      {"gatherOutput",
       OptionHandler::createWithBool(matMulOptions.gatherOutput)},
  };
  for (const auto &entry : options) {
    matMulSpec.parse(entry.first, entry.second);
  }
  return matMulOptions;
}

static poplar::OptionFlags getConvOptionFlags(const MatMulOptions &options) {
  poplar::OptionFlags convOptions;
  convOptions.set("partialsType", options.partialsType.toString());
  convOptions.set("availableMemoryProportion",
                  std::to_string(options.availableMemoryProportion));
  convOptions.set("use128BitConvUnitLoad",
                  options.use128BitConvUnitLoad ? "true" : "false");
  convOptions.set("enableMultiStageReduce",
                  options.enableMultiStageReduce ? "true" : "false");
  convOptions.set("enableFastReduce",
                  options.enableFastReduce ? "true" : "false");
  convOptions.set("remapOutputTensor",
                  options.remapOutputTensor ? "true" : "false");
  convOptions.set("gatherConvOutput", options.gatherOutput ? "true" : "false");
  convOptions.set("planConstraints", options.planConstraints);
  switch (options.fullyConnectedPass) {
  case FullyConnectedPass::NONE:
    convOptions.set("pass", "NONE_MATMUL");
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
    linCache = &cache->getImpl();
  }
  return linCache;
}

// Transform a conv activations tensor to a  grouped matrix tensor view
static Tensor matrixFromConvActivations(const Tensor &A, unsigned numGroups) {
  assert(A.rank() == 3);
  assert(A.dim(2) == 1);
  assert(A.dim(1) % numGroups == 0);
  return A.reshape({A.dim(0), numGroups, A.dim(1) / numGroups})
      .dimShuffle({1, 0, 2});
}

// Transpose a grouped matrix
static Tensor transpose(const Tensor &A) {
  if (A.rank() != 3) {
    throw poputil::poplibs_error("Tensor is not a grouped matrix tensor");
  }
  assert(A.rank() == 3);
  return A.dimShuffle({0, 2, 1});
}

// Transform a conv weights tensor to a grouped matrix tensor view
static Tensor matrixFromConvWeights(const Tensor &A) {
  assert(A.rank() == 4);
  assert(A.dim(3) == 1);
  return A.squeeze({3});
}

// Transform a grouped matrix tensor to an activations tensor view
static Tensor convActivationsFromMatrix(const Tensor &A) {
  assert(A.rank() == 3);
  return A.dimShuffle({1, 0, 2}).reshape({A.dim(1), A.dim(0) * A.dim(2), 1});
}

// Transform a grouped matrix tensor to a weights tensor view with given
// 3D shape containing {numGroups, outputChannels/group, inputChannels/group}
static Tensor convWeightsFromMatrix(const Tensor &A) {
  assert(A.rank() == 3);
  return A.expand({3});
}

// Maps shape of matmul to convolution parameters
static poplin::ConvParams
getConvParams(const Type &inputType, const Type &outputType,
              const std::vector<std::size_t> &aShape,
              const std::vector<std::size_t> &bShape) {
  if (aShape.size() != 3 || bShape.size() != 3) {
    throw poputil::poplibs_error("Operand to matrix multiplication is not a "
                                 "grouped matrix ");
  }
  if (aShape[0] != bShape[0]) {
    throw poputil::poplibs_error("Number of matrix multiplication groups must "
                                 "be the same for both operands");
  }

  if (aShape[2] != bShape[1]) {
    throw poputil::poplibs_error(
        "Third dimension of first operand to matrix "
        "multiplication does not match second dimension "
        "of second operand.");
  }
  const auto inputSize = bShape[1];
  const auto outputSize = bShape[2];
  const auto batchSize = aShape[1];
  const auto numGroups = aShape[0];

  // Matmul is equivalent to a 1-d convolution with
  // input channels = inputSize
  // output channels = outputSize
  // batch = batchSize
  return poplin::ConvParams{
      inputType,  outputType,
      batchSize,  // batch size
      {1},        // input field shape
      {1},        // kernel shape
      inputSize,  // input channels
      outputSize, // output channels
      numGroups   // conv groups
  };
}

MatMulParams toMatMulParams(const std::vector<size_t> &params,
                            poplar::Type dType) {
  const auto groupSize = params[0];
  const auto batchSize = params[1];
  const auto inputSize = params[2];
  const auto outputSize = params[3];
  return {dType,
          dType,
          {groupSize, batchSize, inputSize},
          {groupSize, inputSize, outputSize}};
}

static MatMulParams convertFwdToBwdParams(const MatMulParams &fwdPassParams) {
  MatMulParams bwd = fwdPassParams;
  const auto inputSize = fwdPassParams.aShape[2];
  const auto outputSize = fwdPassParams.bShape[2];
  // Swap the input and output size
  bwd.aShape[2] = outputSize;
  bwd.bShape[1] = outputSize;
  bwd.bShape[2] = inputSize;
  return bwd;
}

static MatMulParams convertFwdToWuParams(const MatMulParams &fwdPassParams) {
  MatMulParams wu = fwdPassParams;
  const auto inputSize = fwdPassParams.aShape[2];
  const auto batchSize = fwdPassParams.aShape[1];
  // Swap the input and batch size
  wu.aShape[2] = batchSize;
  wu.bShape[1] = batchSize;
  wu.aShape[1] = inputSize;
  return wu;
}

// Given a fwd pass parameters and options, return parameters and options for
// backwards and weight update passes
std::vector<std::pair<MatMulParams, poplar::OptionFlags>>
bwdAndWuPassPermutations(std::pair<MatMulParams, poplar::OptionFlags> fwdPass) {
  std::vector<std::pair<MatMulParams, poplar::OptionFlags>> permutations;
  permutations.reserve(2);

  const auto fwdPassParams = fwdPass.first;
  const auto fwdPassOpt = fwdPass.second;

  poplar::OptionFlags bwdPassOpt = fwdPassOpt;
  bwdPassOpt.set("fullyConnectedPass", "TRAINING_BWD");
  const auto bwdPassParams = convertFwdToBwdParams(fwdPassParams);

  poplar::OptionFlags wuPassOpt = fwdPassOpt;
  wuPassOpt.set("fullyConnectedPass", "TRAINING_WU");
  const auto wuPassParams = convertFwdToWuParams(fwdPassParams);

  permutations.push_back(std::make_pair(bwdPassParams, bwdPassOpt));
  permutations.push_back(std::make_pair(wuPassParams, wuPassOpt));
  return permutations;
}

static poplin::ConvParams getConvParams(poplin::MatMulParams params) {
  return getConvParams(params.inputType, params.outputType, params.aShape,
                       params.bShape);
}

// Converts FullyConnectedPass -> Pass
static poplar::OptionFlags
getConvOptionFlags(const poplar::OptionFlags &options) {
  const auto matMulOptions = parseMatMulOptions(options);
  return getConvOptionFlags(matMulOptions);
}

std::set<ConvPlanParams>
matMulGetConvPlanParams(const std::set<MatMulPlanParams> &matmuls,
                        MatMulToConvOptions &matmulOptsPtrToConvOpts) {
  std::set<ConvPlanParams> matmulConvs;

  // Convert all the options to conv options first for lookup
  for (auto &matmul : matmuls) {
    const auto target = std::get<0>(matmul);
    const auto matMulParams = std::get<1>(matmul);
    const auto matMulOpts = std::get<2>(matmul);
    auto res = matmulOptsPtrToConvOpts.emplace(matMulOpts, OptionFlags{});
    // If the pointer wasn't already in the map
    if (res.second) {
      // Create the conv options and store them
      res.first->second = getConvOptionFlags(*matMulOpts);
    }
    const auto convParams = getConvParams(matMulParams);
    // Safe to take pointer to the new option flags in the unordered_map as
    // future insertions don't invalidate this.
    matmulConvs.emplace(target, convParams, &res.first->second);
  }
  return matmulConvs;
}

static poplar::Tensor
matMulImpl(poplar::Graph &graph, const poplar::Tensor &A,
           const poplar::Tensor &B, poplar::program::Sequence &prog,
           const DebugNameAndId &dnai, const MatMulOptions &options,
           matmul::PlanningCache *cache, const Type &outputType) {
  assert(A.rank() == 3 && B.rank() == 3);
  const auto inputType = A.elementType();
  const auto convOptions = getConvOptionFlags(options);
  poplin::PlanningCache *linCache = getLinCache(cache);
  auto convParams = getConvParams(inputType, outputType, A.shape(), B.shape());
  // A matmul is equivalent to a 1-d convolution with
  // input channels = inputSize
  // output channels = outputSize
  // batch = batchSize
  auto weights = B;
  auto acts = A;
  const auto numGroups = acts.dim(0);
  auto actsView = convActivationsFromMatrix(acts);
  auto weightsView = convWeightsFromMatrix(transpose(weights));
  if (options.fullyConnectedPass == FullyConnectedPass::TRAINING_BWD &&
      !options.inputRHSIsPreArranged) {
    weightsView = poplin::fullyConnectedWeightTranspose(
        graph, weightsView.dimShuffle({0, 2, 1, 3}), convParams, prog,
        {dnai, "weightTranspose"}, convOptions, linCache);
  }
  auto out = poplin::convolution(graph, actsView, weightsView, convParams,
                                 false, prog, {dnai}, convOptions, linCache);
  out = matrixFromConvActivations(out, numGroups);
  assert(out.rank() == 3);
  assert(out.dim(0) == A.dim(0));
  assert(out.dim(1) == A.dim(1));
  assert(out.dim(2) == B.dim(2));
  return out;
}

static void matMulDimChecks(const std::vector<std::size_t> &aShape,
                            const std::vector<std::size_t> &bShape) {
  if (aShape.size() != 2 || bShape.size() != 2) {
    throw poputil::poplibs_error("Operand to matrix multiplication is not a "
                                 "matrix.");
  }
  if (aShape[1] != bShape[0]) {
    throw poputil::poplibs_error(
        "Second dimension of first operand to matrix "
        "multiplication does not match first dimension "
        "of second operand.");
  }
}

static void matMulGroupedDimChecks(const std::vector<std::size_t> &aShape,
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

Tensor transposeGroupedMatrix(const Tensor &A) { return transpose(A); }

void matMulAcc(poplar::Graph &graph, const poplar::Tensor &C_, float k,
               const poplar::Tensor &A_, const poplar::Tensor &B_,
               poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext,
               const poplar::OptionFlags &options_,
               matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(C_, A_, B_, k, options_, cache));

  const auto options = parseMatMulOptions(options_);
  logging::poplin::info("matMulAcc {} += {} x {} x {}, pass={}, name={}",
                        C_.shape(), k, A_.shape(), B_.shape(),
                        options.fullyConnectedPass, debugContext.getPathName());

  matMulDimChecks(A_.shape(), B_.shape());
  const auto A = A_.expand({0});
  const auto B = B_.expand({0});
  auto product =
      matMulImpl(graph, A, B, prog, {di}, options, cache, C_.elementType())[0];
  popops::scaledAddTo(graph, C_, product, k, prog, {di});
}

static void scaleTensorChecks(const poplar::Tensor &scale,
                              const poplar::Type &leftTensorType) {
  if (scale.numElements() != 1) {
    throw poputil::poplibs_error(
        "scale k must be a tensor of a single element");
  }
  if (scale.elementType() != leftTensorType) {
    throw poputil::poplibs_error("type for scale (k) tensor should be the "
                                 "same as the type of left hand operand");
  }
}

void matMulAcc(poplar::Graph &graph, const poplar::Tensor &C_,
               const poplar::Tensor &k, const poplar::Tensor &A_,
               const poplar::Tensor &B_, poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext,
               const poplar::OptionFlags &options_,
               matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(C_, k, A_, B_, options_, cache));

  scaleTensorChecks(k, A_.elementType());
  const auto options = parseMatMulOptions(options_);
  logging::poplin::info("matMulAcc {} += k x {} x {}, pass={}, name={}",
                        C_.shape(), A_.shape(), B_.shape(),
                        options.fullyConnectedPass, debugContext.getPathName());

  matMulDimChecks(A_.shape(), B_.shape());
  const auto A = A_.expand({0});
  const auto B = B_.expand({0});
  auto product =
      matMulImpl(graph, A, B, prog, {di}, options, cache, C_.elementType())[0];
  popops::scaledAddTo(graph, C_, product, k, prog, {di});
}

void matMulGroupedAcc(poplar::Graph &graph, const poplar::Tensor &C,
                      const poplar::Tensor &k, const poplar::Tensor &A,
                      const poplar::Tensor &B, poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext,
                      const poplar::OptionFlags &options_,
                      matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(C, k, A, B, options_, cache));

  scaleTensorChecks(k, A.elementType());
  const auto options = parseMatMulOptions(options_);
  logging::poplin::info("matMulGroupedAcc {} x {} + k{}, pass={}, name={}",
                        A.shape(), B.shape(), C.shape(),
                        options.fullyConnectedPass, debugContext.getPathName());

  matMulGroupedDimChecks(A.shape(), B.shape());
  auto product =
      matMulImpl(graph, A, B, prog, {di}, options, cache, C.elementType());
  popops::scaledAddTo(graph, C, product, k, prog, {di});
}

void matMulGroupedAcc(poplar::Graph &graph, const poplar::Tensor &C, float k,
                      const poplar::Tensor &A, const poplar::Tensor &B,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext,
                      const poplar::OptionFlags &options_,
                      matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(C, A, B, k, options_, cache));

  const auto options = parseMatMulOptions(options_);
  logging::poplin::info("matMulGroupedAcc {} x {} + {}{}, pass={}, name={}",
                        A.shape(), B.shape(), k, C.shape(),
                        options.fullyConnectedPass, debugContext.getPathName());

  matMulGroupedDimChecks(A.shape(), B.shape());
  auto product =
      matMulImpl(graph, A, B, prog, {di}, options, cache, C.elementType());
  popops::scaledAddTo(graph, C, product, k, prog, {di});
}

static poplar::Tensor createMatMulInputLHSImpl(
    poplar::Graph &graph, const Type &inputType, const Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, const DebugNameAndId &dnai,
    const MatMulOptions &options, matmul::PlanningCache *cache) {

  auto convParams = getConvParams(inputType, outputType, aShape, bShape);
  auto convOptions = getConvOptionFlags(options);
  auto linCache = getLinCache(cache);
  auto convInput =
      poplin::createInput(graph, convParams, {dnai}, convOptions, linCache);
  return matrixFromConvActivations(convInput, convParams.numConvGroups);
}

poplar::Tensor createMatMulInputRHSImpl(
    poplar::Graph &graph, const Type &inputType, const Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, const DebugNameAndId &dnai,
    const MatMulOptions &options, matmul::PlanningCache *cache) {
  auto convParams = getConvParams(inputType, outputType, aShape, bShape);
  const auto convOptions = getConvOptionFlags(options);
  const auto linCache = getLinCache(cache);

  auto convWeights =
      poplin::createWeights(graph, convParams, {dnai}, convOptions, linCache);
  return transpose(matrixFromConvWeights(convWeights));
}

poplar::Tensor createMatMulInputRHS(poplar::Graph &graph, const Type &inputType,
                                    const Type &outputType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options_,
                                    matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(inputType, outputType, aShape, bShape, options_, cache));

  const auto options = parseMatMulOptions(options_);
  auto output = createMatMulInputRHSImpl(
      graph, inputType, outputType, {1, aShape[0], aShape[1]},
      {1, bShape[0], bShape[1]}, {di}, options, cache)[0];
  di.addOutput(output);
  return output;
}

poplar::Tensor createMatMulInputRHS(poplar::Graph &graph, const Type &dataType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options_,
                                    matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(dataType, aShape, bShape, options_, cache));

  auto output = createMatMulInputRHS(graph, dataType, dataType, aShape, bShape,
                                     {di}, options_, cache);
  di.addOutput(output);
  return output;
}

poplar::Tensor createMatMulGroupedInputRHS(
    poplar::Graph &graph, const Type &inputType, const Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(inputType, outputType, aShape, bShape, options_, cache));

  const auto options = parseMatMulOptions(options_);
  auto output = createMatMulInputRHSImpl(graph, inputType, outputType, aShape,
                                         bShape, {di}, options, cache);
  di.addOutput(output);
  return output;
}

poplar::Tensor matMul(poplar::Graph &graph, const poplar::Tensor &A_,
                      const poplar::Tensor &B_, poplar::program::Sequence &prog,
                      const Type &outputType,
                      const poplar::DebugContext &debugContext,
                      const poplar::OptionFlags &options_,
                      matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(A_, B_, outputType, options_, cache));

  const auto options = parseMatMulOptions(options_);
  logging::poplin::info("matMul {} x {}, pass={}, name={}", A_.shape(),
                        B_.shape(), options.fullyConnectedPass,
                        debugContext.getPathName());

  matMulDimChecks(A_.shape(), B_.shape());
  const auto A = A_.expand({0});
  const auto B = B_.expand({0});
  auto output =
      matMulImpl(graph, A, B, prog, {di}, options, cache, outputType)[0];
  di.addOutput(output);
  return output;
}

poplar::Tensor matMul(poplar::Graph &graph, const poplar::Tensor &A_,
                      const poplar::Tensor &B_, poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext,
                      const poplar::OptionFlags &options_,
                      matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(A_, B_, options_, cache));

  auto output =
      matMul(graph, A_, B_, prog, A_.elementType(), {di}, options_, cache);
  di.addOutput(output);
  return output;
}

void matMulReportPlan(std::ostream &out, const poplar::Graph &graph,
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
  return matMulGroupedReportPlan(out, graph, inputType, outputType, aShape,
                                 bShape, options, cache);
}

poplar::Tensor matMulGrouped(poplar::Graph &graph, const poplar::Tensor &A,
                             const poplar::Tensor &B,
                             poplar::program::Sequence &prog,
                             const Type &outputType,
                             const poplar::DebugContext &debugContext,
                             const poplar::OptionFlags &options_,
                             matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(A, B, outputType, options_, cache));

  const auto options = parseMatMulOptions(options_);
  logging::poplin::info("matMulGrouped {} x {}, pass={}, name={}", A.shape(),
                        B.shape(), options.fullyConnectedPass,
                        debugContext.getPathName());

  matMulGroupedDimChecks(A.shape(), B.shape());
  auto output = matMulImpl(graph, A, B, prog, {di}, options, cache, outputType);
  di.addOutput(output);
  return output;
}

// Gives the serialisation of the the output matrix as a result of doing
// a grouped matmul.
poplibs_support::PlanConstraints groupedMatMulPlanConstraints(
    const poplar::Graph &graph, const Type &inputType, const Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, const poplar::OptionFlags &options_,
    matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  auto convOptions = getConvOptionFlags(options);
  auto convParams = getConvParams(inputType, outputType, aShape, bShape);
  poplin::PlanningCache *linCache = getLinCache(cache);
  return getPlanConstraints(graph, convParams, convOptions, linCache);
}

poplibs_support::PlanConstraints matMulPlanConstraints(
    const poplar::Graph &graph, const Type &inputType, const Type &outputType,
    const std::vector<std::size_t> &aShape_,
    const std::vector<std::size_t> &bShape_, const poplar::OptionFlags &options,
    matmul::PlanningCache *cache) {
  auto aShape = aShape_;
  aShape.insert(aShape.begin(), 1);
  auto bShape = bShape_;
  bShape.insert(bShape.begin(), 1);
  return groupedMatMulPlanConstraints(graph, inputType, outputType, aShape,
                                      bShape, options, cache);
}

void matMulGroupedReportPlan(std::ostream &out, const poplar::Graph &graph,
                             const Type &inputType, const Type &outputType,
                             const std::vector<std::size_t> &aShape,
                             const std::vector<std::size_t> &bShape,
                             const poplar::OptionFlags &options_,
                             matmul::PlanningCache *cache) {
  const auto options = parseMatMulOptions(options_);
  auto convOptions = getConvOptionFlags(options);
  auto convParams = getConvParams(inputType, outputType, aShape, bShape);
  auto linCache = getLinCache(cache);
  if (!bShape[2]) {
    out << "Matrix multiplication result produced via special handling\n";
    return;
  }
  return poplin::reportPlanInfo(out, graph, convParams, convOptions, linCache);
}

poplar::Tensor createMatMulInputLHS(poplar::Graph &graph, const Type &inputType,
                                    const Type &outputType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options_,
                                    matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(inputType, outputType, aShape, bShape, options_, cache));

  const auto options = parseMatMulOptions(options_);
  auto output = createMatMulInputLHSImpl(
      graph, inputType, outputType, {1, aShape[0], aShape[1]},
      {1, bShape[0], bShape[1]}, {di}, options, cache)[0];
  di.addOutput(output);
  return output;
}

poplar::Tensor createMatMulInputLHS(poplar::Graph &graph, const Type &dataType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options_,
                                    matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(dataType, aShape, bShape, options_, cache));

  auto output = createMatMulInputLHS(graph, dataType, dataType, aShape, bShape,
                                     {di}, options_, cache);
  di.addOutput(output);
  return output;
}

poplar::Tensor createMatMulGroupedInputLHS(
    poplar::Graph &graph, const Type &inputType, const Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(inputType, outputType, aShape, bShape, options_, cache));

  const auto options = parseMatMulOptions(options_);
  auto output = createMatMulInputLHSImpl(graph, inputType, outputType, aShape,
                                         bShape, {di}, options, cache);
  di.addOutput(output);
  return output;
}

static poplar::Tensor preArrangeMatMulInputRHSImpl(
    poplar::Graph &graph, const std::vector<std::size_t> &aShape,
    const poplar::Tensor &B, poplar::program::Sequence &prog,
    const DebugNameAndId &dnai, const MatMulOptions &options,
    matmul::PlanningCache *cache, const Type &outputType) {

  if (!options.inputRHSIsPreArranged ||
      options.fullyConnectedPass != FullyConnectedPass::TRAINING_BWD) {
    return B;
  }
  const std::string fPrefix = "PreArrangeMatMulInputRHS";
  const auto inputType = B.elementType();
  const auto convOptions = getConvOptionFlags(options);
  poplin::PlanningCache *linCache = getLinCache(cache);
  auto convParams = getConvParams(inputType, outputType, aShape, B.shape());
  auto weights = B;
  auto fwdWeightsView = convWeightsFromMatrix(weights);
  auto bwdWeights = poplin::fullyConnectedWeightTranspose(
      graph, fwdWeightsView, convParams, prog, {dnai, fPrefix}, convOptions,
      linCache);
  auto arranged = transpose(matrixFromConvWeights(bwdWeights));
  assert(arranged.rank() == 3);
  assert(arranged.dim(0) == B.dim(0));
  assert(arranged.dim(1) == B.dim(1));
  assert(arranged.dim(2) == B.dim(2));
  return arranged;
}

poplar::Tensor preArrangeMatMulInputRHS(
    poplar::Graph &graph, const std::vector<std::size_t> &aShape_,
    const poplar::Tensor &B_, poplar::program::Sequence &prog,
    const Type &outputType, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(B_, aShape_, outputType, options_, cache));

  const auto options = parseMatMulOptions(options_);
  matMulDimChecks(aShape_, B_.shape());
  auto aShape = aShape_;
  aShape.insert(aShape.begin(), 1);
  const auto B = B_.expand({0});
  auto output = preArrangeMatMulInputRHSImpl(graph, aShape, B, prog, {di},
                                             options, cache, outputType)[0];
  di.addOutput(output);
  return output;
}

poplar::Tensor preArrangeMatMulInputRHS(
    poplar::Graph &graph, const std::vector<std::size_t> &aShape_,
    const poplar::Tensor &B_, poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(B_, aShape_, options_, cache));

  auto output = preArrangeMatMulInputRHS(
      graph, aShape_, B_, prog, B_.elementType(), {di}, options_, cache);
  di.addOutput(output);
  return output;
}

poplar::Tensor preArrangeMatMulGroupedInputRHS(
    poplar::Graph &graph, const std::vector<std::size_t> &aShape,
    const poplar::Tensor &B, poplar::program::Sequence &prog,
    const Type &outputType, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, matmul::PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(B, aShape, options_, cache));

  const auto options = parseMatMulOptions(options_);
  matMulGroupedDimChecks(aShape, B.shape());
  return preArrangeMatMulInputRHSImpl(graph, aShape, B, prog, di, options,
                                      cache, outputType);
}

} // namespace poplin
