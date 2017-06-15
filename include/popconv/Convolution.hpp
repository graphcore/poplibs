#ifndef __popconv_Convolution_hpp__
#define __popconv_Convolution_hpp__
#include <tuple>
#include <map>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Engine.hpp>

namespace popconv {

/** Class used to cache the calculation of plans for convolution operations.
 */
class PlanningCache;

enum class WeightUpdateMethod {
  AOP,
  AMP,
  AUTO
};

const char *asString(const WeightUpdateMethod &method);
std::ostream &operator<<(std::ostream &os, const WeightUpdateMethod &method);
std::istream &operator>>(std::istream &is, WeightUpdateMethod &method);

enum class FullyConnectedPass {
  NONE,
  FWD,
  BWD,
  WU,
};

/** Options to control the implementation of a convolution */
struct ConvOptions {
  WeightUpdateMethod weightUpdateMethod = WeightUpdateMethod::AUTO;
  bool useWinograd = false;
  unsigned winogradPatchSize = 4;
  unsigned percentageCyclesExcessForMemOptim = 0;
  /// The fully connected pass this layer corresponds to. If this variable
  /// is not set to NONE look for a joint plan that avoids the need to
  /// exchange weights.
  FullyConnectedPass fullyConnectedPass = FullyConnectedPass::NONE;
  /// Is the fully connected backward pass implemented by calling
  /// convolutionWeightUpdate()?
  bool fullyConnectedBwdAsWU = true;
  std::string partialsType = "float";
  PlanningCache *cache = nullptr;
  bool operator<(const ConvOptions &other) const {
    return std::tie(weightUpdateMethod, useWinograd, winogradPatchSize,
                    percentageCyclesExcessForMemOptim,
                    partialsType) <
             std::tie(other.weightUpdateMethod, other.useWinograd,
                      other.winogradPatchSize,
                      other.percentageCyclesExcessForMemOptim,
                      other.partialsType);
  }
};

struct ConvParams {
  std::string dType;
  // Input shape {B x H x W x inChans}
  std::vector<std::size_t> inputShape;
  // Filter shape {H x W x outChans x inChans }
  std::vector<std::size_t> kernelShape;
  std::vector<unsigned> stride;
  // Padding applied to input after dilation and before input
  std::vector<int> inputPaddingLower;
  std::vector<int> inputPaddingUpper;
  // Dilation applied to the input in spatial dimensions before
  // padding and convolution. Dilation is peformed by placing
  // zeroed elements between the elements of the field.
  std::vector<unsigned> inputDilation;
  // Padding applied to kernel after dilation and before input
  std::vector<int> kernelPaddingLower;
  std::vector<int> kernelPaddingUpper;
  // Dilation applied to the kernel in spatial dimensions before
  // padding and convolution. Dilation is peformed by placing
  // zeroed elements between the elements of the filter.
  std::vector<unsigned> kernelDilation;
  ConvParams() = default;
  ConvParams(std::string dType,
             std::vector<std::size_t> inputShape,
             std::vector<std::size_t> kernelShape,
             std::vector<unsigned> stride,
             std::vector<int> inputPaddingLower,
             std::vector<int> inputPaddingUpper,
             std::vector<unsigned> inputDilation,
             std::vector<int> kernelPaddingLower,
             std::vector<int> kernelPaddingUpper,
             std::vector<unsigned> kernelDilation) :
    dType(std::move(dType)),
    inputShape(std::move(inputShape)),
    kernelShape(std::move(kernelShape)),
    stride(std::move(stride)),
    inputPaddingLower(std::move(inputPaddingLower)),
    inputPaddingUpper(std::move(inputPaddingUpper)),
    inputDilation(std::move(inputDilation)),
    kernelPaddingLower(std::move(kernelPaddingLower)),
    kernelPaddingUpper(std::move(kernelPaddingUpper)),
    kernelDilation(std::move(kernelDilation)) {}
  bool operator<(const ConvParams &other) const {
    return std::tie(dType, inputShape, kernelShape, stride, inputPaddingLower,
                    inputPaddingUpper, inputDilation) <
             std::tie(other.dType, other.inputShape, other.kernelShape,
                      other.stride,
                      other.inputPaddingLower, other.inputPaddingUpper,
                      other.inputDilation);
  }
  std::size_t getOutputSize(unsigned dim) const;
  std::size_t getOutputWidth() const;
  std::size_t getOutputHeight() const;
  std::size_t getOutputDepth() const { return kernelShape[2]; }
  std::size_t getInputWidth() const { return inputShape[2]; }
  std::size_t getInputHeight() const { return inputShape[1]; }
  std::size_t getInputDepth() const { return inputShape[3]; }

  std::size_t getBatchSize() const { return inputShape[0]; }
  int getPaddedDilatedInputSize(unsigned dim) const {
    int inputSize = inputShape[1 + dim];
    int dilatedInputSize = (inputSize - 1) * inputDilation[dim] + 1;
    return inputPaddingLower[dim] + dilatedInputSize + inputPaddingUpper[dim];
  }
  int getPaddedDilatedKernelSize(unsigned dim) const {
    int kernelSize = kernelShape[dim];
    int dilatedKernelSize = (kernelSize - 1) * kernelDilation[dim] + 1;
    return kernelPaddingLower[dim] + dilatedKernelSize +
           kernelPaddingUpper[dim];
  }
  std::vector<size_t> getOutputShape() const;

};

uint64_t getFwdFlops(const ConvParams &params);
uint64_t getBwdFlops(const ConvParams &params);
uint64_t getWuFlops(const ConvParams &params);

double
getFwdPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

double
getBwdPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

double
getWuPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

poplar::Tensor
createWeights(poplar::Graph &graph, const ConvParams &params,
              const std::string &name,
              const ConvOptions &options = ConvOptions());

poplar::Tensor
createBiases(poplar::Graph &graph, const poplar::Tensor &acts);

poplar::Tensor
createInput(poplar::Graph &graph, const ConvParams &params,
            const std::string &name,
            const ConvOptions &options = ConvOptions());

poplar::Tensor
convolution(poplar::Graph &graph,
            const poplar::Tensor &in,
            const poplar::Tensor &weights,
            const ConvParams &params,
            bool transposeAndFlipWeights,
            poplar::program::Sequence &prog,
            const std::string &debugPrefix = "",
            const ConvOptions &options = ConvOptions());

void
mapWeights(poplar::Graph &graph, const poplar::Tensor &weights,
           const ConvParams &params,
           const ConvOptions &options = ConvOptions());

void mapBiases(poplar::Graph &graph, const poplar::Tensor &biases,
               const poplar::Tensor &out);

void
weightsTransposeChansFlipXY(poplar::Graph &graph,
                            const poplar::Tensor &weightsIn,
                            const poplar::Tensor &WeightsOut,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "");

poplar::Tensor
calculateWeightDeltas(poplar::Graph &graph, const poplar::Tensor &zDeltas,
                      const poplar::Tensor &activations,
                      const ConvParams &params,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const ConvOptions &options = ConvOptions());

void
convolutionWeightUpdate(poplar::Graph &graph,
                        const poplar::Tensor &zDeltas,
                        const poplar::Tensor &weights,
                        const poplar::Tensor &activations,
                        const ConvParams &params, float learningRate,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const ConvOptions &options = ConvOptions());

void
convolutionBiasUpdate(poplar::Graph &graph, const poplar::Tensor &zDeltas,
                      const poplar::Tensor &biases,
                      float learningRate,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");

void
addBias(poplar::Graph &graph, const poplar::Tensor &acts,
        const poplar::Tensor &biases,
        poplar::program::Sequence &prog,
        const std::string &debugPrefix = "");

poplar::Tensor
fullyConnectedWeightTranspose(poplar::Graph &graph,
                              poplar::Tensor activations,
                              ConvParams params,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix,
                              const ConvOptions &options);

void reportPlanInfo(std::ostream &out, const poplar::Graph &graph,
                    const ConvParams &params,
                    const ConvOptions &options = ConvOptions());

void reportWeightUpdatePlanInfo(std::ostream &out,
                                const poplar::Graph &graph,
                                const poplar::Tensor &activations,
                                const poplar::Tensor &zDeltas,
                                const ConvParams &params,
                                const ConvOptions &options = ConvOptions());

struct Plan;
class PlanningCacheImpl;
class PlanningCache {
public:
  PlanningCache();
  ~PlanningCache();
  friend Plan getPlan(const poplar::Graph &graph, const ConvParams &params,
                      ConvOptions options);
  std::unique_ptr<PlanningCacheImpl> impl;
};

}
#endif  // __popconv_Convolution_hpp__
