#ifndef popconv_Convolution_hpp
#define popconv_Convolution_hpp
#include "poputil/exceptions.hpp"
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
  AMP,
  AUTO
};

const char *asString(const WeightUpdateMethod &method);
std::ostream &operator<<(std::ostream &os, const WeightUpdateMethod &method);
std::istream &operator>>(std::istream &is, WeightUpdateMethod &method);

enum class Pass {
  NONE,
  INFERENCE_FWD,
  TRAINING_FWD,
  TRAINING_BWD,
  TRAINING_WU,
  FC_INFERENCE_FWD,
  FC_TRAINING_FWD,
  FC_TRAINING_BWD,
  FC_TRAINING_WU
};

/** Options to control the implementation of a convolution */
struct ConvOptions {
  WeightUpdateMethod weightUpdateMethod = WeightUpdateMethod::AUTO;
  bool useWinograd = false;
  unsigned winogradPatchSize = 4;
  unsigned percentageCyclesExcessForMemOptim = 0;
  /// The pass this layer corresponds to.
  Pass pass = Pass::NONE;
  poplar::Type partialsType = poplar::FLOAT;
  poplar::Type interTilePartialsType = poplar::FLOAT;
  poplar::Type interIpuPartialsType = poplar::FLOAT;
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
  struct InputTransform {
    // Amount each spatial dimension is truncated before dilation.
    std::vector<unsigned> truncationLower;
    std::vector<unsigned> truncationUpper;
    // Dilation applied to each spatial dimensions after truncation and before
    // padding. Dilation is peformed by placing zeroed elements between the
    // elements of the field.
    std::vector<unsigned> dilation;
    // Padding applied to each spatial dimension after dilation and before
    // flipping.
    std::vector<unsigned> paddingLower;
    std::vector<unsigned> paddingUpper;
    // Whether to flip each spatial dimension. Flipping is applied after
    // padding.
    std::vector<bool> flip;
    InputTransform() = default;
    InputTransform(std::vector<unsigned> truncationLower,
                   std::vector<unsigned> truncationUpper,
                   std::vector<unsigned> dilation,
                   std::vector<unsigned> paddingLower,
                   std::vector<unsigned> paddingUpper,
                   std::vector<bool> flip);
  };
  struct OutputTransform {
    // Amount each spatial dimension is truncated before striding.
    std::vector<unsigned> truncationLower;
    std::vector<unsigned> truncationUpper;
    // Striding applied to each spatial dimension after truncation and before
    // padding.
    std::vector<unsigned> stride;
    // Padding applied to each spatial dimension after striding.
    std::vector<unsigned> paddingLower;
    std::vector<unsigned> paddingUpper;
    OutputTransform() = default;
    OutputTransform(std::vector<unsigned> truncationLower,
                    std::vector<unsigned> truncationUpper,
                    std::vector<unsigned> striding,
                    std::vector<unsigned> paddingLower,
                    std::vector<unsigned> paddingUpper);
  };
  poplar::Type dType;
  // batch size (B)
  std::size_t batchSize;
  // Input field shape for each channel in a batch
  std::vector<std::size_t> inputFieldShape;
  // kernel shape for each channel
  std::vector<std::size_t> kernelShape;
  // input channels (Ci)
  std::size_t inputChannels;
  // output channels (Co)
  std::size_t outputChannels;
  // number of groups in a grouped convolution (G). The input and output
  // channels are divided by G such that G kernels are applied to an input
  // tensors of size {B, {dims}, Ci/G} to produce output tensors of size
  // {B, O{dims}, Co/G}. O{dims} is the output field dimensions
  std::size_t numConvGroups;

  // The transform applied to the input.
  InputTransform inputTransform;
  // The transform applied to the kernel.
  InputTransform kernelTransform;
  // The transform applied to the output.
  OutputTransform outputTransform;
  ConvParams() = default;
  ConvParams(poplar::Type dType,
             std::size_t batchSize,
             std::vector<std::size_t> inputFieldShape,
             std::vector<std::size_t> kernelShape,
             std::size_t inputChannels,
             std::size_t outputChannels,
             std::size_t numConvGroups,

             std::vector<unsigned> inputTruncationLower,
             std::vector<unsigned> inputTruncationUpper,
             std::vector<unsigned> inputDilation,
             std::vector<unsigned> inputPaddingLower,
             std::vector<unsigned> inputPaddingUpper,
             std::vector<bool> flipInput,

             std::vector<unsigned> kernelTruncationLower,
             std::vector<unsigned> kernelTruncationUpper,
             std::vector<unsigned> kernelDilation,
             std::vector<unsigned> kernelPaddingLower,
             std::vector<unsigned> kernelPaddingUpper,
             std::vector<bool> flipKernel,

             std::vector<unsigned> outputTruncationLower,
             std::vector<unsigned> outputTruncationUpper,
             std::vector<unsigned> stride,
             std::vector<unsigned> outputPaddingLower,
             std::vector<unsigned> outputPaddingUpper);

  /// Return the size of the output of the convolution operation, before
  /// output transformations are applied.
  std::size_t getUntransformedOutputSize(unsigned dim) const;
  /// Return the size of the output.
  std::size_t getOutputSize(unsigned dim) const;
  std::size_t getNumOutputChansPerConvGroup() const { return outputChannels;}
  std::size_t getNumOutputChans() const {
    return outputChannels * numConvGroups;
  }
  std::size_t getInputSize(unsigned dim) const { return inputFieldShape[dim]; }
  std::size_t getNumInputChansPerConvGroup() const { return inputChannels; }
  std::size_t getNumInputChans() const {
    return inputChannels * numConvGroups;
  }
  std::size_t getNumConvGroups() const { return numConvGroups; }
  std::size_t getNumFieldDims() const { return inputFieldShape.size(); }
  std::size_t getBatchSize() const { return batchSize; }

  unsigned getTransformedInputSize(unsigned dim) const;
  unsigned getTransformedKernelSize(unsigned dim) const;
  // Returns the shape of the output field
  std::vector<size_t> getOutputFieldShape() const;
};

inline bool operator<(const ConvParams::InputTransform &a,
                      const ConvParams::InputTransform &b) {
  return std::tie(a.truncationLower,
                  a.truncationUpper,
                  a.dilation,
                  a.paddingLower,
                  a.paddingUpper,
                  a.flip) <
         std::tie(b.truncationLower,
                  b.truncationUpper,
                  b.dilation,
                  b.paddingLower,
                  b.paddingUpper,
                  b.flip);
}

inline bool operator==(const ConvParams::InputTransform &a,
                       const ConvParams::InputTransform &b) {
  return std::tie(a.truncationLower,
                  a.truncationUpper,
                  a.dilation,
                  a.paddingLower,
                  a.paddingUpper,
                  a.flip) ==
         std::tie(b.truncationLower,
                  b.truncationUpper,
                  b.dilation,
                  b.paddingLower,
                  b.paddingUpper,
                  b.flip);
}

inline bool operator!=(const ConvParams::InputTransform &a,
                       const ConvParams::InputTransform &b) {
  return !(a == b);
}

inline bool operator<(const ConvParams::OutputTransform &a,
                      const ConvParams::OutputTransform &b) {
  return std::tie(a.truncationLower,
                  a.truncationUpper,
                  a.stride,
                  a.paddingLower,
                  a.paddingUpper) <
         std::tie(b.truncationLower,
                  b.truncationUpper,
                  b.stride,
                  b.paddingLower,
                  b.paddingUpper);
}

inline bool operator==(const ConvParams::OutputTransform &a,
                       const ConvParams::OutputTransform &b) {
  return std::tie(a.truncationLower,
                  a.truncationUpper,
                  a.stride,
                  a.paddingLower,
                  a.paddingUpper) ==
         std::tie(b.truncationLower,
                  b.truncationUpper,
                  b.stride,
                  b.paddingLower,
                  b.paddingUpper);
}

inline bool operator!=(const ConvParams::OutputTransform &a,
                       const ConvParams::OutputTransform &b) {
  return !(a == b);
}

inline bool operator<(const ConvParams &a, const ConvParams &b) {
  return std::tie(a.dType,
                  a.batchSize,
                  a.inputFieldShape,
                  a.kernelShape,
                  a.inputChannels,
                  a.outputChannels,
                  a.numConvGroups,
                  a.inputTransform,
                  a.kernelTransform,
                  a.outputTransform) <
         std::tie(b.dType,
                  b.batchSize,
                  b.inputFieldShape,
                  b.kernelShape,
                  b.inputChannels,
                  b.outputChannels,
                  b.numConvGroups,
                  b.inputTransform,
                  b.kernelTransform,
                  b.outputTransform);
}

inline bool operator==(const ConvParams &a, const ConvParams &b) {
  return std::tie(a.dType,
                  a.batchSize,
                  a.inputFieldShape,
                  a.kernelShape,
                  a.inputChannels,
                  a.outputChannels,
                  a.numConvGroups,
                  a.inputTransform,
                  a.kernelTransform,
                  a.outputTransform) ==
         std::tie(b.dType,
                  b.batchSize,
                  b.inputFieldShape,
                  b.kernelShape,
                  b.inputChannels,
                  b.outputChannels,
                  b.numConvGroups,
                  b.inputTransform,
                  b.kernelTransform,
                  b.outputTransform);
}

inline bool operator!=(const ConvParams &a, const ConvParams &b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream &os, const ConvParams &p);

uint64_t getFwdFlops(const ConvParams &params);
uint64_t getBwdFlops(const ConvParams &params);
uint64_t getWuFlops(const ConvParams &params);

double
getFwdPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

double
getBwdPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

double
getWuPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

/** Create a weight tensor suitable for use with convolution()
 *
 * The shape of the tensor will be [convGroups x outChans x inChans x H x W]
 *
 * \param graph   The tensor will be added to this graph
 * \param params  The same parameters as used by the convolution()
 * \param name    Debugging name for the tensor
 * \param options Options controlling the implementation
 * \return        The weights tensor suitable for use with convolution()
 */
poplar::Tensor
createWeights(poplar::Graph &graph, const ConvParams &params,
              const std::string &name,
              const ConvOptions &options = ConvOptions());

/** Create a bias tensor suitable for input to addBias() function
 *
 * The tensor will have the shape [outChans]
 *
 * \param graph  The tensor will be added to this graph
 * \param acts   The activation tensor which is output from the convolution
 * \param name   Debugging name for the tensor
 * \return       The tensor of biases
 */
poplar::Tensor
createBiases(poplar::Graph &graph, const poplar::Tensor &acts,
             const std::string &name = "biases");

/** Create an input tensor for a convolution
 *
 * Use this when required to create an input data tensor for a convoution. The
 * same set of parameters which will be passed to the convolution() should also
 * be passed to createInput()
 *
 * The returned tensor has the shape [B x inChans x H x W].
 *
 * \param graph    The tensor will be added to this graph
 * \param params   Parameters as passed to the target convolution.
 * \param name     Debugging name for the tensor
 * \param options  Options controlling the implementation
 * \return         The allocated input tensor
 */
poplar::Tensor
createInput(poplar::Graph &graph, const ConvParams &params,
            const std::string &name,
            const ConvOptions &options = ConvOptions());

/** Convolve an input with a set of weights.
 *
 * This is for a 2D convolution.
 *
 * The input tensor is in the form [B x inChans x H x W], and can be allocated
 * using createInput().  The weights tensor is in the form
 * [convGroups x outChans x inChans x H x W], and can be allocated using
 * createWeights().
 *
 * Padding and striding are specified in the ConvParams structure.
 *
 * \param graph                   The operation will be added to this graph
 * \param in                      Input data tensor
 * \param weights                 Weights tensor
 * \param params                  Parameters for the form of the convolution
 * \param transposeAndFlipWeights For the weight update pass
 * \param prog                    Poplar program sequence to append to op onto
 * \param debugPrefix             Name of the operation, for debugging
 * \param options                 Options that control the implementation
 * \return                        The convolved output tensor
 */
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
                      float learningRate, const poplar::Type &partialsType,
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

// creates a tensor pair of batch normalisation parameters (gamma, beta)
std::pair<poplar::Tensor, poplar::Tensor>
createBatchNormParams(poplar::Graph &graph, const poplar::Tensor &acts);

// Estimates estimates from a batch. The two tensors returned are:
// 1) mean
// 2) inverse of standard deviation
std::pair<poplar::Tensor, poplar::Tensor>
batchNormEstimates(poplar::Graph &graph,
                   const poplar::Tensor &actsUngrouped,
                   float eps,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const std::string &debugPrefix = "");

// Computes and returns the following given mean and inverse of standard
// deviation
// 1) whitened activations
// 2) output of batch normalisation
std::pair<poplar::Tensor, poplar::Tensor>
batchNormalise(poplar::Graph &graph,
               const poplar::Tensor &acts,
               const poplar::Tensor &gamma,
               const poplar::Tensor &beta,
               const poplar::Tensor &mean,
               const poplar::Tensor &invStdDev,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");

// Computes the output of batch normalisation given
//  combinedMultiplicand = gamma / stdDev
//  addend = beta - gamma * mean / stdDev
poplar::Tensor
batchNormalise(poplar::Graph &graph,
               const poplar::Tensor &acts,
               const poplar::Tensor &combinedMultiplicand,
               const poplar::Tensor &addend,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");

// Compute deltas required for both input gradient and parameter
// update computations
std::pair<poplar::Tensor, poplar::Tensor>
batchNormDeltas(poplar::Graph &graph,
                const poplar::Tensor &actsWhitened,
                const poplar::Tensor &gradsIn,
                poplar::program::Sequence &prog,
                const poplar::Type &partialsType = poplar::FLOAT,
                const std::string &debugPrefix = "");


poplar::Tensor
batchNormGradients(poplar::Graph &graph,
                   const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   const poplar::Tensor &gammaDelta,
                   const poplar::Tensor &betaDelta,
                   const poplar::Tensor &invStdDev,
                   const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const std::string &debugPrefix = "");

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
#endif // popconv_Convolution_hpp
