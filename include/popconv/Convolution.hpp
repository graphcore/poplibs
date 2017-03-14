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

/** Options to control the implementation of a convolution */
struct ConvOptions {
  WeightUpdateMethod weightUpdateMethod = WeightUpdateMethod::AUTO;
  bool useWinograd = false;
  unsigned winogradPatchSize = 4;
  unsigned percentageCyclesExcessForMemOptim = 0;
  // Aviod rearrangement of left hand side argument between convolution and
  // weight delta calculation.
  bool noLHSRearrangement = false;
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

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned strideY, unsigned strideX, unsigned paddingY,
             unsigned paddingX);

uint64_t getFwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX, unsigned paddingY,
                     unsigned paddingX, unsigned outNumChans);

uint64_t getBwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX, unsigned paddingY,
                     unsigned paddingX, unsigned outNumChans);

uint64_t getWuFlops(unsigned batchSize,
                    unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSizeY, unsigned kernelSizeX,
                    unsigned strideY, unsigned strideX, unsigned paddingY,
                    unsigned paddingX, unsigned outNumChans);

double getFwdPerfectCycleCount(const poplar::Graph &graph,
                               std::string dType,
                               unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned inNumChans,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX,
                               unsigned outNumChans);

double getBwdPerfectCycleCount(const poplar::Graph &graph,
                               std::string dType,
                               unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned inNumChans,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX,
                               unsigned outNumChans);

double getWuPerfectCycleCount(const poplar::Graph &graph,
                              std::string dType,
                              unsigned batchSize,
                              unsigned inDimY, unsigned inDimX,
                              unsigned inNumChans,
                              unsigned kernelSizeY, unsigned kernelSizeX,
                              unsigned strideY, unsigned strideX,
                              unsigned paddingY, unsigned paddingX,
                              unsigned outNumChans);

poplar::Tensor
createWeights(poplar::Graph &graph, const poplar::Tensor &in,
              unsigned kernelSizeY, unsigned kernelSizeX, unsigned outNumChans,
              unsigned strideY, unsigned strideX,
              unsigned paddingY, unsigned paddingX,
              bool isFractional,
              const ConvOptions &options);

poplar::Tensor
createBiases(poplar::Graph &graph, std::string dType,
             unsigned outNumChans);

poplar::program::Program
convolution(poplar::Graph &graph,
            const std::vector<unsigned> &stride,
            const std::vector<unsigned> &padding,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor out, const std::string &partialsType,
            bool isFractional, bool transposeAndFlipWeights,
            const std::string &debugPrefix = "",
            const ConvOptions &options = ConvOptions());

inline poplar::program::Program
convolution(poplar::Graph &graph,
            unsigned strideY, unsigned strideX,
            unsigned paddingY, unsigned paddingX,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor out, const std::string &partialsType,
            bool isFractional, bool transposeAndFlipWeights,
            const std::string &debugPrefix = "",
            const ConvOptions &options = ConvOptions()) {
  return convolution(graph, {strideY, strideX}, {paddingY, paddingX},
                     in, weights, biases, out, partialsType, isFractional,
                     transposeAndFlipWeights, debugPrefix, options);
}

void mapActivations(poplar::Graph &graph,
                    const poplar::Tensor &in,
                    const poplar::Tensor &weights,
                    unsigned strideY, unsigned strideX,
                    unsigned paddingY, unsigned paddingX,
                    bool isFractional,
                    const ConvOptions &options);

void
mapWeights(poplar::Tensor w, poplar::Graph &graph, const poplar::Tensor &in,
           unsigned strideY, unsigned strideX,
           unsigned paddingY, unsigned paddingX, bool isFractional,
           const ConvOptions &options);

void mapBiases(poplar::Tensor b, poplar::Graph &graph,
               poplar::Tensor activations);

poplar::program::Program
weightsTransposeChansFlipXY(poplar::Graph &graph,
                            poplar::Tensor weightsIn,
                            poplar::Tensor WeightsOut,
                            const std::string &debugPrefix = "");

poplar::Tensor
calculateWeightDeltas(poplar::Graph &graph, poplar::Tensor zDeltas,
                      unsigned kernelSizeY, unsigned kernelSizeX,
                      poplar::Tensor activations,
                      const std::vector<unsigned> &stride,
                      const std::vector<unsigned> &padding,
                      bool isFractional,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const ConvOptions &options = ConvOptions());

poplar::program::Program
convolutionWeightUpdate(poplar::Graph &graph,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        const std::vector<unsigned> &stride,
                        const std::vector<unsigned> &padding,
                        bool isFractional, float learningRate,
                        const std::string &debugPrefix = "",
                        const ConvOptions &options = ConvOptions());

inline poplar::program::Program
convolutionWeightUpdate(poplar::Graph &graph,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        unsigned strideY, unsigned strideX, unsigned paddingY,
                        unsigned paddingX, bool isFractional,
                        float learningRate,
                        const std::string &debugPrefix = "",
                        const ConvOptions &options = ConvOptions()) {
  return convolutionWeightUpdate(graph, zDeltas, weights,
                                 biases, activations, {strideY, strideX},
                                 {paddingY, paddingX}, isFractional,
                                 learningRate,
                                 debugPrefix, options);
}


struct Plan;
class PlanningCacheImpl;
class PlanningCache {
public:
  PlanningCache();
  ~PlanningCache();
  friend Plan getPlan(const poplar::Graph &graph,
                      std::string dType,
                      std::vector<std::size_t> inShape,
                      std::vector<std::size_t> weightsShape,
                      std::vector<std::size_t> stride,
                      std::vector<std::size_t> padding,
                      unsigned numChannels, bool isFractional,
                      bool isWeightUpdate,
                      ConvOptions options);
  std::unique_ptr<PlanningCacheImpl> impl;
};

}
#endif  // __popconv_Convolution_hpp__
