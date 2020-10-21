// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/print.hpp"
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <exception>
#include <fstream>
#include <istream>
#include <ostream>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/MultiArray.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/GeneralMatrixAdd.hpp>
#include <poplibs_test/Multirate.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/MultiConvolution.hpp>
#include <poplin/codelets.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/exceptions.hpp>
#include <random>

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using poplibs_test::Pass;
using namespace poplibs_support;

using Array1d = boost::multi_array<double, 1>;
using Array3d = boost::multi_array<double, 3>;
using Array4d = boost::multi_array<double, 4>;
using Array1dRef = boost::multi_array_ref<double, 1>;
using Array3dRef = boost::multi_array_ref<double, 3>;

struct OctConvUserParams {
  Type inputType;
  Type outputType;
  std::size_t batchSize;
  std::vector<std::size_t> inputFieldShape;
  std::vector<std::size_t> kernelShape;
  std::size_t inChansPerConvGroup;
  std::size_t outChansPerConvGroup;
  std::size_t numConvGroups;
};

// OctConv input and output shapes
struct OctConvData {
  std::string name;
  std::vector<std::size_t> shape;
  std::size_t numFieldElements;
  std::size_t chansPerConvGroup;
  Tensor tensor;
  std::shared_ptr<Array3d> host;
  std::shared_ptr<Array3d> model; // only applicable for output

  OctConvData(std::string name, std::vector<std::size_t> shape,
              std::size_t chansPerConvGroup)
      : name(name), shape(shape), numFieldElements(product(shape)),
        chansPerConvGroup(chansPerConvGroup) {}
};

// Parameters for each component convolution
struct ConvParameters {
  std::string name;
  poplin::ConvParams fwdParams;
  poplin::ConvParams bwdParams;
  unsigned preDownsamplingRate;
  unsigned postUpsamplingRate;

  // OctConv input data (i.e., before pre-downsampling)
  std::shared_ptr<OctConvData> inputData;

  // OctConv output data (i.e., after post-upsampling)
  std::shared_ptr<OctConvData> outputData;

  ConvParameters(std::string name, poplin::ConvParams fwdParams,
                 poplin::ConvParams bwdParams, unsigned preDnsampling,
                 unsigned postUpsampling,
                 std::shared_ptr<OctConvData> inputData,
                 std::shared_ptr<OctConvData> outputData)
      : name(name), fwdParams(fwdParams), bwdParams(bwdParams),
        preDownsamplingRate(preDnsampling), postUpsamplingRate(postUpsampling),
        inputData(inputData), outputData(outputData) {}
};

// Input shapes for a particular frequency
struct InputFreqParameters {
  std::string name;
  std::size_t numChans;
  std::vector<std::size_t> fShape;
};

// Output shapes for a particular frequency
struct OutputFreqParameters {
  std::string name;
  std::size_t numChans;
  std::shared_ptr<OctConvData> data;
};

static std::vector<poplar::Tensor>
createAllWeights(poplar::Graph &graph,
                 const std::vector<poplin::multiconv::CreateTensorArgs> &args,
                 const poplar::OptionFlags &multiConvOptions,
                 poplin::PlanningCache *cache) {
  std::vector<Tensor> weights;
  for (size_t i = 0; i < args.size(); i++) {
    weights.push_back(poplin::multiconv::createWeights(
        graph, args, i, multiConvOptions, cache));
  }
  return weights;
}

static std::vector<poplar::Tensor>
createAllInputs(poplar::Graph &graph,
                const std::vector<poplin::multiconv::CreateTensorArgs> &args,
                const poplar::OptionFlags &multiConvOptions,
                poplin::PlanningCache *cache) {
  std::vector<Tensor> inputs;
  for (size_t i = 0; i < args.size(); i++) {
    inputs.push_back(poplin::multiconv::createInput(graph, args, i,
                                                    multiConvOptions, cache));
  }
  return inputs;
}

// Upsample or downsample a tensor shape of arbitrary number of dimensions
// by an integer factor
static std::vector<std::size_t>
TensorShapeRateChange(const std::vector<std::size_t> &inShape,
                      const unsigned factor, const bool up) {
  if (!up &&
      std::any_of(inShape.begin(), inShape.end(),
                  [factor](const std::size_t v) { return v % factor; })) {
    throw poputil::poplibs_error("At least one field dimension is not "
                                 "divisible by downsampling factor " +
                                 std::to_string(factor));
  }
  std::vector<std::size_t> outShape(inShape);
  std::transform(inShape.begin(), inShape.end(), outShape.begin(),
                 [factor, up](const std::size_t &v) {
                   return up ? (v * factor) : (v / factor);
                 });
  return outShape;
}

//
// The component convolutions are listed in the following table.
//
//    name  |pool2x|num-in-channels|field-size|num-out-channels|upsample2x
//  --------+------+---------------+----------+----------------+-----------
//   FreqLL |  -   |    a_i  * n_i |   f_lo   |    a_o  * n_o  |    -
//   FreqLH |  -   |    a_i  * n_i |   f_lo   | (1-a_o) * n_o  |    X
//   FreqHL |  X   | (1-a_i) * n_i |   f_lo   |    a_o  * n_o  |    -
//   FreqHH |  -   | (1-a_i) * n_i |   f_hi   | (1-a_o) * n_o  |    -
//
// where
//  - a_i and a_o denote alpha_in and alpha_out resply.
//  - n_i and n_o denote number of input channels and output channels resply.
//  - f_lo and f_hi denote field sizes for lo and hi frequencies resply.
//
// As shown above, there are two special cases:
//  - FreqLH convolution generates a smaller output field size (f_lo) than
//    the natural low frequency field size. Hence upsampling by 2x must be
//    applied on the output before it can be added to FreqHH output.
//  - FreqHL convolution uses a smaller input field size (f_lo) than the
//    natural high frequency field size. Hence average-pooling 2x is required
//    prior to the convolution.
//
static std::tuple<std::vector<ConvParameters>,
                  std::vector<std::shared_ptr<OctConvData>>,
                  std::vector<std::shared_ptr<OctConvData>>>
splitConvParamsByFrequency(
    const OctConvUserParams &userParam,
    const poplin::ConvParams::InputTransform &inputTransform,
    const poplin::ConvParams::InputTransform &kernelTransform,
    const poplin::ConvParams::OutputTransform &outputTransform,
    const double alphaIn, const double alphaOut) {
  auto bSize = userParam.batchSize;
  auto nCGroups = userParam.numConvGroups;
  unsigned samplingRate = 2;
  std::vector<ConvParameters> convParams;
  std::vector<std::shared_ptr<OctConvData>> octInputData;
  std::vector<std::shared_ptr<OctConvData>> octOutputData;

  // Calculate Low Frequency parameters
  auto pooledFieldShape =
      TensorShapeRateChange(userParam.inputFieldShape, samplingRate, false);
  std::size_t inChansLo =
      static_cast<std::size_t>(alphaIn * userParam.inChansPerConvGroup);
  std::size_t outChansLo =
      static_cast<std::size_t>(alphaOut * userParam.outChansPerConvGroup);

  // Enumerate shapes for both low and high frequency
  enum FreqType { LO, HI };
  std::map<FreqType, InputFreqParameters> octInputFreq = {
      {LO, {"Lo", inChansLo, pooledFieldShape}},
      {HI,
       {"Hi", userParam.inChansPerConvGroup - inChansLo,
        userParam.inputFieldShape}}};
  std::map<FreqType, OutputFreqParameters> octOutputFreq = {
      {LO, {"Lo", outChansLo, nullptr}},
      {HI, {"Hi", userParam.outChansPerConvGroup - outChansLo, nullptr}}};

  // Iterate over Low and High frequency input shapes
  for (auto &inData : octInputFreq) {
    auto inType = inData.first;
    auto &in = inData.second;
    if (in.numChans > 0) {
      // Allocate input oct-conv data. Model data is not required for input
      auto allocOctData =
          [nCGroups, bSize](const std::string name,
                            const std::size_t nChansPGrp,
                            std::vector<std::size_t> &shape,
                            std::vector<std::shared_ptr<OctConvData>> &storage,
                            const bool allocModel) {
            auto data = std::make_shared<OctConvData>(name, shape, nChansPGrp);
            storage.push_back(data);
            auto nChans = nChansPGrp * nCGroups;
            auto fSize = product(shape);
            data->host =
                std::make_shared<Array3d>(boost::extents[bSize][nChans][fSize]);
            if (allocModel) {
              data->model = std::make_shared<Array3d>(
                  boost::extents[bSize][nChans][fSize]);
            }
            return data;
          };
      allocOctData("freq" + in.name, in.numChans, in.fShape, octInputData,
                   false);

      // Iterate over Low and High frequency output shapes
      for (auto &outData : octOutputFreq) {
        auto outType = outData.first;
        auto &out = outData.second;
        if (out.numChans > 0) {
          auto postUpsampling =
              (inType == LO) && (outType == HI) ? samplingRate : 1;
          auto preDnsampling =
              (inType == HI) && (outType == LO) ? samplingRate : 1;
          auto fShape = (inType == HI) && (outType == HI)
                            ? octInputFreq[HI].fShape
                            : octInputFreq[LO].fShape;
          auto fwdParams = poplin::ConvParams{userParam.inputType,
                                              userParam.outputType,
                                              bSize,
                                              fShape,
                                              userParam.kernelShape,
                                              in.numChans,
                                              out.numChans,
                                              nCGroups,
                                              inputTransform,
                                              kernelTransform,
                                              outputTransform};
          auto bwdParams = getGradientParams(fwdParams);
          auto outFieldShape = fwdParams.getOutputFieldShape();
          if (postUpsampling > 1) {
            outFieldShape =
                TensorShapeRateChange(outFieldShape, postUpsampling, true);
          }
          if (out.data == nullptr) {
            // allocate input oct-conv data
            out.data = allocOctData("freq" + out.name, out.numChans,
                                    outFieldShape, octOutputData, true);
          } else if (product(outFieldShape) != out.data->numFieldElements) {
            throw poputil::poplibs_error(
                "Output field shape for convolution-" + in.name + out.name +
                " is not compatible with OctConv output-" + out.name);
          }
          convParams.emplace_back("freq" + in.name + out.name, fwdParams,
                                  bwdParams, preDnsampling, postUpsampling,
                                  octInputData.back(), out.data);
        }
      }
    }
  }
  return {convParams, octInputData, octOutputData};
}

static void
writeRandomValues(const poplar::Target &target, const poplar::Type type,
                  std::vector<std::shared_ptr<OctConvData>> &octConv,
                  const double min, const double max,
                  std::mt19937 &randomEngine) {
  for (unsigned i = 0; i < octConv.size(); ++i) {
    writeRandomValues(target, type, *(octConv[i]->host), min, max,
                      randomEngine);
  }
}

static void copy(const poplar::Target &target, const poplar::Type &type,
                 const std::vector<std::shared_ptr<OctConvData>> &octConv,
                 const std::vector<std::unique_ptr<char[]>> &dst) {
  for (unsigned i = 0; i < octConv.size(); ++i) {
    copy(target, *(octConv[i]->host), type, dst[i].get());
  }
}

static void copy(const poplar::Target &target, const poplar::Type &type,
                 const std::vector<std::unique_ptr<char[]>> &src,
                 const std::vector<std::shared_ptr<OctConvData>> &octConv) {
  for (unsigned i = 0; i < octConv.size(); ++i) {
    copy(target, type, src[i].get(), *(octConv[i]->host));
  }
}

static std::vector<Array3d>
createMultiArrayInput(const std::vector<ConvParameters> &octParams,
                      const bool bwdFlag) {
  auto numConvolutions = octParams.size();
  std::vector<Array3d> multiArray;
  for (unsigned i = 0; i < numConvolutions; ++i) {
    auto params = bwdFlag ? octParams[i].bwdParams : octParams[i].fwdParams;
    auto numConvGroups = params.getNumConvGroups();
    auto numInChans = params.getNumInputChansPerConvGroup() * numConvGroups;
    auto inputFieldShape = params.getInputFieldShape();
    auto batchSize = params.getBatchSize();
    auto numInFieldElems = product(inputFieldShape);
    multiArray.push_back(
        Array3d(boost::extents[batchSize][numInChans][numInFieldElems]));
  }
  return multiArray;
}

static std::vector<Array3d>
createMultiArrayOutput(const std::vector<ConvParameters> &params) {
  auto numConvolutions = params.size();
  std::vector<Array3d> multiArray;
  for (unsigned i = 0; i < numConvolutions; ++i) {
    auto numConvGroups = params[i].fwdParams.getNumConvGroups();
    auto numOutChans =
        params[i].fwdParams.getNumOutputChansPerConvGroup() * numConvGroups;
    auto batchSize = params[i].fwdParams.getBatchSize();
    auto numOutFieldElems = product(params[i].fwdParams.getOutputFieldShape());
    multiArray.push_back(
        Array3d(boost::extents[batchSize][numOutChans][numOutFieldElems]));
  }
  return multiArray;
}

static std::vector<Array4d>
createMultiArrayWeights(const std::vector<ConvParameters> &params,
                        const std::vector<std::size_t> kernelShape) {
  auto numConvolutions = params.size();
  std::vector<Array4d> multiArray;
  for (unsigned i = 0; i < numConvolutions; ++i) {
    auto numConvGroups = params[i].fwdParams.getNumConvGroups();
    auto numInChans = params[i].fwdParams.getNumInputChansPerConvGroup();
    auto numOutChans = params[i].fwdParams.getNumOutputChansPerConvGroup();
    auto numKernelElems = product(kernelShape);
    multiArray.push_back(Array4d(boost::extents[numConvGroups][numOutChans]
                                               [numInChans][numKernelElems]));
  }
  return multiArray;
}

// Create OctConv input tensors for each frequency
static void
createOctConvTensors(poplar::Graph &graph, poplin::PlanningCache &cache,
                     const OctConvUserParams &userParam,
                     const poplin::ConvParams::InputTransform &inputTransform,
                     const poplin::ConvParams::InputTransform &kernelTransform,
                     const poplin::ConvParams::OutputTransform &outputTransform,
                     const OptionFlags &fwdOptions,
                     const OptionFlags &bwdOptions,
                     std::vector<std::shared_ptr<OctConvData>> &octData,
                     const poplar::OptionFlags &multiConvOptions) {
  auto numConvGroups = userParam.numConvGroups;
  auto batchSize = userParam.batchSize;

  // Create up to two Oct-convolution inputs for Low and High Frequencies
  // respectively.
  using CreateTensorArgs = poplin::multiconv::CreateTensorArgs;
  std::vector<CreateTensorArgs> prevActArgs(octData.size());
  for (unsigned i = 0; i < prevActArgs.size(); ++i) {
    prevActArgs[i].params = poplin::ConvParams{userParam.inputType,
                                               userParam.outputType,
                                               batchSize,
                                               octData[i]->shape,
                                               userParam.kernelShape,
                                               octData[i]->chansPerConvGroup,
                                               userParam.outChansPerConvGroup,
                                               numConvGroups,
                                               inputTransform,
                                               kernelTransform,
                                               outputTransform};
    prevActArgs[i].options = fwdOptions;
    prevActArgs[i].name = octData[i]->name + "_prevAct";
  }

  // Create input tensors and model multiarray
  auto prevAct = createAllInputs(graph, prevActArgs, multiConvOptions, &cache);
  for (unsigned i = 0; i < octData.size(); ++i) {
    octData[i]->tensor = prevAct[i];
  }
}

// Create tensors for weights and zDeltas for forward, backward and weight
// update passes.
static std::tuple<std::vector<Tensor>, std::vector<Tensor>>
createConvInputTensors(poplar::Graph &graph, poplin::PlanningCache &cache,
                       const std::vector<ConvParameters> &params,
                       const OptionFlags &fwdOptions,
                       const OptionFlags &bwdOptions, const bool createZDeltas,
                       const poplar::OptionFlags &multiConvOptions) {
  // Create the required number of convolution inputs
  auto numConvolutions = params.size();
  using CreateTensorArgs = poplin::multiconv::CreateTensorArgs;
  std::vector<CreateTensorArgs> weightsArgs(numConvolutions);
  std::vector<CreateTensorArgs> zDeltasArgs(numConvolutions);
  for (unsigned i = 0; i < numConvolutions; ++i) {
    weightsArgs[i].name = params[i].name + "_weights";
    weightsArgs[i].params = params[i].fwdParams;
    weightsArgs[i].options = fwdOptions;
    zDeltasArgs[i].name = params[i].name + "_zDeltas";
    zDeltasArgs[i].params = params[i].bwdParams;
    zDeltasArgs[i].options = bwdOptions;
  }
  auto weights = createAllWeights(graph, weightsArgs, multiConvOptions, &cache);
  std::vector<Tensor> zDeltas;
  if (createZDeltas) {
    zDeltas = createAllInputs(graph, zDeltasArgs, multiConvOptions, &cache);
  }
  return {weights, zDeltas};
}

static std::vector<poplin::multiconv::ConvolutionArgs>
getConvolutionArguments(const std::vector<Tensor> &inA,
                        const std::vector<Tensor> &inB, const bool bwdFlag,
                        const std::vector<ConvParameters> &octParams,
                        const OptionFlags &options) {
  std::vector<poplin::multiconv::ConvolutionArgs> convArgs;
  auto numConvolutions = octParams.size();
  for (unsigned i = 0; i < numConvolutions; ++i) {
    poplin::ConvParams params =
        bwdFlag ? octParams[i].bwdParams : octParams[i].fwdParams;
    convArgs.push_back({inA[i], inB[i], params, options});
  }
  return convArgs;
}

static std::vector<poplin::multiconv::ConvWeightUpdateArgs>
getConvolutionWeightUpdateArguments(
    const std::vector<Tensor> &prevAct, const std::vector<Tensor> &zDeltas,
    const std::vector<Tensor> &weights, const Tensor &scale,
    const std::vector<ConvParameters> &octParams, const OptionFlags &options) {
  std::vector<poplin::multiconv::ConvWeightUpdateArgs> wuArgs;
  auto numConvolutions = octParams.size();
  for (unsigned i = 0; i < numConvolutions; ++i) {
    wuArgs.push_back({zDeltas[i], weights[i], prevAct[i], scale,
                      octParams[i].fwdParams, options});
  }
  return wuArgs;
}

static std::vector<std::unique_ptr<char[]>>
allocateOctHostMemory(Graph &graph, const std::string name,
                      const std::vector<ConvParameters> &params,
                      const std::vector<Tensor> &tensors,
                      boost::optional<Sequence &> uploadProg,
                      boost::optional<Sequence &> downloadProg,
                      std::vector<std::pair<std::string, char *>> &tmap) {
  std::vector<std::unique_ptr<char[]>> rawHostMem(tensors.size());
  for (unsigned i = 0; i < tensors.size(); ++i) {
    auto prefix = params[i].name;
    rawHostMem[i] = allocateHostMemoryForTensor(
        tensors[i], prefix + name, graph, uploadProg, downloadProg, tmap);
  }
  return rawHostMem;
}

// Upsample tensor by a given factor along every dimension
static Tensor upsampleTensor(Graph &graph, Tensor &input,
                             const unsigned samplingRate,
                             const std::size_t numFieldDims) {
  auto inputShape = input.shape();
  std::vector<std::size_t> upSamplingPartial(inputShape);
  auto size = product<std::size_t>(inputShape);
  std::size_t elemsToDupl = 1;
  std::size_t numDupl = size / elemsToDupl;
  Tensor inShuffled = input;
  int currDim = upSamplingPartial.size() - 1;
  int firstFieldDim = upSamplingPartial.size() - numFieldDims;
  while (currDim >= firstFieldDim) {
    upSamplingPartial[currDim] *= samplingRate;
    inShuffled = inShuffled.reshape({numDupl, elemsToDupl})
                     .broadcast(samplingRate, 1)
                     .reshape(upSamplingPartial);
    elemsToDupl *= upSamplingPartial[currDim];
    numDupl = product<std::size_t>(upSamplingPartial) / elemsToDupl;
    currDim--;
  }
  return inShuffled;
}

// Use Average-Pooling to sub-sample tensor by a given factor along every
// dimension.
static Tensor downsampleTensor(Graph &graph, const Tensor &input,
                               const unsigned samplingRate,
                               const std::size_t numFieldDims, Sequence &prog,
                               const OptionFlags &poolOptions = {}) {
  auto inputShape = input.shape();
  popnn::PoolingType poolingType = popnn::PoolingType::AVG;
  std::vector<std::size_t> inputFieldShape;
  std::vector<std::size_t> kernelShape;
  std::vector<unsigned> stride;
  std::vector<int> paddingLower;
  std::vector<int> paddingUpper;
  unsigned firstFieldDim = inputShape.size() - numFieldDims;
  for (unsigned i = firstFieldDim; i < inputShape.size(); ++i) {
    inputFieldShape.push_back(inputShape[i]);
    kernelShape.push_back(samplingRate);
    stride.push_back(samplingRate);
    paddingLower.push_back(0);
    paddingUpper.push_back(0);
  }
  const auto poolParams = popnn::pooling::PoolParams(
      poolingType, inputFieldShape, kernelShape, stride, paddingLower,
      paddingUpper, inputShape[1], inputShape[0], input.elementType());
  return popnn::pooling::pool(graph, poolParams, input, prog, "", poolOptions);
}

static std::vector<Tensor> preProcess(Graph &graph,
                                      const std::vector<ConvParameters> &params,
                                      Sequence &prog,
                                      const OptionFlags poolOptions = {}) {
  std::vector<Tensor> output(params.size());
  for (unsigned i = 0; i < params.size(); ++i) {
    auto &input = params[i].inputData->tensor;
    auto samplingRate = params[i].preDownsamplingRate;
    if (samplingRate > 1) {
      auto numFieldDims = params[i].fwdParams.getNumFieldDims();
      output[i] = downsampleTensor(graph, input, samplingRate, numFieldDims,
                                   prog, poolOptions);
    } else {
      output[i] = input;
    }
  }
  return output;
}

static void postProcess(Graph &graph, std::vector<ConvParameters> &params,
                        std::vector<Tensor> &input, Sequence &prog) {
  for (unsigned i = 0; i < params.size(); ++i) {
    Tensor tensor = input[i];
    if (params[i].postUpsamplingRate > 1) {
      auto name = params[i].name + "_upsampled";
      tensor = upsampleTensor(graph, input[i], params[i].postUpsamplingRate,
                              params[i].fwdParams.getNumFieldDims());
    }

    // Accumulate outputs of individual convolution paths which are of the same
    // shape.
    if (!params[i].outputData->tensor.valid()) {
      params[i].outputData->tensor = tensor;
    } else {
      params[i].outputData->tensor =
          popops::add(graph, params[i].outputData->tensor, tensor, prog,
                      params[i].outputData->name);
    }
  }
  return;
}

static void convolutionForwardModel(const std::vector<ConvParameters> &params,
                                    std::vector<Array3d> &prevAct,
                                    std::vector<Array4d> &weights,
                                    std::vector<Array3d> &nextAct,
                                    Array1dRef &bias) {
  for (unsigned i = 0; i < nextAct.size(); ++i) {
    auto fieldShape = params[i].fwdParams.getInputFieldShape();
    auto kernelShape = params[i].fwdParams.getKernelShape();
    auto &inTransf = params[i].fwdParams.inputTransform;
    auto &kTransf = params[i].fwdParams.kernelTransform;
    auto &outTransf = params[i].fwdParams.outputTransform;
    poplibs_test::conv::convolution(
        vectorConvert<unsigned>(fieldShape), inTransf.truncationLower,
        inTransf.truncationUpper, inTransf.dilation, inTransf.paddingLower,
        inTransf.paddingUpper, inTransf.flip,
        vectorConvert<unsigned>(kernelShape), kTransf.truncationLower,
        kTransf.truncationUpper, kTransf.dilation, kTransf.paddingLower,
        kTransf.paddingUpper, kTransf.flip, outTransf.truncationLower,
        outTransf.truncationUpper, outTransf.stride, outTransf.paddingLower,
        outTransf.paddingUpper, prevAct[i], weights[i], bias, nextAct[i]);
  }
}

static void convolutionBackwardModel(const std::vector<ConvParameters> &params,
                                     std::vector<Array3d> &zDeltas,
                                     std::vector<Array4d> &weights,
                                     std::vector<Array3d> &prevDeltas) {
  for (unsigned i = 0; i < prevDeltas.size(); ++i) {
    auto fieldShape = params[i].fwdParams.getInputFieldShape();
    auto kernelShape = params[i].fwdParams.getKernelShape();
    auto &inTransf = params[i].fwdParams.inputTransform;
    auto &kTransf = params[i].fwdParams.kernelTransform;
    auto &outTransf = params[i].fwdParams.outputTransform;
    poplibs_test::conv::convolutionBackward(
        vectorConvert<unsigned>(fieldShape), inTransf.truncationLower,
        inTransf.truncationUpper, inTransf.dilation, inTransf.paddingLower,
        inTransf.paddingUpper, inTransf.flip,
        vectorConvert<unsigned>(kernelShape), kTransf.truncationLower,
        kTransf.truncationUpper, kTransf.dilation, kTransf.paddingLower,
        kTransf.paddingUpper, kTransf.flip, outTransf.truncationLower,
        outTransf.truncationUpper, outTransf.stride, outTransf.paddingLower,
        outTransf.paddingUpper, zDeltas[i], weights[i], prevDeltas[i]);
  }
}

static void weightUpdateModel(const std::vector<ConvParameters> &params,
                              const std::vector<Array3d> &prevAct,
                              const std::vector<Array3d> &zDeltas,
                              const std::vector<Array4d> &weights,
                              const double learningRate,
                              const Array1dRef &bias) {
  for (unsigned i = 0; i < weights.size(); ++i) {
    auto fieldShape = params[i].fwdParams.getInputFieldShape();
    auto kernelShape = params[i].fwdParams.getKernelShape();
    auto &inTransf = params[i].fwdParams.inputTransform;
    auto &kTransf = params[i].fwdParams.kernelTransform;
    auto &outTransf = params[i].fwdParams.outputTransform;
    poplibs_test::conv::weightUpdate(
        vectorConvert<unsigned>(fieldShape), inTransf.truncationLower,
        inTransf.truncationUpper, inTransf.dilation, inTransf.paddingLower,
        inTransf.paddingUpper, inTransf.flip,
        vectorConvert<unsigned>(kernelShape), kTransf.truncationLower,
        kTransf.truncationUpper, kTransf.dilation, kTransf.paddingLower,
        kTransf.paddingUpper, kTransf.flip, outTransf.truncationLower,
        outTransf.truncationUpper, outTransf.stride, outTransf.paddingLower,
        outTransf.paddingUpper, learningRate, prevAct[i], zDeltas[i],
        weights[i], bias);
  }
}

static void PreProcessModel(const std::vector<ConvParameters> &params,
                            std::vector<Array3d> &outputs) {
  for (unsigned i = 0; i < params.size(); ++i) {
    auto &input = params[i].inputData->host;
    if (params[i].preDownsamplingRate > 1) {
      auto inShape = params[i].fwdParams.getInputFieldShape();
      poplibs_test::downsample(inShape, params[i].preDownsamplingRate, *input,
                               outputs[i]);
    } else {
      outputs[i] = *input;
    }
  }
}

static void PostProcessModel(std::vector<ConvParameters> &params,
                             const std::size_t numConvGroups,
                             const std::size_t batchSize,
                             const std::vector<Array3d> &inputs) {
  for (unsigned i = 0; i < params.size(); ++i) {
    auto &output = params[i].outputData->model;
    auto numOutElems = params[i].outputData->numFieldElements;
    auto numOutChans =
        params[i].fwdParams.getNumOutputChansPerConvGroup() * numConvGroups;
    Array3d tensor(boost::extents[batchSize][numOutChans][numOutElems]);
    if (params[i].postUpsamplingRate > 1) {
      auto outShape = params[i].fwdParams.getOutputFieldShape();
      poplibs_test::upsample(outShape, params[i].postUpsamplingRate, inputs[i],
                             tensor);
    } else {
      tensor = inputs[i];
    }
    // Accumulate outputs of individual convolution paths which are of the same
    // shape.
    if (output->size() == 0) {
      *output = tensor;
    } else {
      auto outputRef = Array3dRef(
          output->data(), boost::extents[batchSize][numOutChans][numOutElems]);
      auto tensorRef = Array3dRef(
          tensor.data(), boost::extents[batchSize][numOutChans][numOutElems]);
      poplibs_test::axpby::add(outputRef, tensorRef, outputRef);
    }
  }
}

static void printTensors(const std::string name,
                         const std::vector<ConvParameters> &params,
                         const std::vector<Tensor> &tensors, Sequence &prog) {
  for (unsigned i = 0; i < tensors.size(); ++i) {
    prog.add(PrintTensor(name + params[i].name, tensors[i]));
  }
}

template <unsigned N>
static bool
multipleCheckIsClose(const std::vector<ConvParameters> params,
                     const std::vector<boost::multi_array<double, N>> &inA,
                     const std::vector<boost::multi_array<double, N>> &inB,
                     const double relativeTol, const double absoluteTol) {
  for (unsigned i = 0; i < inA.size(); ++i) {
    if (!checkIsClose(params[i].name, inA[i], inB[i], relativeTol, absoluteTol))
      return false;
  }
  return true;
}

static bool
multipleCheckIsClose(const std::vector<std::shared_ptr<OctConvData>> &data,
                     const double relativeTol, const double absoluteTol) {
  for (unsigned i = 0; i < data.size(); ++i) {
    if (!checkIsClose(data[i]->name, *(data[i]->host), *(data[i]->model),
                      relativeTol, absoluteTol))
      return false;
  }
  return true;
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  bool doPrintTensors = false;
  unsigned fwdInChansPerConvGroup;
  unsigned fwdOutChansPerConvGroup;
  ShapeOption<std::size_t> inputFieldShapeOption;
  ShapeOption<std::size_t> kernelShapeOption;
  unsigned numConvGroups = 1;
  ShapeOption<unsigned> truncationLowerOption, truncationUpperOption,
      truncationOption;
  ShapeOption<unsigned> inDilationOption;
  ShapeOption<unsigned> paddingLowerOption, paddingUpperOption, paddingOption;
  ShapeOption<bool> flipInputOption;
  ShapeOption<unsigned> kernelTruncationLowerOption,
      kernelTruncationUpperOption, kernelTruncationOption;
  ShapeOption<unsigned> kernelDilationOption;
  ShapeOption<unsigned> kernelPaddingLowerOption, kernelPaddingUpperOption,
      kernelPaddingOption;
  ShapeOption<bool> flipKernelOption;
  ShapeOption<unsigned> outputTruncationOption, outputTruncationLowerOption,
      outputTruncationUpperOption;
  ShapeOption<unsigned> strideOption;
  ShapeOption<unsigned> outputPaddingOption, outputPaddingLowerOption,
      outputPaddingUpperOption;
  unsigned batchSize;
  Type inputType;
  Type outputType;
  double absoluteTolerance, relativeTolerance;
  double alpha = 1.0;
  double alphaIn = 1.0;
  double alphaOut = 1.0;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  bool reportPlan;
  bool reportVarStorage;
  bool remapOutputTensor;
  bool poolOptimizeForSpeed;
  std::string multiConvOptionsString;

  Pass pass = Pass::ALL;
  poplin::PlanningCache cache;

  boost::optional<std::string> jsonProfileOut;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling report")
    ("use-unstable-format", "Use the unstable profile format")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("print",
     po::value<bool>(&doPrintTensors)->default_value(doPrintTensors),
     "Print the tensors")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("input-channels", po::value<unsigned>(&fwdInChansPerConvGroup)->required(),
     "Number of input channels per grouped convolution")
    ("output-channels",
     po::value<unsigned>(&fwdOutChansPerConvGroup)->required(),
     "Number of output channels per grouped convolution")
    ("field",
     po::value<ShapeOption<std::size_t>>(&inputFieldShapeOption)->required(),
      "Field size")
    ("kernel-size",
      po::value<ShapeOption<std::size_t>>(&kernelShapeOption)->default_value(1),
     "Size of square kernel. If set, it is an error to also set either "
     "kernel-height and/or kernel-width")
    ("data-type",
     po::value<Type>(&inputType)->default_value(HALF),
     "Type of the input and output data")
    ("input-type",
     po::value<Type>(&inputType),
     "Type of the input data")
    ("output-type",
     po::value<Type>(&outputType),
     "Type of the output data and the parameters")
    ("truncation",
     po::value<ShapeOption<unsigned>>(&truncationOption)->default_value(0),
     "Amount to truncate the start and end of each dimension of the input")
    ("truncation-upper",
     po::value<ShapeOption<unsigned>>(&truncationUpperOption)->default_value(0),
     "Amount to truncate the end of each dimension of the input")
    ("truncation-lower",
     po::value<ShapeOption<unsigned>>(&truncationLowerOption)->default_value(0),
     "Amount to truncate the start of each dimension of the input")
    ("in-dilation",
     po::value<ShapeOption<unsigned>>(&inDilationOption)->default_value(1),
     "Input dilation")
    ("padding",
     po::value<ShapeOption<unsigned>>(&paddingOption)->default_value(0),
     "Amount of zero padding to add to the start and end of each dimension")
    ("padding-upper",
     po::value<ShapeOption<unsigned>>(&paddingUpperOption)->default_value(0),
     "Amount of zero padding to add at the end of each dimension")
    ("padding-lower",
     po::value<ShapeOption<unsigned>>(&paddingLowerOption)->default_value(0),
     "Amount of zero padding to add at the start of each dimension")
    ("flip-input",
     po::value<ShapeOption<bool>>(&flipInputOption)->default_value(false),
     "Whether to flip each input spatial field")
    ("kernel-truncation",
     po::value<ShapeOption<unsigned>>(&kernelTruncationOption)
         ->default_value(0),
     "Amount to truncate the start and end of each dimension of the kernel")
    ("kernel-truncation-upper",
     po::value<ShapeOption<unsigned>>(&kernelTruncationUpperOption)
         ->default_value(0),
     "Amount to truncate the end of each dimension of the kernel")
    ("kernel-truncation-lower",
     po::value<ShapeOption<unsigned>>(&kernelTruncationLowerOption)
         ->default_value(0),
     "Amount to truncate the start of each dimension of the kernel")
    ("kernel-dilation",
     po::value<ShapeOption<unsigned>>(&kernelDilationOption)
         ->default_value(1),
     "Kernel dilation")
    ("kernel-padding",
     po::value<ShapeOption<unsigned>>(&kernelPaddingOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the start and end of each"
     "dimension")
    ("kernel-padding-upper",
     po::value<ShapeOption<unsigned>>(&kernelPaddingUpperOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the start of each dimension")
    ("kernel-padding-lower",
     po::value<ShapeOption<unsigned>>(&kernelPaddingLowerOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the end of each dimension")
    ("flip-kernel",
     po::value<ShapeOption<bool>>(&flipKernelOption)->default_value(false),
     "Whether to flip each kernel spatial field")
    ("output-truncation",
     po::value<ShapeOption<unsigned>>(&outputTruncationOption)
         ->default_value(0),
     "Number of output elements to truncate")
    ("output-truncation-upper",
     po::value<ShapeOption<unsigned>>(&outputTruncationUpperOption)
         ->default_value(0),
     "Number of output elements to truncate at the end of each dimension")
    ("output-truncation-lower",
     po::value<ShapeOption<unsigned>>(&outputTruncationLowerOption)
         ->default_value(0),
     "Number of output elements to truncate at the start of each dimension")
    ("stride",
     po::value<ShapeOption<unsigned>>(&strideOption)->default_value(1),
     "Stride")
    ("output-padding",
     po::value<ShapeOption<unsigned>>(&outputPaddingOption)->default_value(0),
     "Number of output elements to truncate")
    ("output-padding-upper",
     po::value<ShapeOption<unsigned>>(&outputPaddingUpperOption)
         ->default_value(0),
     "Number of output elements to truncate at the end of each dimension")
    ("output-padding-lower",
     po::value<ShapeOption<unsigned>>(&outputPaddingLowerOption)
         ->default_value(0),
     "Number of output elements to truncate at the start of each dimension")
    ("alpha",
     po::value<double>(&alpha),
     "Fraction of input channels to use for Lower Frequency")
    ("alpha-in",
     po::value<double>(&alphaIn),
     "Fraction of input channels to use for Lower Frequency")
    ("alpha-out",
     po::value<double>(&alphaOut),
     "Fraction of output channels to use for Lower Frequency")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("plan-only", "Only plan the requested passes, don't build or run a graph")
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("absolute-tolerance",
     po::value<double>(&absoluteTolerance),
     "Absolute tolerance to use when validating results against the reference "
     "model")
    ("ipus",
     po::value<unsigned>(&numIPUs)->default_value(numIPUs),
     "Number of IPUs")
    ("tiles-per-ipu", po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("workers-per-tile",
     po::value<unsigned>(),
     "Number of worker contexts per tile")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("conv-groups",
     po::value<unsigned>(&numConvGroups)->default_value(1),
     "Number of convolution groups in grouped convolution")
    ("report-plan", po::value<bool>(&reportPlan)->default_value(false),
     "Display plan")
    ("report-var-storage",
     po::value<bool>(&reportVarStorage)->default_value(false),
     "Report variable storage information")
    ("remap-output-tensor",
     po::value<bool>(&remapOutputTensor)->default_value(false),
     "Remap output tensor if layout is detected to be poor")
    ("pool-optimize-for-speed",
     po::value<bool>(&poolOptimizeForSpeed)->default_value(false),
     "Optimize any pooling operation for speed, not memory")
    ("options", po::value<std::string>(&multiConvOptionsString),
    "Options to use for the multi-convolution, specified as a JSON string, "
    "e.g. {\"key\":\"value\"}")
    ;

  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      std::cout << "A multi-dimensional shape can be specified using a brace "
                   "enclosed comma\n"
                   "separated list, for example --stride={1,2}. You may also "
                   "specify a single\n"
                   "number without braces in which case that value is used for "
                   "each dimension,\n"
                   "for example --stride=2\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  OptionFlags poolOptions = {
      {"optimizeForSpeed", poolOptimizeForSpeed ? "true" : "false"}};
  if (alpha != 1.0) {
    alphaIn = alphaOut = alpha;
  }

  // alpha parameter range checking
  if ((alphaIn < 0.0) || (alphaOut < 0.0) || (alphaIn > 1.0) ||
      (alphaOut > 1.0)) {
    std::cerr << "error: alphaIn (" << alphaIn << ") and alphaOut (" << alphaOut
              << ") must be within the interval [0, 1]\n";
    return 1;
  }
  auto &inputFieldShape = inputFieldShapeOption.val;
  const auto numFieldDims = inputFieldShape.size();

  kernelShapeOption.broadcast(numFieldDims);
  auto &kernelShape = kernelShapeOption.val;

  struct UpperLowerOption {
    ShapeOption<unsigned> &lowerOption;
    ShapeOption<unsigned> &upperOption;
    std::string name;
  } upperLowerOptionTriples[] = {
      {paddingLowerOption, paddingUpperOption, "padding"},
      {truncationLowerOption, truncationUpperOption, "truncation"},
      {kernelTruncationLowerOption, kernelTruncationUpperOption,
       "kernel-truncation"},
      {kernelPaddingLowerOption, kernelPaddingUpperOption, "kernel-padding"},
      {outputTruncationLowerOption, outputTruncationUpperOption,
       "output-truncation"},
      {outputPaddingLowerOption, outputPaddingUpperOption, "output-padding"}};
  for (const auto &entry : upperLowerOptionTriples) {
    if (!vm[entry.name].defaulted()) {
      std::string conflictingOptions[] = {entry.name + "-lower",
                                          entry.name + "-upper"};
      for (auto option : conflictingOptions) {
        if (!vm[option].defaulted()) {
          std::cerr << "--" << entry.name << " as well as --";
          std::cerr << option << " set\n";
          return 1;
        }
      }
      entry.lowerOption = vm[entry.name].as<ShapeOption<unsigned>>();
      entry.upperOption = vm[entry.name].as<ShapeOption<unsigned>>();
    }
    entry.lowerOption.broadcast(numFieldDims);
    entry.upperOption.broadcast(numFieldDims);
  }
  auto &truncationLower = truncationLowerOption.val;
  auto &truncationUpper = truncationUpperOption.val;
  auto &paddingLower = paddingLowerOption.val;
  auto &paddingUpper = paddingUpperOption.val;

  auto &outputTruncationLower = outputTruncationLowerOption.val;
  auto &outputTruncationUpper = outputTruncationUpperOption.val;
  auto &outputPaddingLower = outputPaddingLowerOption.val;
  auto &outputPaddingUpper = outputPaddingUpperOption.val;

  auto &kernelTruncationLower = kernelTruncationLowerOption.val;
  auto &kernelTruncationUpper = kernelTruncationUpperOption.val;
  auto &kernelPaddingLower = kernelPaddingLowerOption.val;
  auto &kernelPaddingUpper = kernelPaddingUpperOption.val;

  inDilationOption.broadcast(numFieldDims);
  auto &inDilation = inDilationOption.val;
  flipInputOption.broadcast(numFieldDims);
  auto &flipInput = flipInputOption.val;

  kernelDilationOption.broadcast(numFieldDims);
  auto &kernelDilation = kernelDilationOption.val;
  flipKernelOption.broadcast(numFieldDims);
  auto &flipKernel = flipKernelOption.val;

  strideOption.broadcast(numFieldDims);
  auto &stride = strideOption.val;
  const auto fwdOutChans = fwdOutChansPerConvGroup * numConvGroups;

  const bool planOnly = vm.count("plan-only");
  const bool inferenceOnly = vm.count("inference-only");
  const bool ignoreData = vm.count("ignore-data");
  const bool useUnstableFormat = vm.count("use-unstable-format");

  bool doFwdPass = pass == Pass::ALL || pass == Pass::FWD;
  bool doBwdPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::BWD);
  bool doWuPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::WU);

  if ((vm["output-type"].empty() != vm["input-type"].empty()) ||
      (!vm["data-type"].defaulted() && !vm["output-type"].empty())) {
    throw poputil::poplibs_error("Please specify either --data-type OR "
                                 "(--input-type AND --output-type), not both.");
  }
  if (vm["output-type"].empty()) {
    outputType = inputType;
  }

  if (vm["tolerance"].empty()) {
    if (outputType == FLOAT) {
      relativeTolerance = FLOAT_REL_TOL;
    } else {
      relativeTolerance = HALF_REL_TOL;
    }
  }
  if (vm["absolute-tolerance"].empty()) {
    if (outputType == FLOAT) {
      absoluteTolerance = FLOAT_ABS_TOL;
    } else {
      absoluteTolerance = HALF_ABS_TOL;
    }
  }

  OptionFlags multiConvOptions;
  if (!multiConvOptionsString.empty()) {
    poplar::readJSON(multiConvOptionsString, multiConvOptions);
  }

  auto dev = [&]() -> TestDevice {
    if (isIpuModel(deviceType)) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      IPUModel ipuModel(deviceTypeToIPUName(deviceType));
      ipuModel.numIPUs = numIPUs;
      if (vm.count("profile") || jsonProfileOut) {
        ipuModel.compileIPUCode = true;
      }
      if (vm.count("workers-per-tile"))
        ipuModel.numWorkerContexts = vm["workers-per-tile"].as<unsigned>();
      if (tilesPerIPU)
        ipuModel.tilesPerIPU = *tilesPerIPU;
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      if (tilesPerIPU)
        return createTestDevice(deviceType, numIPUs, *tilesPerIPU);
      else
        return createTestDeviceFullSize(deviceType, numIPUs);
    }
  }();

  Graph graph(dev.getTarget());
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);
  const poplin::ConvParams::InputTransform inputTransform{
      truncationLower, truncationUpper, inDilation,
      paddingLower,    paddingUpper,    flipInput};
  const poplin::ConvParams::InputTransform kernelTransform{
      kernelTruncationLower, kernelTruncationUpper, kernelDilation,
      kernelPaddingLower,    kernelPaddingUpper,    flipKernel};
  const poplin::ConvParams::OutputTransform outputTransform{
      outputTruncationLower, outputTruncationUpper, stride, outputPaddingLower,
      outputPaddingUpper};
  const OctConvUserParams userParams{inputType,
                                     outputType,
                                     batchSize,
                                     inputFieldShape,
                                     kernelShape,
                                     fwdInChansPerConvGroup,
                                     fwdOutChansPerConvGroup,
                                     numConvGroups};
  OptionFlags convOptions;
  convOptions.set(
      {{"remapOutputTensor", remapOutputTensor ? "true" : "false"}});
  auto fwdOptions = convOptions;
  fwdOptions.set("pass", inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  auto bwdOptions = convOptions;
  bwdOptions.set("pass", "TRAINING_BWD");
  auto wuOptions = convOptions;
  wuOptions.set("pass", "TRAINING_WU");
  std::vector<ConvParameters> octConvParams;
  std::vector<std::shared_ptr<OctConvData>> octInData;
  std::vector<std::shared_ptr<OctConvData>> octOutData;
  std::tie(octConvParams, octInData, octOutData) =
      splitConvParamsByFrequency(userParams, inputTransform, kernelTransform,
                                 outputTransform, alphaIn, alphaOut);
  if (reportPlan) {
    for (unsigned i = 0; i < octConvParams.size(); ++i) {
      const auto outFieldShape =
          octConvParams[i].fwdParams.getOutputFieldShape();
      std::cout << "Convolution parameters for " << i
                << "'th :\n"
                   " Batch size: "
                << octConvParams[i].fwdParams.batchSize
                << "\n"
                   " Kernel:"
                << octConvParams[i].fwdParams.kernelShape
                << "\n"
                   " Stride:"
                << octConvParams[i].fwdParams.outputTransform.stride
                << "\n"
                   " Padding Lower: "
                << octConvParams[i].fwdParams.inputTransform.paddingLower
                << "\n"
                   " Padding Upper: "
                << octConvParams[i].fwdParams.inputTransform.paddingUpper
                << "\n"
                   " Group size: "
                << octConvParams[i].fwdParams.numConvGroups
                << "\n"
                   " Input: "
                << octConvParams[i].fwdParams.inputChannelsPerConvGroup << "x"
                << octConvParams[i].fwdParams.inputFieldShape
                << "\n"
                   " Output: "
                << octConvParams[i].fwdParams.outputChannelsPerConvGroup << "x"
                << outFieldShape << "\n";

      if (doFwdPass) {
        std::cout << "Forward plan:\n";
        poplin::reportPlanInfo(std::cout, graph, octConvParams[i].fwdParams,
                               fwdOptions, &cache);
        std::cout << "Forward FLOPs: "
                  << getFwdFlops(octConvParams[i].fwdParams) << "\n";
      }

      if (doBwdPass) {
        std::cout << "Backward plan:\n";
        poplin::reportPlanInfo(std::cout, graph, octConvParams[i].bwdParams,
                               bwdOptions, &cache);
        std::cout << "Backward FLOPs: "
                  << getBwdFlops(octConvParams[i].bwdParams) << "\n";
      }

      if (doWuPass) {
        std::cout << "WU plan:\n";
        poplin::reportWeightUpdatePlanInfo(
            std::cout, graph, octConvParams[i].fwdParams, wuOptions, &cache);
        std::cout << "WU FLOPs: " << getWuFlops(octConvParams[i].fwdParams)
                  << "\n";
      }
    }
  }
  if (planOnly) {
    return 0;
  }
  // Create OctConv inputs and outputs for target as well as model
  createOctConvTensors(graph, cache, userParams, inputTransform,
                       kernelTransform, outputTransform, fwdOptions, bwdOptions,
                       octInData, multiConvOptions);
  std::vector<Tensor> octConvInput;
  for (auto octConvIn : octInData) {
    octConvInput.push_back(octConvIn->tensor);
  }

  // Create and individual convolution inputs
  std::vector<Tensor> weights, zDeltas;
  std::tie(weights, zDeltas) = createConvInputTensors(
      graph, cache, octConvParams, fwdOptions, bwdOptions,
      (doBwdPass || doWuPass), multiConvOptions);
  auto fwdProg = Sequence();

  // Convert OctConv inputs to individual convolution prevAct tensors.
  auto prevAct = preProcess(graph, octConvParams, fwdProg, poolOptions);
  auto fwdArgs = getConvolutionArguments(prevAct, weights, false, octConvParams,
                                         fwdOptions);
  auto nextAct = poplin::multiconv::convolution(
      graph, fwdArgs, false, fwdProg, "fwd", multiConvOptions, &cache);
  std::vector<Tensor> octConvOutput;
  auto revProg = Sequence();
  if (doFwdPass) {
    // Generate OctConv outputs
    postProcess(graph, octConvParams, nextAct, fwdProg);
    for (auto octConvOut : octOutData) {
      octConvOutput.push_back(octConvOut->tensor);
    }
  } else {
    fwdProg = Sequence();
    // Generate convolution inputs
    prevAct = preProcess(graph, octConvParams, revProg, poolOptions);
  }
  const auto learningRate = 0.05;
  std::vector<Tensor> prevDeltas;
  if (doBwdPass) {
    auto bwdArgs = getConvolutionArguments(zDeltas, weights, true,
                                           octConvParams, bwdOptions);
    prevDeltas = poplin::multiconv::convolution(
        graph, bwdArgs, true, revProg, "bwd", multiConvOptions, &cache);
  }
  if (doWuPass) {
    auto scale = graph.addConstant(weights[0].elementType(), {}, -learningRate);
    graph.setTileMapping(scale, 0);
    auto wuArgs = getConvolutionWeightUpdateArguments(
        prevAct, zDeltas, weights, scale, octConvParams, wuOptions);
    poplin::multiconv::convolutionWeightUpdate(graph, wuArgs, revProg, "wu",
                                               multiConvOptions, &cache);
  }
  if (doPrintTensors) {
    if (doFwdPass || doWuPass) {
      printTensors("octConvInput", octConvParams, octConvInput, fwdProg);
      printTensors("prevAct", octConvParams, prevAct, fwdProg);
    }
    printTensors("weights", octConvParams, weights, fwdProg);
    if (doFwdPass) {
      printTensors("nextAct", octConvParams, nextAct, fwdProg);
      printTensors("octConvOutput", octConvParams, octConvOutput, fwdProg);
    }
    if (doBwdPass || doWuPass) {
      printTensors("zDeltas", octConvParams, zDeltas, revProg);
    }
    if (doBwdPass) {
      printTensors("prevDeltas", octConvParams, prevDeltas, revProg);
    }
  }

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostOctConvInput =
      allocateOctHostMemory(graph, "octConvInput", octConvParams, octConvInput,
                            uploadProg, boost::none, tmap);
  auto rawHostWeights = allocateOctHostMemory(
      graph, "weights", octConvParams, weights, uploadProg, downloadProg, tmap);
  auto rawHostOctConvOutput =
      allocateOctHostMemory(graph, "octConvOutput", octConvParams,
                            octConvOutput, boost::none, downloadProg, tmap);
  std::vector<std::unique_ptr<char[]>> rawHostZDeltas;
  std::vector<std::unique_ptr<char[]>> rawHostPrevDeltas;
  if (doBwdPass || doWuPass) {
    rawHostZDeltas =
        allocateOctHostMemory(graph, "zDeltas", octConvParams, zDeltas,
                              uploadProg, downloadProg, tmap);
  }
  if (doBwdPass) {
    rawHostPrevDeltas =
        allocateOctHostMemory(graph, "prevDeltas", octConvParams, prevDeltas,
                              boost::none, downloadProg, tmap);
  }

  std::vector<Program> programs;
  const auto fwdProgIndex = programs.size(); // 0
  programs.push_back(std::move(fwdProg));
  const auto revProgIndex = programs.size(); // 1
  programs.push_back(std::move(revProg));
  const auto uploadProgIndex = programs.size(); // 2
  programs.push_back(std::move(uploadProg));
  const auto downloadProgIndex = programs.size(); // 3
  programs.push_back(std::move(downloadProg));

  OptionFlags engineOptions;
  if (vm.count("profile") || jsonProfileOut) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (useUnstableFormat) {
      engineOptions.set("profiler.useUnstableFormat", "true");
    }
  }

  Engine engine(graph, std::move(programs), engineOptions);
  attachStreams(engine, tmap);

  auto hostPrevAct = createMultiArrayInput(octConvParams, false);
  auto hostWeights = createMultiArrayWeights(octConvParams, kernelShape);
  auto hostZDeltas = createMultiArrayInput(octConvParams, true);
  auto hostPrevDeltas = createMultiArrayInput(octConvParams, false);
  Array1d hostBiases(boost::extents[fwdOutChans]);
  std::fill(hostBiases.begin(), hostBiases.end(), 0);
  std::mt19937 randomEngine;
  auto target = graph.getTarget();
  writeRandomValues(target, inputType, octInData, -1.0, +5.0, randomEngine);
  copy(target, inputType, octInData, rawHostOctConvInput);
  writeRandomValues<double, 4>(target, inputType, hostWeights, -1.0, +7.0,
                               randomEngine);
  copy<double, 4>(target, inputType, hostWeights, rawHostWeights);
  if (doBwdPass || doWuPass) {
    writeRandomValues<double, 3>(target, inputType, hostZDeltas, -3.0, +7.0,
                                 randomEngine);
    copy<double, 3>(target, inputType, hostZDeltas, rawHostZDeltas);
  }

  // Run all configured convolution passes and pre/post processings.
  dev.bind([&](const Device &d) {
    engine.load(d);
    if (!ignoreData) {
      engine.run(uploadProgIndex);
    }
    if (doFwdPass) {
      engine.run(fwdProgIndex);
    }
    if (doBwdPass || doWuPass) {
      engine.run(revProgIndex);
    }
    if (!ignoreData) {
      engine.run(downloadProgIndex);
    }
  });

  // Validate against a reference model.
  bool matchesModel = true;
  if (!ignoreData) {
    PreProcessModel(octConvParams, hostPrevAct);
    auto modelNextAct = createMultiArrayOutput(octConvParams);
    convolutionForwardModel(octConvParams, hostPrevAct, hostWeights,
                            modelNextAct, hostBiases);
    if (doFwdPass) {
      copy(target, outputType, rawHostOctConvOutput, octOutData);
      PostProcessModel(octConvParams, numConvGroups, batchSize, modelNextAct);
      matchesModel &= multipleCheckIsClose(octOutData, relativeTolerance,
                                           absoluteTolerance);
    }
  }

  if (doBwdPass || doWuPass) {
    // Save biases and weights before they are overwritten with device contents.
    auto modelBiases = hostBiases;
    auto modelWeights = hostWeights;
    copy<double, 3>(target, outputType, rawHostZDeltas, hostZDeltas);
    if (doBwdPass) {
      // Copy result of Backward Pass from device
      copy<double, 3>(target, outputType, rawHostPrevDeltas, hostPrevDeltas);
    }
    if (!ignoreData) {
      // Validate against a reference model.
      if (doBwdPass) {
        auto modelPrevDeltas = createMultiArrayInput(octConvParams, false);
        convolutionBackwardModel(octConvParams, hostZDeltas, modelWeights,
                                 modelPrevDeltas);
        matchesModel &= multipleCheckIsClose<3>(
            octConvParams, hostPrevDeltas, modelPrevDeltas, relativeTolerance,
            absoluteTolerance);
      }
      if (doWuPass) {
        // Copy result of Weight Update Pass from device
        copy<double, 4>(target, inputType, rawHostWeights, hostWeights);
        weightUpdateModel(octConvParams, hostPrevAct, hostZDeltas, modelWeights,
                          learningRate, modelBiases);
        matchesModel &=
            multipleCheckIsClose<4>(octConvParams, hostWeights, modelWeights,
                                    relativeTolerance, absoluteTolerance);
      }
    }
  }
  if (!ignoreData && !matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  if (jsonProfileOut) {
    const auto pr = engine.getProfile();
    std::ofstream os(*jsonProfileOut);
    poplar::serializeToJSON(os, pr);
  }
  if (vm.count("profile")) {
    auto reportOptions = OptionFlags{{"showExecutionSteps", "true"}};
    if (reportVarStorage) {
      reportOptions.set("showVarStorage", "true");
    }
    engine.printProfileSummary(std::cout, reportOptions);
  }
  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
