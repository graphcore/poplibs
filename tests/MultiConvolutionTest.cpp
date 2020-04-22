// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MultiConvolution.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>

#define BOOST_TEST_MODULE MultiConvolution
#include <boost/test/unit_test.hpp>

#include "TestDevice.hpp"
#include <poplar/Program.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

void fwd_convolution(poplin::ConvParams fwdParams,
                     boost::const_multi_array_ref<double, 3> in,
                     boost::const_multi_array_ref<double, 4> kernel,
                     boost::const_multi_array_ref<double, 1> biases,
                     boost::multi_array_ref<double, 3> out) {
  const auto &fwd = fwdParams;
  const auto &i = fwd.inputTransform;
  const auto &k = fwd.kernelTransform;
  const auto &o = fwd.outputTransform;
  poplibs_test::conv::convolution(
      vectorConvert<unsigned>(fwd.inputFieldShape), i.truncationLower,
      i.truncationUpper, i.dilation, i.paddingLower, i.paddingUpper, i.flip,
      vectorConvert<unsigned>(fwd.kernelShape), k.truncationLower,
      k.truncationUpper, k.dilation, k.paddingLower, k.paddingUpper, k.flip,
      o.truncationLower, o.truncationUpper, o.stride, o.paddingLower,
      o.paddingUpper, in, kernel, biases, out);
}

void bwd_convolution(poplin::ConvParams fwdParams,
                     boost::const_multi_array_ref<double, 3> in,
                     boost::const_multi_array_ref<double, 4> weights,
                     boost::multi_array_ref<double, 3> out) {
  const auto &fwd = fwdParams;
  const auto &i = fwd.inputTransform;
  const auto &k = fwd.kernelTransform;
  const auto &o = fwd.outputTransform;
  poplibs_test::conv::convolutionBackward(
      vectorConvert<unsigned>(fwd.inputFieldShape), i.truncationLower,
      i.truncationUpper, i.dilation, i.paddingLower, i.paddingUpper, i.flip,
      vectorConvert<unsigned>(fwd.kernelShape), k.truncationLower,
      k.truncationUpper, k.dilation, k.paddingLower, k.paddingUpper, k.flip,
      o.truncationLower, o.truncationUpper, o.stride, o.paddingLower,
      o.paddingUpper, in, weights, out);
}

void wu(poplin::ConvParams fwdParams, double learningRate,
        boost::const_multi_array_ref<double, 3> activations,
        boost::const_multi_array_ref<double, 3> deltas,
        boost::multi_array_ref<double, 4> weights,
        boost::multi_array_ref<double, 1> biases) {
  const auto &fwd = fwdParams;
  const auto &i = fwd.inputTransform;
  const auto &k = fwd.kernelTransform;
  const auto &o = fwd.outputTransform;
  poplibs_test::conv::weightUpdate(
      vectorConvert<unsigned>(fwd.inputFieldShape), i.truncationLower,
      i.truncationUpper, i.dilation, i.paddingLower, i.paddingUpper, i.flip,
      vectorConvert<unsigned>(fwd.kernelShape), k.truncationLower,
      k.truncationUpper, k.dilation, k.paddingLower, k.paddingUpper, k.flip,
      o.truncationLower, o.truncationUpper, o.stride, o.paddingLower,
      o.paddingUpper, learningRate, activations, deltas, weights, biases);
}

BOOST_AUTO_TEST_CASE(SimpleMultiConvFwdPassWithSameConvParams) {

  auto device = createTestDevice(TEST_TARGET, 1, 2);
  poplar::Graph graph(device.getTarget());
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  poplin::PlanningCache cache;

  const auto dataType = poplar::FLOAT;
  const auto batchSize = 1;
  const std::vector<std::size_t> inputFieldShape{32, 32};
  const std::vector<std::size_t> kernelShape{3, 3};
  const auto inputChannels = 1;
  const auto outputChannels = 1;
  const auto numConvGroups = 1;
  const auto learningRate = 0.05;

  const auto fwdParams = poplin::ConvParams{
      dataType,      batchSize,      inputFieldShape, kernelShape,
      inputChannels, outputChannels, numConvGroups};
  const auto bwdParams = poplin::getGradientParams(fwdParams);

  const auto outputFieldShape = fwdParams.getOutputFieldShape();

  OptionFlags convOptions;
  auto fwdOptions = convOptions;
  fwdOptions.set("pass", "TRAINING_FWD");
  auto bwdOptions = convOptions;
  bwdOptions.set("pass", "TRAINING_BWD");
  auto wuOptions = convOptions;
  wuOptions.set("pass", "TRAINING_WU");

  auto prevAct =
      poplin::multiconv::createInput(graph,
                                     {{fwdParams, fwdOptions, "prevAct0"},
                                      {fwdParams, fwdOptions, "prevAct1"}},
                                     &cache);
  auto weights =
      poplin::multiconv::createWeights(graph,
                                       {{fwdParams, fwdOptions, "weights0"},
                                        {fwdParams, fwdOptions, "weights1"}},
                                       &cache);

  auto zDeltas =
      poplin::multiconv::createInput(graph,
                                     {{bwdParams, bwdOptions, "zDeltas0"},
                                      {bwdParams, bwdOptions, "zDeltas1"}},
                                     &cache);

  auto fwdProg = Sequence();
  auto revProg = Sequence();

  auto nextAct = poplin::multiconv::convolution(
      graph,
      {{prevAct[0], weights[0], fwdParams, fwdOptions},
       {prevAct[1], weights[1], fwdParams, fwdOptions}},
      fwdProg, "fwd/", &cache);

  // Create transposed/flipped weights for bwd pass
  std::vector<Tensor> bwdWeights{
      poplin::createWeights(graph, bwdParams, "bwdWeights0", bwdOptions,
                            &cache),
      poplin::createWeights(graph, bwdParams, "bwdWeights1", bwdOptions,
                            &cache),
  };
  poplin::weightsTransposeChansFlipXY(graph, weights[0], bwdWeights[0], revProg,
                                      "bwd/");
  poplin::weightsTransposeChansFlipXY(graph, weights[1], bwdWeights[1], revProg,
                                      "bwd/");

  auto prevDeltas = poplin::multiconv::convolution(
      graph,
      {{zDeltas[0], bwdWeights[0], bwdParams, bwdOptions},
       {zDeltas[1], bwdWeights[1], bwdParams, bwdOptions}},
      revProg, "bwd/", &cache);

  auto scale = graph.addConstant(dataType, {}, -learningRate);
  graph.setTileMapping(scale, 0);
  poplin::multiconv::convolutionWeightUpdate(
      graph,
      {{zDeltas[0], weights[0], prevAct[0], scale, fwdParams, wuOptions},
       {zDeltas[1], weights[1], prevAct[1], scale, fwdParams, wuOptions}},
      revProg, "wu/", &cache);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  auto rawHostActivations0 = allocateHostMemoryForTensor(
      prevAct[0], "prevAct0", graph, uploadProg, downloadProg, tmap);
  auto rawHostActivations1 = allocateHostMemoryForTensor(
      prevAct[1], "prevAct1", graph, uploadProg, downloadProg, tmap);

  auto rawHostWeights0 = allocateHostMemoryForTensor(
      weights[0], "weights0", graph, uploadProg, downloadProg, tmap);
  auto rawHostWeights1 = allocateHostMemoryForTensor(
      weights[1], "weights1", graph, uploadProg, downloadProg, tmap);

  auto rawHostNextAct0 = allocateHostMemoryForTensor(
      nextAct[0], "nextAct0", graph, uploadProg, downloadProg, tmap);
  auto rawHostNextAct1 = allocateHostMemoryForTensor(
      nextAct[1], "nextAct1", graph, uploadProg, downloadProg, tmap);

  auto rawHostZDeltas0 = allocateHostMemoryForTensor(
      zDeltas[0], "zDeltas0", graph, uploadProg, downloadProg, tmap);
  auto rawHostZDeltas1 = allocateHostMemoryForTensor(
      zDeltas[1], "zDeltas1", graph, uploadProg, downloadProg, tmap);

  auto rawHostPrevDeltas0 = allocateHostMemoryForTensor(
      prevDeltas[0], "prevDeltas0", graph, uploadProg, downloadProg, tmap);
  auto rawHostPrevDeltas1 = allocateHostMemoryForTensor(
      prevDeltas[1], "prevDeltas1", graph, uploadProg, downloadProg, tmap);

  std::vector<Program> programs;
  programs.push_back(std::move(uploadProg));
  programs.push_back(std::move(fwdProg));
  programs.push_back(std::move(revProg));
  programs.push_back(std::move(downloadProg));

  Engine engine(graph, std::move(programs), {});
  attachStreams(engine, tmap);

  boost::multi_array<double, 3> hostPrevAct0(
      boost::extents[batchSize][inputChannels][product(inputFieldShape)]);
  boost::multi_array<double, 3> hostPrevAct1(
      boost::extents[batchSize][inputChannels][product(inputFieldShape)]);

  boost::multi_array<double, 4> hostWeights0(
      boost::extents[numConvGroups][outputChannels][inputChannels]
                    [product(kernelShape)]);
  boost::multi_array<double, 4> hostWeights1(
      boost::extents[numConvGroups][outputChannels][inputChannels]
                    [product(kernelShape)]);

  boost::multi_array<double, 3> hostNextAct0(
      boost::extents[batchSize][outputChannels][product(outputFieldShape)]);
  boost::multi_array<double, 3> hostNextAct1(
      boost::extents[batchSize][outputChannels][product(outputFieldShape)]);

  std::mt19937 randomEngine;
  auto target = graph.getTarget();
  writeRandomValues(target, dataType, hostPrevAct0, -1.0, +5.0, randomEngine);
  writeRandomValues(target, dataType, hostPrevAct1, -1.0, +5.0, randomEngine);

  writeRandomValues(target, dataType, hostWeights0, -1.0, +7.0, randomEngine);
  writeRandomValues(target, dataType, hostWeights1, -1.0, +7.0, randomEngine);

  copy(target, hostPrevAct0, dataType, rawHostActivations0.get());
  copy(target, hostPrevAct1, dataType, rawHostActivations1.get());
  copy(target, hostWeights0, dataType, rawHostWeights0.get());
  copy(target, hostWeights1, dataType, rawHostWeights1.get());

  // Not actually using biases
  boost::multi_array<double, 1> hostBiases0(boost::extents[outputChannels]);
  boost::multi_array<double, 1> hostBiases1(boost::extents[outputChannels]);
  std::fill(hostBiases0.data(), hostBiases0.data() + hostBiases0.num_elements(),
            0.0);
  std::fill(hostBiases1.data(), hostBiases1.data() + hostBiases1.num_elements(),
            0.0);

  // Run the forward pass.
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0); // Upload
    engine.run(1); // Fwd
    engine.run(3); // Download
  });

  boost::multi_array<double, 3> modelNextAct0(
      boost::extents[batchSize][outputChannels]
                    [product(fwdParams.getOutputFieldShape())]);
  boost::multi_array<double, 3> modelNextAct1(
      boost::extents[batchSize][outputChannels]
                    [product(fwdParams.getOutputFieldShape())]);
  // Actual
  copy(target, dataType, rawHostNextAct0.get(), hostNextAct0);
  copy(target, dataType, rawHostNextAct1.get(), hostNextAct1);

  // Expected
  fwd_convolution(fwdParams, hostPrevAct0, hostWeights0, hostBiases0,
                  modelNextAct0);
  fwd_convolution(fwdParams, hostPrevAct1, hostWeights1, hostBiases1,
                  modelNextAct1);

  double relativeTolerance = 0.01;
  double absoluteTolerance = 1e-6;
  BOOST_CHECK(checkIsClose("fwd0", hostNextAct0, modelNextAct0,
                           relativeTolerance, absoluteTolerance));
  BOOST_CHECK(checkIsClose("fwd1", hostNextAct1, modelNextAct1,
                           relativeTolerance, absoluteTolerance));

  // =====

  boost::multi_array<double, 3> hostZDeltas0(
      boost::extents[batchSize][bwdParams.getNumInputChans()]
                    [product(outputFieldShape)]);
  boost::multi_array<double, 3> hostZDeltas1(
      boost::extents[batchSize][bwdParams.getNumInputChans()]
                    [product(outputFieldShape)]);

  boost::multi_array<double, 3> hostPrevDeltas0(
      boost::extents[batchSize][fwdParams.getNumInputChans()]
                    [product(inputFieldShape)]);
  boost::multi_array<double, 3> hostPrevDeltas1(
      boost::extents[batchSize][fwdParams.getNumInputChans()]
                    [product(inputFieldShape)]);

  // Run the backwards and/or weight update passes.
  writeRandomValues(target, dataType, hostZDeltas0, -3.0, 7.0, randomEngine);
  writeRandomValues(target, dataType, hostZDeltas1, -3.0, 7.0, randomEngine);
  copy(target, hostZDeltas0, dataType, rawHostZDeltas0.get());
  copy(target, hostZDeltas1, dataType, rawHostZDeltas1.get());

  // Run the reverse pass.
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0); // Upload
    engine.run(2); // Rev
    engine.run(3); // Download
  });

  copy(target, dataType, rawHostZDeltas0.get(), hostZDeltas0);
  copy(target, dataType, rawHostZDeltas1.get(), hostZDeltas1);
  copy(target, dataType, rawHostPrevDeltas0.get(), hostPrevDeltas0);
  copy(target, dataType, rawHostPrevDeltas1.get(), hostPrevDeltas1);

  boost::multi_array<double, 3> modelPrevDeltas0(
      boost::extents[batchSize][inputChannels][product(inputFieldShape)]);
  boost::multi_array<double, 3> modelPrevDeltas1(
      boost::extents[batchSize][inputChannels][product(inputFieldShape)]);

  bwd_convolution(fwdParams, hostZDeltas0, hostWeights0, modelPrevDeltas0);
  bwd_convolution(fwdParams, hostZDeltas1, hostWeights1, modelPrevDeltas1);

  BOOST_CHECK(checkIsClose("bwd0", hostPrevDeltas0, modelPrevDeltas0,
                           relativeTolerance, absoluteTolerance));
  BOOST_CHECK(checkIsClose("bwd1", hostPrevDeltas1, modelPrevDeltas1,
                           relativeTolerance, absoluteTolerance));

  auto modelBiases0 = hostBiases0;
  auto modelBiases1 = hostBiases1;
  wu(fwdParams, learningRate, hostPrevAct0, hostZDeltas0, hostWeights0,
     modelBiases0);
  wu(fwdParams, learningRate, hostPrevAct1, hostZDeltas1, hostWeights1,
     modelBiases1);

  copy(target, dataType, rawHostWeights0.get(), hostWeights0);
  copy(target, dataType, rawHostWeights1.get(), hostWeights1);

  BOOST_CHECK(checkIsClose("weights0", hostPrevDeltas0, modelPrevDeltas0,
                           relativeTolerance, absoluteTolerance));
  BOOST_CHECK(checkIsClose("weights1", hostPrevDeltas1, modelPrevDeltas1,
                           relativeTolerance, absoluteTolerance));
}
