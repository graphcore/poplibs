// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
// Simple test case for test log of roialign.
//

#define BOOST_TEST_MODULE ROIAlignTest
#include "poputil/exceptions.hpp"
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/ROIAlign.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popnn/experimental/ROIAlign.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <vector>

#define TOL 1e-1
#define FLOAT_ATOL 1e-3
#define HALF_ATOL 1e-2

void generateInput(unsigned inputH, unsigned inputW, unsigned poolH,
                   unsigned poolW, unsigned batchSize, unsigned channel,
                   unsigned numRois, boost::multi_array<float, 4> &hInputFeat,
                   boost::multi_array<float, 2> &hInputROI,
                   boost::multi_array<int, 1> &hBatchIndex,
                   boost::multi_array<float, 4> &hInputGrad,
                   boost::multi_array<float, 2> &hBIndexAndROI) {
  std::vector<float> ROI = {1.6, 1.6, 9.2, 11.0};
  std::vector<int> ROIIndex = {0};
  if (ROI.size() % 4 != 0 || ROI.size() / 4 != ROIIndex.size()) {
    throw poputil::poplibs_error(
        "The number of ROI does not match the batch index");
  }
  float count = 0;
  for (unsigned int b = 0; b < batchSize; b++) {
    for (unsigned int c = 0; c < channel; c++) {
      for (unsigned int i = 0; i < inputH; i++) {
        for (unsigned int j = 0; j < inputW; j++) {
          hInputFeat[b][c][i][j] = count;
          count++;
        }
      }
    }
  }

  count = 0;
  for (unsigned int n = 0; n < numRois; n++) {
    for (unsigned int c = 0; c < channel; c++) {
      for (unsigned int i = 0; i < poolH; i++) {
        for (unsigned int j = 0; j < poolW; j++) {
          hInputGrad[n][c][i][j] = count;
          count++;
        }
      }
    }
  }

  for (unsigned n = 0; n < numRois; n++) {
    for (unsigned l = 0; l < 5; l++) {
      if (l == 0) {
        hBatchIndex[n] = ROIIndex[n];
        hBIndexAndROI[n][l] = ROIIndex[n];
      } else {
        hInputROI[n][l - 1] = ROI[n * 4 + l - 1];
        hBIndexAndROI[n][l] = ROI[n * 4 + l - 1];
      }
    }
  }
}

void checkResult(float scale, bool aligned, unsigned sample,
                 boost::multi_array<float, 4> &hInputFeat,
                 boost::multi_array<float, 2> &hInputROI,
                 boost::multi_array<int, 1> &hBatchIndex,
                 boost::multi_array<float, 4> &hInputGrad,
                 boost::multi_array<float, 2> &hBIndexAndROI) {
  unsigned inputH = hInputFeat.shape()[2];
  unsigned inputW = hInputFeat.shape()[3];
  unsigned poolH = hInputGrad.shape()[2];
  unsigned poolW = hInputGrad.shape()[3];
  unsigned batchSize = hInputFeat.shape()[0];
  unsigned channel = hInputFeat.shape()[1];
  unsigned numRois = hInputGrad.shape()[0];
  const auto roiAlignParams =
      popnn::experimental::roiAlignParams(sample, poolH, poolW, aligned, scale);

  auto device = createTestDevice(TEST_TARGET);
  auto &target = device.getTarget();

  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);
  poplin::addCodelets(graph);
  poplar::program::Sequence prog, uploadProg, downloadProg;
  std::vector<std::pair<std::string, poplar_test::HostMemory>> tmap;

  poplar::Tensor inputFeatF = graph.addVariable(
      poplar::FLOAT, {batchSize, channel, inputH, inputW}, "inputFeatF");
  poplar::Tensor inputFeatH = graph.addVariable(
      poplar::HALF, {batchSize, channel, inputH, inputW}, "inputFeatH");
  poputil::mapTensorLinearly(graph, inputFeatF);
  poputil::mapTensorLinearly(graph, inputFeatH);
  poplar::Tensor inputROIF =
      graph.addVariable(poplar::FLOAT, {numRois, 4}, "inputROIF");
  poplar::Tensor inputROIH =
      graph.addVariable(poplar::HALF, {numRois, 4}, "inputROIH");
  poputil::mapTensorLinearly(graph, inputROIF);
  poputil::mapTensorLinearly(graph, inputROIH);
  poplar::Tensor batchIndex =
      graph.addVariable(poplar::INT, {numRois}, "batchIndex");
  poputil::mapTensorLinearly(graph, batchIndex);
  poplar::Tensor inputGradF = graph.addVariable(
      poplar::FLOAT, {numRois, channel, poolH, poolW}, "inputGradF");
  poplar::Tensor inputGradH = graph.addVariable(
      poplar::HALF, {numRois, channel, poolH, poolW}, "inputGradH");
  poputil::mapTensorLinearly(graph, inputGradF);
  poputil::mapTensorLinearly(graph, inputGradH);

  graph.createHostWrite("inputFeatF", inputFeatF);
  graph.createHostWrite("inputFeatH", inputFeatH);
  graph.createHostWrite("inputROIF", inputROIF);
  graph.createHostWrite("inputROIH", inputROIH);
  graph.createHostWrite("batchIndex", batchIndex);
  graph.createHostWrite("inputGradF", inputGradF);
  graph.createHostWrite("inputGradH", inputGradH);
  graph.createHostRead("inputFeatF", inputFeatF);
  graph.createHostRead("inputFeatH", inputFeatH);
  graph.createHostRead("inputROIF", inputROIF);
  graph.createHostRead("inputROIH", inputROIH);
  graph.createHostRead("batchIndex", batchIndex);
  graph.createHostRead("inputGradF", inputGradF);
  graph.createHostRead("inputGradH", inputGradH);

  auto rawHInputFeatF = poplar_test::allocateHostMemoryForTensor(
      inputFeatF, "inputFeatF", graph, uploadProg, downloadProg, tmap);
  auto rawHInputFeatH = poplar_test::allocateHostMemoryForTensor(
      inputFeatH, "inputFeatH", graph, uploadProg, downloadProg, tmap);
  auto rawHinputROIF = poplar_test::allocateHostMemoryForTensor(
      inputROIF, "inputROIF", graph, uploadProg, downloadProg, tmap);
  auto rawHinputROIH = poplar_test::allocateHostMemoryForTensor(
      inputROIH, "inputROIH", graph, uploadProg, downloadProg, tmap);
  auto rawHbatchIndex = poplar_test::allocateHostMemoryForTensor(
      batchIndex, "batchIndex", graph, uploadProg, downloadProg, tmap);
  auto rawHinputGradF = poplar_test::allocateHostMemoryForTensor(
      inputGradF, "inputGradF", graph, uploadProg, downloadProg, tmap);
  auto rawHinputGradH = poplar_test::allocateHostMemoryForTensor(
      inputGradH, "inputGradH", graph, uploadProg, downloadProg, tmap);

  poplar_test::copy(target, hInputFeat, poplar::FLOAT, rawHInputFeatF.get());
  poplar_test::copy(target, hInputFeat, poplar::HALF, rawHInputFeatH.get());
  poplar_test::copy(target, hInputROI, poplar::FLOAT, rawHinputROIF.get());
  poplar_test::copy(target, hInputROI, poplar::HALF, rawHinputROIH.get());
  poplar_test::copy(target, hInputGrad, poplar::FLOAT, rawHinputGradF.get());
  poplar_test::copy(target, hInputGrad, poplar::HALF, rawHinputGradH.get());
  poplar_test::copy(target, hBatchIndex, poplar::INT, rawHbatchIndex.get());

  poplar::Tensor outF, outGradF, outH, outGradH;
  outF = popnn::experimental::roiAlignFwd(graph, prog, inputFeatF, inputROIF,
                                          batchIndex, roiAlignParams);
  outGradF = popnn::experimental::roiAlignInputGradient(
      graph, prog, inputFeatF, inputROIF, batchIndex, inputGradF,
      roiAlignParams);
  outH = popnn::experimental::roiAlignFwd(graph, prog, inputFeatH, inputROIH,
                                          batchIndex, roiAlignParams);
  outGradH = popnn::experimental::roiAlignInputGradient(
      graph, prog, inputFeatH, inputROIH, batchIndex, inputGradH,
      roiAlignParams);

  auto rawHOutF = poplar_test::allocateHostMemoryForTensor(
      outF, "outF", graph, uploadProg, downloadProg, tmap);
  auto rawHOutGradF = poplar_test::allocateHostMemoryForTensor(
      outGradF, "outGradF", graph, uploadProg, downloadProg, tmap);
  auto rawHOutH = poplar_test::allocateHostMemoryForTensor(
      outH, "outH", graph, uploadProg, downloadProg, tmap);
  auto rawHOutGradH = poplar_test::allocateHostMemoryForTensor(
      outGradH, "outGradH", graph, uploadProg, downloadProg, tmap);

  poplar::Engine engine{
      graph, poplar::program::Sequence{uploadProg, prog, downloadProg}};
  poplar_test::attachStreams(engine, tmap);
  device.bind([&](const poplar::Device &d) { engine.loadAndRun(d); });

  boost::multi_array<float, 4> hOutRef = roi_align_forward_cpu(
      hInputFeat, hBIndexAndROI, poolH, poolW, scale, sample);
  boost::multi_array<float, 4> hOutGradRef =
      roi_align_backward_cpu(hBIndexAndROI, hInputGrad, batchSize, channel,
                             inputH, inputW, poolH, poolW, scale, sample);

  boost::multi_array<float, 4> hOutF(
      boost::extents[numRois][channel][poolH][poolW]);
  boost::multi_array<float, 4> hOutGradF(
      boost::extents[batchSize][channel][inputH][inputW]);
  boost::multi_array<float, 4> hOutH(
      boost::extents[numRois][channel][poolH][poolW]);
  boost::multi_array<float, 4> hOutGradH(
      boost::extents[batchSize][channel][inputH][inputW]);
  poplar_test::copy(target, poplar::FLOAT, rawHOutF.get(), hOutF);
  poplar_test::copy(target, poplar::FLOAT, rawHOutGradF.get(), hOutGradF);
  poplar_test::copy(target, poplar::HALF, rawHOutH.get(), hOutH);
  poplar_test::copy(target, poplar::HALF, rawHOutGradH.get(), hOutGradH);

  BOOST_TEST(poplibs_test::util::checkIsClose("ROIAlignForward", hOutRef, hOutF,
                                              TOL, FLOAT_ATOL));
  BOOST_TEST(poplibs_test::util::checkIsClose("ROIAlignBackward", hOutGradRef,
                                              hOutGradF, TOL, FLOAT_ATOL));
  BOOST_TEST(poplibs_test::util::checkIsClose("ROIAlignForward", hOutRef, hOutH,
                                              TOL, HALF_ATOL));
  BOOST_TEST(poplibs_test::util::checkIsClose("ROIAlignBackward", hOutGradRef,
                                              hOutGradH, TOL, HALF_ATOL));
}

BOOST_AUTO_TEST_CASE(ROIAlign_BS_1_C_1) {
  unsigned inputH = 7, inputW = 7;
  unsigned poolH = 2, poolW = 2, sample = 1, batchSize = 1, channel = 1,
           numRois = 1;
  float scale = 0.5;
  bool aligned = false;
  boost::multi_array<float, 4> hInputFeat(
      boost::extents[batchSize][channel][inputH][inputW]);
  boost::multi_array<float, 2> hInputROI(boost::extents[numRois][4]);
  boost::multi_array<int, 1> hBatchIndex(boost::extents[numRois]);
  boost::multi_array<float, 4> hInputGrad(
      boost::extents[numRois][channel][poolH][poolW]);
  boost::multi_array<float, 2> hBIndexAndROI(boost::extents[numRois][5]);
  generateInput(inputH, inputW, poolH, poolW, batchSize, channel, numRois,
                hInputFeat, hInputROI, hBatchIndex, hInputGrad, hBIndexAndROI);
  checkResult(scale, aligned, sample, hInputFeat, hInputROI, hBatchIndex,
              hInputGrad, hBIndexAndROI);
}

BOOST_AUTO_TEST_CASE(ROIAlign_BS_1_C_2) {
  unsigned inputH = 7, inputW = 7;
  unsigned poolH = 2, poolW = 2, sample = 1, batchSize = 1, channel = 2,
           numRois = 1;
  float scale = 0.5;
  bool aligned = false;
  boost::multi_array<float, 4> hInputFeat(
      boost::extents[batchSize][channel][inputH][inputW]);
  boost::multi_array<float, 2> hInputROI(boost::extents[numRois][4]);
  boost::multi_array<int, 1> hBatchIndex(boost::extents[numRois]);
  boost::multi_array<float, 4> hInputGrad(
      boost::extents[numRois][channel][poolH][poolW]);
  boost::multi_array<float, 2> hBIndexAndROI(boost::extents[numRois][5]);
  generateInput(inputH, inputW, poolH, poolW, batchSize, channel, numRois,
                hInputFeat, hInputROI, hBatchIndex, hInputGrad, hBIndexAndROI);
  checkResult(scale, aligned, sample, hInputFeat, hInputROI, hBatchIndex,
              hInputGrad, hBIndexAndROI);
}

BOOST_AUTO_TEST_CASE(ROIAlign_BS_2_C_2) {
  unsigned inputH = 7, inputW = 7;
  unsigned poolH = 2, poolW = 2, sample = 1, batchSize = 2, channel = 2,
           numRois = 1;
  float scale = 0.5;
  bool aligned = false;
  boost::multi_array<float, 4> hInputFeat(
      boost::extents[batchSize][channel][inputH][inputW]);
  boost::multi_array<float, 2> hInputROI(boost::extents[numRois][4]);
  boost::multi_array<int, 1> hBatchIndex(boost::extents[numRois]);
  boost::multi_array<float, 4> hInputGrad(
      boost::extents[numRois][channel][poolH][poolW]);
  boost::multi_array<float, 2> hBIndexAndROI(boost::extents[numRois][5]);
  generateInput(inputH, inputW, poolH, poolW, batchSize, channel, numRois,
                hInputFeat, hInputROI, hBatchIndex, hInputGrad, hBIndexAndROI);
  checkResult(scale, aligned, sample, hInputFeat, hInputROI, hBatchIndex,
              hInputGrad, hBIndexAndROI);
}