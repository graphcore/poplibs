// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LossTests
#include <boost/test/unit_test.hpp>

#include "../popnn/NonLinearityInternal.hpp"
#include "poplar/Engine.hpp"
#include "poplar/IPUModel.hpp"
#include "poplibs_test/Util.hpp"
#include "popnn/Loss.hpp"
#include "popnn/NonLinearityDef.hpp"
#include "popops/EncodingConstants.hpp"
#include <poplibs_support/TestDevice.hpp>

// codelets
#include "popnn/codelets.hpp"
#include "popops/codelets.hpp"

#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/random.hpp>
#include <cassert>
#include <iostream>
#include <limits>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace popnn;
using namespace poplibs_test::util;
using namespace poplibs_support;

namespace {

static unsigned getExpected(boost::multi_array<double, 2> &activations,
                            std::vector<std::uint64_t> &expected,
                            std::mt19937 &randomEngine, bool maskLabels,
                            const std::size_t minCorrect = 1,
                            const double proportionCorrect = 0.5) {
  const auto batchSize = activations.size();
  const auto numClasses = activations[0].size();
  assert(proportionCorrect <= 1.0);
  assert(minCorrect <= batchSize);

  // Work out what predictions have been made for each batch
  std::vector<std::uint64_t> predictions(batchSize);
  for (std::size_t b = 0; b < batchSize; b++) {
    double max = activations[b][0];
    std::size_t maxLabel = 0;
    for (std::size_t t = 0; t < numClasses; t++) {
      if (activations[b][t] > max) {
        max = activations[b][t];
        maxLabel = t;
      }
    }
    predictions[b] = maxLabel;
  }

  auto numCorrect =
      std::max(minCorrect, std::size_t(batchSize * proportionCorrect));
  // Make exactly a certain number of predictions correct for randomly
  // selected batches
  expected.resize(batchSize);
  {
    std::vector<std::size_t> shuffledBatches(batchSize);
    std::iota(shuffledBatches.begin(), shuffledBatches.end(), 0);
    std::shuffle(shuffledBatches.begin(), shuffledBatches.end(), randomEngine);
    for (std::size_t b = 0; b < numCorrect; b++) {
      const auto actualBatch = shuffledBatches[b];
      expected[actualBatch] = predictions[actualBatch];
    }
    // Random labels not equal the predicted for the rest
    auto numClassesToGen = maskLabels ? numClasses : numClasses - 1;
    boost::random::uniform_int_distribution<std::uint64_t> labelDist(
        0, numClassesToGen);
    for (std::size_t b = numCorrect; b < batchSize; b++) {
      const auto actualBatch = shuffledBatches[b];
      auto randLabel = predictions[actualBatch];
      while (randLabel == predictions[actualBatch]) {
        randLabel = labelDist(randomEngine);
        if (randLabel == numClasses) {
          randLabel = MASKED_LABEL_CODE;
        }
      }
      expected[shuffledBatches[b]] = randLabel;
    }
  }
  return numCorrect;
}

template <typename LabelType>
static inline void copyLabels(const std::vector<std::uint64_t> &labels,
                              char *out) {
  auto *typed = reinterpret_cast<LabelType *>(out);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    std::int64_t max =
        static_cast<std::int64_t>(std::numeric_limits<LabelType>::max());
    std::int64_t min =
        -static_cast<std::int64_t>(std::numeric_limits<LabelType>::min());
    const auto range = max + min;
    BOOST_CHECK(labels[i] <= static_cast<std::uint64_t>(range));
    typed[i] = static_cast<LabelType>(labels[i]);
  }
}

static inline void copyLabels(const Type &labelType,
                              const std::vector<std::uint64_t> &labels,
                              char *out) {
  if (labelType == UNSIGNED_INT) {
    copyLabels<unsigned>(labels, out);
  } else if (labelType == INT) {
    copyLabels<int>(labels, out);
  }
}

static void getModelLossAndDeltas(
    const LossType lossType, const boost::multi_array<double, 2> &activations,
    const std::vector<uint64_t> &expected,
    boost::multi_array<double, 2> &deltas, boost::multi_array<double, 1> &loss,
    const poplar::Type &dataType, const float scalingForDeltas,
    const float modelOutputScaling) {
  const auto batchSize = activations.size();
  const auto numClasses = activations[0].size();
  switch (lossType) {
  case LossType::SUM_SQUARED_LOSS: {
    for (std::size_t b = 0; b < batchSize; b++) {
      for (std::size_t t = 0; t < numClasses; t++) {
        if (expected[b] == MASKED_LABEL_CODE) {
          BOOST_FAIL("Cannot have masked expected code for sum squared loss");
        }
        double expect = (t == expected[b] ? 1 : 0);
        double delta = activations[b][t] - expect;
        deltas[b][t] = delta;
        loss[b] += 0.5 * delta * delta;
      }
    }
    break;
  }
  case LossType::CROSS_ENTROPY_LOSS: {
    const double scaleOut = scalingForDeltas;
    const double logModelOutputScaling = log(modelOutputScaling);
    const double eps =
        dataType == poplar::FLOAT ? EPS_LOG_N_FLOAT : EPS_LOG_N_HALF;
    for (std::size_t b = 0; b < batchSize; b++) {
      for (std::size_t t = 0; t < numClasses; t++) {
        double expect = (t == expected[b] ? 1 : 0);
        double delta =
            (activations[b][t] / modelOutputScaling - expect) * scaleOut;
        if (expected[b] == MASKED_LABEL_CODE) {
          delta = 0;
        }
        deltas[b][t] = delta;
        loss[b] +=
            -expect * (log(activations[b][t] + eps) - logModelOutputScaling);
      }
    }
    break;
  }
  default:
    BOOST_FAIL("calculateExpectedResults unimplemented for given LossType");
    break;
  }
}

static bool lossTest(const LossType lossType, std::size_t batchSize,
                     std::size_t numClasses, const Type &fpType,
                     const Type &expectedType, bool transposedActs,
                     bool maskLabels, bool scaling,
                     const float modelOutputScaling) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  std::vector<std::size_t> actsShape = {batchSize, numClasses};
  if (transposedActs) {
    std::swap(actsShape[0], actsShape[1]);
  }
  auto activations = graph.addVariable(
      fpType, {actsShape}, VariableMappingMethod::LINEAR, "activations");
  if (transposedActs) {
    activations = activations.transpose();
  }

  auto expected = graph.addVariable(expectedType, {batchSize},
                                    VariableMappingMethod::LINEAR, "expected");
  auto deltas = graph.addVariable(fpType, {batchSize, numClasses},
                                  VariableMappingMethod::LINEAR, "deltas");
  auto loss = graph.addVariable(fpType, {batchSize},
                                VariableMappingMethod::LINEAR, "loss");

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActivations = allocateHostMemoryForTensor(
      activations, "activations", graph, uploadProg, downloadProg, tmap);
  auto rawHostExpected = allocateHostMemoryForTensor(
      expected, "expected", graph, uploadProg, downloadProg, tmap);
  auto rawHostDeltas = allocateHostMemoryForTensor(
      deltas, "deltas", graph, uploadProg, downloadProg, tmap);
  auto rawHostLoss = allocateHostMemoryForTensor(
      loss, "loss", graph, uploadProg, downloadProg, tmap);

  std::mt19937 randomEngine;
  boost::multi_array<double, 2> hostActivations(
      boost::extents[batchSize][numClasses]);
  // cross entropy requires a probability distribution
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  const auto scaleForDeltas = 1000.0f;

  for (std::size_t b = 0; b < batchSize; ++b) {
    double batchSum = 0.0;
    for (std::size_t c = 0; c < numClasses; ++c) {
      hostActivations[b][c] = dist(randomEngine);
      batchSum += hostActivations[b][c];
    }
    for (std::size_t c = 0; c < numClasses; ++c) {
      // Scaled activations for the target.
      hostActivations[b][c] =
          ((lossType == LossType::CROSS_ENTROPY_LOSS) ? modelOutputScaling
                                                      : 1.0f) *
          hostActivations[b][c] / batchSum;
    }
  }
  copy(target, hostActivations, fpType, rawHostActivations.get());
  std::vector<std::uint64_t> hostExpected;
  getExpected(hostActivations, hostExpected, randomEngine, maskLabels);
  copyLabels(expectedType, hostExpected, rawHostExpected.get());

  auto deltasScale =
      graph.addConstant(deltas.elementType(), {}, scaleForDeltas);
  auto tModelOutputScaling =
      graph.addConstant(deltas.elementType(), {}, modelOutputScaling);
  graph.setTileMapping(deltasScale, 0);
  graph.setTileMapping(tModelOutputScaling, 0);
  auto prog =
      (lossType == LossType::CROSS_ENTROPY_LOSS && scaling)
          ?
          // (modelOutputScaling != 1.0)) ?
          calcLoss(graph, activations, expected, loss, deltas, deltasScale,
                   tModelOutputScaling, lossType)
          : calcLoss(graph, activations, expected, loss, deltas, lossType);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  boost::multi_array<double, 2> hostDeltas(
      boost::extents[batchSize][numClasses]);
  boost::multi_array<double, 1> hostLoss(boost::extents[batchSize]);
  copy(target, fpType, rawHostDeltas.get(), hostDeltas);
  copy(target, fpType, rawHostLoss.get(), hostLoss);

  boost::multi_array<double, 2> modelDeltas(
      boost::extents[batchSize][numClasses]);
  boost::multi_array<double, 1> modelLoss(boost::extents[batchSize]);
  bool scaledLoss = (lossType == LossType::CROSS_ENTROPY_LOSS) && scaling;
  //(modelOutputScaling != 1.0);
  getModelLossAndDeltas(lossType, hostActivations, hostExpected, modelDeltas,
                        modelLoss, fpType, scaledLoss ? scaleForDeltas : 1.0f,
                        scaledLoss ? modelOutputScaling : 1.0f);

  const double relativeTolerance = fpType == FLOAT ? 0.01 : 0.1;
  const double absoluteTolerance = fpType == FLOAT ? 1e-6 : 1e-5;

  bool matchesModel = true;
  matchesModel &= checkIsClose("deltas", hostDeltas, modelDeltas,
                               relativeTolerance, absoluteTolerance);
  matchesModel &= checkIsClose("loss", hostLoss, modelLoss, relativeTolerance,
                               absoluteTolerance);
  return matchesModel;
}

static bool accuracyTest(const Type &fpType, const Type &labelType,
                         std::size_t batchSize, std::size_t numClasses,
                         bool maskLabels) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  auto activations =
      graph.addVariable(fpType, {batchSize, numClasses},
                        VariableMappingMethod::LINEAR, "activations");
  auto expected = graph.addVariable(labelType, {batchSize},
                                    VariableMappingMethod::LINEAR, "expected");
  auto numCorrect = graph.addVariable(
      UNSIGNED_INT, {}, VariableMappingMethod::LINEAR, "numCorrect");
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActivations = allocateHostMemoryForTensor(
      activations, "activations", graph, uploadProg, downloadProg, tmap);
  auto rawHostExpected = allocateHostMemoryForTensor(
      expected, "expected", graph, uploadProg, downloadProg, tmap);
  auto rawHostNumCorrect = allocateHostMemoryForTensor(
      numCorrect, "numCorrect", graph, uploadProg, downloadProg, tmap);

  std::mt19937 randomEngine;
  boost::multi_array<double, 2> hostActivations(
      boost::extents[batchSize][numClasses]);
  writeRandomValues(target, fpType, hostActivations, 0.0, 1.0, randomEngine);
  copy(target, hostActivations, fpType, rawHostActivations.get());

  std::vector<std::uint64_t> hostExpected;
  auto modelNumCorrect =
      getExpected(hostActivations, hostExpected, randomEngine, maskLabels);
  copyLabels(labelType, hostExpected, rawHostExpected.get());

  auto prog = calcAccuracy(graph, activations, expected, numCorrect);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  auto *hostNumCorrect = reinterpret_cast<unsigned *>(rawHostNumCorrect.get());
  unsigned actualNumCorrect = *hostNumCorrect;
  return modelNumCorrect == actualNumCorrect;
}

static bool argMinMaxTest(bool max, const Type &inType, std::size_t batchSize,
                          std::size_t numClasses) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  auto activations =
      graph.addVariable(inType, {batchSize, numClasses},
                        VariableMappingMethod::LINEAR, "activations");

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActivations = allocateHostMemoryForTensor(
      activations, "activations", graph, uploadProg, downloadProg, tmap);

  std::mt19937 randomEngine;
  boost::multi_array<double, 2> hostActivations(
      boost::extents[batchSize][numClasses]);
  const bool isFpType = inType == HALF || inType == FLOAT;
  const bool isInt = inType == INT;
  writeRandomValues(target, inType, hostActivations,
                    isInt ? std::numeric_limits<int>::min() : 0.0,
                    isFpType ? 1.0 : std::numeric_limits<int>::max(),
                    randomEngine);
  copy(target, hostActivations, inType, rawHostActivations.get());

  Sequence prog;
  auto indices =
      max ? argMax(graph, activations, prog) : argMin(graph, activations, prog);

  auto rawHostIndices = allocateHostMemoryForTensor(
      indices, "indices", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  const auto indicesHost = reinterpret_cast<unsigned *>(rawHostIndices.get());

  bool matches = true;
  for (unsigned b = 0; b != batchSize; ++b) {
    auto elementItr = hostActivations[b].end();
    if (max) {
      elementItr = std::max_element(hostActivations[b].begin(),
                                    hostActivations[b].end());
    } else {
      elementItr = std::min_element(hostActivations[b].begin(),
                                    hostActivations[b].end());
    }

    matches = matches && std::distance(hostActivations[b].begin(),
                                       elementItr) == indicesHost[b];
  }
  return matches;
}

static bool argMaxTest(const Type &inType, std::size_t batchSize,
                       std::size_t numClasses) {
  return argMinMaxTest(/*max=*/true, inType, batchSize, numClasses);
}

static bool argMinTest(const Type &inType, std::size_t batchSize,
                       std::size_t numClasses) {
  return argMinMaxTest(/*max=*/false, inType, batchSize, numClasses);
}

static bool maxMinArgMinMaxTest(bool max, const Type &inType,
                                std::size_t batchSize, std::size_t numClasses) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  auto activations =
      graph.addVariable(inType, {batchSize, numClasses},
                        VariableMappingMethod::LINEAR, "activations");

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActivations = allocateHostMemoryForTensor(
      activations, "activations", graph, uploadProg, downloadProg, tmap);

  std::mt19937 randomEngine;
  boost::multi_array<double, 2> hostActivations(
      boost::extents[batchSize][numClasses]);
  const bool isFpType = inType == HALF || inType == FLOAT;
  const bool isInt = inType == INT;
  writeRandomValues(target, inType, hostActivations,
                    isInt ? std::numeric_limits<int>::min() : 0.0,
                    isFpType ? 1.0 : std::numeric_limits<int>::max(),
                    randomEngine);
  copy(target, hostActivations, inType, rawHostActivations.get());

  Sequence prog;
  auto [values, indices] = max ? maxAndArgMax(graph, activations, prog)
                               : minAndArgMin(graph, activations, prog);

  auto rawHostValues = allocateHostMemoryForTensor(
      values, "values", graph, uploadProg, downloadProg, tmap);
  auto rawHostIndices = allocateHostMemoryForTensor(
      indices, "indices", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  const auto indicesHost = reinterpret_cast<unsigned *>(rawHostIndices.get());
  std::vector<double> valuesHost(batchSize);
  copy(target, inType, rawHostValues.get(), valuesHost.data(), batchSize);

  bool matches = true;
  for (unsigned b = 0; b != batchSize; ++b) {
    auto elementItr = hostActivations[b].end();
    if (max) {
      elementItr = std::max_element(hostActivations[b].begin(),
                                    hostActivations[b].end());
    } else {
      elementItr = std::min_element(hostActivations[b].begin(),
                                    hostActivations[b].end());
    }
    matches = matches && std::fabs(static_cast<double>(*elementItr) -
                                   valuesHost[b]) < 0.00001;
    matches = matches && std::distance(hostActivations[b].begin(),
                                       elementItr) == indicesHost[b];
  }
  return matches;
}

static bool maxAndArgMaxTest(const Type &inType, std::size_t batchSize,
                             std::size_t numClasses) {
  return maxMinArgMinMaxTest(/*max=*/true, inType, batchSize, numClasses);
}

static bool minAndArgMinTest(const Type &inType, std::size_t batchSize,
                             std::size_t numClasses) {
  return maxMinArgMinMaxTest(/*max=*/false, inType, batchSize, numClasses);
}

static std::vector<double> model_topK(std::vector<double> &acts, int numK) {
  std::make_heap(acts.begin(), acts.end());
  std::vector<double> output(numK);

  for (int i = 0; i < numK; ++i) {
    std::pop_heap(acts.begin(), acts.end());
    output[i] = acts.back();
    acts.pop_back();
  }

  return output;
}

static bool topKTest(const Type &fpType, std::size_t batchSize,
                     std::size_t numClasses, std::size_t numK,
                     bool sort = false) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  auto activations =
      graph.addVariable(fpType, {batchSize, numClasses},
                        VariableMappingMethod::LINEAR, "activations");

  auto outputIndices = graph.addVariable(
      fpType, {batchSize, numK}, VariableMappingMethod::LINEAR, "indices");

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostActivations = allocateHostMemoryForTensor(
      activations, "activations", graph, uploadProg, downloadProg, tmap);

  auto rawHostIndices = allocateHostMemoryForTensor(
      outputIndices, "indices", graph, uploadProg, downloadProg, tmap);

  std::mt19937 randomEngine;
  boost::multi_array<double, 2> hostActivations(
      boost::extents[batchSize][numClasses]);

  writeRandomValues(target, fpType, hostActivations, 0.0, 1.0, randomEngine);
  copy(target, hostActivations, fpType, rawHostActivations.get());

  Sequence prog;
  auto values = topK(graph, activations, outputIndices, numK, sort, prog);

  auto rawHostOut = allocateHostMemoryForTensor(values, "output", graph,
                                                uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0);
  });

  std::vector<double> outHost(batchSize * numK);
  copy(target, fpType, rawHostOut.get(), outHost.data(), outHost.size());

  bool matches = true;

  for (unsigned b = 0; b != batchSize; ++b) {
    std::vector<double> tmp(hostActivations[b].begin(),
                            hostActivations[b].end());
    auto maxElement = model_topK(tmp, numK);

    std::vector<double> deviceActs(numK);
    for (unsigned i = 0; i < numK; ++i) {
      deviceActs[i] = outHost[b * numK + i];
    }

    // We can only check against sorted output so if the operation wasn't
    // required to sort the output we should sort it now. Otherwise don't sort
    // so we can test that the operation sorted it correctly.
    if (!sort) {
      std::sort(deviceActs.begin(), deviceActs.end(), std::greater<double>());
    }

    for (unsigned i = 0; i < numK; ++i) {
      matches &= std::fabs(maxElement[i] - deviceActs[i]) < 0.00001;
    }
  }
  return matches;
}

} // end anonymous namespace

BOOST_AUTO_TEST_SUITE(ArgMinMax)

BOOST_AUTO_TEST_CASE(argMaxFloat) {
  auto matchesModel = argMaxTest(FLOAT, 2, 10);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(argMaxHalf) {
  auto matchesModel = argMaxTest(HALF, 3, 15);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(argMaxInt) {
  auto matchesModel = argMaxTest(INT, 4, 20);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(argMaxUnsignedInt) {
  auto matchesModel = argMaxTest(UNSIGNED_INT, 5, 25);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(argMinFloat) {
  auto matchesModel = argMinTest(FLOAT, 2, 10);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(argMinHalf) {
  auto matchesModel = argMinTest(HALF, 3, 15);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(argMinInt) {
  auto matchesModel = argMinTest(INT, 4, 20);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(argMinUnsignedInt) {
  auto matchesModel = argMinTest(UNSIGNED_INT, 5, 25);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(maxAndArgMaxFloat) {
  auto matchesModel = maxAndArgMaxTest(FLOAT, 2, 10);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(maxAndArgMaxHalf) {
  auto matchesModel = maxAndArgMaxTest(HALF, 3, 15);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(minAndArgMinFloat) {
  auto matchesModel = minAndArgMinTest(FLOAT, 3, 15);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(minAndArgMinHalf) {
  auto matchesModel = minAndArgMinTest(HALF, 2, 10);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(maxAndArgMaxUnsignedInt) {
  auto matchesModel = maxAndArgMaxTest(UNSIGNED_INT, 5, 25);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_CASE(maxAndArgMaxInt) {
  auto matchesModel = maxAndArgMaxTest(INT, 5, 25);
  BOOST_CHECK(matchesModel);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TopK)

BOOST_AUTO_TEST_CASE(topKFloat) {
  BOOST_CHECK(topKTest(FLOAT, 2, 10, 2));
  BOOST_CHECK(topKTest(FLOAT, 2, 10, 3));

  BOOST_CHECK(topKTest(FLOAT, 10, 14, 8));
  BOOST_CHECK(topKTest(FLOAT, 10, 100, 24));

  BOOST_CHECK(topKTest(FLOAT, 1, 12, 5));
  BOOST_CHECK(topKTest(FLOAT, 1, 30, 11));

  // Test that we correctly sort the input if requested.
  BOOST_CHECK(topKTest(FLOAT, 1, 10, 1, true));
  BOOST_CHECK(topKTest(FLOAT, 1, 8, 2, true));
  BOOST_CHECK(topKTest(FLOAT, 2, 9, 3, true));
  BOOST_CHECK(topKTest(FLOAT, 2, 12, 4, true));

  // Test some large 'K' sizes.
  BOOST_CHECK(topKTest(FLOAT, 1, 200, 150, false));
  BOOST_CHECK(topKTest(FLOAT, 1, 200, 150, true));

  BOOST_CHECK(topKTest(FLOAT, 1, 1200, 24, true));

  // Test K==Size
  BOOST_CHECK(topKTest(FLOAT, 1, 20, 20, false));
  BOOST_CHECK(topKTest(FLOAT, 1, 20, 20, true));
}

BOOST_AUTO_TEST_CASE(topKHalf) {
  BOOST_CHECK(topKTest(HALF, 2, 10, 2));
  BOOST_CHECK(topKTest(HALF, 2, 10, 3));

  BOOST_CHECK(topKTest(HALF, 10, 14, 8));
  BOOST_CHECK(topKTest(HALF, 10, 100, 24));

  BOOST_CHECK(topKTest(HALF, 1, 12, 5));
  BOOST_CHECK(topKTest(HALF, 1, 30, 11));

  // Test that we correctly sort the input if requested.
  BOOST_CHECK(topKTest(HALF, 1, 10, 1, true));
  BOOST_CHECK(topKTest(HALF, 1, 8, 2, true));
  BOOST_CHECK(topKTest(HALF, 2, 9, 3, true));
  BOOST_CHECK(topKTest(HALF, 2, 12, 4, true));

  // Test some large 'K' sizes.
  BOOST_CHECK(topKTest(HALF, 1, 200, 150, false));
  BOOST_CHECK(topKTest(HALF, 1, 200, 150, true));

  BOOST_CHECK(topKTest(HALF, 1, 1200, 24, true));

  // Test K==Size
  BOOST_CHECK(topKTest(HALF, 1, 20, 20, false));
  BOOST_CHECK(topKTest(HALF, 1, 20, 20, true));
}

BOOST_AUTO_TEST_SUITE_END()

#define LOSS_TEST_NAME(lossType, b, n, tr, ml, fpType, lType, scaling)         \
  lossType##_##b##x##n##_##tr##_##ml##_##fpType##_##lType##_##scaling

#define LOSS_TEST_TYPE(lossType, b, n, tr, ml, fpType, lType, scaling, scale)  \
  BOOST_AUTO_TEST_CASE(                                                        \
      LOSS_TEST_NAME(lossType, b, n, tr, ml, fpType, lType, scaling)) {        \
    auto matchesModel =                                                        \
        lossTest(lossType, b, n, fpType, lType, tr, ml, scaling, scale);       \
    BOOST_CHECK(matchesModel);                                                 \
  }

#define ENUMERATE_VALID_LOSS_TYPE_TESTS(lossType, b, n, tr, ml, scaling)       \
  BOOST_AUTO_TEST_SUITE(lossType##_suite)                                      \
  LOSS_TEST_TYPE(lossType, b, n, tr, ml, FLOAT, UNSIGNED_INT, scaling,         \
                 SOFTMAX_SCALING)                                              \
  LOSS_TEST_TYPE(lossType, b, n, tr, ml, HALF, UNSIGNED_INT, scaling, 16384)   \
  LOSS_TEST_TYPE(lossType, b, n, tr, ml, FLOAT, INT, scaling, 32768)           \
  LOSS_TEST_TYPE(lossType, b, n, tr, ml, HALF, INT, scaling, 2)                \
  BOOST_AUTO_TEST_SUITE_END()

#define ENUMERATE_LOSS_TYPE_TESTS(b, n, tr, ml)                                \
  ENUMERATE_VALID_LOSS_TYPE_TESTS(SUM_SQUARED_LOSS, b, n, tr, false, false)    \
  ENUMERATE_VALID_LOSS_TYPE_TESTS(CROSS_ENTROPY_LOSS, b, n, tr, ml, false)     \
  ENUMERATE_VALID_LOSS_TYPE_TESTS(CROSS_ENTROPY_LOSS, b, n, tr, ml, true)

ENUMERATE_LOSS_TYPE_TESTS(1, 1, true, false)
ENUMERATE_LOSS_TYPE_TESTS(100, 20, true, true)
ENUMERATE_LOSS_TYPE_TESTS(1, 1, false, false)
ENUMERATE_LOSS_TYPE_TESTS(100, 24, false, true)

#define ACCURACY_TEST_NAME(name, b, n, ml, fpType, labelType)                  \
  name##_##b##x##n##_##ml##_##fpType##_##labelType

#define ACCURACY_TEST_TYPE(name, b, n, ml, fpType, labelType)                  \
  BOOST_AUTO_TEST_CASE(                                                        \
      ACCURACY_TEST_NAME(name, b, n, ml, fpType, labelType)) {                 \
    auto matchesModel = accuracyTest(fpType, labelType, b, n, ml);             \
    BOOST_CHECK(matchesModel);                                                 \
  }

#define ENUMERATE_VALID_ACCURACY_TYPE_TESTS(b, n, ml)                          \
  ACCURACY_TEST_TYPE(Accuracy, b, n, ml, FLOAT, UNSIGNED_INT)                  \
  ACCURACY_TEST_TYPE(Accuracy, b, n, ml, HALF, UNSIGNED_INT)                   \
  ACCURACY_TEST_TYPE(Accuracy, b, n, ml, FLOAT, INT)                           \
  ACCURACY_TEST_TYPE(Accuracy, b, n, ml, HALF, INT)

BOOST_AUTO_TEST_SUITE(Accuracy)

ENUMERATE_VALID_ACCURACY_TYPE_TESTS(1, 1, false)
ENUMERATE_VALID_ACCURACY_TYPE_TESTS(100, 20, true)

BOOST_AUTO_TEST_SUITE_END()
