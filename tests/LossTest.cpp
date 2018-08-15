#define BOOST_TEST_MODULE LossTests

#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include "poplar/Engine.hpp"
#include "poplar/IPUModel.hpp"
#include "popnn/Loss.hpp"
#include "poplibs_test/Util.hpp"

// codelets
#include "popops/codelets.hpp"
#include "popnn/codelets.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <boost/random.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popnn;
using namespace poplibs_test::util;

const OptionFlags options {
  {"target.textSectionSizeInBytes", "0x9000"}
};

namespace {

static unsigned
getExpected(boost::multi_array<double, 2> &activations,
            std::vector<std::uint64_t> &expected,
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
    std::shuffle(shuffledBatches.begin(), shuffledBatches.end(),
                 std::mt19937{std::random_device{}()});
    for (std::size_t b = 0; b < numCorrect; b++) {
      const auto actualBatch = shuffledBatches[b];
      expected[actualBatch] = predictions[actualBatch];
    }
    // Random labels not equal the predicted for the rest
    std::mt19937 labelEngine;
    boost::random::uniform_int_distribution<std::uint64_t>
        labelDist(0, numClasses - 1);
    for (std::size_t b = numCorrect; b < batchSize; b++) {
      const auto actualBatch = shuffledBatches[b];
      auto randLabel = predictions[actualBatch];
      while (randLabel == predictions[actualBatch])
        randLabel = labelDist(labelEngine);
      expected[shuffledBatches[b]] = randLabel;
    }
  }
  return numCorrect;
}

template <typename LabelType>
static inline void
copyLabels(const std::vector<std::uint64_t> &labels,
           char *out) {
  auto *typed = reinterpret_cast<LabelType*>(out);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    BOOST_CHECK(labels[i] <= std::numeric_limits<LabelType>::max());
    typed[i] = static_cast<LabelType>(labels[i]);
  }
}

static inline void
copyLabels(const Type &labelType,
           const std::vector<std::uint64_t> &labels,
           char *out) {
  if (labelType == UNSIGNED_INT) {
    copyLabels<unsigned>(labels, out);
  } else if (labelType == INT) {
    copyLabels<int>(labels, out);
  }
}

static void
getModelLossAndDeltas(const LossType lossType,
                      const boost::multi_array<double, 2> &activations,
                      const std::vector<uint64_t> &expected,
                      boost::multi_array<double, 2> &deltas,
                      boost::multi_array<double, 1> &loss) {
  const auto batchSize = activations.size();
  const auto numClasses = activations[0].size();
  switch (lossType) {
    case LossType::SUM_SQUARED_LOSS: {
      for (std::size_t b = 0; b < batchSize; b++) {
        for (std::size_t t = 0; t < numClasses; t++) {
          double expect = (t == expected[b] ? 1 : 0);
          double delta = activations[b][t] - expect;
          deltas[b][t] = delta;
          loss[b] += 0.5 * delta * delta;
        }
      }
      break;
    }
    case LossType::SOFTMAX_CROSS_ENTROPY_LOSS: {
      for (std::size_t b = 0; b < batchSize; b++) {
        std::vector<float> transformed(numClasses);
        double sum = 0;
        for (std::size_t t = 0; t < numClasses; t++)
          transformed[t] = exp(activations[b][t]);
        for (std::size_t t = 0; t < numClasses; t++)
          sum += transformed[t];
        for (std::size_t t = 0; t < numClasses; t++)
          transformed[t] /= sum;
        for (std::size_t t = 0; t < numClasses; t++) {
          double expect = (t == expected[b] ? 1 : 0);
          double delta = transformed[t] - expect;
          deltas[b][t] = delta;
          loss[b] += -expect * log(transformed[t]);
        }
      }
      break;
    }
    default:
      BOOST_FAIL("calculateExpectedResults unimplemented for given LossType");
      break;
  }
}

static bool lossTest(const LossType lossType,
                     std::size_t batchSize,
                     std::size_t numClasses,
                     const Type &fpType,
                     const Type &expectedType) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  auto target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  auto activations = graph.addVariable(fpType, {batchSize, numClasses},
      VariableMappingMethod::LINEAR, "activations");
  auto expected = graph.addVariable(expectedType, {batchSize},
      VariableMappingMethod::LINEAR, "expected");
  auto deltas = graph.addVariable(fpType, {batchSize, numClasses},
      VariableMappingMethod::LINEAR, "deltas");
  auto loss = graph.addVariable(fpType, {batchSize},
      VariableMappingMethod::LINEAR, "loss");

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto rawHostActivations =
    allocateHostMemoryForTensor(activations, "activations", graph, uploadProg,
                                downloadProg, tmap);
  auto rawHostExpected =
    allocateHostMemoryForTensor(expected, "expected", graph, uploadProg,
                                downloadProg, tmap);
  auto rawHostDeltas =
    allocateHostMemoryForTensor(deltas, "deltas", graph, uploadProg,
                                downloadProg, tmap);
  auto rawHostLoss =
    allocateHostMemoryForTensor(loss, "loss", graph, uploadProg, downloadProg,
                                tmap);

  std::mt19937 randomEngine;
  boost::multi_array<double, 2>
    hostActivations(boost::extents[batchSize][numClasses]);
  writeRandomValues(target, fpType, hostActivations, 0.0, 1.0, randomEngine);
  copy(target, hostActivations, fpType, rawHostActivations.get());

  std::vector<std::uint64_t> hostExpected;
  getExpected(hostActivations, hostExpected);
  copyLabels(expectedType, hostExpected, rawHostExpected.get());

  auto prog = calcLoss(graph,
                       activations, expected,
                       loss, deltas,
                       fpType, expectedType,
                       lossType);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
  engine.load(device);
  attachStreams(engine, tmap);

  engine.run(0);

  boost::multi_array<double, 2>
    hostDeltas(boost::extents[batchSize][numClasses]);
  boost::multi_array<double, 1>
    hostLoss(boost::extents[batchSize]);
  copy(target, fpType, rawHostDeltas.get(), hostDeltas);
  copy(target, fpType, rawHostLoss.get(), hostLoss);

  boost::multi_array<double, 2>
    modelDeltas(boost::extents[batchSize][numClasses]);
  boost::multi_array<double, 1>
    modelLoss(boost::extents[batchSize]);
  getModelLossAndDeltas(lossType,
                        hostActivations,
                        hostExpected,
                        modelDeltas,
                        modelLoss);

  const double relativeTolerance = fpType == FLOAT ? 0.01 : 0.1;
  const double absoluteTolerance = fpType == FLOAT ? 1e-6 : 1e-5;

  bool matchesModel = true;
  matchesModel &= checkIsClose("deltas", hostDeltas, modelDeltas,
                               relativeTolerance, absoluteTolerance);
  matchesModel &= checkIsClose("loss", hostLoss, modelLoss,
                               relativeTolerance, absoluteTolerance);
  return matchesModel;
}

static bool accuracyTest(const Type &fpType,
                         const Type &labelType,
                         std::size_t batchSize,
                         std::size_t numClasses) {
  IPUModel ipuModel;
  auto device = ipuModel.createDevice();
  auto target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  auto activations = graph.addVariable(fpType, {batchSize, numClasses},
      VariableMappingMethod::LINEAR, "activations");
  auto expected = graph.addVariable(labelType, {batchSize},
      VariableMappingMethod::LINEAR, "expected");
  auto numCorrect = graph.addVariable(UNSIGNED_INT, {},
      VariableMappingMethod::LINEAR, "numCorrect");
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto rawHostActivations =
    allocateHostMemoryForTensor(activations, "activations", graph, uploadProg,
                                downloadProg, tmap);
  auto rawHostExpected =
    allocateHostMemoryForTensor(expected, "expected", graph, uploadProg,
                                downloadProg, tmap);
  auto rawHostNumCorrect =
    allocateHostMemoryForTensor(numCorrect, "numCorrect", graph, uploadProg,
                                downloadProg, tmap);

  std::mt19937 randomEngine;
  boost::multi_array<double, 2>
    hostActivations(boost::extents[batchSize][numClasses]);
  writeRandomValues(target, fpType, hostActivations, 0.0, 1.0, randomEngine);
  copy(target, hostActivations, fpType, rawHostActivations.get());

  std::vector<std::uint64_t> hostExpected;
  auto modelNumCorrect = getExpected(hostActivations, hostExpected);
  copyLabels(labelType, hostExpected, rawHostExpected.get());

  auto prog = calcAccuracy(graph, activations,
                           expected, numCorrect,
                           fpType, labelType);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
  engine.load(device);
  attachStreams(engine, tmap);

  engine.run(0);

  auto *hostNumCorrect = reinterpret_cast<unsigned*>(rawHostNumCorrect.get());
  unsigned actualNumCorrect = *hostNumCorrect;
  return modelNumCorrect == actualNumCorrect;
}

} // end anonymous namespace

#define LOSS_TEST_NAME(lossType, b, n, fpType, lType) \
  lossType ## _ ## b ## x ## n ## _ ## fpType ## _ ## lType

#define LOSS_TEST_TYPE(lossType, b, n, fpType, lType) \
  BOOST_AUTO_TEST_CASE(LOSS_TEST_NAME(lossType, b, n, fpType, lType)) { \
    auto matchesModel = lossTest(lossType, b, n, fpType, lType); \
    BOOST_CHECK(matchesModel); \
  }

#define ENUMERATE_VALID_LOSS_TYPE_TESTS(lossType, b, n) \
  LOSS_TEST_TYPE(lossType, b, n, FLOAT, UNSIGNED_INT) \
  LOSS_TEST_TYPE(lossType, b, n, HALF, UNSIGNED_INT) \
  LOSS_TEST_TYPE(lossType, b, n, FLOAT, INT) \
  LOSS_TEST_TYPE(lossType, b, n, HALF, INT)

#define ENUMERATE_LOSS_TYPE_TESTS(b, n) \
  ENUMERATE_VALID_LOSS_TYPE_TESTS(SUM_SQUARED_LOSS, b, n) \
  ENUMERATE_VALID_LOSS_TYPE_TESTS(SOFTMAX_CROSS_ENTROPY_LOSS, b, n)

ENUMERATE_LOSS_TYPE_TESTS(1, 1)
ENUMERATE_LOSS_TYPE_TESTS(4, 1024)

#define ACCURACY_TEST_NAME(name, b, n, fpType, labelType) \
  name ## _ ## b ## x ## n ## _ ## fpType ## _ ## labelType

#define ACCURACY_TEST_TYPE(name, b, n, fpType, labelType) \
  BOOST_AUTO_TEST_CASE(ACCURACY_TEST_NAME(name, b, n, fpType, labelType)) { \
    auto matchesModel = accuracyTest(fpType, labelType, b, n); \
    BOOST_CHECK(matchesModel); \
  }

#define ENUMERATE_VALID_ACCURACY_TYPE_TESTS(b, n) \
  ACCURACY_TEST_TYPE(Accuracy, b, n, FLOAT, UNSIGNED_INT) \
  ACCURACY_TEST_TYPE(Accuracy, b, n, HALF, UNSIGNED_INT) \
  ACCURACY_TEST_TYPE(Accuracy, b, n, FLOAT, INT) \
  ACCURACY_TEST_TYPE(Accuracy, b, n, HALF, INT)

ENUMERATE_VALID_ACCURACY_TYPE_TESTS(1, 1)
ENUMERATE_VALID_ACCURACY_TYPE_TESTS(4, 1024)
