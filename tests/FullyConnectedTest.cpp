#define BOOST_TEST_MODULE FullyConnectedTest
#include <boost/test/unit_test.hpp>
#include <poplar/IPUModelEngine.hpp>
#include <popnn/FullyConnected.hpp>
#include <popnn/FullyConnectedPlan.hpp>
#include <popnn/ActivationMapping.hpp>
#include <popnn/Net.hpp>
using namespace poplar;
using namespace poplar::program;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(FullyConnected,
                       *utf::tolerance<float>(
                          fpc::percent_tolerance<float>(0.01))) {
  GraphProgEnv env(popnn::findGraphProg(), GraphProgFileType::Object);
  Graph graph(env);
  IPUModelEngineBuilder ipuEB(env);
  DeviceInfo deviceInfo(ipuEB, 64, 4, 8, 4, true, false, true);
  IPUModelEngineBuilder::TileMapping mapping(graph);
  std::size_t inSize = 1000;
  std::size_t outSize = 10;
  auto in = graph.addTensor("float", {inSize}, "in");
  auto out = graph.addTensor("float", {outSize}, "out");
  mapActivations(in, mapping, deviceInfo);
  mapActivations(out, mapping, deviceInfo);
  Tensor weights, biases;
  std::tie(weights, biases) = fc::createParams(graph, "float", inSize, outSize);
  auto outMapping = computeActivationsMapping(out, deviceInfo);
  auto plan = fc::createPlan(deviceInfo, "float", inSize, outMapping, true);
  auto fc = fc::fullyConnected(graph, mapping, deviceInfo,
                               outSize, NON_LINEARITY_NONE,
                               "float", in, weights, biases, out, plan);
  float hIn[inSize];
  float hOut[outSize], hRefOut[outSize];
  float hWeights[outSize][inSize];
  float hBiases[outSize];
  auto prog = Sequence(Copy(in, &hIn[0]),
                       Copy(weights, &hWeights[0]),
                       Copy(biases, &hBiases[0]),
                       fc,
                       Copy(&hOut[0], out));
  IPUModelEngineBuilder::UserTilePartitioner p(mapping);
  ipuEB.setTilePartitioner(p);
  auto eng = ipuEB.makeEngine(graph, prog);
  for (unsigned i = 0; i < inSize; ++i)
    hIn[i] = i;
  for (unsigned i = 0; i < outSize; ++i)
    hBiases[i] = i;
  for (unsigned i = 0; i < outSize; ++i) {
    for (unsigned j = 0; j < inSize; ++j) {
      hWeights[i][j] = static_cast<float>(i * j) / (inSize * outSize);
    }
  }

  // Model of fully connected layer
  for (unsigned i = 0; i < outSize; ++i) {
    float sum = 0;
    for (unsigned j = 0; j < inSize; ++j) {
      sum += hWeights[i][j] * hIn[j];
    }
    hRefOut[i] = sum;
  }
  eng->run();
  for (unsigned i = 0; i < outSize; ++i)
    BOOST_TEST(hOut[i] == hRefOut[i]);
}


