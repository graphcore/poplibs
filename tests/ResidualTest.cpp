#define BOOST_TEST_MODULE ResidualTest
#include <boost/test/unit_test.hpp>
#include <popnn/ActivationMapping.hpp>
#include <popnn/Net.hpp>
#include <popnn_ref/Util.hpp>
#include "Residual.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace ref::util;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(JoinDeltasTest) {
  GraphProgEnv env(popnn::findGraphProg(), GraphProgFileType::Object);
  Graph graph(env, createIPUModelDevice());
  const unsigned batchSize = 7;
  const unsigned height = 7;
  const unsigned width = 7;
  const unsigned depth = 64;
  const unsigned chansPerGroup1 = 64;
  const unsigned chansPerGroup2 = 16;
  auto deltas1 = graph.addTensor("float", {batchSize, depth / chansPerGroup1,
                                           height, width, chansPerGroup1});
  mapActivations(graph, deltas1);
  auto deltas2 = graph.addTensor("float", {batchSize, depth / chansPerGroup2,
                                           height, width, chansPerGroup2});
  mapActivations(graph, deltas2);
  auto prog = residual::joinDeltas(graph, deltas1, deltas2, "");
  Sequence upload, download;
  auto rawHostDeltas1 = allocateHostMemoryForTensor(graph, deltas1, upload,
                                                    download);
  auto rawHostDeltas2 = allocateHostMemoryForTensor(graph, deltas2, upload,
                                                    download);
  Engine engine(graph, {std::move(upload), std::move(download),
                        std::move(prog)});

  boost::multi_array<double, 4>
      hostDeltas1(boost::extents[batchSize][depth][height][width]);
  boost::multi_array<double, 4>
      hostDeltas2(boost::extents[batchSize][depth][height][width]);
  boost::multi_array<double, 4>
      expected(boost::extents[batchSize][depth][height][width]);
  std::mt19937 randomEngine;
  writeRandomValues(hostDeltas1, -5.0, 5.0, randomEngine);
  writeRandomValues(hostDeltas2, -5.0, 5.0, randomEngine);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned z = 0; z != depth; ++z) {
      for (unsigned y = 0; y != height; ++y) {
        for (unsigned x = 0; x != width; ++x) {
          expected[b][z][y][x] = hostDeltas1[b][z][y][x] +
                                 hostDeltas2[b][z][y][x];
        }
      }
    }
  }
  groupActivations(hostDeltas1, "float", deltas1.shape(),
                   rawHostDeltas1.get());
  groupActivations(hostDeltas2, "float", deltas2.shape(),
                   rawHostDeltas2.get());
  engine.run(0); // Upload.
  engine.run(2); // Run.
  engine.run(1); // Download.
  ungroupActivations("float", deltas1.shape(), rawHostDeltas1.get(),
                     hostDeltas1);

  const double relativeTolerance = 0.1;
  BOOST_CHECK(checkIsClose("fwd", hostDeltas1, expected, relativeTolerance));
}
