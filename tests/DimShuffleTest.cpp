#define BOOST_TEST_MODULE DimShuffleTest
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include "DimShuffle.hpp"
#include <popnn/ActivationMapping.hpp>
#include <popnn/Net.hpp>

using namespace poplar;
using namespace poplar::program;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(DimShuffle) {
  GraphProgEnv env(popnn::findGraphProg(), GraphProgFileType::Object);
  Graph graph(env, createIPUModelDevice());
  auto in = graph.addTensor("float", {16, 2, 50, 8}, "in");
  auto out = graph.addTensor("float", {50, 16, 2, 8}, "out");
  std::vector<unsigned> permutation = {2, 0, 1, 3};
  mapTensor(graph, in);
  mapTensor(graph, out);
  auto outMapping = computeActivationsMapping(graph, out);
  auto shuffle = dimShuffle(graph, in, out, permutation,
                            outMapping);
  std::vector<float> hIn(in.numElements());
  std::vector<float> hOut(out.numElements());
  auto prog = Sequence(Copy(in, &hIn[0]),
                       shuffle,
                       Copy(&hOut[0], out));
  Engine eng(graph, prog);

  for (unsigned i = 0; i != in.numElements(); ++i) {
    hIn[i] = i;
  }
  eng.run();
  const auto &inDims = in.dims();
  const auto &outDims = out.dims();
  for (unsigned i = 0; i != outDims[0]; ++i) {
    for (unsigned j = 0; j != outDims[1]; ++j) {
      for (unsigned k = 0; k != outDims[2]; ++k) {
        for (unsigned l = 0; l != outDims[3]; ++l) {
          const auto outIndex =
              l + outDims[3] * (k + outDims[2] * (j + outDims[1] * i));
          const auto inIndex =
              l + inDims[3] * (i + inDims[2] * (k + inDims[1] * j));
          BOOST_CHECK_EQUAL(hOut[outIndex], hIn[inIndex]);
        }
      }
    }
  }
}
