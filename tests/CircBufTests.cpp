#define BOOST_TEST_MODULE StdOperatorTest
#include <popops/CircBuf.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;

const OptionFlags options {
  {"target.textSectionSizeInBytes", "0x9000"},
  {"target.workerStackSizeInBytes", "0x1000"}
};

BOOST_AUTO_TEST_CASE(CircBufIncrIndex) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popops::addCodelets(graph);
  const unsigned circBufSize = 20;
  const unsigned indexBufSize = 25;
  auto cb = CircBuf(graph, FLOAT, circBufSize, {1});
  auto indexStore = graph.addVariable(UNSIGNED_INT, {indexBufSize});
  mapTensorLinearly(graph, indexStore);
  auto dummy = graph.addVariable(FLOAT, {1});
  mapTensorLinearly(graph, dummy);

  auto prog = Sequence();
  for (auto i = 0U; i != indexBufSize; ++i) {
    prog.add(Copy(cb.getIndex(), indexStore[i]));
    cb.add(dummy, prog);
  }
  graph.createHostRead("out", indexStore);

  unsigned cbOut[indexBufSize];

  Engine eng(device, graph, prog, options);
  eng.run();
  eng.readTensor("out", cbOut);

  for (unsigned i = 0; i != indexBufSize; ++i) {
    BOOST_TEST(i % circBufSize == cbOut[i]);
  }
}

BOOST_AUTO_TEST_CASE(CircBufIncrIndex2d) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popops::addCodelets(graph);
  const unsigned circBufSize = 20;
  const unsigned indexBufSize = 25;
  auto cb = CircBuf(graph, FLOAT, circBufSize, {5, 3});
  auto indexStore = graph.addVariable(UNSIGNED_INT, {indexBufSize});
  mapTensorLinearly(graph, indexStore);
  auto dummy = graph.addVariable(FLOAT, {5, 3});
  mapTensorLinearly(graph, dummy);

  auto prog = Sequence();
  for (auto i = 0U; i != indexBufSize; ++i) {
    prog.add(Copy(cb.getIndex(), indexStore[i]));
    cb.add(dummy, prog);
  }
  graph.createHostRead("out", indexStore);

  unsigned cbOut[indexBufSize];

  Engine eng(device, graph, prog, options);
  eng.run();
  eng.readTensor("out", cbOut);

  for (unsigned i = 0; i != indexBufSize; ++i) {
    BOOST_TEST(i % circBufSize == cbOut[i]);
  }
}

BOOST_AUTO_TEST_CASE(CircBufCheckAdd) {
  auto device = createTestDevice(TEST_TARGET, 1, 16);
  Graph graph(device);

  popops::addCodelets(graph);
  const unsigned circBufSize = 20;
  const unsigned srcBufSize = 25;
  const unsigned numElemsA = 33, numElemsB = 2;
  auto cb = CircBuf(graph, FLOAT, circBufSize, {numElemsA, numElemsB});

  auto src = graph.addVariable(FLOAT, {srcBufSize, numElemsA, numElemsB});
  mapTensorLinearly(graph, src);
  auto dst = graph.addVariable(FLOAT, {circBufSize, numElemsA, numElemsB});
  mapTensorLinearly(graph, dst);


  auto prog = Sequence();
  for (auto i = 0U; i != srcBufSize; ++i) {
    cb.add(src[i], prog);
  }

 for (auto i = 0U; i != circBufSize; ++i) {
    prog.add(Copy(cb.prev(i, prog), dst[i]));
  }

  graph.createHostWrite("in", src);
  graph.createHostRead("out", dst);

  float cbSrc[srcBufSize][numElemsA][numElemsB];
  float cbDst[circBufSize][numElemsA][numElemsB];

  for (auto s = 0U; s != srcBufSize; ++s) {
    for (auto r = 0U; r != numElemsA; ++r) {
      for (auto c = 0U; c != numElemsB; ++c) {
        cbSrc[s][r][c] = 1000 * s + 10 * r + c;
      }
    }
  }

  Engine eng(device, graph, prog, options);
  eng.writeTensor("in", cbSrc);
  eng.run();
  eng.readTensor("out", cbDst);

  for (unsigned i = 0; i != circBufSize; ++i) {
    for (unsigned j = 0; j != numElemsA; ++j) {
      for (unsigned k = 0; k != numElemsB; ++k) {
        BOOST_TEST(cbDst[i][j][k] == (srcBufSize - 1 - i) * 1000 + j * 10 + k);
      }
    }
  }
}
