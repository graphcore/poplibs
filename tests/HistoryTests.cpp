#define BOOST_TEST_MODULE StdOperatorTest
#include <popstd/History.hpp>
#include <popstd/TileMapping.hpp>
#include <popstd/exceptions.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <poplar/Engine.hpp>
#include <popstd/codelets.hpp>
#include <iostream>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

BOOST_AUTO_TEST_CASE(HistoryIncrIndex) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);
  const unsigned historySize = 20;
  const unsigned indexBufSize = 25;
  auto h = History(graph, "float", historySize, {1});
  auto indexStore = graph.addTensor("unsigned", {indexBufSize});
  mapTensorLinearly(graph, indexStore);
  auto dummy = graph.addTensor("float", {1});
  mapTensorLinearly(graph, dummy);

  auto prog = Sequence();
  for (auto i = 0U; i != indexBufSize; ++i) {
    prog.add(Copy(h.getIndex(), indexStore[i]));
    h.add(dummy, prog);
  }
  graph.createHostRead("out", indexStore);

  unsigned hOut[indexBufSize];

  Engine eng(graph, prog);
  eng.run();
  eng.readTensor("out", hOut);

  for (unsigned i = 0; i != indexBufSize; ++i) {
    BOOST_TEST(i % historySize == hOut[i]);
  }
}

BOOST_AUTO_TEST_CASE(HistoryCheckAdd) {
  DeviceInfo info;
  info.tilesPerIPU = 16;

  Graph graph(createIPUModelDevice(info));
  popstd::addCodelets(graph);
  const unsigned historySize = 20;
  const unsigned srcBufSize = 25;
  const unsigned numElems = 64;
  auto h = History(graph, "float", historySize, {numElems});

  auto src = graph.addTensor("float", {srcBufSize, numElems});
  mapTensorLinearly(graph, src);
  auto dst = graph.addTensor("float", {historySize, numElems});
  mapTensorLinearly(graph, dst);


  auto prog = Sequence();
  for (auto i = 0U; i != srcBufSize; ++i) {
    h.add(src[i], prog);
  }

 for (auto i = 0U; i != historySize; ++i) {
    prog.add(Copy(h.prev(i, prog), dst[i]));
  }

  graph.createHostWrite("in", src);
  graph.createHostRead("out", dst);

  float hSrc[srcBufSize][numElems];
  float hDst[historySize][numElems];

  for (auto r = 0U; r != srcBufSize; ++r) {
    for (auto c = 0U; c != numElems; ++c) {
      hSrc[r][c] = 100 * r + c;
    }
  }

  Engine eng(graph, prog);
  eng.writeTensor("in", hSrc);
  eng.run();
  eng.readTensor("out", hDst);

  for (unsigned i = 0; i != historySize; ++i) {
    for (unsigned j = 0; j != numElems; ++j) {
      BOOST_TEST(hDst[i][j] == (srcBufSize - 1 - i) * 100 + j) ;
    }
  }
}
