#define BOOST_TEST_MODULE BroadcastToMatchTest
#include <poplar/Graph.hpp>
#include <boost/test/unit_test.hpp>
#include <poputil/Broadcast.hpp>

using namespace poplar;

static void matchTest(Graph &graph,
                      const std::vector<std::size_t> &shapeA,
                      const std::vector<std::size_t> &shapeB,
                      const std::vector<std::size_t> &shapeC) {
  auto t1 = graph.addVariable(FLOAT, shapeA);
  auto t2 = graph.addVariable(FLOAT, shapeB);
  poputil::broadcastToMatch(t1, t2);
  auto t1shape = t1.shape();
  auto t2shape = t2.shape();
  BOOST_CHECK_EQUAL_COLLECTIONS(t1shape.begin(), t1shape.end(),
                                shapeC.begin(), shapeC.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(t2shape.begin(), t2shape.end(),
                                shapeC.begin(), shapeC.end());
}

static void matchTest(Graph &graph,
                      const std::vector<std::size_t> &shapeA,
                      const std::vector<std::size_t> &shapeB) {
  auto t1 = graph.addVariable(FLOAT, shapeA);
  poputil::broadcastToMatch(t1, shapeB);
  auto t1shape = t1.shape();
  BOOST_CHECK_EQUAL_COLLECTIONS(t1shape.begin(), t1shape.end(),
                                shapeB.begin(), shapeB.end());
}


BOOST_AUTO_TEST_CASE(BroadcastToMatchTest) {
  Graph g(Target::createCPUTarget());
  matchTest(g, {1,4},{4,4});
  matchTest(g, {9},{5,9},{5,9});

  matchTest(g, {1,4},{3,1},{3,4});
  matchTest(g, {1,9},{1,9},{1,9});
  matchTest(g, {22},{7,22},{7,22});
}
