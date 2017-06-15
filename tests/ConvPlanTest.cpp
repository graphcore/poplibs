#define BOOST_TEST_MODULE ConvPlanTest
#include <boost/test/unit_test.hpp>
#include <ConvPlan.hpp>
#include <popnn/codelets.hpp>
#include <vector>

BOOST_AUTO_TEST_CASE(getPlan){
  poplar::Graph graph(poplar::createCPUDevice());

  auto params = popconv::ConvParams("float",
                                    {0, 0, 0, 0},
                                    {3, 3, 1, 0},
                                    {1, 1},
                                    {0, 0}, {0, 0}, {1, 1},
                                    {0, 0}, {0, 0}, {1, 1});
  popconv::getPlan(graph, params, popconv::ConvOptions());
}

BOOST_AUTO_TEST_CASE(getWeightUpdatePlan){

  poplar::Graph graph(poplar::createCPUDevice());
  popnn::addCodelets(graph);

  const std::vector<size_t> inShape = {1, 3, 3, 1};
  const std::vector<size_t> weightsShape = {3, 3, 1, 1}; // y, x, out, in
  const std::vector<unsigned> stride = {1, 1};           // y, x
  const std::vector<int> padding = {0, 0};               // y, x
  auto params =
      popconv::ConvParams("float", inShape, weightsShape, stride,
                          padding, padding, {1, 1},
                          {0, 0}, {0, 0}, {1, 1});

  auto activations = popconv::createInput(graph, params, "activations");

  auto deltas = popconv::createInput(graph, params, "deltas");

  popconv::getWeightUpdatePlan(graph, activations,
                               deltas, params,
                               popconv::ConvOptions());
}
