#define BOOST_TEST_MODULE ConvPlanTest
#include <boost/test/unit_test.hpp>
#include <ConvPlan.hpp>
#include <popnn/codelets.hpp>
#include <vector>

BOOST_AUTO_TEST_CASE(getPlan){
  poplar::Graph graph(poplar::createCPUDevice());

  auto params = popconv::ConvParams("float",
                                    // batch size
                                    1,
                                    // input field shape
                                    {4, 4},
                                    // kernel shape
                                    {3, 3},
                                    // input channels
                                    1,
                                    // output channels
                                    1,
                                    // stride
                                    {1, 1},
                                    // lower input padding
                                    {0, 0},
                                    // upper input padding
                                    {0, 0},
                                    // input dilation
                                    {1, 1},
                                    // lower kernel padding
                                    {0, 0},
                                    // upper kernel padding
                                    {0, 0},
                                    // kernel dilation
                                    {1, 1});
  popconv::getPlan(graph, params, popconv::ConvOptions());
}

BOOST_AUTO_TEST_CASE(getWeightUpdatePlan){

  poplar::Graph graph(poplar::createCPUDevice());
  popnn::addCodelets(graph);

  const std::vector<size_t> inFieldShape = {3, 3};
  const std::vector<size_t> weightsShape = {3, 3};       // ky, kx
  const std::vector<unsigned> stride = {1, 1};           // y, x
  const std::vector<int> padding = {0, 0};               // y, x
  auto params =
      popconv::ConvParams("float",
                          // batch size
                          1,
                          inFieldShape,
                          weightsShape,
                          // input channels
                          1,
                          // output channels
                          1,
                          stride,
                          // lower input padding
                          padding,
                          // upper input padding
                          padding,
                          // input diation
                          {1, 1},
                          // lower kernel padding
                          {0, 0},
                          // upper kernel padding
                          {0, 0},
                          // kernel dilation
                          {1, 1});

  auto activations = popconv::createInput(graph, params, "activations");

  auto deltas = popconv::createInput(graph, params, "deltas");

  popconv::getWeightUpdatePlan(graph, activations,
                               deltas, params,
                               popconv::ConvOptions());
}
