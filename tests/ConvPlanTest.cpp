#define BOOST_TEST_MODULE ConvPlanTest
#include <boost/test/unit_test.hpp>
#include <popconv/internal/ConvPlan.hpp>
#include <popnn/codelets.hpp>
#include <vector>

BOOST_AUTO_TEST_CASE(getPlan){
  poplar::Graph graph(poplar::Target::createCPUTarget());

  auto params = popconv::ConvParams(poplar::FLOAT,
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
                                    // flip input
                                    {false, false},
                                    // lower kernel padding
                                    {0, 0},
                                    // upper kernel padding
                                    {0, 0},
                                    // kernel dilation
                                    {1, 1},
                                    // flip kernel
                                    {false, false});
  popconv::getPlan(graph, params, popconv::ConvOptions());
}
