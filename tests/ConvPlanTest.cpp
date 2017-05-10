#define BOOST_TEST_MODULE ConvPlanTest
#include <boost/test/unit_test.hpp>
#include <ConvPlan.hpp>
#include <popnn/codelets.hpp>
#include <vector>

BOOST_AUTO_TEST_CASE(getPlan){
  poplar::Graph graph(poplar::createCPUDevice());

  popconv::getPlan(graph,
                   "float",
                   0,
                   0,
                   0,
                   0,
                   {3, 3, 1}, // weightsShape
                   {1, 1}, // stride
                   {0, 0}, // paddingLower
                   {0, 0}, // paddingUpper
                   false, // isFractional
                   popconv::ConvOptions ());
}

BOOST_AUTO_TEST_CASE(getWeightUpdatePlan){

  poplar::Graph graph(poplar::createCPUDevice());
  popnn::addCodelets(graph);

  const std::vector<size_t> weightsShape = {3, 3, 1}; // y, x, channels
  const std::vector<unsigned> stride = {1, 1};         // y, x
  const std::vector<unsigned> padding = {0, 0};        // y, x
  const bool isFractional = false;

  auto activations = popconv::createInput(graph,
                                          "float",
                                          1, // batchSize
                                          3, // height
                                          3, // width
                                          1, // num chans
                                          weightsShape [1],
                                          weightsShape [0],
                                          weightsShape [2],
                                          stride,
                                          padding,
                                          padding,
                                          isFractional,
                                          "activations");

  auto deltas = popconv::createInput(graph,
                                     "float",
                                     1, // batchSize
                                     3, // height
                                     3, // width
                                     1, // num chans
                                     weightsShape [0],
                                     weightsShape [1],
                                     weightsShape [2],
                                     stride,
                                     padding,
                                     padding,
                                     isFractional,
                                     "deltas");

  popconv::getWeightUpdatePlan(graph,
                               activations,
                               deltas,
                               weightsShape,
                               stride,
                               padding,
                               padding,
                               isFractional,
                               popconv::ConvOptions());
}
