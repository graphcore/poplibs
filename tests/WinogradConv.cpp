#define BOOST_TEST_MODULE FullyConnectedTest
#include <boost/test/unit_test.hpp>
#include <popconv/Convolution.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/codelets.hpp>
#include <popconv/codelets.hpp>
#include <poplin/codelets.hpp>
#include <string>
#include <random>
#include <Winograd.hpp>
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

const OptionFlags options {
  {"target.textSectionSizeInBytes", "0x6000"},
  {"target.workerStackSizeInBytes", "0x400"}

};

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

static unsigned filterLengthPre(unsigned a, unsigned kernel) {
  unsigned hLen = (kernel - 1)/2;
  unsigned filtLen = hLen;

  if (a < hLen)
    filtLen -= hLen - a;

  return filtLen;
}

static unsigned filterLengthPost(unsigned a, unsigned kernel, unsigned aLim) {
  unsigned hLen = (kernel - 1)/2;
  unsigned filtLen = hLen;

  if (a + hLen >= aLim)
    filtLen -= a + hLen - aLim + 1;

  return filtLen;
}


/* Reference convolution layer implementation using naive convolution method */
static void computeReference(Tensor in, Tensor weights, Tensor activations,
                             const float *inpBuffer, const float *weightBuffer,
                             float *outBuffer,
                             unsigned paddingY, unsigned paddingX) {

  unsigned numInpChanGroups = in.dim(1);
  unsigned numInpChansInGroup = in.dim(4);
  unsigned featureX           = in.dim(3);
  unsigned featureY           = in.dim(2);
  unsigned kernelX            = weights.dim(3);
  unsigned kernelY            = weights.dim(2);

  unsigned numInpChansInGroupWeight = weights.dim(5);
  unsigned numOutChansInGroupWeight = weights.dim(4);
  unsigned numInpChanGroupsWeight   = weights.dim(1);

  unsigned numOutChansInGroup = activations.dim(4);
  unsigned numOutChanGroups   = activations.dim(1);

  for (unsigned ozg = 0; ozg <  numOutChanGroups; ++ozg) {
    for (unsigned ozc = 0; ozc < numOutChansInGroup; ++ozc) {

      const auto wozc = (ozg * numOutChansInGroup + ozc) %
                            numOutChansInGroupWeight;
      const auto wozg = (ozg * numOutChansInGroup + ozc) /
                            numOutChansInGroupWeight;
      const auto outIdx = ozg * (numOutChansInGroup * featureX * featureY)
                          + ozc;

      for (unsigned y = 0; y < featureY; ++y) {
        for (unsigned x = 0; x < featureX; ++x) {

          /* pre and post filter segments from centre tap: needed to include
           * padding
           */
          const unsigned filtLenLX = filterLengthPre(x, kernelX);
          const unsigned filtLenLY = filterLengthPre(y, kernelY);
          const unsigned filtLenRX = filterLengthPost(x, kernelX, featureX);
          const unsigned filtLenRY = filterLengthPost(y, kernelY, featureY);

          float outRes = 0;

          for (unsigned izg = 0; izg < numInpChanGroups; ++izg) {
            for (unsigned izc = 0; izc < numInpChansInGroup; ++izc) {
              const auto wizc = (izg * numInpChansInGroup + izc) %
                                     numInpChansInGroupWeight;
              const auto wizg = (izg * numInpChansInGroup + izc) /
                                     numInpChansInGroupWeight;

              const auto wIdx = wozg * (numInpChanGroupsWeight * kernelX
                                        * kernelY * numOutChansInGroupWeight
                                        * numInpChansInGroupWeight)
                     + wizg * (kernelX * kernelY * numOutChansInGroupWeight
                                       * numInpChansInGroupWeight)
                     + wozc * (numInpChansInGroupWeight)
                     + wizc;

              const auto inIdx = izg * (featureX * featureY
                                     * numInpChansInGroup) + izc;
              float acc = 0;

              for (unsigned ix = (kernelX - 1)/2 - filtLenLX;
                            ix <= (kernelX - 1)/2 + filtLenRX;
                            ++ix) {
                for (unsigned iy = (kernelY - 1)/2 - filtLenLY;
                              iy <= (kernelY - 1)/2 + filtLenRY;
                              ++iy) {
                  const auto finIdx = inIdx
                                + (x + ix - (kernelX - 1)/2)
                                   * numInpChansInGroup
                                + (y + iy - (kernelY - 1)/2)
                                   * featureX * numInpChansInGroup;

                  const auto fwIdx  = wIdx
                                + ix * numOutChansInGroupWeight
                                     * numInpChansInGroupWeight
                                + iy * kernelX * numOutChansInGroupWeight
                                     * numInpChansInGroupWeight;

                  acc += inpBuffer[finIdx] * weightBuffer[fwIdx];

                }
              }
              outRes += acc;
            }
          }

          const auto foIdx = outIdx + x * numOutChansInGroup + y
                                        * numOutChansInGroup * featureX;
          outBuffer[foIdx] = outRes;
        }
      }
    }
  }
}



BOOST_AUTO_TEST_CASE(WinogradConvolution,
                       *utf::tolerance<float>(
                          fpc::percent_tolerance<float>(1))) {
  auto device = createTestDevice(TEST_TARGET, 1, 24);
  Graph graph(device);
  popops::addCodelets(graph);
  popconv::addCodelets(graph);

  /* Test configuration */

  const auto dType = FLOAT;
  const unsigned numOutPartialChanGroups = 128/8;
  const unsigned numOutPartialChansInGroup = 8;
  const unsigned numInpChanGroups = 32/16;
  const unsigned featureX = 13;
  const unsigned featureY = 13;
  const unsigned numInpChansInGroup = 16;
  const unsigned kernelSizeX = 3;
  const unsigned kernelSizeY = 3;
  const unsigned numOutChanGroups = 128/8;
  const unsigned numOutChansInGroup = 8;
  const unsigned patchSizeX = 4;
  const unsigned patchSizeY = 4;
  const unsigned paddingY = 1;
  const unsigned paddingX = 1;
  const float    mean = 0.2;
  const float    stdDev = 1.0;


  auto in = graph.addVariable(
                  dType,
                  {1, numInpChanGroups, featureY, featureX, numInpChansInGroup},
                  "in");
  auto weights = graph.addVariable(
          dType,
          {numOutPartialChanGroups, numInpChanGroups, kernelSizeY,
           kernelSizeX, numOutPartialChansInGroup, numInpChansInGroup},
          "weights");
  auto activations = graph.addVariable(
          dType,
          {1, numOutChanGroups, featureY, featureX, numOutChansInGroup},
          "activations");
  Tensor residual;


  mapTensorLinearly(graph, in);
  mapTensorLinearly(graph, activations);

  const std::size_t inSize = numInpChanGroups * featureY
                             * featureX * numInpChansInGroup;
  const std::size_t wSize = numOutPartialChanGroups * numInpChanGroups
                            * kernelSizeY * kernelSizeX
                            * numOutPartialChansInGroup * numInpChansInGroup;
  const std::size_t outSize = numOutChanGroups
                             * featureX * featureY * numOutChansInGroup;

  std::vector<float> inBuffer(inSize);
  std::vector<float> outBuffer(outSize);
  std::vector<float> outBufferRef(outSize);
  std::vector<float> weightsBuffer(wSize);
  std::vector<float> debugBuffer(inSize*16);
  std::fill(debugBuffer.begin(), debugBuffer.end(), 0);

  std::mt19937 randomEngine;
  std::normal_distribution<> dist(mean, stdDev);

  for (unsigned i = 0; i < inSize; ++i) {
    inBuffer[i] = dist(randomEngine);
  }


  for (unsigned i = 0; i < wSize; ++i) {
    weightsBuffer[i] = dist(randomEngine);
  }

  auto params =
      popconv::ConvParams(dType,
                          1,
                          {featureY, featureX},
                          {kernelSizeY, kernelSizeX},
                          numInpChansInGroup * numInpChanGroups,
                          numOutChanGroups * numOutChansInGroup,
                          1,
                          // Input
                          {0, 0}, {0, 0},
                          {1, 1},
                          {paddingY, paddingX}, {paddingY, paddingX},
                          {false, false},
                          // Kernel
                          {0, 0}, {0, 0},
                          {1, 1},
                          {0, 0}, {0, 0},
                          {false, false},
                          // Output
                          {0, 0}, {0, 0},
                          {1, 1},
                          {0, 0}, {0, 0});

  auto wgdConv = popconv::winogradConvolution(graph, params, in, weights,
                                              activations,
                                              patchSizeX, patchSizeY,
                                              FLOAT);

  graph.createHostWrite("in", in);
  graph.createHostWrite("weights", weights);
  graph.createHostRead("out", activations);

  Engine eng(device, graph, wgdConv, options);
  eng.writeTensor("in", inBuffer.data());
  eng.writeTensor("weights", weightsBuffer.data());
  eng.run();
  eng.readTensor("out", outBuffer.data());

  computeReference(in, weights, activations, &inBuffer[0],
                   &weightsBuffer[0], &outBufferRef[0],
                   paddingX, paddingY);


  for (unsigned i = 0; i < outSize; ++i) {
    //float relDiff = fabs((outBuffer[i]-outBufferRef[i])/outBufferRef[i]*100);
    // std::cout << outBuffer[i] << "  " << outBufferRef[i] << "  ";
    //std::cout << relDiff << std::endl;
    //if (relDiff > 0.01)
    //std::cout << " error ... " << i << std::endl;
    BOOST_TEST(outBuffer[i] == outBufferRef[i]);
  }
}
