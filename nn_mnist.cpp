#include <poplar/GraphProgEnv.hpp>
#include <poplar/GraphBuilder.hpp>
#include <poplar/CPUEngine.hpp>
#include <poplar/IPUModelEngine.hpp>
#include <initializer_list>
#include <vector>
#include <mnist.h>
#include <iostream>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <chrono>
#include <memory>
#include "neural_net_common.h"
#include "VTileMapper.hpp"
#include "Net.hpp"
#include "FullyConnectedLayer.hpp"

#define LARGE_DNN_MODEL 1

int main() {
  DataSet MNIST;
  std::cerr << "Reading MNIST data\n";
  MNIST.dataSize = 784;
  MNIST.numTraining = 60000;
  MNIST.numTest = 10000;
  MNIST.testLabels = readMNISTLabels(MNIST.numTest,
                                     "t10k-labels-idx1-ubyte");
  MNIST.testData = readMNISTData(MNIST.numTest,
                                 MNIST.dataSize,
                                 "t10k-images-idx3-ubyte");
  MNIST.trainingData = readMNISTData(MNIST.numTraining,
                                     MNIST.dataSize,
                                     "train-images-idx3-ubyte");
  MNIST.trainingLabels = readMNISTLabels(MNIST.numTraining,
                                         "train-labels-idx1-ubyte");

  NetType netType = TrainingNet;

  #if LARGE_DNN_MODEL
  NetOptions options;
  options.useIPUModel = true;
  options.singleBatchProfile = true;
  options.useSuperTiles = true;
  Net net(MNIST,
          100, // batch size
          makeHiddenLayers({
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(10, NON_LINEARITY_SIGMOID),
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9, // learning rate
          netType,
          options
          );
  #else
  Net net(MNIST,
          10, // batch size
          makeHiddenLayers({
            new FullyConnectedLayer(30, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(10, NON_LINEARITY_SIGMOID),
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9, // learning rate
          netType
          );
  #endif

  net.run(5000);

  return 0;
}
