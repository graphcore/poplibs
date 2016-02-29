#include <initializer_list>
#include "Net.hpp"
#include "FullyConnectedLayer.hpp"
#include "mnist.h"

#define LARGE_DNN_MODEL 0

int main() {
  DataSet MNIST;
  std::cerr << "Reading MNIST data\n";
  MNIST.dim = std::vector<std::size_t>{28, 28};
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

#if 0
  for (unsigned x = 0; x < 10; ++x) {
    for (unsigned i = 0; i < 28; ++i) {
      for (unsigned j = 0; j < 28; ++j) {
        float a = MNIST.testData[x * 28*28 + i * 28 + j];
        std::cout << (a > 0.5 ? "*" : " ");
      }
      std::cout << "\n";
    }
    std::cout << "\n----------------------------\n";
  }
#endif

  NetType netType = TrainingNet;

  #if LARGE_DNN_MODEL
  NetOptions options;
  options.useIPUModel = true;
  options.singleBatchProfile = true;
  options.useSuperTiles = false;
  Net net(MNIST,
          100, // batch size
          makeLayers({
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
          FP32,
          options
          );
  #else
  Net net(MNIST,
          10, // batch size
          makeLayers({
            new FullyConnectedLayer(30, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(10, NON_LINEARITY_SIGMOID),
          }),
          SUM_SQUARED_LOSS,
          //          SOFTMAX_CROSS_ENTROPY_LOSS,
          3.0, // learning rate
          netType,
          FP32
          );
  #endif

  net.run(50000);

  return 0;
}
