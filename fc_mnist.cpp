#include <initializer_list>
#include "Net.hpp"
#include "mnist.h"

#define LARGE_DNN_MODEL 0

int main() {
  DataSet MNIST;
  std::cerr << "Reading MNIST data\n";
  MNIST.dim = std::vector<std::size_t>{28, 28, 1};
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
  Net net(MNIST,
          1, // batch size
          makeLayers({
            new FullyConnectedLayer(30, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(10, NON_LINEARITY_SIGMOID),
          }),
          SUM_SQUARED_LOSS,
          0.3, // learning rate
          netType,
          FP32
          );
  net.run(10*50000);
  return 0;
}
