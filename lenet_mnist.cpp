#include <initializer_list>
#include "Net.hpp"
#include "FullyConnectedLayer.hpp"
#include "mnist.h"

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

  NetType netType = TrainingNet;

  Net net(MNIST,
          10, // batch size
          makeLayers({
            new ConvLayer(5, 1, 1, 1, 20,
                          NON_LINEARITY_RELU, NORMALIZATION_NONE),
            new MaxPoolLayer(2, 2),
            new ConvLayer(5, 1, 1, 1, 50,
                          NON_LINEARITY_RELU, NORMALIZATION_NONE),
            new MaxPoolLayer(2, 2),
            new FullyConnectedLayer(500, NON_LINEARITY_RELU),
            new FullyConnectedLayer(10, NON_LINEARITY_SIGMOID),
          }),
          SUM_SQUARED_LOSS,
          //          SOFTMAX_CROSS_ENTROPY_LOSS,
          3.0, // learning rate
          netType,
          FP32
          );

  net.run(50000);

  return 0;
}
