#include <initializer_list>
#include "popnn/Net.hpp"
#include "mnist.h"

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
            new MaxPoolLayer(2,2),
            new ConvLayer(5, 5, 1, 1, 2, 2, 2, NON_LINEARITY_NONE),
            new MaxPoolLayer(2,2),
            new ConvLayer(7, 7, 1, 1, 0, 0, 10, NON_LINEARITY_SIGMOID),
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.01,
          netType,
          FP32
          );
  net.run(10*50000);
  return 0;
}
