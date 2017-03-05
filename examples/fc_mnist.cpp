#include <initializer_list>
#include "popnn/Optimizer.hpp"
#include "mnist.h"

using namespace popnn::optimizer;

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

  Context context;
  auto in   = feed(MNIST, context);
  auto act1 = sigmoid(fullyconnected(30, in));
  auto out  = sigmoid(fullyconnected(10, act1));
  auto loss = sumSquaredLoss(in, out);

  OptimizerOptions options;
  options.learningRate = 0.3;
  options.dataType = FP32;
  options.training = true;
  options.batchSize = 1;

  Optimizer optimizer(loss, options);
  optimizer.run(10*50000);

  return 0;
}
