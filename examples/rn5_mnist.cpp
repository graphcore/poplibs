#include <initializer_list>
#include "enigma/Optimizer.hpp"
#include "mnist.h"

/** This model is based on conv_mnist, with an additional residual layer */

using namespace enigma;

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
  auto in    = feed(MNIST, context);
  auto pool1 = maxPool(2, 2, in);
  auto conv1 = relu(conv2d(5, 1, 2, 2, pool1));
  auto conv2 = conv2d(5, 1, 2, 2, conv1);
  auto res1  = relu(residualAdd(conv2, conv1));
  auto conv3 = relu(conv2d(5, 1, 2, 2, res1));
  auto pool2 = maxPool(2, 2, conv3);
  auto out   = sigmoid(conv2d(7, 1, 0, 10, pool2));
  auto loss = softMaxCrossEntropyLoss(in, out);

  OptimizerOptions options;
  options.learningRate = 0.002;
  options.dataType = FP32;
  options.training = true;
  options.batchSize = 1;

  Optimizer optimizer(loss, options);
  optimizer.run(10*50000);
  return 0;
}
