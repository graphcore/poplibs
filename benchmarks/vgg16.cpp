#include <initializer_list>
#include "popnn/Optimizer.hpp"

/** This model is derived from the paper:

    Very deep convolutional networks for large scale image recognition
    Karen Simonyan & Andrew Zisserman

    https://arxiv.org/pdf/1409.1556.pdf

*/

using namespace popnn::optimizer;

Exp module(unsigned channels, unsigned count, Exp in) {
  auto out = in;
  for (unsigned i = 0; i < count; ++i) {
    out = relu(conv2d(3, 1,  1, channels, out));
  }
  out = maxPool(2, 2, out);
  return out;
}

int main(int argc, char **argv) {
  DataSet IMAGENET;
  IMAGENET.dataSize = 224*224*4;
  IMAGENET.dim = std::vector<std::size_t>{224,224,4};
  IMAGENET.numTraining = 1;
  IMAGENET.numTest = 1;
  IMAGENET.testLabels =
    std::unique_ptr<unsigned[]>(new unsigned[IMAGENET.numTest]);
  IMAGENET.testData =
    std::unique_ptr<float[]>(new float[IMAGENET.dataSize * IMAGENET.numTest]);
  IMAGENET.trainingLabels =
    std::unique_ptr<unsigned[]>(new unsigned[IMAGENET.numTraining]);
  IMAGENET.trainingData =
    std::unique_ptr<float[]>(new float[IMAGENET.dataSize *
                                       IMAGENET.numTraining]);
  OptimizerOptions options;
  options.doComputation = true;
  options.useIPUModel = true;
  options.doTestsDuringTraining = false;
  options.ignoreData = true;
  options.learningRate = 0.9;
  options.dataType = FP16;
  if (!parseCommandLine(argc, argv, options))
    return 1;

  Context context;
  auto in   = feed(IMAGENET, context);
  auto v1   = module(64, 2, in);
  auto v2   = module(128, 2, v1);
  auto v3   = module(256, 3, v2);
  auto v4   = module(512, 3, v3);
  auto v5   = module(512, 3, v4);
  auto fc1  = relu(fullyconnected(4096, v5));
  auto fc2  = relu(fullyconnected(4096, fc1));
  auto out  = relu(fullyconnected(1000, fc2));
  auto loss = softMaxCrossEntropyLoss(in, out);
  Optimizer optimizer(loss, options);
  optimizer.run(1);

  return 0;
}
