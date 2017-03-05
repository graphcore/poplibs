#include <initializer_list>
#include "popnn/Optimizer.hpp"

using namespace popnn::optimizer;

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
  auto in    = feed(IMAGENET, context);
  auto act1  = relu(conv2d(11, 4, 3, 64, in));
  auto pool1 = maxPool(3, 2, act1);
  auto act2  = relu(conv2d(5, 1, 2, 192, pool1));
  auto pool2 = maxPool(3, 2, act2);
  auto act3  = relu(conv2d(3, 1, 1, 384, pool2));
  auto act4  = relu(conv2d(3, 1, 1, 256, act3));
  auto act5  = relu(conv2d(3, 1, 1, 256, act4));
  auto pool4 = maxPool(3, 2, act5);
  auto act6  = relu(fullyconnected(4096, pool4));
  auto act7  = relu(fullyconnected(4096, act6));
  auto out   = relu(fullyconnected(1000, act7));
  auto loss  = softMaxCrossEntropyLoss(in, out);
  Optimizer optimizer(loss, options);
  optimizer.run(1);

  return 0;
}
