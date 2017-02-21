#include <initializer_list>
#include "popnn/Net.hpp"

/** This model is derived from the paper:

    Very deep convolutional networks for large scale image recognition
    Karen Simonyan & Andrew Zisserman

    https://arxiv.org/pdf/1409.1556.pdf

*/

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
  NetOptions options;
  options.doComputation = true;
  options.useIPUModel = true;
  options.doTestsDuringTraining = false;
  options.ignoreData = true;
  bool doTraining = false;
  if (!parseCommandLine(argc, argv, options, doTraining))
    return 1;
  NetType netType = doTraining ? TrainingNet : TestOnlyNet;


  Net net(IMAGENET,
          options.batchSize,
          makeLayers({
            new ConvLayer(3, 1, 1, 64, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 64, NON_LINEARITY_RELU),
            new MaxPoolLayer(2, 2),

            new ConvLayer(3, 1, 1, 128, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 128, NON_LINEARITY_RELU),
            new MaxPoolLayer(2, 2),

            new ConvLayer(3, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 256, NON_LINEARITY_RELU),
            new MaxPoolLayer(2, 2),

            new ConvLayer(3, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 512, NON_LINEARITY_RELU),
            new MaxPoolLayer(2, 2),

            new ConvLayer(3, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvLayer(3, 1, 1, 512, NON_LINEARITY_RELU),
            new MaxPoolLayer(2, 2),

            new FullyConnectedLayer(4096, NON_LINEARITY_RELU),
            new FullyConnectedLayer(4096, NON_LINEARITY_RELU),
            new FullyConnectedLayer(1000, NON_LINEARITY_RELU),
          }),

          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9, // learning rate
          netType,
          FP16,
          options
          );

  net.run(1);
  return 0;
}
