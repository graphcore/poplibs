#include <initializer_list>
#include "popnn/Net.hpp"

/** This model is derived from the paper:

    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    http://arxiv.org/abs/1512.03385

    The details are summarized in a internal spreadsheet comparing different
    imagenet models (congidox document XM-010286-UN).
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

  auto resMethod = RESIDUAL_PAD;

  Net net(IMAGENET,
          options.batchSize,
          makeLayers({
            new ConvLayer(7, 7, 2, 2, 3, 3, 64,
                          NON_LINEARITY_RELU),
            new MaxPoolLayer(3, 2, 1),

            new ConvLayer(3, 3, 1, 1, 1, 1, 64, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 64, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 64, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 64, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 64, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 64, NON_LINEARITY_RELU, 2,
                             resMethod),

            new ConvLayer(3, 3, 2, 2, 1, 1, 128, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 128, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 128, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 128, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 128, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 128, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 128, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 128, NON_LINEARITY_RELU, 2,
                             resMethod),

            new ConvLayer(3, 3, 2, 2, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 256, NON_LINEARITY_RELU, 2,
                             resMethod),

            new ConvLayer(3, 3, 2, 2, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 512, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 512, NON_LINEARITY_RELU, 2,
                             resMethod),
            new ConvLayer(3, 3, 1, 1, 1, 1, 512, NON_LINEARITY_RELU),
            new ConvResLayer(3, 3, 1, 1, 1, 1, 512, NON_LINEARITY_RELU, 2,
                             resMethod),

            new MaxPoolLayer(7, 7),
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
