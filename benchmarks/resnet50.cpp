#include <initializer_list>
#include "enigma/Optimizer.hpp"

/** This model is derived from the paper:

    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    http://arxiv.org/abs/1512.03385

    The details are summarized in a internal spreadsheet comparing different
    imagenet models (congidox document XM-010286-UN).
*/

using namespace enigma;

Exp bottleneckModule(unsigned channels, unsigned initialStride,
           unsigned count, Exp in) {
  auto out = in;
  for (unsigned i = 0; i < count; ++i) {
    auto a = relu(conv2d(1, i == 0 ? initialStride : 1,  0, channels, out));
    auto b = relu(conv2d(3, 1, 1, channels, a));
    auto c = conv2d(1, 1, 0, channels * 4, b);
    out = relu(residualAdd(c, out));
  }
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
  auto in    = feed(IMAGENET, context);
  auto res1  = relu(conv2d(7, 2, 3, 64, in));
  auto pool1 = maxPool(3, 2, 1, res1);
  auto res2  = bottleneckModule(64, 1, 3, pool1);
  auto res3  = bottleneckModule(128, 2, 4, res2);
  auto res4  = bottleneckModule(256, 2, 6, res3);
  auto res5  = bottleneckModule(512, 2, 3, res4);
  auto pool2 = maxPool(7, 7, res5);
  auto out   = fullyconnected(1000, pool2);
  auto loss  = softMaxCrossEntropyLoss(in, out);
  Optimizer optimizer(loss, options);
  optimizer.run(1);

  return 0;
}
