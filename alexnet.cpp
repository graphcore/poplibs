#include <initializer_list>
#include "Net.hpp"
#include "FullyConnectedLayer.hpp"
#include "ConvLayer.hpp"
#include "MaxPoolLayer.hpp"


#define USE_CONVNET_BENCH_NET 1

int main() {
  DataSet IMAGENET;
  IMAGENET.dataSize = 224*224*3;
  IMAGENET.dim = std::vector<std::size_t>{224,224,3};
  IMAGENET.numTraining = 1000;
  IMAGENET.numTest = 1000;
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
  options.numIPUs = 1;
  options.singleBatchProfile = true;

#if USE_CONVNET_BENCH_NET
  Net net(IMAGENET,
          1, // batch size
          makeLayers({
            new ConvLayer(11, 4, 4, 1, 64,
                          NON_LINEARITY_RELU, NORMALIZATION_LR),
            new MaxPoolLayer(3, 2),
            new ConvLayer(5, 1, 4, 1, 192,
                          NON_LINEARITY_RELU, NORMALIZATION_LR),
            new MaxPoolLayer(3, 2),
            new ConvLayer(3, 1, 2, 1, 384, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new ConvLayer(3, 1, 2, 1, 256, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new ConvLayer(3, 1, 2, 1, 256, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new MaxPoolLayer(3, 2),
            new FullyConnectedLayer(2048*2, NON_LINEARITY_RELU),
            new FullyConnectedLayer(2048*2, NON_LINEARITY_RELU),
            new FullyConnectedLayer(1000, NON_LINEARITY_RELU),
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9, // learning rate
          TestOnlyNet,
          FP16,
          options
          );
#else
  Net net(IMAGENET,
          1, // batch size
          makeLayers({
            new ConvLayer(11, 4, 4, 1, 48*2,
                          NON_LINEARITY_RELU, NORMALIZATION_LR),
              #if 0
            new MaxPoolLayer(3, 2),
            new ConvLayer(5, 1, 4, 2, 128*2,
                          NON_LINEARITY_RELU, NORMALIZATION_LR),
            new MaxPoolLayer(3, 2),
            new ConvLayer(3, 1, 2, 1, 192*2, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new ConvLayer(3, 1, 2, 2, 192*2, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new ConvLayer(3, 1, 2, 2, 128*2, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new MaxPoolLayer(3, 2),
            new FullyConnectedLayer(2048*2, NON_LINEARITY_RELU),
            new FullyConnectedLayer(2048*2, NON_LINEARITY_RELU),
            new FullyConnectedLayer(1000, NON_LINEARITY_RELU),
              #endif
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9, // learning rate
          TestOnlyNet,
          FP16,
          options
          );
#endif

  net.run(1);
  return 0;
}
