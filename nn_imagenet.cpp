#include <initializer_list>
#include "Net.hpp"
#include "FullyConnectedLayer.hpp"
#include "ConvLayer.hpp"
#include "MaxPoolLayer.hpp"

int main() {
  DataSet IMAGENET;
  IMAGENET.dataSize = 224*224*3;
  IMAGENET.dim = std::vector<unsigned>{224,224,3};
  IMAGENET.numTraining = 1000;
  IMAGENET.numTest = 1000;
  IMAGENET.testLabels =
    std::unique_ptr<unsigned char[]>(new unsigned char[IMAGENET.numTest]);
  IMAGENET.testData =
    std::unique_ptr<float[]>(new float[IMAGENET.dataSize * IMAGENET.numTest]);
  IMAGENET.trainingLabels =
    std::unique_ptr<unsigned char[]>(new unsigned char[IMAGENET.numTraining]);
  IMAGENET.trainingData =
    std::unique_ptr<float[]>(new float[IMAGENET.dataSize *
                                       IMAGENET.numTraining]);

  NetOptions options;
  options.doComputation = false;
  options.useIPUModel = true;
  options.numIPUs = 2;
  //  options.useSuperTiles = true;
  Net net(IMAGENET,
          10, // batch size
          makeHiddenLayers({
            new ConvLayer(11, 4, 4, 48*2, NON_LINEARITY_RELU, NORMALIZATION_LR),
            new MaxPoolLayer(3, 2),
            new ConvLayer(5, 1, 4, 128*2, NON_LINEARITY_RELU, NORMALIZATION_LR),
            new MaxPoolLayer(3, 2),
            new ConvLayer(3, 1, 2, 192*2, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new ConvLayer(3, 1, 2, 192*2, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new ConvLayer(3, 1, 2, 128*2, NON_LINEARITY_RELU,
                          NORMALIZATION_NONE),
            new MaxPoolLayer(3, 2),
            new FullyConnectedLayer(2048*2, NON_LINEARITY_RELU),
            new FullyConnectedLayer(2048*2, NON_LINEARITY_RELU),
            new FullyConnectedLayer(1000, NON_LINEARITY_RELU),
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9, // learning rate
          TestOnlyNet,
          options
          );

  net.run(5000);
  return 0;
}
