#include <initializer_list>
#include "Net.hpp"
#include "FullyConnectedLayer.hpp"

int main() {
  DataSet DATA;
  DATA.dataSize = 9216;
  DATA.dim = std::vector<std::size_t>{9216};
  DATA.numTraining = 1000;
  DATA.numTest = 1000;
  DATA.testLabels =
    std::unique_ptr<unsigned[]>(new unsigned[DATA.numTest]);
  DATA.testData =
    std::unique_ptr<float[]>(new float[DATA.dataSize * DATA.numTest]);
  DATA.trainingLabels =
    std::unique_ptr<unsigned[]>(new unsigned[DATA.numTraining]);
  DATA.trainingData =
    std::unique_ptr<float[]>(new float[DATA.dataSize *
                                       DATA.numTraining]);
  NetOptions options;
  options.doComputation = true;
  options.useIPUModel = true;
  options.numIPUs = 2;
  options.singleBatchProfile = true;
  //  options.useSuperTiles = true;
  Net net(DATA,
          128, // batch size
          makeLayers({
            new FullyConnectedLayer(4096, NON_LINEARITY_RELU),
            new FullyConnectedLayer(4096, NON_LINEARITY_RELU),
            new FullyConnectedLayer(1000, NON_LINEARITY_RELU),
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9, // learning rate
          TestOnlyNet,
          //TrainingNet,
          FP16,
          options
          );

  net.run(1);
  return 0;
}
