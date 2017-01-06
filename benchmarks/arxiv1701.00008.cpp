#include <initializer_list>
#include "popnn/Net.hpp"

/*
--  Deep Neural Networks to Enable Real-time Multimessenger Astrophysics
--  https://arxiv.org/abs/1701.00008
*/

int main(int argc, char **argv) {
  DataSet LIGO;
  LIGO.dataSize = 8192;
  LIGO.dim = std::vector<std::size_t>{1, 8192,1};
  LIGO.numTraining = 1;
  LIGO.numTest = 1;
  LIGO.testLabels =
    std::unique_ptr<unsigned[]>(new unsigned[LIGO.numTest]);
  LIGO.testData =
    std::unique_ptr<float[]>(new float[LIGO.dataSize * LIGO.numTest]);
  LIGO.trainingLabels =
    std::unique_ptr<unsigned[]>(new unsigned[LIGO.numTraining]);
  LIGO.trainingData =
    std::unique_ptr<float[]>(new float[LIGO.dataSize * LIGO.numTraining]);
  NetOptions options;
  options.doComputation = true;
  options.useIPUModel = true;
  options.doTestsDuringTraining = false;
  options.ignoreData = true;
  bool doTraining = false;
  if (!parseCommandLine(argc, argv, options, doTraining))
    return 1;
  NetType netType = doTraining ? TrainingNet : TestOnlyNet;

  Net net(LIGO,
          options.batchSize,
          makeLayers({
            new ConvLayer({1,16}, {1,1}, {0,0}, 16, NON_LINEARITY_RELU),
            new MaxPoolLayer({1,4}, {1,4}, {0,2}),
            new ConvLayer({1,8}, {1,1}, {0,0}, 32, NON_LINEARITY_RELU),
            new MaxPoolLayer({1,4}, {1,4}),
            new ConvLayer({1,8}, {1,1}, {0,0}, 64, NON_LINEARITY_RELU),
            new MaxPoolLayer({1,4}, {1,4}),
            new FullyConnectedLayer(64, NON_LINEARITY_RELU),
            new FullyConnectedLayer(2, NON_LINEARITY_NONE),
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
