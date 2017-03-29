#include <initializer_list>
#include "enigma/Optimizer.hpp"

/*
--  Deep Neural Networks to Enable Real-time Multimessenger Astrophysics
--  https://arxiv.org/abs/1701.00008
*/

using namespace enigma;

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
  auto in    = feed(LIGO, context);
  auto act1  = relu(conv2d({1, 16}, {1, 1}, {0, 0}, 16, in));
  auto pool1 = maxPool({1, 4}, {1, 4}, {0, 2}, act1);
  auto act2  = relu(conv2d({1, 8}, {1, 1}, {0, 0}, 32, pool1));
  auto pool2 = maxPool({1, 4}, {1, 4}, {0, 2}, act2);
  auto act3  = relu(conv2d({1, 8}, {1, 1}, {0, 0}, 64, pool2));
  auto pool3 = maxPool({1, 4}, {1, 4}, {0, 2}, act3);
  auto act4  = relu(fullyconnected(64, pool3));
  auto out   = fullyconnected(2, act4);
  auto loss  = softMaxCrossEntropyLoss(in, out);
  Optimizer optimizer(loss, options);
  optimizer.run(1);

  return 0;
}
