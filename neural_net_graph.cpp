#include <poplar/Vertex.hpp>
#include <poplar/ComputeSet.hpp>
#include <poplar/DataElement.hpp>
#include <poplar/DataArray.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "neural_net_common.h"

using namespace poplar;

#define NOTHING_TO_PROCESS (-2)
#define DONE_PROCESSING    (-1)
#define START_WEIGHT_UPDATE (-3)

typedef enum nn_state_t {
  INIT,
  TRAIN,
  TEST,
  WEIGHT_SYNC
} nn_state_t;

static float sigmoid(float x)
{
  return (1. / (1. + exp(-x)));
}

static float sigmoid_derivative(float x)
{
  return sigmoid(x) * (1. - sigmoid(x));
}

static float relu(float x)
{
  if (x > 0)
    return x;
  return 0;
}

static float relu_derivative(float x)
{
  if (x > 0)
    return 1;
  return 0;
}

static float nonlinearity(NonLinearityType t, float x) {
  switch (t) {
  case NON_LINEARITY_SIGMOID:
    return sigmoid(x);
  case NON_LINEARITY_RELU:
    return relu(x);
  case NON_LINEARITY_NONE:
    return x;
  }
}

static float nonlinearity_derivative(NonLinearityType t, float x) {
  switch (t) {
  case NON_LINEARITY_SIGMOID:
    return sigmoid_derivative(x);
  case NON_LINEARITY_RELU:
    return relu_derivative(x);
  case NON_LINEARITY_NONE:
    return 1;
  }
}


class InputLayerVertex : public Vertex {
public:
  Vector<float> data;
  float activationOut;
  float z = 0;
  int indexOut;

  Input<nn_state_t> state;
  unsigned batchSize;

  bool compute() {
    if (state == INIT) {
      indexOut = NOTHING_TO_PROCESS;
      return true;
    }

    if (indexOut == DONE_PROCESSING)
      return true;

    if (indexOut == NOTHING_TO_PROCESS)
      indexOut = 0;
    else
      indexOut++;

    if (indexOut == batchSize) {
      indexOut = DONE_PROCESSING;
      return true;
    }

    /* The input layer will simple output data values until a whole
       batch has been output. */
    activationOut = data[indexOut];
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};

/** This vertex gathers together a vector of inputs into
    a dense vector.
    This will gather all the activations from a previous layer
    to pass on to all the vertices in the next layer.
 */
class InnerProductFwdGatherVertex : public Vertex {
public:
  Vector<Input<float>> activationIn;
  Vector<float> activationOut;

  Input<nn_state_t> state;
  Input<int> indexIn;

  bool compute() {
    if (state == INIT)
      return true;

    if (indexIn == NOTHING_TO_PROCESS ||
        indexIn == DONE_PROCESSING)
      return true;

    for (unsigned i = 0; i < activationOut.size(); ++i) {
      activationOut[i] = activationIn[i];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return (activationIn.size() + 10);
  }
};

class InnerProductFwdVertex : public Vertex {
public:
  NonLinearityType nonLinearityType;

#if USE_GATHER_VERTEX
  Input<Vector<float>> activationIn;
#else
  Vector<Input<float>> activationIn;
#endif
  Input<int> indexIn;
  float z;
  float activationOut;
  int indexOut;

  Input<Vector<float>> weights;
  Input<float> bias;

  Input<nn_state_t> state;

  bool compute() {
    if (state == INIT) {
      indexOut = NOTHING_TO_PROCESS;
      return true;
    }

    if (indexIn == NOTHING_TO_PROCESS)
      return false;

    if (indexIn == DONE_PROCESSING) {
      indexOut = DONE_PROCESSING;
      return true;
    }

    float sum = 0;
    for (unsigned i = 0;  i < activationIn.size(); ++i) {
      sum += activationIn[i] * weights[i];
    }
    z = sum + bias;
    activationOut = nonlinearity(nonLinearityType, z);
    indexOut = indexIn;
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }

};

#if MEM_OPTIMIZED_WEIGHT_SYNC

class InnerProductParamsGatherVertex : public Vertex {
public:
  Vector<Input<float>> weightsIn;
  Vector<float> weightsOut;

  bool compute() {
    for (unsigned i = 0; i < weightsOut.size(); ++i) {
      weightsOut[i] = weightsIn[i];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }

};


class InnerProductParamsVertex : public Vertex {
public:
  Input<Vector<float>> weightsIn;
  Vector<float> weightsOut;
  Input<float> biasIn;
  float biasOut;

  unsigned currentRank;
  unsigned myRank;
  bool updated;

  Input<nn_state_t> state;

  bool compute() {
    if (state == INIT) {
      currentRank = 0;
      updated = false;
      return true;
    }

    if (state != WEIGHT_SYNC)
      return true;

    if (myRank == currentRank) {
      for (unsigned i = 0; i < weightsOut.size(); ++i) {
        weightsOut[i] = weightsIn[i];
      }
      biasOut = biasIn;
      updated = true;
    }
    currentRank++;
    return updated;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};



#else

class InnerProductParamsVertex : public Vertex {
public:
  Vector<Input<float>> weightsIn;
  Vector<float> weightsOut;
  Input<float> biasIn;
  float biasOut;
 
  bool compute() {
    for (unsigned i = 0; i < weightsOut.size(); ++i) {
      weightsOut[i] = weightsIn[i];
    }
    biasOut = biasIn;
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};

#endif

class InnerProductBwdGatherVertex : public Vertex {
public:
  Vector<Input<float>> deltaIn;
  Vector<float> deltaOut;

  Input<nn_state_t> state;
  Input<int> indexIn;

  bool compute() {
    if (state == INIT || state == TEST)
      return true;

    if (indexIn == NOTHING_TO_PROCESS ||
        indexIn == DONE_PROCESSING ||
        indexIn == START_WEIGHT_UPDATE)
      return true;

    for (unsigned i = 0; i < deltaOut.size(); ++i) {
      deltaOut[i] = deltaIn[i];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};


class InnerProductBwdVertex : public Vertex {
public:
  NonLinearityType nonLinearityType;

#if USE_GATHER_VERTEX
  Input<Vector<float>> deltaIn;
#else
  Vector<Input<float>> deltaIn;
#endif
  Input<int> indexIn;
  float deltaOut;
  int indexOut;

  Vector<float> weights;

  Input<float> activationIn;
  Input<float> zIn;
  Input<int> actIndexIn;
  Vector<float> zRecord, actRecord, bwdRecord;

  Input<float> eta;
  Input<nn_state_t> state;

  #if MEM_OPTIMIZED_WEIGHT_SYNC
  float weightSyncOutput;
  unsigned currentRank;
  #endif

  bool doingWeightUpdate;

  bool compute() {

    if (state == INIT) {
      indexOut = NOTHING_TO_PROCESS;
      doingWeightUpdate = false;
      #if MEM_OPTIMIZED_WEIGHT_SYNC
      currentRank = 0;
      #endif
      return true;
    }

    #if MEM_OPTIMIZED_WEIGHT_SYNC
    if (state == WEIGHT_SYNC) {
      if (currentRank >= weights.size())
        return true;
      weightSyncOutput = weights[currentRank];
      currentRank++;
      return false;
    }
    #endif

    if (state == TEST)
      return true;

    // During the forward pass the backwards vertex needs to record 
    // the activations going through the network.
    if (actIndexIn != NOTHING_TO_PROCESS &&
	actIndexIn != DONE_PROCESSING) {
      actRecord[actIndexIn] = activationIn;
      zRecord[actIndexIn] = zIn;
    }


    if (doingWeightUpdate) {

      if (indexIn == DONE_PROCESSING) {
        indexOut = DONE_PROCESSING;
        return true;
      }

      unsigned batchSize = actRecord.size();
      for (unsigned i = 0;  i < deltaIn.size(); ++i) {
        weights[i] += eta * actRecord[indexIn] * deltaIn[i] / batchSize;
      }

      deltaOut = bwdRecord[indexIn];
      indexOut = indexIn;

    } else {

      if (indexIn == NOTHING_TO_PROCESS)
        return false;

      if (indexIn == START_WEIGHT_UPDATE) {
        doingWeightUpdate = true;
        indexOut = START_WEIGHT_UPDATE;
        return false;
      }

      float sum = 0;
      for (unsigned i = 0;  i < deltaIn.size(); ++i) {
        sum += deltaIn[i] * weights[i];
      }
      float nlGradient = nonlinearity_derivative(nonLinearityType,
                                                 zRecord[indexIn]);

      // Pass on the error to the previous layer.
      deltaOut = nlGradient * sum;

      // Record the output for the weight update phase.
      bwdRecord[indexIn] = deltaOut;

      // Pass on the batch index to process to the previous layer.
      indexOut = indexIn;
    }
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};


class InnerProductBwdBiasVertex : public Vertex {
public:
  Input<float> deltaIn;
  Input<int> indexIn;

  float bias;
  
  Input<float> eta;
  Input<nn_state_t> state;
  unsigned batchSize;

  float update;
  bool updated;

  bool compute() {

    if (state == INIT) {
      update = 0;
      updated = false;
      return true;
    }

    if (state == TEST)
      return true;

    if (updated)
      return true;

    if (indexIn == NOTHING_TO_PROCESS)
      return false;

    if (indexIn == START_WEIGHT_UPDATE) {
      bias += eta * update / (float) batchSize;
      updated = true;
      return true;
    }

    if (indexIn == DONE_PROCESSING)
      return true;

    update += deltaIn;
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};


class SumSquaredErrorVertex : public Vertex {
public:
  Vector<Input<float>> zIn;
  Input<int> indexIn;
  float error;

  Vector<float> deltaOut;
  int indexOut;

  Vector<unsigned char> labels;

  Input<nn_state_t> state;

  unsigned numCorrect = 0;

  bool doingWeightUpdate;

  unsigned batchSize;

  NonLinearityType nonLinearityType;

  bool compute() {
    if (state == INIT) {
      error = 0;
      indexOut = NOTHING_TO_PROCESS;
      doingWeightUpdate = false;
      return true;
    }

    if (doingWeightUpdate) {

      if (indexOut == DONE_PROCESSING)
        return true;

      if (indexOut == START_WEIGHT_UPDATE) {
        indexOut = 0;
      } else {
        indexOut++;
        if (indexOut == batchSize) {
          indexOut = DONE_PROCESSING;
          return true;
        }
      }

    } else {

      if (indexIn == NOTHING_TO_PROCESS)
        return false;

      if (indexIn == DONE_PROCESSING) {
        indexOut = START_WEIGHT_UPDATE;
        doingWeightUpdate = true;
        return true;
      }

    }

    unsigned index = doingWeightUpdate ? indexOut : indexIn;

    unsigned E = labels[index];

    float sum = 0;
    float max = nonlinearity(nonLinearityType, zIn[0]);
    unsigned maxIndex = 0;
    for (unsigned i = 0;  i < zIn.size(); ++i) {
      float expected = (i == E ? 1 : 0);
      float actual = nonlinearity(nonLinearityType, zIn[i]);
      float nlGradient = nonlinearity_derivative(nonLinearityType,
                                                 zIn[i]);
      deltaOut[i] = (expected - actual) * nlGradient;
      sum += (expected - actual) *  (expected - actual);
      if (actual > max) {
        max = actual;
	maxIndex = i;
      }
    }
    indexOut = index;
    bool correct = (maxIndex == E);
    if (correct)
      numCorrect++;
    error += sum;
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};

void weightSync(DataElement<nn_state_t> state,
                ComputeSet weightSyncCS)
{
  state = INIT;
  weightSyncCS.compute();
  state = WEIGHT_SYNC;
  bool complete = false;
  while (!complete) {
    complete = weightSyncCS.compute();
  }
}

void trainOnBatch(DataElement<nn_state_t> state,
                  ComputeSet trainCS, ComputeSet weightSyncCS,
                  DataArray trainingData,  DataArray trainingLabels)
{
  trainingData.copyIn();
  trainingLabels.copyIn();
  state = INIT;
  trainCS.compute();
  state = TRAIN;
  bool complete = false;
  while (!complete) {
    complete = trainCS.compute();
  }
  weightSync(state, weightSyncCS);
}

void testOnBatch(DataElement<nn_state_t> state,
                 ComputeSet testCS,
                 DataArray testData,  DataArray testLabels)
{
  testData.copyIn();
  testLabels.copyIn();
  state = INIT;
  testCS.compute();
  state = TEST;
  bool complete = false;
  while (!complete) {
    complete = testCS.compute();
  }
}
	   

__control__
void runTest(ComputeSet trainCS, ComputeSet testCS, ComputeSet weightSyncCS,
             DataElement<nn_state_t> state,
             DataArray initialParams,
             DataArray trainingData,  DataArray trainingLabels,
             DataArray testData,  DataArray testLabels,
	     unsigned batchSize,
             DataElement<unsigned> numBatches,
             DataElement<unsigned> numCorrect,
             unsigned numTestBatches,
             unsigned numBatchesBetweenTests) {
  std::cout << "-- Initializing params.\n";
  initialParams.copyIn();
  weightSync(state, weightSyncCS);

  std::cout << "-- Training with batch size " << batchSize << ".\n";
  for (unsigned i = 0; i < numBatches; i++) {
    trainOnBatch(state, trainCS, weightSyncCS, trainingData, trainingLabels);
    if (i % numBatchesBetweenTests == 0) {
      numCorrect = 0;
      for (unsigned j = 0; j < numTestBatches; j++) {
        testOnBatch(state, trainCS, testData, testLabels);
      }
      unsigned numTests = (numTestBatches * batchSize);
      float percentCorrect = 100 * ((float) numCorrect) / numTests;
      std::cout << "--- Accuracy after " << i << " batches = "
                << percentCorrect << "%\n";
    }
  }
}
