#include <Vertex.hpp>
#include <ComputeSet.hpp>
#include <DataElement.hpp>
#include <DataArray.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

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


class InputLayerVertex : public Vertex {
public:
  Vector<float> data;
  float activation_out;
  int index_out;

  Input<nn_state_t> state;
  Input<unsigned> base_index;
  Input<unsigned> batch_size;

  bool compute() {
    if (state == INIT) {
      index_out = NOTHING_TO_PROCESS;
      return true;
    }
    if (index_out == DONE_PROCESSING)
      return true;

    if (index_out == NOTHING_TO_PROCESS)
      index_out = 0;
    else
      index_out++;

    if (index_out == batch_size) {
      index_out = DONE_PROCESSING;
      return true;
    }

    /* The input layer will simple output data values until a whole
       batch has been output. */
    activation_out = data[base_index + index_out];
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};

class InnerProductFwdGatherVertex : public Vertex {
public:
  Vector<Input<float>> activation_in;
  Vector<float> activation_out;

  Input<nn_state_t> state;
  Input<int> index_in;

  bool compute() {
    if (state == INIT)
      return true;

    if (index_in == NOTHING_TO_PROCESS ||
        index_in == DONE_PROCESSING)
      return true;

    for (unsigned i = 0; i < activation_out.size(); ++i) {
      activation_out[i] = activation_in[i];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};

class InnerProductFwdVertex : public Vertex {
public:
#if USE_GATHER_VERTEX
  Input<Vector<float>> activation_in;
#else
  Vector<Input<float>> activation_in;
#endif
  Input<int> index_in;
  float activation_out;
  int index_out;

  Input<Vector<float>> weights;
  Input<float> bias;

  Input<nn_state_t> state;

  int debug = 0;

  bool compute() {
    if (state == INIT) {
      index_out = NOTHING_TO_PROCESS;
      return true;
    }

    if (index_in == NOTHING_TO_PROCESS)
      return false;

    if (index_in == DONE_PROCESSING) {
      index_out = DONE_PROCESSING;
      return true;
    }

    int do_debug = (debug && index_in == 0);

    if (do_debug) 
      std::cout << "----FORWARD---\n";
    float sum = 0;
    for (unsigned i = 0;  i < activation_in.size(); ++i) {
      if (do_debug) 
	std::cout << "W:" << weights[i] << " * " << activation_in[i] << "\n";
      sum += activation_in[i] * weights[i];
    }
    activation_out = sum + bias;
    if (do_debug) 
      std::cout << "Sum:" << sum << " + bias:" << bias << " = " << activation_out << "\n"; 
    index_out = index_in;
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }

};

#if MEM_OPTIMIZED_WEIGHT_SYNC

class InnerProductParamsGatherVertex : public Vertex {
public:
  Vector<Input<float>> weights_in;
  Vector<float> weights_out;

  bool compute() {
    for (unsigned i = 0; i < weights_out.size(); ++i) {
      weights_out[i] = weights_in[i];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }

};


class InnerProductParamsVertex : public Vertex {
public:
  Input<Vector<float>> weights_in;
  Vector<float> weights_out;
  Input<float> bias_in;
  float bias_out;

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
      for (unsigned i = 0; i < weights_out.size(); ++i) {
        weights_out[i] = weights_in[i];
      }
      bias_out = bias_in;
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
  Vector<Input<float>> weights_in;
  Vector<float> weights_out;
  Input<float> bias_in;
  float bias_out;
 
  bool compute() {
    for (unsigned i = 0; i < weights_out.size(); ++i) {
      weights_out[i] = weights_in[i];
    }
    bias_out = bias_in;
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};

#endif

class InnerProductBwdGatherVertex : public Vertex {
public:
  Vector<Input<float>> delta_in;
  Vector<float> delta_out;

  Input<nn_state_t> state;
  Input<int> index_in;

  bool compute() {
    if (state == INIT || state == TEST)
      return true;

    if (index_in == NOTHING_TO_PROCESS ||
        index_in == DONE_PROCESSING ||
        index_in == START_WEIGHT_UPDATE)
      return true;

    for (unsigned i = 0; i < delta_out.size(); ++i) {
      delta_out[i] = delta_in[i];
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};


class InnerProductBwdVertex : public Vertex {
public:
#if USE_GATHER_VERTEX
  Input<Vector<float>> delta_in;
#else
  Vector<Input<float>> delta_in;
#endif
  Input<int> index_in;
  float delta_out;
  int index_out;

  Vector<float> weights;

  Input<float> activation_in;
  Input<int> act_index_in;
  Vector<float> fwdRecord, bwdRecord;

  Input<float> eta;
  Input<nn_state_t> state;

  #if MEM_OPTIMIZED_WEIGHT_SYNC
  float weightSyncOutput;
  unsigned currentRank;
  #endif

  int debug = 0;
  bool doingWeightUpdate;
  bool compute() {

    if (state == INIT) {
      index_out = NOTHING_TO_PROCESS;
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
    if (act_index_in != NOTHING_TO_PROCESS &&
	act_index_in != DONE_PROCESSING) {
      fwdRecord[act_index_in] = activation_in;
    }

    int do_debug = debug;

    if (do_debug)
      std::cout << "---BACKWARD---- (" << index_in << ")\n";

    if (doingWeightUpdate) {
      if (index_in == DONE_PROCESSING) {
        index_out = DONE_PROCESSING;
        return true;
      }
      unsigned batchSize = fwdRecord.size();
      for (unsigned i = 0;  i < delta_in.size(); ++i) {
        weights[i] += eta * fwdRecord[index_in] * delta_in[i] / batchSize;
      }
      delta_out = bwdRecord[index_in];
      index_out = index_in;
    } else {
      if (index_in == NOTHING_TO_PROCESS)
        return false;

      if (index_in == START_WEIGHT_UPDATE) {
        doingWeightUpdate = true;
        index_out = START_WEIGHT_UPDATE;
        return false;
      }

      float sum = 0;
      for (unsigned i = 0;  i < delta_in.size(); ++i) {
        if (do_debug) 
          std::cout << "W:" << weights[i] << " * " << delta_in[i] << "\n";
        sum += delta_in[i] * weights[i];
      }
      delta_out = sum;
      bwdRecord[index_in] = delta_out;
      index_out = index_in;
    }
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};


class InnerProductBwdBiasVertex : public Vertex {
public:
  Input<float> delta_in;
  Input<int> index_in;

  float bias;
  
  Input<float> eta;
  Input<nn_state_t> state;
  Input<unsigned> batch_size;

  float update;
  bool updated;

  unsigned debug = 0;

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

    if (index_in == NOTHING_TO_PROCESS)
      return false;

    if (index_in == START_WEIGHT_UPDATE) {
      if (debug)
        std::cout << "BIAS update: " << bias << " + " << eta << " * " << update << " / " << batch_size << "\n";
      bias += eta * update / (float) batch_size;
      if (debug)
        std::cout << "Result: " << bias << "\n";
      updated = true;
      return true;
    }

    if (index_in == DONE_PROCESSING)
      return true;

    if (debug) {
      std::cout << "Bias error:" << delta_in << "\n";
      std::cout << "Update:" << update << "\n";
    }
    update += delta_in;
    if (debug) {
      std::cout << "Update:" << update << "\n";
    }
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};


class ReLUFwdVertex : public Vertex {
public:
  Input<float> activation_in;
  Input<int> index_in;
  float activation_out;
  int index_out;

  Input<nn_state_t> state;

  bool compute() {
    if (state == INIT) {
      index_out = NOTHING_TO_PROCESS;
      return true;
    }

    if (index_in == NOTHING_TO_PROCESS)
      return false;

    if (index_in == DONE_PROCESSING) {
      index_out = DONE_PROCESSING;
      return true;
    }

    if (activation_in > 0)
      activation_out = activation_in;
    else
      activation_out = 0;

    index_out = index_in;
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};

class ReLUBwdVertex : public Vertex {
public:
  Input<float> delta_in;
  Input<int> index_in;
  float delta_out;
  int index_out;

  Input<float> activation_in;
  Input<int> act_index_in;
  Vector<float> record;

  Input<nn_state_t> state;

  bool compute() {

    if (state == INIT) {
      index_out = NOTHING_TO_PROCESS;
      return true;
    }

    if (state == TEST)
      return true;

    // During the forward pass the backwards vertex needs to record 
    // the activations going through the network.
    if (act_index_in != NOTHING_TO_PROCESS &&
	act_index_in != DONE_PROCESSING) {
      record[act_index_in] = activation_in;
    }

    if (index_in == NOTHING_TO_PROCESS ||
        index_in == START_WEIGHT_UPDATE) {
      index_out = index_in;
      return false;
    }

    if (index_in == DONE_PROCESSING) {
      index_out = DONE_PROCESSING;
      return true;
    }

    float activation_in = record[index_in];
    float gradient;
    if (activation_in > 0)
      gradient = 1;
    else
      gradient = 0;
    delta_out = gradient * delta_in;
    index_out = index_in;
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};


class SigmoidFwdVertex : public Vertex {
public:
  Input<float> activation_in;
  Input<int> index_in;
  float activation_out;
  int index_out;

  Input<nn_state_t> state;

  bool compute() {
    if (state == INIT) {
      index_out = NOTHING_TO_PROCESS;
      return true;
    }

    if (index_in == NOTHING_TO_PROCESS)
      return false;

    if (index_in == DONE_PROCESSING) {
      index_out = DONE_PROCESSING;
      return true;
    }

    activation_out = sigmoid(activation_in);

    index_out = index_in;
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};

class SigmoidBwdVertex : public Vertex {
public:
  Input<float> delta_in;
  Input<int> index_in;
  float delta_out;
  int index_out;

  Input<float> activation_in;
  Input<int> act_index_in;
  Vector<float> record;

  Input<nn_state_t> state;

  bool compute() {
    if (state == INIT) {
      index_out = NOTHING_TO_PROCESS;
      return true;
    }

    if (state == TEST)
      return true;

    // During the forward pass the backwards vertex needs to record 
    // the activations going through the network.
    if (act_index_in != NOTHING_TO_PROCESS &&
	act_index_in != DONE_PROCESSING) {
      record[act_index_in] = activation_in;
    }


    if (index_in == NOTHING_TO_PROCESS ||
        index_in == START_WEIGHT_UPDATE) {
      index_out = index_in;
      return false;
    }

    if (index_in == DONE_PROCESSING) {
      index_out = DONE_PROCESSING;
      return true;
    }

    float activation_in = record[index_in];
    float gradient = sigmoid_derivative(activation_in);
    delta_out = gradient * delta_in;
    index_out = index_in;
    return false;
  }

  uint64_t getCycleEstimate() const {
    return 30;
  }
};



class SumSquaredErrorVertex : public Vertex {
public:
  Vector<Input<float>> activation_in;
  Input<int> index_in;
  float error;

  Vector<float> delta_out;
  int index_out;

  Vector<unsigned char> labels;

  Input<nn_state_t> state;
  Input<unsigned> base_index;

  unsigned numCorrect = 0;

  bool doingWeightUpdate;

  Input<unsigned> batchSize;

  bool compute() {
    if (state == INIT) {
      error = 0;
      index_out = NOTHING_TO_PROCESS;
      doingWeightUpdate = false;
      return true;
    }

    if (doingWeightUpdate) {

      if (index_out == DONE_PROCESSING)
        return true;

      if (index_out == START_WEIGHT_UPDATE) {
        index_out = 0;
      } else {
        index_out++;
        if (index_out == batchSize) {
          index_out = DONE_PROCESSING;
          return true;
        }
      }

    } else {

      if (index_in == NOTHING_TO_PROCESS)
        return false;

      if (index_in == DONE_PROCESSING) {
        index_out = START_WEIGHT_UPDATE;
        doingWeightUpdate = true;
        return true;
      }

    }

    unsigned index = doingWeightUpdate ? index_out : index_in;

    unsigned E = labels[base_index + index];

    float sum = 0;
    float max = activation_in[0];
    unsigned max_index = 0;
    for (unsigned i = 0;  i < activation_in.size(); ++i) {  
      float expected = (i == E ? 1 : 0);
      float actual = activation_in[i];
      delta_out[i] = expected - actual;
      sum += (expected - actual) *  (expected - actual);
      if (activation_in[i] > max) {
	max = activation_in[i];
	max_index = i;
      }
    }
    index_out = index;
    bool correct = (max_index == E);
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
	     DataElement<unsigned> baseIndex,
	     unsigned batchSize,
             DataElement<unsigned> numBatches,
             DataElement<unsigned> numCorrect,
             unsigned numTestBatches,
             unsigned numBatchesBetweenTests) {
  std::cout << "-- Initializing params.\n";
  initialParams.copyIn();
  weightSync(state, weightSyncCS);

  std::cout << "-- Training with batch size " << batchSize << ".\n";
  baseIndex = 0;
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
