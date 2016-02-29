#ifndef _net_hpp_
#define _net_hpp_
#include <poplar/GraphProgEnv.hpp>
#include <poplar/GraphBuilder.hpp>
#include <poplar/CPUEngine.hpp>
#include <poplar/IPUModelEngine.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include "neural_net_common.h"

using namespace poplar;

typedef enum NetType {
  TrainingNet,
  TestOnlyNet
} NetType;

class NetOptions {
public:
  bool useIPUModel = false;
  bool doComputation = true;
  bool singleBatchProfile = false;
  unsigned numIPUs = 1;
  bool useSuperTiles = false;
};

class Net;

/* A data set full of test and training data along with its dimensions */
class DataSet {
public:
  std::unique_ptr<float[]> testData, trainingData;
  std::unique_ptr<unsigned char[]> testLabels, trainingLabels;
  unsigned dataSize, numTest, numTraining;
  std::vector<unsigned> dim;
};

/* The hidden layer class represents all non-input layers in the net.
 */
class HiddenLayer {
public:
  virtual void addForward(Net &net) = 0;

  virtual void addBackward(Net &net) = 0;

  virtual bool requiresLayeredInput() = 0;
  virtual bool providesLayeredOutput() = 0;
};

/* This utility function wraps a vector of normal pointers as unique_ptrs.
   It allows the hidden layer array to be initializes with an
   initializer list. */
std::vector<std::unique_ptr<HiddenLayer>>
makeHiddenLayers(std::vector<HiddenLayer *> vs)
{
  std::vector<std::unique_ptr<HiddenLayer>> xs;
  for (auto p: vs)
    xs.push_back(std::unique_ptr<HiddenLayer>(p));
  return xs;
}

/* This class represent the entire network. */
class Net {
public:
  NetType netType;
  NetOptions options;

  unsigned batchSize;
  float eta;
  std::vector<std::unique_ptr<HiddenLayer>> hiddenLayers;

  /* This field references are parameters in the graph that can be
     changed by the control program */
  FieldRef numBatchesField;
  FieldRef stateField;
  FieldRef etaField;

  /* Three compute sets since different vertices run during training, testing
     and weight synchronization */
  ComputeSetRef trainCS;
  ComputeSetRef testCS;
  ComputeSetRef weightSyncCS;

  /* The following data arrays represent data to be copied into and out
     of the network */
  DataArrayRef daParams;  /* The parameters (weights and bias) */
  DataArrayRef daTrainingData;
  DataArrayRef daTrainingLabels;
  DataArrayRef daTestData;
  DataArrayRef daTestLabels;

  /* These parts of the graph are made by the Net object. The hidden
     layer vertices are created via the addForward and addBackward method
     calls of the hidden layer objects. */
  std::vector<VertexRef> inputLayer;
  VertexRef errorVertex;

  /* The following state is used when building up the graph.
     Calling addForward or addBackward will read and update them which
     allows each layer to pass information to the next layer when
     building the graph. */
  NonLinearityType prevNonLinearityType;
  std::vector<VertexRef> fwd;
  unsigned xDim, yDim;
  unsigned prevLayers, prevChunks;

  std::vector<FieldRef> bwdDeltaOut;
  std::vector<FieldRef> bwdIndexOut;

  /* The training and testing data needs to be copied in a different
     layout to be copied into the graph correctly. */
  std::unique_ptr<float[]> shuffledTrainingData;
  std::unique_ptr<float[]> shuffledTestData;
  std::unique_ptr<float[]> params;

  /* Poplar graph creation state. */
  std::unique_ptr<GraphBuilder> graphBuilder;
  std::unique_ptr<EngineBuilder> engineBuilder;
  std::unique_ptr<Engine> engine;
  unsigned numTiles;

  /* This function shuffles data into a good layout to be transferred into
     the graph.
     The data is arranged into batches and each batch.
  */
  std::unique_ptr<float[]> shuffleData(const float *data,
                                       unsigned numItems,
                                       unsigned dataSize,
                                       unsigned batchSize) {
    auto shuffled = std::unique_ptr<float[]>(new float[numItems * dataSize]);
    for (unsigned i = 0; i < numItems/batchSize; ++i) {
      for (unsigned j = 0; j < batchSize; ++j) {
        for (unsigned k = 0; k < dataSize; ++k) {
          float x = data[i * batchSize * dataSize + j * dataSize + k];
          shuffled[i * batchSize * dataSize + k * batchSize + j] = x;
        }
      }
    }
    return std::move(shuffled);
  }

  void initialize(DataSet &data, LossType lossType) {
    unsigned inputSize = data.dataSize;
    GraphProgEnv env("obj/neural_net_graph.ppo", GraphProgFileType::Object);
    graphBuilder = std::unique_ptr<GraphBuilder>(new GraphBuilder(env));
    GraphBuilder &builder = *graphBuilder;

    std::cerr << "Constructing graph\n";

    /* First, the global data vertices, compute sets and data arrays
       are created in the graph. */
    numBatchesField = builder.addDataVertex("unsigned")["data"];
    stateField = builder.addDataVertex("nn_state_t")["data"];
    etaField = builder.addDataVertex(FPTypeStr)["data"];

    daTrainingData = builder.createDataArray();
    daTrainingLabels = builder.createDataArray();
    daTestData = builder.createDataArray();
    daTestLabels = builder.createDataArray();
    daParams = builder.createDataArray();

    trainCS = builder.createComputeSet();
    testCS = builder.createComputeSet();
    weightSyncCS = builder.createComputeSet();

    bool layeredInput = data.dim.size() == 3;

    /* Now the input layer is added to the graph. Both the
       test and training data arrays are built up on top of the
       input vertices. */
    std::cerr << "-- Adding input layer\n";
    for (unsigned i = 0; i < inputSize; ++i) {
      VertexRef v;
      if (layeredInput) {
        v = builder.addVertex("LayeredInputVertex");
        builder.setFieldSize(v["activationOut"], data.dim[2]);
        prevLayers = data.dim[2];
        prevChunks = 1;
      } else {
        v = builder.addVertex("InputVertex");
        prevLayers = 0;
        prevChunks = 1;
      }

      builder.addToComputeSet(trainCS, v);
      builder.addToComputeSet(testCS, v);
      builder.setFieldSize(v["data"], batchSize);
      builder.addToDataArray(daTrainingData, v["data"]);
      builder.addToDataArray(daTestData, v["data"]);
      builder.setInitialFieldValue<unsigned>(v["batchSize"], batchSize);
      builder.addEdge(stateField, v["state"], false);
      fwd.push_back(v);
      inputLayer.push_back(v);
    }

    if (data.dim.size() >= 2) {
      xDim = data.dim[0];
      yDim = data.dim[1];
    }

    /* The hidden layers are added in order. Each layer
       will use and then update
       the 'fwd' and 'prevNonLinearityType' fields of this
       object. */
    prevNonLinearityType = NON_LINEARITY_NONE;
    for (unsigned i = 0; i < hiddenLayers.size(); ++i) {
      std::cerr << "-- Adding forward layer " << i << "\n";
      hiddenLayers[i]->addForward(*this);
    }

    /* The loss layer connects to the final hidden layer. */
    std::cerr << "-- Adding loss layer\n";
    errorVertex = builder.addVertex("ErrorVertex");
    builder.setInitialFieldValue<LossType>(errorVertex["lossType"], lossType);
    builder.addEdge(stateField, errorVertex["state"], false);
    builder.setInitialFieldValue<NonLinearityType>(
        errorVertex["nonLinearityType"],
        prevNonLinearityType);
    builder.addToComputeSet(trainCS, errorVertex);
    builder.addToComputeSet(testCS, errorVertex);
    builder.addEdge(fwd[0]["indexOut"], errorVertex["indexIn"], true);
    builder.setInitialFieldValue<unsigned>(errorVertex["batchSize"], batchSize);
    builder.setFieldSize(errorVertex["deltaOut"], fwd.size());
    builder.setFieldSize(errorVertex["probs"], fwd.size());
    builder.setFieldSize(errorVertex["labels"], batchSize);
    builder.addToDataArray(daTrainingLabels, errorVertex["labels"]);
    builder.addToDataArray(daTestLabels, errorVertex["labels"]);
    builder.setFieldSize(errorVertex["zIn"], fwd.size());
    for (unsigned i = 0; i < fwd.size(); i++) {
      builder.addEdge(fwd[i]["z"],
                      errorVertex["zIn"][i],
                      true);
      bwdDeltaOut.push_back(errorVertex["deltaOut"][i]);
      bwdIndexOut.push_back(errorVertex["indexOut"]);
    }

    /* Finally the backward pass vertices are adding in reverse
       layer order.
       Each layer will use and then update the 'bwdDeltaOut' and
       'bwdIndexOut' fields of this object. */
    if (netType == TrainingNet) {
      for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
        std::cerr << "-- Adding backward layer " << i << "\n";
        hiddenLayers[i]->addBackward(*this);
      }
    }

    /* Now that the graph is constructed, a poplar engine is created. */

    if (options.useIPUModel) {
      engineBuilder =
        std::unique_ptr<EngineBuilder>(new IPUModelEngineBuilder(builder));
    } else {
      engineBuilder =
        std::unique_ptr<EngineBuilder>(new CPUEngineBuilder(builder));
    }

    EngineBuilder &eb = *engineBuilder;

    /* All the data arrays, compute sets and parameter fields need to be
       passed to the control program. */
    if (netType == TrainingNet) {
      eb.setControlProgram("doTraining");
      eb.setControlProgramArg(0, trainCS);
      eb.setControlProgramArg(1, testCS);
      eb.setControlProgramArg(2, weightSyncCS);
      eb.setControlProgramArg(3, stateField);
      eb.setControlProgramArg(4, daParams);
      eb.setControlProgramArg(5, daTrainingData);
      eb.setControlProgramArg(6, daTrainingLabels);
      eb.setControlProgramArg(7, daTestData);
      eb.setControlProgramArg(8, daTestLabels);
      eb.setControlProgramArg<unsigned>(9, batchSize);
      eb.setControlProgramArg(10, numBatchesField);
      eb.setControlProgramArg(11, errorVertex["numCorrect"]);
      eb.setControlProgramArg(12, data.numTest / batchSize);
      eb.setControlProgramArg(13, 500);
      eb.setControlProgramArg<bool>(14, options.singleBatchProfile);
    } else {
      eb.setControlProgram("doTest");
      eb.setControlProgramArg(0, trainCS);
      eb.setControlProgramArg(1, stateField);
      eb.setControlProgramArg(2, daParams);
      eb.setControlProgramArg(3, daTestData);
      eb.setControlProgramArg(4, daTestLabels);
      eb.setControlProgramArg<unsigned>(5, batchSize);
      eb.setControlProgramArg(6, numBatchesField);
      eb.setControlProgramArg(7, errorVertex["numCorrect"]);
      eb.setControlProgramArg(8, data.numTest / batchSize);
      eb.setControlProgramArg<bool>(9, options.singleBatchProfile);
    }



    /* The parameters (i.e. weights and biases) data array is linked to
       an array of randomly created values based on a normal distribution. */
    unsigned numParams = eb.dataArrayBufferSize(daParams) / sizeof(float);
    params = std::unique_ptr<float[]>(new float[numParams]);
    unsigned seed = time(0);
    boost::variate_generator< boost::mt19937, boost::normal_distribution<> >
      generator(boost::mt19937(seed), boost::normal_distribution<>(0, 0.01));
    for (unsigned i = 0; i < numParams; ++i)
      params[i] = generator();
    eb.linkDataArrayToBuffer(daParams, (char *) &params[0]);


    /* Link the training data array to the shuffled training data on the
       host. */
    shuffledTrainingData = shuffleData(&data.trainingData[0],
                                       data.numTraining,
                                       inputSize,
                                       batchSize);
    eb.linkDataArrayToCircularBuffer(
      daTrainingData,
      (char *) &shuffledTrainingData[0],
      (char *) &shuffledTrainingData[data.numTraining * inputSize]);

    /* Link the training labels to the correct array on the host. */
    eb.linkDataArrayToCircularBuffer(
      daTrainingLabels,
      (char *) &data.trainingLabels[0],
      (char *) &data.trainingLabels[data.numTraining]);

    /* Link the test data array to the shuffled test data on the
       host. */
    shuffledTestData = shuffleData(&data.testData[0],
                                   data.numTest,
                                   inputSize,
                                   batchSize);
    eb.linkDataArrayToCircularBuffer(
      daTestData,
      (char *) &shuffledTestData[0],
      (char *) &shuffledTestData[data.numTest * inputSize]);

    /* Link the test labels to the correct array on the host. */
    eb.linkDataArrayToCircularBuffer(
      daTestLabels,
      (char *) &data.testLabels[0],
      (char *) &data.testLabels[data.numTest]);

    if (options.useIPUModel) {
      IPUModelEngineBuilder *ipuEB =
        static_cast<IPUModelEngineBuilder *>(&eb);
      ipuEB->setNumIPUs(options.numIPUs);
      unsigned superTileDiv = options.useSuperTiles ? 4 : 1;
      ipuEB->setTilesPerIPU(1152/superTileDiv);
      ipuEB->setNumBytesPerTile(256*superTileDiv*1024);
      numTiles = ipuEB->getTilesPerIPU() * ipuEB->getNumIPUs();
      ipuEB->setIPUExchangeImplementation(IPUModelEngineBuilder::OPTIMISTIC_WITH_MULTICAST);

      std::cerr << builder.getNumVertices() << " vertices, "
                << numTiles << " tiles.\n";
    }

    std::cerr << "Creating graph engine\n";
    engine = eb.makeEngine();
  }

  /* When a Net object is constructed the corrensponding poplar graph is
     made */
  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<HiddenLayer>> &hiddenLayers,
      LossType lossType,
      float learningRate,
      NetType netType,
      NetOptions options = NetOptions()) : netType(netType), options(options),
                         batchSize(batchSize),
                         hiddenLayers(std::move(hiddenLayers)),
                         eta(learningRate) {
    initialize(data, lossType);
  }

  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<HiddenLayer>> &&hiddenLayers,
      LossType lossType,
      float learningRate,
      NetType netType,
      NetOptions options = NetOptions()) : netType(netType), options(options),
                         batchSize(batchSize),
                         hiddenLayers(std::move(hiddenLayers)),
                         eta(learningRate) {
    initialize(data, lossType);
  }

  void run(unsigned numBatches) {
    /* All this method needs to do is set the relevant parameters and
       run the control program. */
    engine->setValue<float>(etaField, eta);
    engine->setValue<unsigned>(numBatchesField, numBatches);
    std::cerr << "Running graph program\n";


    if (options.doComputation) {
      engine->run();
    }

    if (options.useIPUModel) {
      IPUModelEngine *ipuEngine = static_cast<IPUModelEngine *>(&*engine);
      ipuEngine->report(std::cout);
      #if 0
      std::vector<unsigned> tileMapping = engine->getTileMapping();
      for (unsigned i = 0; i < graphBuilder->getNumVertices(); ++i) {
        if (tileMapping[i] == 756)
          engine->dumpVertexInfo(i, std::cout);
      }
      #endif
    }

  }

};

#endif //_net_hpp_
