#include <poplar/GraphProgEnv.hpp>
#include <poplar/GraphBuilder.hpp>
#include <poplar/CPUEngine.hpp>
#include <poplar/IPUModelEngine.hpp>
#include <initializer_list>
#include <vector>
#include <mnist.h>
#include <iostream>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <chrono>
#include <memory>
#include "neural_net_common.h"
#include "VTileMapper.hpp"
using namespace poplar;

class Net;

/* A data set full of test and training data along with its dimensions */
class DataSet {
public:
  std::unique_ptr<float[]> testData, trainingData;
  std::unique_ptr<unsigned char[]> testLabels, trainingLabels;
  unsigned dataSize, numTest, numTraining;
};

/* The hidden layer class represents all non-input layers in the net.
 */
class HiddenLayer {
protected:
  std::vector<VertexRef> vertices;
public:
  virtual void addForward(Net &net) = 0;

  virtual void addBackward(Net &net) = 0;

  std::vector<VertexRef> getVertices() const {
    return vertices;
  }
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
    GraphProgEnv env("neural_net_graph.ppo", GraphProgFileType::Object);
    graphBuilder = std::unique_ptr<GraphBuilder>(new GraphBuilder(env));
    GraphBuilder &builder = *graphBuilder;

    std::cerr << "Constructing graph\n";

    /* First, the global data vertices, compute sets and data arrays
       are created in the graph. */
    numBatchesField = builder.addDataVertex("unsigned")["data"];
    stateField = builder.addDataVertex("nn_state_t")["data"];
    etaField = builder.addDataVertex("float")["data"];

    daTrainingData = builder.createDataArray();
    daTrainingLabels = builder.createDataArray();
    daTestData = builder.createDataArray();
    daTestLabels = builder.createDataArray();
    daParams = builder.createDataArray();

    trainCS = builder.createComputeSet();
    testCS = builder.createComputeSet();
    weightSyncCS = builder.createComputeSet();

    /* Now the input layer is added to the graph. Both the
       test and training data arrays are built up on top of the
       input vertices. */
    std::cerr << "-- Adding input layer\n";
    for (unsigned i = 0; i < inputSize; ++i) {
      auto v = builder.addVertex("InputLayerVertex");
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
    for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
      std::cerr << "-- Adding backward layer " << i << "\n";
      hiddenLayers[i]->addBackward(*this);
    }

    /* Now that the graph is constructed, a poplar engine is created. */

    #if IPU_MODEL
    engineBuilder =
      std::unique_ptr<EngineBuilder>(new IPUModelEngineBuilder(builder));
    #else
    engineBuilder =
      std::unique_ptr<EngineBuilder>(new CPUEngineBuilder(builder));
    #endif

    EngineBuilder &eb = *engineBuilder;

    /* All the data arrays, compute sets and parameter fields need to be
       passed to the control program. */
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

    #if IPU_MODEL
    /* When modelling the IPU, a tile mapping is created based on the
       VTileMapper, each layer is placed on a different 'virtual' tile. */
    IPUModelEngineBuilder *ipuEB =
      static_cast<IPUModelEngineBuilder *>(&eb);
    ipuEB->setNumIPUs(1);
    unsigned superTileDiv = 4;
    ipuEB->setTilesPerIPU(1152/superTileDiv);
    ipuEB->setNumBytesPerTile(256*superTileDiv*1024);
    numTiles = ipuEB->getTilesPerIPU() * ipuEB->getNumIPUs();
    std::cerr << builder.getNumVertices() << " vertices, "
              << numTiles << " tiles.\n";
    VTileMapper mapper;
    for (const std::unique_ptr<HiddenLayer> &layer : hiddenLayers) {
      auto vTile = mapper.createVTile();
      for (VertexRef &v : layer->getVertices()) {
        mapper.addToVTile(vTile, v.getId());
      }
    }
    auto mapping = mapper.createTileMapping(builder.getNumVertices(),
                                            numTiles);
    IPUModelEngineBuilder::UserTilePartitioner p(mapping);
    ipuEB->setTilePartitioner(p);
    #endif

    std::cerr << "Creating graph engine\n";
    engine = eb.makeEngine();
  }

  /* When a Net object is constructed the corrensponding poplar graph is
     made */
  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<HiddenLayer>> &hiddenLayers,
      LossType lossType,
      float learningRate) : batchSize(batchSize),
                            hiddenLayers(std::move(hiddenLayers)),
                            eta(learningRate) {
    initialize(data, lossType);
  }

  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<HiddenLayer>> &&hiddenLayers,
      LossType lossType,
      float learningRate) : batchSize(batchSize),
                            hiddenLayers(std::move(hiddenLayers)),
                            eta(learningRate) {
    initialize(data, lossType);
  }

  void train(unsigned numBatches) {
    /* All this method needs to do is set the relevant parameters and
       run the control program. */
    engine->setValue<float>(etaField, eta);
    engine->setValue<unsigned>(numBatchesField, numBatches);
    std::cerr << "Running graph program\n";

    #if DO_COMPUTATION
    engine->run();
    #endif

    #if IPU_MODEL
    IPUModelEngine *ipuEngine = static_cast<IPUModelEngine *>(&*engine);
    ipuEngine->report(std::cout);
    #if 0
    std::vector<unsigned> tileMapping = engine->getTileMapping();
    for (unsigned i = 0; i < graphBuilder->getNumVertices(); ++i) {
      if (tileMapping[i] == 756)
        engine->dumpVertexInfo(i, std::cout);
    }
    #endif
    #endif

    #if DEBUG_INPUT_OUTPUT_DATA
    for (unsigned j = 0; j < batchSize; ++j) {
      unsigned l = engine->getValue<unsigned char>(errorVertex["labels"][j]);
      std::cout << "LABEL: " << l << "\n";
      for (unsigned x = 0; x < 28; ++x) {
        for (unsigned y = 0; y < 28; ++y) {
          FieldRef f = inputLayer[x*28+y]["data"][j];
          float d = engine->getValue<float>(f);
          if (d == 0)
            std::cout << " ";
          else
            std::cout << ".";
        }
        std::cout << "\n";
      }
    }
    #endif
  }

};


class FullyConnectedLayer : public HiddenLayer {
public:
  unsigned size;
  std::vector<VertexRef> fwd, bwd, weightSyncVertices, prev, biasVertices;
  std::vector<VertexRef> paramGatherVertices;
  unsigned numParamGathers = 20;
  NonLinearityType nonLinearityType, prevNonLinearityType;

  FullyConnectedLayer(unsigned size,
                      NonLinearityType nonLinearityType) :
    size(size),
    nonLinearityType(nonLinearityType) {
    if (numParamGathers > size)
      numParamGathers = size;

    if (size % numParamGathers != 0) {
      numParamGathers = size / (size/numParamGathers + 1);
    }
  }

  void addForward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    prev = net.fwd;
    unsigned prevSize = prev.size();
    prevNonLinearityType = net.prevNonLinearityType;
    net.prevNonLinearityType = nonLinearityType;

    #if USE_GATHER_VERTEX
    VertexRef gatherVertex = builder.addVertex("InnerProductFwdGatherVertex");
    vertices.push_back(gatherVertex);
    builder.addEdge(net.stateField, gatherVertex["state"], false);
    builder.addToComputeSet(net.trainCS, gatherVertex);
    builder.addToComputeSet(net.testCS, gatherVertex);
    builder.setFieldSize(gatherVertex["activationIn"], prevSize);
    builder.setFieldSize(gatherVertex["activationOut"], prevSize);
    builder.addEdge(prev[0]["indexOut"], gatherVertex["indexIn"], true);
    for (unsigned j = 0; j < prevSize; j++) {
      builder.addEdge(prev[j]["activationOut"],
                      gatherVertex["activationIn"][j],
                      true);
    }
    #endif

    VertexRef vCurParamGather;
    for (unsigned i = 0; i < size; ++i) {
      if (i % (size/numParamGathers) == 0) {
        VertexRef v = builder.addVertex("InnerProductParamsGatherVertex");
        vertices.push_back(v);
        paramGatherVertices.push_back(v);
        builder.addToComputeSet(net.weightSyncCS, v);
        builder.setFieldSize(v["weightsIn"], prevSize);
        builder.setFieldSize(v["weightsOut"], prevSize);
        vCurParamGather = v;
      }

      VertexRef v = builder.addVertex("InnerProductFwdVertex");
      builder.addEdge(net.stateField, v["state"], false);
      vertices.push_back(v);
      builder.addToComputeSet(net.trainCS, v);
      builder.addToComputeSet(net.testCS, v);
      fwd.push_back(v);
      builder.setInitialFieldValue<NonLinearityType>(v["nonLinearityType"],
                                                     nonLinearityType);

      builder.addEdge(prev[0]["indexOut"], v["indexIn"], true);
      #if USE_GATHER_VERTEX
      builder.addEdge(gatherVertex["activationOut"],
                      v["activationIn"],
                      false);
      #else
      builder.setFieldSize(v["activationIn"], prevSize);
      for (unsigned j = 0; j < prevSize; j++) {
        builder.addEdge(prev[j]["activationOut"], v["activationIn"][j],
                        true);
      }
      #endif

      VertexRef pv = builder.addVertex("InnerProductParamsVertex");
      vertices.push_back(pv);
      weightSyncVertices.push_back(pv);
      builder.addEdge(net.stateField, pv["state"], false);
      builder.addToComputeSet(net.weightSyncCS, pv);
      builder.setFieldSize(pv["weightsOut"], prevSize);
      VertexRef paramsGatherVertex = vCurParamGather;
      builder.addEdge(paramsGatherVertex["weightsOut"], pv["weightsIn"],
                      false);
      builder.addEdge(pv["weightsOut"], v["weights"], false);
      builder.addEdge(pv["biasOut"], v["bias"], false);
      builder.setInitialFieldValue<unsigned>(pv["myRank"],
                                             i % (size / numParamGathers));

      if (i < prevSize) {
        VertexRef bv = builder.addVertex("InnerProductBwdVertex");
        vertices.push_back(bv);
        bwd.push_back(bv);
      }
    }
    for (unsigned i = size; i < prevSize; ++i) {
        VertexRef bv = builder.addVertex("InnerProductBwdVertex");
        vertices.push_back(bv);
        bwd.push_back(bv);
    }

    net.fwd = fwd;
  }

  void addBackward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    unsigned prevSize = prev.size();
    std::vector<FieldRef> bwdDeltaOut, bwdIndexOut;

    #if USE_GATHER_VERTEX
    VertexRef gatherVertex = builder.addVertex("InnerProductBwdGatherVertex");
    builder.addEdge(net.stateField, gatherVertex["state"], false);
    vertices.push_back(gatherVertex);
    builder.addToComputeSet(net.trainCS, gatherVertex);
    builder.setFieldSize(gatherVertex["deltaIn"], size);
    builder.setFieldSize(gatherVertex["deltaOut"], size);
    builder.addEdge(net.bwdIndexOut[0], gatherVertex["indexIn"], true);
    for (unsigned j = 0; j < size; j++) {
      builder.addEdge(net.bwdDeltaOut[j],
                      gatherVertex["deltaIn"][j],
                      true);
    }
    #endif

    for (unsigned i = 0; i < prevSize; i++) {
      VertexRef v = bwd[i];
      builder.addEdge(net.stateField, v["state"], false);
      builder.addEdge(net.etaField, v["eta"], false);
      builder.addToComputeSet(net.trainCS, v);

      bwdDeltaOut.push_back(v["deltaOut"]);
      bwdIndexOut.push_back(v["indexOut"]);

      builder.setInitialFieldValue<NonLinearityType>(v["nonLinearityType"],
                                                     prevNonLinearityType);

      builder.addEdge(net.bwdIndexOut[0], v["indexIn"], true);

      builder.setFieldSize(v["weights"], size);
      builder.setFieldSize(v["bwdRecord"], net.batchSize);
      builder.setFieldSize(v["actRecord"], net.batchSize);
      builder.setFieldSize(v["zRecord"], net.batchSize);

      #if USE_GATHER_VERTEX
      builder.addEdge(gatherVertex["deltaOut"],
                      v["deltaIn"],
                      false);
      #else
      builder.setFieldSize(v["deltaIn"], size);
      for (unsigned j = 0; j < size; ++j) {
        builder.addEdge(net.bwdDeltaOut[j], v["deltaIn"][j], true);
      }
      #endif

      builder.setFieldSize(v["weightSyncOutput"], numParamGathers);
      for (unsigned j = 0; j < numParamGathers; ++j) {
        builder.addEdge(v["weightSyncOutput"][j],
                        paramGatherVertices[j]["weightsIn"][i],
                        false);
      }
      builder.addToComputeSet(net.weightSyncCS, v);

      builder.addEdge(prev[i]["activationOut"], v["activationIn"], true);
      builder.addEdge(prev[i]["z"], v["zIn"], true);
      builder.addEdge(prev[i]["indexOut"], v["actIndexIn"], true);

      builder.addToDataArray(net.daParams, v["weights"]);
    }

    for (unsigned i = 0; i < size; ++i) {
      VertexRef v = builder.addVertex("InnerProductBwdBiasVertex");
      builder.setInitialFieldValue<unsigned>(v["batchSize"], net.batchSize);
      builder.addEdge(net.stateField, v["state"], false);
      builder.addEdge(net.etaField, v["eta"], false);
      vertices.push_back(v);
      biasVertices.push_back(v);
      builder.addToComputeSet(net.trainCS, v);
      builder.addEdge(net.bwdDeltaOut[i], v["deltaIn"], true);
      builder.addEdge(net.bwdIndexOut[i], v["indexIn"], true);
      builder.addEdge(v["bias"], weightSyncVertices[i]["biasIn"], false);
      builder.addToDataArray(net.daParams, v["bias"]);
    }

    net.bwdDeltaOut = bwdDeltaOut;
    net.bwdIndexOut = bwdIndexOut;
  }
};

int main() {
  std::vector<std::vector<float>> MNISTTestData, MNISTTrainData;
  std::vector<int> MNISTTestLabels, MNISTTrainLabels;
  DataSet MNIST;
  std::cerr << "Reading MNIST data\n";
  MNIST.dataSize = 784;
  MNIST.numTraining = 60000;
  MNIST.numTest = 10000;
  MNIST.testLabels = readMNISTLabels(MNIST.numTest,
                                     "t10k-labels-idx1-ubyte");
  MNIST.testData = readMNISTData(MNIST.numTest,
                                 MNIST.dataSize,
                                 "t10k-images-idx3-ubyte");
  MNIST.trainingData = readMNISTData(MNIST.numTraining,
                                     MNIST.dataSize,
                                     "train-images-idx3-ubyte");
  MNIST.trainingLabels = readMNISTLabels(MNIST.numTraining,
                                         "train-labels-idx1-ubyte");

  #if LARGE_DNN_MODEL
  Net net(MNIST,
          100, // batch size
          makeHiddenLayers({
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(2000, NON_LINEARITY_SIGMOID),

            new FullyConnectedLayer(10, NON_LINEARITY_SIGMOID),
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9 // learning rate
          );
  #else
  Net net(MNIST,
          10, // batch size
          makeHiddenLayers({
            new FullyConnectedLayer(30, NON_LINEARITY_SIGMOID),
            new FullyConnectedLayer(10, NON_LINEARITY_SIGMOID),
          }),
          SOFTMAX_CROSS_ENTROPY_LOSS,
          0.9 // learning rate
          );
  #endif

  net.train(5000);

  return 0;
}
