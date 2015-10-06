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
using namespace poplar;

#define MEM_EXPERIMENT 0

class Net;

class DataSet {
public:
  std::unique_ptr<float[]> testData, trainingData;
  std::unique_ptr<unsigned char[]> testLabels, trainingLabels;
  unsigned dataSize, numTest, numTraining;
};

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

class VTileMapper {
  std::vector<std::vector<unsigned>> vTiles;
public:
  unsigned createVTile() {
    vTiles.push_back(std::vector<unsigned>());
    return vTiles.size() - 1;
  }

  void addToVTile(unsigned vTile, unsigned vertex_id) {
    vTiles[vTile].push_back(vertex_id);
  }

  std::vector<unsigned> createTileMapping(unsigned numVertices,
                                          unsigned numTiles) {
    unsigned invalidTile = numTiles + 1;
    std::vector<unsigned> map(numVertices, invalidTile);
    double vertsPerTile = (double) numVertices / numTiles;
    double vertsPerTileFrac = vertsPerTile - (double ((unsigned) vertsPerTile));
    double remainder = 0;
    unsigned curVert = 0;
    std::vector<unsigned> vTileOrdered;
    for (auto v : vTiles)
      vTileOrdered.insert(vTileOrdered.end(), v.begin(), v.end());

    auto vTileIter = vTileOrdered.begin();

    for (unsigned i = 0; i < numTiles; i++) {
      unsigned vertsOnThisTile = (unsigned) vertsPerTile;
      remainder += vertsPerTileFrac;
      if (remainder > 1) {
        remainder -= 1;
        vertsOnThisTile += 1;
      }
      for (unsigned j = 0; j < vertsOnThisTile; ++j) {
        if (vTileIter != vTileOrdered.end()) {
          map[*vTileIter] = i;
          ++vTileIter;
        } else {
          while (map[curVert] != invalidTile)
            curVert++;
          map[curVert] = i;
          curVert++;
        }
      }
    }
    std::cout << vTiles[0].size() << "\n";
    for (unsigned i = 0; i < vTiles[0].size(); i++) {
      //      std::cout << map[vTiles[0][i]] << "\n";
    }
    return map;
  }
};

class Net {
public:
  unsigned batchSize;
  std::vector<HiddenLayer *> hiddenLayers;

  FieldRef baseIndexField;
  FieldRef batchSizeField;
  FieldRef numBatchesField;
  FieldRef stateField;
  FieldRef etaField;

  ComputeSetRef trainCS;
  ComputeSetRef testCS;
  ComputeSetRef weightSyncCS;

  DataArrayRef daParams;
  DataArrayRef daTrainingData;
  DataArrayRef daTrainingLabels;
  DataArrayRef daTestData;
  DataArrayRef daTestLabels;

  std::vector<VertexRef> inputLayer;
  std::vector<VertexRef> fwd;
  VertexRef errorVertex;

  std::vector<FieldRef> bwdDeltaOut;
  std::vector<FieldRef> bwdIndexOut;

  std::unique_ptr<GraphBuilder> graphBuilder;
  std::unique_ptr<IPUModelEngine> engine;

  std::unique_ptr<float[]> shuffledTrainingData;
  std::unique_ptr<float[]> shuffledTestData;
  std::unique_ptr<float[]> params;

  unsigned numTiles;

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

  Net(DataSet &data, unsigned batchSize,
      std::vector<HiddenLayer *> hiddenLayers) : batchSize(batchSize),
                                                 hiddenLayers(hiddenLayers) {
    GraphProgEnv env("neural_net_graph.ppo", GraphProgFileType::Object);
    graphBuilder = std::unique_ptr<GraphBuilder>(new GraphBuilder(env));
    GraphBuilder &builder = *graphBuilder;

    std::cerr << "Constructing graph\n";

    unsigned inputSize = data.dataSize;

    baseIndexField = builder.addDataVertex("unsigned")["data"];
    batchSizeField = builder.addDataVertex("unsigned")["data"];
    numBatchesField = builder.addDataVertex("unsigned")["data"];
    stateField = builder.addDataVertex("nn_state_t")["data"];
    etaField = builder.addDataVertex("float")["data"];

    for (auto vType: {"InputLayerVertex",
                      "InnerProductFwdVertex", "InnerProductBwdVertex",
                      "InnerProductFwdGatherVertex",
                      "InnerProductBwdGatherVertex",
                      "InnerProductBwdBiasVertex",
                      #if MEM_OPTIMIZED_WEIGHT_SYNC
                      "InnerProductParamsVertex",
                      #endif
                      "ReLUFwdVertex", "ReLUBwdVertex",
                      "SigmoidFwdVertex", "SigmoidBwdVertex",
                      "SumSquaredErrorVertex"}) {
      builder.setAutoInEdge(vType, "state", stateField, false);
    }

    for (auto vType: {"InnerProductBwdVertex", "InnerProductBwdBiasVertex"}) {
      builder.setAutoInEdge(vType, "eta", etaField, false);
    }

    for (auto vType: {"InnerProductBwdBiasVertex"}) {
      builder.setAutoInEdge(vType, "batch_size", batchSizeField, false);
    }

    daTrainingData = builder.createDataArray();
    daTrainingLabels = builder.createDataArray();
    daTestData = builder.createDataArray();
    daTestLabels = builder.createDataArray();
    daParams = builder.createDataArray();

    trainCS = builder.createComputeSet();
    testCS = builder.createComputeSet();
    weightSyncCS = builder.createComputeSet();

    std::cerr << "-- Adding input layer\n";
    for (unsigned i = 0; i < inputSize; ++i) {
      auto v = builder.addVertex("InputLayerVertex");
      builder.addToComputeSet(trainCS, v);
      builder.addToComputeSet(testCS, v);
      builder.setFieldSize(v["data"], batchSize);
      builder.addToDataArray(daTrainingData, v["data"]);
      builder.addToDataArray(daTestData, v["data"]);
      builder.addEdge(baseIndexField, v["base_index"], false);
      builder.addEdge(batchSizeField, v["batch_size"], false);
      fwd.push_back(v);
      inputLayer.push_back(v);
    }

    for (unsigned i = 0; i < hiddenLayers.size(); ++i) {
      std::cerr << "-- Adding forward layer " << i << "\n";
      hiddenLayers[i]->addForward(*this);
    }

    std::cerr << "-- Adding loss layer\n";
    errorVertex = builder.addVertex("SumSquaredErrorVertex");
    builder.addToComputeSet(trainCS, errorVertex);
    builder.addToComputeSet(testCS, errorVertex);
    builder.addEdge(fwd[0]["index_out"], errorVertex["index_in"], true);
    builder.addEdge(baseIndexField, errorVertex["base_index"], false);
    builder.addEdge(batchSizeField, errorVertex["batchSize"], false);
    builder.setFieldSize(errorVertex["delta_out"], fwd.size());
    builder.setFieldSize(errorVertex["labels"], batchSize);
    builder.addToDataArray(daTrainingLabels, errorVertex["labels"]);
    builder.addToDataArray(daTestLabels, errorVertex["labels"]);
    builder.setFieldSize(errorVertex["activation_in"], fwd.size());
    for (unsigned i = 0; i < fwd.size(); i++) {
      builder.addEdge(fwd[i]["activation_out"],
                      errorVertex["activation_in"][i],
                      true);
      bwdDeltaOut.push_back(errorVertex["delta_out"][i]);
      bwdIndexOut.push_back(errorVertex["index_out"]);
    }

    for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
      std::cerr << "-- Adding backward layer " << i << "\n";
      hiddenLayers[i]->addBackward(*this);
    }

    IPUModelEngineBuilder eb(builder);

    eb.setControlProgram("runTest");
    eb.setControlProgramArg(0, trainCS);
    eb.setControlProgramArg(1, testCS);
    eb.setControlProgramArg(2, weightSyncCS);
    eb.setControlProgramArg(3, stateField);
    eb.setControlProgramArg(4, daParams);
    eb.setControlProgramArg(5, daTrainingData);
    eb.setControlProgramArg(6, daTrainingLabels);
    eb.setControlProgramArg(7, daTestData);
    eb.setControlProgramArg(8, daTestLabels);
    eb.setControlProgramArg(9, baseIndexField);
    eb.setControlProgramArg<unsigned>(10, batchSize);
    eb.setControlProgramArg(11, numBatchesField);
    eb.setControlProgramArg(12, errorVertex["numCorrect"]);
    eb.setControlProgramArg(13, data.numTest / batchSize);
    eb.setControlProgramArg(14, 500);

    unsigned numParams = eb.dataArrayBufferSize(daParams) / sizeof(float);
    params = std::unique_ptr<float[]>(new float[numParams]);
    unsigned seed = time(0);
    boost::variate_generator< boost::mt19937, boost::normal_distribution<> >
      generator(boost::mt19937(seed), boost::normal_distribution<>(0, 0.01));

    for (unsigned i = 0; i < numParams; ++i)
      params[i] = generator();

    eb.linkDataArrayToBuffer(daParams, (char *) &params[0]);

    shuffledTrainingData = shuffleData(&data.trainingData[0],
                                       data.numTraining,
                                       inputSize,
                                       batchSize);

    eb.linkDataArrayToCircularBuffer(
      daTrainingData,
      (char *) &shuffledTrainingData[0],
      (char *) &shuffledTrainingData[data.numTraining * inputSize]);

    eb.linkDataArrayToCircularBuffer(
      daTrainingLabels,
      (char *) &data.trainingLabels[0],
      (char *) &data.trainingLabels[data.numTraining]);

    shuffledTestData = shuffleData(&data.testData[0],
                                   data.numTest,
                                   inputSize,
                                   batchSize);

    eb.linkDataArrayToCircularBuffer(
      daTestData,
      (char *) &shuffledTestData[0],
      (char *) &shuffledTestData[data.numTest * inputSize]);

    eb.linkDataArrayToCircularBuffer(
      daTestLabels,
      (char *) &data.testLabels[0],
      (char *) &data.testLabels[data.numTest]);

    eb.setIPUExchangeImplementation(IPUModelEngineBuilder::OPTIMISTIC);

    numTiles = eb.getTilesPerIPU() * eb.getNumIPUs();

    std::cerr << builder.getNumVertices() << " vertices, "
              << numTiles << " tiles.\n";

    VTileMapper mapper;
    for (HiddenLayer *layer : hiddenLayers) {
      auto vTile = mapper.createVTile();
      for (VertexRef &v : layer->getVertices()) {
        mapper.addToVTile(vTile, v.getId());
      }
    }

    auto mapping = mapper.createTileMapping(builder.getNumVertices(),
                                            numTiles);

    IPUModelEngineBuilder::UserTilePartitioner p(mapping);
    eb.setTilePartitioner(p);

    std::cerr << "Creating graph engine\n";
    engine = eb.makeIPUModelEngine();
  }

  void test(unsigned numBatches) {
    engine->setValue<unsigned>(baseIndexField, 0);
    engine->setValue<unsigned>(batchSizeField, batchSize);
    engine->setValue<float>(etaField, 0.9);
    engine->setValue<unsigned>(numBatchesField, numBatches);
    std::cerr << "Running graph program\n";

    #if MEM_EXPERIMENT
    engine->report(std::cout);
    std::vector<unsigned> tileMapping = engine->getTileMapping();
    for (unsigned i = 0; i < graphBuilder->getNumVertices(); ++i) {
      if (tileMapping[i] == 487)
        engine->dumpVertexInfo(i, std::cout);
    }
    #else
    engine->run();
    #endif

#if 0
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
  #if MEM_OPTIMIZED_WEIGHT_SYNC
  VertexRef paramsGatherVertex;
  #endif

  FullyConnectedLayer(unsigned size) : size(size) {}

  void addForward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    prev = net.fwd;
    unsigned prevSize = prev.size();

    #if USE_GATHER_VERTEX
    VertexRef gatherVertex = builder.addVertex("InnerProductFwdGatherVertex");
    vertices.push_back(gatherVertex);
    builder.addToComputeSet(net.trainCS, gatherVertex);
    builder.addToComputeSet(net.testCS, gatherVertex);
    builder.setFieldSize(gatherVertex["activation_in"], prevSize);
    builder.setFieldSize(gatherVertex["activation_out"], prevSize);
    builder.addEdge(prev[0]["index_out"], gatherVertex["index_in"], true);
    for (unsigned j = 0; j < prevSize; j++) {
      builder.addEdge(prev[j]["activation_out"],
                      gatherVertex["activation_in"][j],
                      true);
    }
    #endif

    #if MEM_OPTIMIZED_WEIGHT_SYNC
    paramsGatherVertex = builder.addVertex("InnerProductParamsGatherVertex");
    vertices.push_back(paramsGatherVertex);
    builder.addToComputeSet(net.weightSyncCS, paramsGatherVertex);
    builder.setFieldSize(paramsGatherVertex["weights_in"], prevSize);
    builder.setFieldSize(paramsGatherVertex["weights_out"], prevSize);
    #endif


    for (unsigned i = 0; i < size; ++i) {
      VertexRef v = builder.addVertex("InnerProductFwdVertex");
      vertices.push_back(v);
      builder.addToComputeSet(net.trainCS, v);
      builder.addToComputeSet(net.testCS, v);
      fwd.push_back(v);
      builder.addEdge(prev[0]["index_out"], v["index_in"], true);
      #if USE_GATHER_VERTEX
      builder.addEdge(gatherVertex["activation_out"],
                      v["activation_in"],
                      false);
      #else
      builder.setFieldSize(v["activation_in"], prevSize);
      for (unsigned j = 0; j < prevSize; j++) {
        builder.addEdge(prev[j]["activation_out"], v["activation_in"][j],
                        true);
      }
      #endif

      #if MEM_OPTIMIZED_WEIGHT_SYNC
      VertexRef pv = builder.addVertex("InnerProductParamsVertex");
      vertices.push_back(pv);
      builder.addToComputeSet(net.weightSyncCS, pv);
      weightSyncVertices.push_back(pv);
      builder.setFieldSize(pv["weights_out"], prevSize);
      builder.addEdge(paramsGatherVertex["weights_out"], pv["weights_in"],
                      false);
      builder.addEdge(pv["weights_out"], v["weights"], false);
      builder.addEdge(pv["bias_out"], v["bias"], false);
      builder.setInitialFieldValue<unsigned>(pv["myRank"], i);
      #else
      VertexRef pv = builder.addVertex("InnerProductParamsVertex");
      vertices.push_back(pv);
      builder.addToComputeSet(net.weightSyncCS, pv);
      weightSyncVertices.push_back(pv);
      builder.setFieldSize(pv["weights_in"], prevSize);
      builder.setFieldSize(pv["weights_out"], prevSize);
      builder.addEdge(pv["weights_out"], v["weights"], false);
      builder.addEdge(pv["bias_out"], v["bias"], false);
      #endif

    }
    net.fwd = fwd;
  }

  void addBackward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    unsigned prevSize = prev.size();
    std::vector<FieldRef> bwdDeltaOut, bwdIndexOut;

    #if USE_GATHER_VERTEX
    VertexRef gatherVertex = builder.addVertex("InnerProductBwdGatherVertex");
    vertices.push_back(gatherVertex);
    builder.addToComputeSet(net.trainCS, gatherVertex);
    builder.setFieldSize(gatherVertex["delta_in"], size);
    builder.setFieldSize(gatherVertex["delta_out"], size);
    builder.addEdge(net.bwdIndexOut[0], gatherVertex["index_in"], true);
    for (unsigned j = 0; j < size; j++) {
      builder.addEdge(net.bwdDeltaOut[j],
                      gatherVertex["delta_in"][j],
                      true);
    }
    #endif

    for (unsigned i = 0; i < prevSize; i++) {
      VertexRef v = builder.addVertex("InnerProductBwdVertex");
      vertices.push_back(v);
      builder.addToComputeSet(net.trainCS, v);
      bwd.push_back(v);
      bwdDeltaOut.push_back(v["delta_out"]);
      bwdIndexOut.push_back(v["index_out"]);

      builder.addEdge(net.bwdIndexOut[0], v["index_in"], true);

      builder.setFieldSize(v["weights"], size);
      builder.setFieldSize(v["bwdRecord"], net.batchSize);
      builder.setFieldSize(v["fwdRecord"], net.batchSize);

      #if USE_GATHER_VERTEX
      builder.addEdge(gatherVertex["delta_out"],
                      v["delta_in"],
                      false);
      #else
      builder.setFieldSize(v["delta_in"], size);
      for (unsigned j = 0; j < size; ++j) {
        builder.addEdge(net.bwdDeltaOut[j], v["delta_in"][j], true);
      }
      #endif

      #if MEM_OPTIMIZED_WEIGHT_SYNC
      builder.addEdge(v["weightSyncOutput"],
                      paramsGatherVertex["weights_in"][i],
                      false);
      builder.addToComputeSet(net.weightSyncCS, v);
      #else
      for (unsigned j = 0; j < size; ++j) {
        builder.addEdge(v["weights"][j], weightSyncVertices[j]["weights_in"][i],
                        false);
      }
      #endif

      builder.addEdge(prev[i]["activation_out"], v["activation_in"], true);
      builder.addEdge(prev[i]["index_out"], v["act_index_in"], true);

      builder.addToDataArray(net.daParams, v["weights"]);
    }

    for (unsigned i = 0; i < size; ++i) {
      VertexRef v = builder.addVertex("InnerProductBwdBiasVertex");
      vertices.push_back(v);
      biasVertices.push_back(v);
      builder.addToComputeSet(net.trainCS, v);
      builder.addEdge(net.bwdDeltaOut[i], v["delta_in"], true);
      builder.addEdge(net.bwdIndexOut[i], v["index_in"], true);
      builder.addEdge(v["bias"], weightSyncVertices[i]["bias_in"], false);
      builder.addToDataArray(net.daParams, v["bias"]);
    }

    net.bwdDeltaOut = bwdDeltaOut;
    net.bwdIndexOut = bwdIndexOut;
  }
};

class OneToOneLayer : public HiddenLayer {
public:
  std::string fwdVertex, bwdVertex;
  std::vector<VertexRef> prev;
  unsigned size;

  OneToOneLayer(std::string fwdVertex,
                std::string bwdVertex) :
    fwdVertex(fwdVertex),
    bwdVertex(bwdVertex) { }

  void addForward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    prev = net.fwd;
    size = prev.size();

    std::vector<VertexRef> fwd;
    for (unsigned i = 0; i < size; ++i) {
      VertexRef v = builder.addVertex(fwdVertex);
      vertices.push_back(v);
      builder.addToComputeSet(net.trainCS, v);
      builder.addToComputeSet(net.testCS, v);
      fwd.push_back(v);
      VertexRef prev = net.fwd[i];
      builder.addEdge(prev["index_out"], v["index_in"], true);
      builder.addEdge(prev["activation_out"], v["activation_in"], true);
    }
    net.fwd = fwd;
  }

  void addBackward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    std::vector<FieldRef> bwdDeltaOut, bwdIndexOut;
    for (unsigned i = 0; i < size; ++i) {
      VertexRef v = builder.addVertex(bwdVertex);
      vertices.push_back(v);
      builder.addToComputeSet(net.trainCS, v);
      bwdDeltaOut.push_back(v["delta_out"]);
      bwdIndexOut.push_back(v["index_out"]);
      builder.addEdge(net.bwdIndexOut[i], v["index_in"], true);
      builder.addEdge(net.bwdDeltaOut[i], v["delta_in"], true);
      builder.setFieldSize(v["record"], net.batchSize);
      builder.addEdge(prev[i]["activation_out"], v["activation_in"], true);
      builder.addEdge(prev[i]["index_out"], v["act_index_in"], true);
    }
    net.bwdDeltaOut = bwdDeltaOut;
    net.bwdIndexOut = bwdIndexOut;
  }

};

class ReLULayer : public OneToOneLayer
{
public:
  ReLULayer() : OneToOneLayer("ReLUFwdVertex", "ReLUBwdVertex") {}
};

class SigmoidLayer : public OneToOneLayer
{
public:
  SigmoidLayer() : OneToOneLayer("SigmoidFwdVertex", "SigmoidBwdVertex") {}
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

  #if !MEM_EXPERIMENT
  std::vector<HiddenLayer *> layers({
      new FullyConnectedLayer(30),
      new SigmoidLayer(),
      new FullyConnectedLayer(10),
      new SigmoidLayer()
  });
  #else
  std::vector<HiddenLayer *> layers({
      new FullyConnectedLayer(2000),
      new SigmoidLayer(),
      new FullyConnectedLayer(2000),
      new SigmoidLayer(),
      new FullyConnectedLayer(2000),
      new SigmoidLayer(),
      new FullyConnectedLayer(2000),
      new SigmoidLayer(),
      new FullyConnectedLayer(2000),
      new SigmoidLayer(),
      new FullyConnectedLayer(10),
      new SigmoidLayer(),
  });
  #endif
  Net net(MNIST, 10, layers);
#if 0
  net.engine->setValue<unsigned>(
      ((FullyConnectedLayer *) net.hiddenLayers[2])->fwd[0]["debug"],
      1);
  /*  net.engine->setValue<unsigned>(
      ((FullyConnectedLayer *) net.hiddenLayers[2])->bwd[0]["debug"],
      1);*/
  net.engine->setValue<unsigned>(
      ((FullyConnectedLayer *) net.hiddenLayers[2])->biasVertices[0]["debug"],
      1);
  net.test(3);
#else
  net.test(5000);
#endif

  return 0;
}
