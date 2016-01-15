#ifndef _fully_connected_layer_hpp_
#define _fully_connected_layer_hpp_
#include "Net.hpp"

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

  virtual bool requiresLayeredInput() {return false;}
  virtual bool providesLayeredOutput() {return false;}

  void addForward(Net &net)  {
    Graph &graph = *net.graph;
    prev = net.fwd;
    unsigned prevSize = prev.size();
    prevNonLinearityType = net.prevNonLinearityType;
    net.prevNonLinearityType = nonLinearityType;


    VertexRef gatherVertex;
    if (net.prevLayers) {
      prevSize  = prevSize * net.prevLayers / net.prevChunks;
      gatherVertex = graph.addVertex("InnerProductFwdLayeredGatherVertex");
    } else {
      gatherVertex = graph.addVertex("InnerProductFwdGatherVertex");
    }
    graph.setFieldSize(gatherVertex["activationOut"], prevSize);
    graph.addEdge(net.stateField, gatherVertex["state"], false);
    graph.addToComputeSet(net.trainCS, gatherVertex);
    graph.addToComputeSet(net.testCS, gatherVertex);
    graph.setFieldSize(gatherVertex["activationIn"], prev.size());
    graph.addEdge(prev[0]["indexOut"], gatherVertex["indexIn"], true);
    for (unsigned j = 0; j < prev.size(); j++) {
      graph.addEdge(prev[j]["activationOut"],
                    gatherVertex["activationIn"][j],
                    true);
    }

    VertexRef vCurParamGather;
    for (unsigned i = 0; i < size; ++i) {
      if (net.netType == TrainingNet &&
          i % (size/numParamGathers) == 0) {
        VertexRef v = graph.addVertex("InnerProductParamsGatherVertex");
        paramGatherVertices.push_back(v);
        graph.addToComputeSet(net.weightSyncCS, v);
        graph.setFieldSize(v["weightsIn"], prevSize);
        graph.setFieldSize(v["weightsOut"], prevSize);
        vCurParamGather = v;
      }

      VertexRef v = graph.addVertex("InnerProductFwdVertex");
      graph.addEdge(net.stateField, v["state"], false);
      graph.addToComputeSet(net.trainCS, v);
      graph.addToComputeSet(net.testCS, v);
      fwd.push_back(v);
      graph.setInitialFieldValue<NonLinearityType>(v["nonLinearityType"],
                                                     nonLinearityType);

      graph.addEdge(prev[0]["indexOut"], v["indexIn"], true);
      graph.addEdge(gatherVertex["activationOut"],
                    v["activationIn"],
                    false);

      if (net.netType == TrainingNet) {
        VertexRef pv = graph.addVertex("InnerProductParamsVertex");
        weightSyncVertices.push_back(pv);
        graph.addEdge(net.stateField, pv["state"], false);
        graph.addToComputeSet(net.weightSyncCS, pv);
        graph.setFieldSize(pv["weightsOut"], prevSize);
        VertexRef paramsGatherVertex = vCurParamGather;
        graph.addEdge(paramsGatherVertex["weightsOut"], pv["weightsIn"],
                      false);
        graph.addEdge(pv["weightsOut"], v["weights"], false);
        graph.addEdge(pv["biasOut"], v["bias"], false);
        graph.setInitialFieldValue<unsigned>(pv["myRank"],
                                             i % (size / numParamGathers));
      } else {
        VertexRef pv = graph.addVertex("InnerProductParamsFwdOnlyVertex");
        graph.addToComputeSet(net.trainCS, pv);
        graph.addToComputeSet(net.testCS, pv);
        graph.addEdge(pv["weights"], v["weights"], false);
        graph.addEdge(pv["bias"], v["bias"], false);
        graph.setFieldSize(pv["weights"], prevSize);
        graph.addToDataArray(net.daParams, pv["weights"]);
        graph.addToDataArray(net.daParams, pv["bias"]);
      }

      if (net.netType == TrainingNet && i < prevSize) {
        VertexRef bv = graph.addVertex("InnerProductBwdVertex");
        bwd.push_back(bv);
      }
    }

    if (net.netType == TrainingNet) {
      for (unsigned i = size; i < prevSize; ++i) {
        VertexRef bv = graph.addVertex("InnerProductBwdVertex");
        bwd.push_back(bv);
      }
    }

    std::cout << "   -- Added fully connected layer:\n"
              << "        Input: "  << prevSize << "\n"
              << "        Output: " << size << "\n"
              << "        Params: " << size * (prevSize + 1) << "\n";

    net.fwd = fwd;
    net.prevLayers = 0;
  }

  void addBackward(Net &net)  {
    Graph &graph = *net.graph;
    unsigned prevSize = prev.size();
    std::vector<FieldRef> bwdDeltaOut, bwdIndexOut;

    VertexRef gatherVertex = graph.addVertex("InnerProductBwdGatherVertex");
    graph.addEdge(net.stateField, gatherVertex["state"], false);
    graph.addToComputeSet(net.trainCS, gatherVertex);
    graph.setFieldSize(gatherVertex["deltaIn"], size);
    graph.setFieldSize(gatherVertex["deltaOut"], size);
    graph.addEdge(net.bwdIndexOut[0], gatherVertex["indexIn"], true);
    for (unsigned j = 0; j < size; j++) {
      graph.addEdge(net.bwdDeltaOut[j],
                    gatherVertex["deltaIn"][j],
                    true);
    }

    for (unsigned i = 0; i < prevSize; i++) {
      VertexRef v = bwd[i];
      graph.addEdge(net.stateField, v["state"], false);
      graph.addEdge(net.etaField, v["eta"], false);
      graph.addToComputeSet(net.trainCS, v);

      bwdDeltaOut.push_back(v["deltaOut"]);
      bwdIndexOut.push_back(v["indexOut"]);

      graph.setInitialFieldValue<NonLinearityType>(v["nonLinearityType"],
                                                     prevNonLinearityType);

      graph.addEdge(net.bwdIndexOut[0], v["indexIn"], true);

      graph.setFieldSize(v["weights"], size);
      graph.setFieldSize(v["bwdRecord"], net.batchSize);
      graph.setFieldSize(v["actRecord"], net.batchSize);
      graph.setFieldSize(v["zRecord"], net.batchSize);

      graph.addEdge(gatherVertex["deltaOut"],
                    v["deltaIn"],
                    false);

      graph.setFieldSize(v["weightSyncOutput"], numParamGathers);
      for (unsigned j = 0; j < numParamGathers; ++j) {
        graph.addEdge(v["weightSyncOutput"][j],
                      paramGatherVertices[j]["weightsIn"][i],
                      false);
      }
      graph.addToComputeSet(net.weightSyncCS, v);

      graph.addEdge(prev[i]["activationOut"], v["activationIn"], true);
      graph.addEdge(prev[i]["z"], v["zIn"], true);
      graph.addEdge(prev[i]["indexOut"], v["actIndexIn"], true);

      graph.addToDataArray(net.daParams, v["weights"]);
    }

    for (unsigned i = 0; i < size; ++i) {
      VertexRef v = graph.addVertex("InnerProductBwdBiasVertex");
      graph.setInitialFieldValue<unsigned>(v["batchSize"], net.batchSize);
      graph.addEdge(net.stateField, v["state"], false);
      graph.addEdge(net.etaField, v["eta"], false);
      biasVertices.push_back(v);
      graph.addToComputeSet(net.trainCS, v);
      graph.addEdge(net.bwdDeltaOut[i], v["deltaIn"], true);
      graph.addEdge(net.bwdIndexOut[i], v["indexIn"], true);
      graph.addEdge(v["bias"], weightSyncVertices[i]["biasIn"], false);
      graph.addToDataArray(net.daParams, v["bias"]);
    }

    net.bwdDeltaOut = bwdDeltaOut;
    net.bwdIndexOut = bwdIndexOut;
  }
};


#endif // _fully_connected_layer_hpp_
