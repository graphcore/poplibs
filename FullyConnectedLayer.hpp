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
    GraphBuilder &builder = *net.graphBuilder;
    prev = net.fwd;
    unsigned prevSize = prev.size();
    prevNonLinearityType = net.prevNonLinearityType;
    net.prevNonLinearityType = nonLinearityType;


    VertexRef gatherVertex;
    if (net.prevLayers) {
      prevSize  = prevSize * net.prevLayers;
      gatherVertex = builder.addVertex("InnerProductFwdLayeredGatherVertex");
    } else {
      gatherVertex = builder.addVertex("InnerProductFwdGatherVertex");
    }
    builder.setFieldSize(gatherVertex["activationOut"], prevSize);
    vertices.push_back(gatherVertex);
    builder.addEdge(net.stateField, gatherVertex["state"], false);
    builder.addToComputeSet(net.trainCS, gatherVertex);
    builder.addToComputeSet(net.testCS, gatherVertex);
    builder.setFieldSize(gatherVertex["activationIn"], prev.size());
    builder.addEdge(prev[0]["indexOut"], gatherVertex["indexIn"], true);
    for (unsigned j = 0; j < prev.size(); j++) {
      builder.addEdge(prev[j]["activationOut"],
                      gatherVertex["activationIn"][j],
                      true);
    }

    VertexRef vCurParamGather;
    for (unsigned i = 0; i < size; ++i) {
      if (net.netType == TrainingNet &&
          i % (size/numParamGathers) == 0) {
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
      builder.addEdge(gatherVertex["activationOut"],
                      v["activationIn"],
                      false);

      if (net.netType == TrainingNet) {
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
      } else {
        VertexRef pv = builder.addVertex("InnerProductParamsFwdOnlyVertex");
        vertices.push_back(pv);
        builder.addToComputeSet(net.trainCS, pv);
        builder.addToComputeSet(net.testCS, pv);
        builder.addEdge(pv["weights"], v["weights"], false);
        builder.addEdge(pv["bias"], v["bias"], false);
        builder.setFieldSize(pv["weights"], prevSize);
        builder.addToDataArray(net.daParams, pv["weights"]);
        builder.addToDataArray(net.daParams, pv["bias"]);
      }

      if (net.netType == TrainingNet && i < prevSize) {
        VertexRef bv = builder.addVertex("InnerProductBwdVertex");
        vertices.push_back(bv);
        bwd.push_back(bv);
      }
    }

    if (net.netType == TrainingNet) {
      for (unsigned i = size; i < prevSize; ++i) {
        VertexRef bv = builder.addVertex("InnerProductBwdVertex");
        vertices.push_back(bv);
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
    GraphBuilder &builder = *net.graphBuilder;
    unsigned prevSize = prev.size();
    std::vector<FieldRef> bwdDeltaOut, bwdIndexOut;

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

      builder.addEdge(gatherVertex["deltaOut"],
                      v["deltaIn"],
                      false);

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


#endif // _fully_connected_layer_hpp_
