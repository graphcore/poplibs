#ifndef _layer_hpp_
#define _layer_hpp_

#include <cassert>
#include <poplar/Graph.hpp>
#include <poplar/CPUEngine.hpp>
#include <poplar/IPUModelEngine.hpp>
#include <random>
#include <string>
#include "neural_net_common.h"
#include "VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

class Net;
class NetOptions;

/* A data set full of test and training data along with its dimensions */
class DataSet {
public:
  std::unique_ptr<float[]> testData, trainingData;
  std::unique_ptr<unsigned[]> testLabels, trainingLabels;
  unsigned dataSize, numTest, numTraining;
  std::vector<std::size_t> dim;
};

enum NetType {
  TrainingNet,
  TestOnlyNet
};

/* The layer class represents a single layer in the net.
 */
class Layer {
  const Net &net;
  int index;
protected:
  Layer(const Net &net, int index) : net(net), index(index) {}
  unsigned getWorkerContextsPerTile() const;
  IPUModelEngineBuilder &getIPUModelEngineBuilder() const;
  unsigned getNumIPUs() const;
  unsigned getTilesPerIPU() const;
  const std::string &getDType() const;
  unsigned getDTypeSize() const;
  enum NetType getNetType() const;
  const NetOptions &getNetOptions() const;
  unsigned getBatchSize() const;
  bool targetSharedConvWeights() const;
  float getLearningRate() const;
  void mapTensor(Tensor t, IPUModelEngineBuilder::TileMapping &mapping);
  void mapComputeSet(const Graph &graph, ComputeSet c,
                     IPUModelEngineBuilder::TileMapping &mapping);
  std::vector<unsigned> computeActivationsMapping(Tensor t);
  void mapActivations(Tensor t, IPUModelEngineBuilder::TileMapping &mapping);
public:
  Layer *getNextLayer() const;
  Layer *getPrevLayer() const;
  std::string makeLayerName(const std::string &name);
  virtual void init(Graph &graph, std::mt19937 &randomEngine,
                    IPUModelEngineBuilder::TileMapping &mapping) = 0;
  virtual Program initParams(Graph &graph) = 0;
  virtual Program forward(Graph &graph,
                          IPUModelEngineBuilder::TileMapping &mapping) = 0;
  virtual Program backward(Graph &graph,
                           IPUModelEngineBuilder::TileMapping &mapping) = 0;
  virtual Program weightUpdate(Graph &graph,
                               IPUModelEngineBuilder::TileMapping &mapping) = 0;
  virtual void describe(std::ostream &out) = 0;
  /// Return the number of FLOPs required for a naive implementation of the
  /// forward pass. A FLOP is a basic arithmetic operation such as multiply,
  /// add, subtract. A fused multiply accumulate counts as 2 FLOPs.
  /// The count of the number of FLOPs does not include operations used to
  /// compute the non-linearity at the end of the layer.
  virtual std::uint64_t getNumberOfFlops() = 0;
  /// Return the number of cycles you would expect the forward pass to require
  /// based only on amount of compute required, ignoring any overheads. The
  /// return value may be fractional in cases where the number of operations
  /// required is not an exact multiple of the ideal number of operations per
  /// cycle.
  virtual double getPerfectCycleCount() = 0;
  virtual Tensor getFwdActivations() const = 0;
  virtual Tensor getFwdZs() const = 0;
  virtual NonLinearityType getNonLinearityType() const {
    return NON_LINEARITY_NONE;
  };
  virtual Tensor getBwdDeltas() const = 0;

  // Called if the previous layer provides a 3D volume as output.
  // A layer can request that the previous layer provides the z-axis
  // (or channel-axis) to be split into a number of groups. So the provided
  // tensor from the previous layer will be a 4D tensor of dimension
  // {numGroups, x, y, z / numGroups}.
  //
  // If this function returns 0 then it indicates that it does not care how
  // the previous output is grouped.
  virtual size_t getNumChannelGroupsIn(size_t xPrev, size_t yPrev,
                                       size_t zPrev) const {
    return 0;
  }
};

class LayerSpec {
public:
  virtual std::unique_ptr<Layer>
  makeLayer(Net &net, int index) = 0;
};

/* This utility function wraps a vector of normal pointers as unique_ptrs.
   It allows the hidden layer array to be initializes with an
   initializer list. */
static std::vector<std::unique_ptr<LayerSpec>>
makeLayers(std::vector<LayerSpec *> vs)
{
  std::vector<std::unique_ptr<LayerSpec>> xs;
  for (auto p: vs)
    xs.push_back(std::unique_ptr<LayerSpec>(p));
  return xs;
}

std::unique_ptr<float[]>
createRandomWeightInitializers(Tensor t, float mean, float variance,
                               std::mt19937 &randomEngine);

#endif // _layer_hpp_
