#include "poplin/Norms.hpp"
#include "popnn/Norms.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/TileMapping.hpp"
#include <cassert>

using namespace poplar;

namespace popnn {

void checkTensorShape(Tensor acts) {
  const auto rank = acts.rank();
  if (rank < 2 ) {
    throw poputil::poplibs_error(
      "Norm supported for tensors of rank > 1");
  }
}

Tensor preProcessNormActs(const Tensor &acts) {
  return acts.rank() == 2 ? acts.expand({2}) : acts;
}

Tensor postProcessNormActs(const Tensor &acts, unsigned originalActsRank) {
  if (originalActsRank == 2) {
    assert(acts.rank() == 3 && acts.dim(2) == 1);
    return acts.squeeze({2});
  }
  return acts;
}

std::size_t normNumChannels(Tensor acts) {
  return acts.dim(1);
}

std::size_t normNumActsPerChannel(Tensor acts) {
  return acts.numElements() / normNumChannels(acts);
}


// This function is used to create normalisaton parameters for layers which
// have input activations of dimension 2 (i.e. fully connected layers)
static Tensor
createNormParam(Graph &graph,
                const Tensor& acts,
                const std::string &name) {
  const unsigned numActs = acts.shape()[1];
  const auto dType = acts.elementType();
  // This should be replaced by a clone of the channel dimension of the
  // activations. It is not done because cloning part of a tensor is currently
  // expensive.
  auto param = graph.addVariable(dType, {numActs}, name);
  poputil::mapTensorLinearly(graph, param);
  return param;
}

Tensor
createNormGamma(Graph &graph, const Tensor &acts) {
  const auto rank = acts.rank();
  checkTensorShape(acts);
  if (rank > 2) {
    return poplin::createNormGamma(graph, acts);
  } else {
    return createNormParam(graph, acts, "gamma");
  }
}

Tensor
createNormBeta(Graph &graph, const Tensor &acts) {
  const auto rank = acts.rank();
  checkTensorShape(acts);
  if (rank > 2) {
    return poplin::createNormBeta(graph, acts);
  } else {
    return createNormParam(graph, acts, "beta");
  }
}

std::pair<Tensor, Tensor>
createNormParams(Graph &graph, const Tensor acts) {
  Tensor gamma = createNormGamma(graph, acts);
  Tensor beta = createNormBeta(graph, acts);
  return std::make_pair(gamma, beta);
}

} // namespace popnn
