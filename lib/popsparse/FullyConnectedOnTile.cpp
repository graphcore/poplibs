// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "FullyConnectedOnTile.hpp"

#include "FullyConnectedOptions.hpp"
#include "FullyConnectedPlan.hpp"
#include "FullyConnectedUtils.hpp"

#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_support/logging.hpp>

using namespace poplar;
using namespace poputil;
using namespace poplibs_support;

namespace popsparse {
namespace fullyconnected {

static std::string getVertexClass(const OnTileMethod &method,
                                  const Type &inputType,
                                  const Type &partialsType) {
  switch (method) {
  case OnTileMethod::Forward:
    return templateVertex("popsparse::SparseDenseMatMulElementWise", inputType,
                          partialsType);
  case OnTileMethod::GradA:
    return templateVertex("popsparse::SparseDenseMatMulGradAElementWise",
                          inputType, partialsType);
  case OnTileMethod::Transpose:
    return templateVertex("popsparse::SparseDenseMatMulElementWiseTranspose",
                          inputType, partialsType);
  case OnTileMethod::GradW:
    return templateVertex("popsparse::SparseDenseMatMulGradWElementWise",
                          inputType, partialsType);
  default:
    throw poplibs_error("Unhandled on-tile sparse fc method");
  }
}

// Operation to perform sparse fully connected pass on a tile. (Only handles fwd
// for now).
void onTileImpl(Graph &graph, const ComputeSet &cs, unsigned tile,
                const OnTileMethod &method, bool zeroPartials,
                const boost::variant<unsigned, Tensor> &subGroupIdToProcess,
                const Vector<std::size_t> &shape, const Tensor &metaInfoBuckets,
                const Tensor &weights, const Tensor &acts,
                const Tensor &partials, const std::string &debugPrefix) {
  // Verify input shapes with respect to shape of on-tile partition
  assert(acts.elementType() == weights.elementType());
  assert(metaInfoBuckets.elementType() == UNSIGNED_SHORT);
  assert(metaInfoBuckets.rank() == 2);

  const auto &inputType = acts.elementType();
  const auto &partialsType = partials.elementType();
  const auto vertexClass = getVertexClass(method, inputType, partialsType);
  const auto v = graph.addVertex(cs, vertexClass);
  if (method != OnTileMethod::GradW) {
    assert(acts.rank() == 6);
    const auto actsUngrouped = unfactorDims(acts, 3);
    assert(weights.rank() == 2);
    // Dimension 0 is number of buckets.
    // These must be equal for both meta-info and non-zero values.
    assert(metaInfoBuckets.dim(0) == weights.dim(0));

    // Instantiate vertex
    graph.connect(v["q"], method == OnTileMethod::Transpose
                              ? actsUngrouped.flatten()
                              : partials.flatten());
    graph.connect(v["r"], weights);
    graph.connect(v["s"], method == OnTileMethod::Transpose
                              ? partials.flatten()
                              : actsUngrouped.flatten());
    graph.connect(v["metaInfo"], metaInfoBuckets);
    graph.setInitialValue(v["subGroupIdToProcess"],
                          boost::get<unsigned>(subGroupIdToProcess));
  } else {
    assert(partials.rank() == 6);
    assert(weights.rank() == acts.rank());
    assert(weights.rank() == 6);
    assert(weights.dim(2) == acts.dim(1));
    assert(weights.dim(5) == acts.dim(4));
    const auto actsUngrouped = unfactorDims(acts, 3);
    const auto weightsUngrouped = unfactorDims(weights, 3);
    const auto subGroupIdToProcessTensor =
        boost::get<Tensor>(subGroupIdToProcess);
    assert(subGroupIdToProcessTensor.rank() <= 1);
    const auto subGroupIdToProcessScalar =
        subGroupIdToProcessTensor.flatten().squeeze({0});

    std::vector<std::size_t> squeezePartialsDims(5);
    std::iota(squeezePartialsDims.begin(), squeezePartialsDims.end(), 0);
    graph.connect(v["qGrad"], weightsUngrouped.flatten());
    graph.connect(v["rGrad"], partials.squeeze(squeezePartialsDims));
    graph.connect(v["metaInfo"], metaInfoBuckets.squeeze({0}));
    graph.connect(v["s"], actsUngrouped.dimRoll(1, 2).flatten());
    graph.connect(v["subGroupIdToProcess"], subGroupIdToProcessScalar);
    graph.setInitialValue(v["numZ"], weightsUngrouped.dim(2));
  }
  graph.setInitialValue(v["zeroInfo"],
                        zeroPartials ? partials.numElements() : 0u);
  graph.setTileMapping(v, tile);
}

} // end namespace fullyconnected
} // end namespace popsparse
