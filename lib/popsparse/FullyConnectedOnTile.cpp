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

static std::string
getVertexClass(const OnTileMethod &method, const Type &inputType,
               const Type &partialsType,
               const std::array<std::size_t, 2> &blockDimensions) {
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
  case OnTileMethod::ForwardAMPBlock:
    return templateVertex("popsparse::SparseDenseMatMulBlock", inputType,
                          partialsType, blockDimensions[0], blockDimensions[1]);
  default:
    throw poplibs_error("Unhandled on-tile sparse fc method");
  }
}

// Operation to perform sparse fully connected pass on a tile.
void onTileImpl(Graph &graph, const ComputeSet &cs, unsigned tile,
                const OnTileMethod &method, bool zeroPartials,
                const boost::variant<unsigned, Tensor> &subGroupIdToProcess,
                const Vector<std::size_t> &shape, const Tensor &metaInfoBuckets,
                const Tensor &weights, const Tensor &acts,
                const Tensor &partials,
                const std::array<std::size_t, 2> &blockDimensions,
                const std::string &debugPrefix) {
  // Verify input shapes with respect to shape of on-tile partition
  assert(acts.elementType() == weights.elementType());
  assert(metaInfoBuckets.elementType() == UNSIGNED_SHORT);
  assert(metaInfoBuckets.rank() == 2);

  const auto &inputType = acts.elementType();
  const auto &partialsType = partials.elementType();
  const auto &target = graph.getTarget();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto vertexClass =
      getVertexClass(method, inputType, partialsType, blockDimensions);
  const auto v = graph.addVertex(cs, vertexClass);
  switch (method) {
  case OnTileMethod::Forward:
  case OnTileMethod::GradA:
  case OnTileMethod::Transpose: {
    assert(acts.rank() == 6);
    assert(partials.rank() == 6);
    const auto actsUngrouped = unfactorDims(acts, 3);
    const auto partialsUngrouped = unfactorDims(partials, 3);
    assert(weights.rank() == 2);
    // Dimension 0 is number of buckets.
    // These must be equal for both meta-info and non-zero values.
    assert(metaInfoBuckets.dim(0) == weights.dim(0));

    // Instantiate vertex
    graph.connect(v["q"], method == OnTileMethod::Transpose
                              ? actsUngrouped.flatten()
                              : partialsUngrouped.flatten());
    graph.connect(v["r"], weights);
    graph.connect(v["s"], method == OnTileMethod::Transpose
                              ? partialsUngrouped.flatten()
                              : actsUngrouped.flatten());
    graph.connect(v["metaInfo"], metaInfoBuckets);
    graph.setInitialValue(v["subGroupIdToProcess"],
                          boost::get<unsigned>(subGroupIdToProcess));
    graph.setInitialValue(v["zeroInfo"],
                          zeroPartials ? partials.numElements() : 0u);
    break;
  }
  case OnTileMethod::GradW: {
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
    graph.setInitialValue(v["zeroInfo"],
                          zeroPartials ? partials.numElements() : 0u);
    break;
  }
  case OnTileMethod::ForwardAMPBlock: {
    assert(acts.rank() == 6);
    assert(partials.rank() == 6);
    const auto actsUngrouped = unfactorDims(acts, 3);
    const auto partialsUngrouped = unfactorDims(partials, 3);
    assert(weights.rank() == 2);
    // Dimension 0 is number of buckets.
    // These must be equal for both meta-info and non-zero values.
    assert(metaInfoBuckets.dim(0) == weights.dim(0));

    // Expect X as innermost dimension
    graph.connect(v["q"], partialsUngrouped.dimRoll(1, 2).flatten());
    graph.connect(v["r"], weights);
    // Expect Y as innermost dimension
    graph.connect(v["s"], actsUngrouped.dimRoll(1, 2).flatten());
    graph.connect(v["metaInfo"], metaInfoBuckets);
    graph.setInitialValue(v["subGroupIdToProcess"],
                          boost::get<unsigned>(subGroupIdToProcess));

    // Work is divided between workers statically along z dimension.
    const auto workerTiles =
        splitTileBetweenWorkers(1, acts.dim(2), numWorkers);
    std::vector<unsigned short> offsetAndNumZByWorker(numWorkers * 2);
    for (std::size_t worker = 0; worker < workerTiles.size(); ++worker) {
      const auto &workerZInterval = workerTiles[worker].getColumns();
      offsetAndNumZByWorker[worker * 2 + 0] = workerZInterval.begin();
      offsetAndNumZByWorker[worker * 2 + 1] = workerZInterval.size();
    }

    const auto tOffsetAndNumZByWorker = graph.addConstant(
        UNSIGNED_SHORT, {offsetAndNumZByWorker.size()},
        offsetAndNumZByWorker.data(), debugPrefix + "/offsetAndNumZByWorker");
    graph.setTileMapping(tOffsetAndNumZByWorker, tile);
    graph.connect(v["offsetAndNumZByWorker"], tOffsetAndNumZByWorker);

    const auto inputElemsPer64Bits = 8 / target.getTypeSize(inputType);
    const auto partialElemsPer64Bits = 8 / target.getTypeSize(partialsType);
    const auto zStrideInQ = partialsUngrouped.dim(1) / partialElemsPer64Bits;
    const auto zStrideInS = actsUngrouped.dim(1) / inputElemsPer64Bits;
    graph.setInitialValue(v["zStrideInQ"], zStrideInQ);
    graph.setInitialValue(v["zStrideInS"], zStrideInS);

    const auto partialElemsPer32Bits = 4 / target.getTypeSize(partialsType);
    assert(partials.numElements() % partialElemsPer32Bits == 0);
    graph.setInitialValue(
        v["zeroInfo"],
        zeroPartials ? partials.numElements() / partialElemsPer32Bits : 0u);
    break;
  }
  default:
    throw poplibs_error("Unhandled OnTileMethod");
  }
  graph.setTileMapping(v, tile);
}

std::vector<unsigned> getOnTileActsOrdering(const OnTileMethod &method) {
  switch (method) {
  case OnTileMethod::Forward:
  case OnTileMethod::GradA:
  case OnTileMethod::Transpose:
  case OnTileMethod::GradW:
    return {0, 1, 2};
    break;
  case OnTileMethod::ForwardAMPBlock:
    return {0, 2, 1};
    break;
  default:
    throw poplibs_error("Unhandled OnTileMethod");
  }
}

std::vector<unsigned> getOnTilePartialsOrdering(const OnTileMethod &method) {
  switch (method) {
  case OnTileMethod::Forward:
  case OnTileMethod::GradA:
  case OnTileMethod::Transpose:
  case OnTileMethod::GradW:
    return {0, 1, 2};
    break;
  case OnTileMethod::ForwardAMPBlock:
    return {0, 2, 1};
    break;
  default:
    throw poplibs_error("Unhandled OnTileMethod");
  }
}

} // end namespace fullyconnected
} // end namespace popsparse
