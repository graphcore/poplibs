// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "FullyConnectedOnTile.hpp"

#include "FullyConnectedOptions.hpp"
#include "FullyConnectedPlan.hpp"
#include "FullyConnectedUtils.hpp"

#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_support/Algorithm.hpp>
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
  case OnTileMethod::TransposeAMPBlock:
    return templateVertex("popsparse::SparseDenseMatMulBlockGradA", inputType,
                          partialsType, blockDimensions[0], blockDimensions[1]);
  case OnTileMethod::GradWBlock:
    return templateVertex("popsparse::SparseDenseMatMulBlockGradW", inputType,
                          partialsType, blockDimensions[0], blockDimensions[1]);
  case OnTileMethod::GradWAMPBlock:
    return templateVertex("popsparse::SparseDenseMatMulBlockAmpGradW",
                          inputType, partialsType, blockDimensions[0],
                          blockDimensions[1]);
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

  auto checkRptBounds = [&](std::size_t numElements, const Type &partialsType) {
    const auto num64BitValues = 8 / target.getTypeSize(partialsType);
    const auto numElemsPerWorker =
        poplibs_support::ceildiv(num64BitValues, target.getNumWorkerContexts());
    if (numElemsPerWorker > target.getRptCountMax()) {
      throw poputil::poplibs_error("Number of elements to zero "
                                   "exceeds rpt count bound");
    }
  };

  auto getZeroInfo = [&](std::size_t numElements, const Type &partialsType) {
    std::size_t multipleOf = 8 / target.getTypeSize(partialsType);
    auto valid = (numElements % multipleOf) == 0;
    return std::make_pair(valid, numElements / multipleOf);
  };

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
  case OnTileMethod::ForwardAMPBlock:
  case OnTileMethod::TransposeAMPBlock: {
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
        splitTileBetweenWorkers(1, actsUngrouped.dim(2), numWorkers);
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
    checkRptBounds(partials.numElements(), partialsType);
    const auto zeroInfo = getZeroInfo(partials.numElements(), partialsType);
    if (!zeroInfo.first) {
      throw poputil::poplibs_error("Number of zero elements is not a "
                                   "multiple of 64-bits ");
    }
    graph.setInitialValue(v["zeroInfo"], zeroPartials ? zeroInfo.second : 0u);
    break;
  }
  case OnTileMethod::GradWBlock:
  case OnTileMethod::GradWAMPBlock: {
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
    auto weightsInCodeletOrder = weightsUngrouped;
    auto actsInCodeletOrder = actsUngrouped;
    if (method == OnTileMethod::GradWBlock) {
      // Expect X as innermost dimension
      weightsInCodeletOrder = weightsInCodeletOrder.dimRoll(1, 2);
    } else if (method == OnTileMethod::GradWAMPBlock) {
      actsInCodeletOrder = actsInCodeletOrder.dimRoll(1, 2);
    }
    graph.connect(v["qGrad"], weightsInCodeletOrder.flatten());
    graph.connect(v["rGrad"], partials.squeeze(squeezePartialsDims));
    graph.connect(v["metaInfo"], metaInfoBuckets.squeeze({0}));
    // Expect Y as innermost dimension
    graph.connect(v["s"], actsInCodeletOrder.flatten());
    graph.connect(v["subGroupIdToProcess"], subGroupIdToProcessScalar);
    checkRptBounds(partials.numElements(), partialsType);
    const auto zeroInfo = getZeroInfo(partials.numElements(), partialsType);
    if (!zeroInfo.first) {
      throw poputil::poplibs_error("Number of zero elements is not a "
                                   "multiple of 64-bits");
    }
    graph.setInitialValue(v["zeroInfo"], zeroPartials ? zeroInfo.second : 0);
    graph.setInitialValue(v["numZ"], weightsUngrouped.dim(2));
    if (method == OnTileMethod::GradWBlock) {
      const auto inputElemsPer64Bits = 8 / target.getTypeSize(inputType);
      const auto zStrideInQ = weightsUngrouped.dim(1) / inputElemsPer64Bits;
      const auto zStrideInS = actsUngrouped.dim(2) / inputElemsPer64Bits;
      graph.setInitialValue(v["zStrideInQ"], zStrideInQ);
      graph.setInitialValue(v["zStrideInS"], zStrideInS);
    }
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
  case OnTileMethod::GradWBlock:
    return {0, 1, 2};
    break;
  case OnTileMethod::GradW:
  case OnTileMethod::ForwardAMPBlock:
  case OnTileMethod::TransposeAMPBlock:
  case OnTileMethod::GradWAMPBlock:
    return {0, 2, 1};
    break;
  default:
    throw poplibs_error("Unhandled OnTileMethod");
  }
}

std::vector<unsigned> getOnTileWeightsOrdering(const OnTileMethod &method) {
  switch (method) {
  case OnTileMethod::GradWBlock:
    return {0, 2, 1};
  case OnTileMethod::GradW:
  case OnTileMethod::GradWAMPBlock:
    return {0, 1, 2};
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
  case OnTileMethod::TransposeAMPBlock:
    return {0, 2, 1};
    break;
  default:
    throw poplibs_error("Unhandled OnTileMethod");
  }
}

} // end namespace fullyconnected
} // end namespace popsparse
