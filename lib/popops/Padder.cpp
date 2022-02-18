// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "Padder.hpp"
#include "poputil/exceptions.hpp"
#include <cstdlib>
#include <poplar/Graph.hpp>
#include <sstream>

namespace popops {
namespace padding {

void mapPadding(poplar::Graph &graph, MappingMethod mappingMethod,
                const poplar::Tensor &tPrepad, const poplar::Tensor &padding,
                unsigned dim, bool padIsLow) {
  switch (mappingMethod) {
  case MappingMethod::NONE:
    break;
  case MappingMethod::ZERO:
    graph.setTileMapping(padding, 0);
    break;
  case MappingMethod::EDGE: {
    if (tPrepad.numElements() == 0) {
      throw poputil::poplibs_error(
          "Attempting to pad with EDGE mapping method, but tensor "
          "prior to padding has no elements");
    }
    auto edgeIdx = padIsLow ? 0 : tPrepad.dim(dim) - 1;
    auto edgeMapping =
        graph.getTileMapping(tPrepad.slice(edgeIdx, edgeIdx + 1, dim));
    for (unsigned i = 0; i < padding.dim(dim); ++i) {
      graph.setTileMapping(padding.slice(i, i + 1, dim), edgeMapping);
    }
    break;
  }
  default:
    throw poputil::poplibs_error("Unsupported mapping method");
  }
}

poplar::Tensor Padder::getPaddedTensor(const poplar::Tensor &tIn,
                                       const std::vector<ptrdiff_t> &pLows,
                                       const std::vector<ptrdiff_t> &pUpps) {

  const auto rank = tIn.rank();
  if (pLows.size() != rank || pUpps.size() != rank) {
    std::stringstream errss;
    errss << "size of [lower, upper] padding vectors = [" << pLows.size()
          << ", " << pUpps.size() << "], "
          << "and the rank of the tensor being padded is " << rank
          << ". These should all be equal.";
    throw poputil::poplibs_error(errss.str());
  }

  poplar::Tensor tOut = tIn;
  for (unsigned i = 0; i < tIn.rank(); ++i) {
    tOut = getPartPaddedTensor(tOut, i, pLows[i], pUpps[i]);
  }
  return tOut;
}

void Padder::validatePadArgs(const poplar::Tensor &in, unsigned d,
                             ptrdiff_t pLow, ptrdiff_t pUpp) {

  if (d >= in.rank()) {
    std::stringstream errss;
    errss << "Dimension " << d << " is oob where rank is " << in.rank();
    throw poputil::poplibs_error(errss.str());
  }

  auto dsize = static_cast<int>(in.dim(d));
  if (pLow + dsize < 0 || pUpp + dsize < 0 || pLow + pUpp + dsize < 0) {
    std::stringstream errss;
    errss << "pad [lower, upper] = [" << pLow << ", " << pUpp << "], "
          << "and tensor width in dimension " << d << " is " << dsize
          << ". This padding either (i) results in negative tensor width or "
          << "(ii) selects a region off the tensor. ";
    throw poputil::poplibs_error(errss.str());
  }
}

poplar::Tensor Padder::getPartPaddedTensor(const poplar::Tensor &tIn,
                                           unsigned d, ptrdiff_t pLow,
                                           ptrdiff_t pUpp) {
  poplar::Tensor tInMetadata;
  if (tIn.elementType() == poplar::QUARTER) {
    tInMetadata = tIn.getMetadata();
  }
  poplar::Tensor t = tIn;

  validatePadArgs(t, d, pLow, pUpp);
  if (pLow > 0) {
    auto padding = getPaddingTensor(t, d, pLow, true);
    t = concat(padding, t, d);
  } else if (pLow < 0) {
    t = t.slice(static_cast<size_t>(-pLow), t.dim(d), d);
  }
  if (pUpp > 0) {
    auto padding = getPaddingTensor(t, d, pUpp, false);
    t = concat(t, padding, d);
  } else if (pUpp < 0) {
    long until = static_cast<long>(t.dim(d)) + pUpp;
    // we have confirmed that t.dim(d) + pUpp >= 0 in validatePadArgs,
    // so the static_cast below is safe.
    t = t.slice(0, static_cast<size_t>(until), d);
  }
  if (tInMetadata.valid()) {
    t.associateMetadata(tInMetadata);
  }
  return t;
}

poplar::Tensor EdgePadder::getPaddingTensor(const poplar::Tensor &t, unsigned d,
                                            ptrdiff_t padSize, bool padIsLow) {
  if (t.dim(d) == 0) {
    throw poputil::poplibs_error("cannot do edge padding: dimension size is 0");
  }

  poplar::Tensor edgePadding;

  unsigned padSize32 = static_cast<unsigned>(padSize);
  if (padIsLow) {
    edgePadding = t.slice(0, 1, d).broadcast(padSize32, d);
  } else {
    edgePadding = t.slice(t.dim(d) - 1, t.dim(d), d).broadcast(padSize32, d);
  }
  return edgePadding;
}

poplar::Tensor ReflectPadder::getPaddingTensor(const poplar::Tensor &t,
                                               unsigned d, ptrdiff_t padSize,
                                               bool padIsLow) {
  size_t padSizeUnsigned = static_cast<size_t>(padSize);
  if (padSizeUnsigned >= t.dim(d)) {
    throw poputil::poplibs_error("padSize too large for reflection padding");
  }

  poplar::Tensor reflPadding;

  if (padIsLow) {
    reflPadding = t.slice(1, padSizeUnsigned + 1, d);
  } else {
    reflPadding = t.slice(t.dim(d) - padSizeUnsigned - 1, t.dim(d) - 1, d);
  }
  reflPadding = reflPadding.reverse(d);
  return reflPadding;
}

std::unique_ptr<Padder> getPtrPadder(Type type) {
  std::unique_ptr<Padder> ptrPadder;

  switch (type) {
  case Type::EDGE: {
    ptrPadder.reset(new EdgePadder());
    break;
  }

  case Type::REFLECT: {
    ptrPadder.reset(new ReflectPadder());
  }
  }
  return ptrPadder;
}

} // namespace padding
} // namespace popops
