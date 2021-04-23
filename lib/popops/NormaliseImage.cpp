// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popops/NormaliseImage.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

static void commonChecks(std::string fName, const Type &type,
                         const ArrayRef<std::size_t> &shape) {
  if (type != UNSIGNED_CHAR && type != HALF && type != FLOAT)
    throw poputil::poplibs_error(fName +
                                 "() only supports "
                                 "UNSIGNED_CHAR, HALF and FLOAT inputs");
  if (shape.size() < 2 || shape[shape.size() - 1] != 3)
    throw poputil::poplibs_error(fName +
                                 "() requires the inner dimension to be 3");
}

Tensor createNormaliseImageInput(Graph &graph, const Type &type,
                                 const ArrayRef<std::size_t> shape,
                                 const DebugContext &debugContext) {
  commonChecks("createNormaliseImageInput", type, shape);
  auto t = graph.addVariable(type, shape,
                             {debugContext, "createNormaliseImageInput"});
  // The grainsize is chosen to ensure no pixels are split between tiles. Also
  // we want to ensure there are no subword accesses.
  auto atomSize = graph.getTarget().getTypeSize(type);
  auto grainSize = 3 * (4 / atomSize);
  auto numWorkers = graph.getTarget().getNumWorkerContexts();
  mapTensorLinearly(graph, t, grainSize * numWorkers, grainSize);
  return t;
}

Tensor normaliseImage(Graph &graph, Sequence &seq, Tensor tIn, float inScale,
                      Tensor offsets, Tensor scales,
                      const DebugContext &debugContext) {
  auto inType = tIn.elementType();
  auto outType = inType == UNSIGNED_CHAR ? HALF : inType;
  commonChecks("normaliseImage", inType, tIn.shape());
  if (offsets.elementType() != outType ||
      offsets.elementType() != scales.elementType())
    throw poputil::poplibs_error(
        "normaliseImage() requires the offsets and scales tensors to "
        "have the same type as the output");
  if (inScale == 0.)
    throw poputil::poplibs_error(
        "normaliseImage() must have a non-zero inScale");

  auto outShape = tIn.shape();
  outShape[outShape.size() - 1] = 4;
  auto tOut = graph.addVariable(outType, outShape, {debugContext, "normalise"});
  auto inMapping = graph.getTileMapping(tIn);
  Graph::TileToTensorMapping outMapping(inMapping.size());

  auto cs = graph.addComputeSet({debugContext, "normaliseImage"});
  auto templateVertexName =
      templateVertex("popops::NormaliseImage", inType, outType);

  const auto numWorkers = graph.getTarget().getNumWorkerContexts();
  for (unsigned tile = 0; tile != inMapping.size(); ++tile) {
    if (inMapping[tile].size() == 0)
      continue;
    if (inMapping[tile].size() > 2)
      throw poputil::poplibs_error(
          "normaliseImage() expects a single region per tile. Did you create "
          "the input using createNormaliseImageInput()?");
    auto &interval = inMapping[tile].front();
    // We're expanding the innermost dimension from 3 channels to 4.
    outMapping[tile].emplace_back(interval.begin() * 4 / 3,
                                  interval.end() * 4 / 3);
    auto &outInterval = outMapping[tile].back();
    auto v = graph.addVertex(cs, templateVertexName,
                             {{"in", tIn.flatten().slice(interval)},
                              {"out", tOut.flatten().slice(outInterval)},
                              {"scales", scales},
                              {"offsets", offsets}});
    graph.setInitialValue(v["inScale"], inScale);
    auto nPixels = interval.size() / 3;
    assert(interval.size() % 3 == 0);
    unsigned packedNPixels =
        ((nPixels / numWorkers) << 3) | (nPixels % numWorkers);
    graph.setInitialValue(v["packedNPixels"], packedNPixels);
    graph.setTileMapping(v, tile);
  }
  seq.add(Execute(cs));
  graph.setTileMapping(tOut, outMapping);

  return tOut;
}

} // namespace popops
