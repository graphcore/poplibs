// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popops/SequenceSlice.hpp"
#include "poplar/Interval.hpp"
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include "poplibs_support/ContiguousRegionsByTile.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/DynamicSlice.hpp"
#include "popops/Rearrange.hpp"
#include "popops/Zero.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VarStructure.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <boost/optional.hpp>

#include <cassert>
#include <numeric>
#include <type_traits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;
using namespace poplibs;

namespace {
struct SliceOptions {
  SliceOptions() = default;

  // The target maximum temporary memory usage for the operation. This
  // may not be satisfiable.
  double availableMemoryProportion = 0.6;
};

static SliceOptions parseSliceOptions(const OptionFlags &optionFlags) {
  SliceOptions options;

  using poplibs::OptionHandler;
  using poplibs::OptionSpec;

  /*
   * Any changes to spec must be reflected in the documentation comment in
   * the header.
   */
  const OptionSpec spec{
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(options.availableMemoryProportion, 0.)}};

  for (const auto &entry : optionFlags) {
    spec.parse(entry.first, entry.second);
  }
  return options;
}

} // unnamed namespace

namespace popops {

// Generate sequenceSlice vertices on the specified tile. These are
// supervisor vertices.
static void generateSequenceSliceVertexOnTile(
    Graph &graph, const ComputeSet &cs, unsigned tile, const Tensor &tSrc,
    const Tensor &tDst, const Tensor &tSrcOffset, const Tensor &tDstOffset,
    const Tensor &tN, const std::string &vertexName,
    boost::optional<unsigned> baseOffset, const DebugNameAndId &dnai) {
  assert(tSrc.rank() >= 2);
  assert(tDst.rank() >= 2);
  auto tSrc2d = tSrc.flatten(1, tSrc.rank());
  auto tDst2d = tDst.flatten(1, tDst.rank());
  assert(tSrc2d.dim(1) == tDst2d.dim(1));
  assert(tSrcOffset.rank() == 1);
  assert(tDstOffset.rank() == 1);
  assert(tN.rank() == 1);
  const auto dType = tSrc.elementType();
  assert(dType == tDst.elementType());
  const auto &target = graph.getTarget();

  auto bytesPerElement = tSrc.dim(1) * target.getTypeSize(dType);
  if (bytesPerElement % graph.getTarget().getAtomicStoreGranularity() != 0)
    logging::popops::warn("sequenceSlice with {} bytesPerElement on tile {}",
                          bytesPerElement, tile);
  Tensor tTmp = graph.addVariable(UNSIGNED_INT, {3 * tN.numElements()},
                                  {dnai, "temp copy info"});
  auto v = graph.addVertex(cs, templateVertex(vertexName, dType),
                           {{"srcT", tSrc2d.flatten()},
                            {"dstT", tDst2d.flatten()},
                            {"srcOffsetT", tSrcOffset},
                            {"dstOffsetT", tDstOffset},
                            {"nElementsT", tN},
                            {"tmpT", tTmp}});
  graph.setInitialValue(v["srcFirst"], 0);
  graph.setInitialValue(v["dstFirst"], 0);
  graph.setInitialValue(v["numSrcElements"], tSrc2d.dim(0));
  graph.setInitialValue(v["numDstElements"], tDst2d.dim(0));
  graph.setInitialValue(v["regionSize"], tSrc2d.dim(1));

  graph.setTileMapping(v, tile);
  graph.setTileMapping(tTmp, tile);
}

// Generate vertices for sequenceSlice.
// This is based on generateMultiSliceVertices.
// Enhancements may be needed to split the slicing dimension across tiles.
static void generateSequenceSliceVertices(
    Graph &graph, Sequence &prog, Tensor tSrc, Tensor tDst, Tensor tN,
    Tensor tSrcOffsets, Tensor tDstOffsets, boost::optional<unsigned> srcOffset,
    const OptionFlags &optionFlags, const DebugNameAndId &dnai) {
  const auto options = parseSliceOptions(optionFlags);

  auto cs = graph.addComputeSet({dnai, "SequenceSlice"});

  assert(tSrc.rank() == 2);
  assert(tDst.rank() == 2);
  assert(tSrc.dim(1) == tDst.dim(1));
  assert(tSrc.elementType() == tDst.elementType());
  assert(tSrcOffsets.rank() == 1);
  assert(tDstOffsets.shape() == tDstOffsets.shape());
  assert(tN.shape() == tSrcOffsets.shape());

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto type = tDst.elementType();

  // We will map vertices based on whichever of the input and output is
  // the biggest in the hope of minimising exchange. Exchanging the destination
  // is more expensive as it's an InOut.
  // TODO: When zeroing unused elements it may be better to specialise the
  // vertices to include the zeroing. This would give them an Output rather than
  // InOut and remove the need for a precopy.
  bool baseIsSrc = tSrc.numElements() > tDst.numElements() * 2;

  // When the source isn't the base it may be exchanged, which is very
  // expensive for subword entities. regroupIfBeneficial only rearranges when a
  // group of at least 4 halves is requested.
  if (!baseIsSrc && target.getTypeSize(type) <= 4) {
    auto preferredGrouping = 8 / target.getTypeSize(type);
    if ((tSrc.numElements() / tSrc.dim(0)) % preferredGrouping == 0) {
      tSrc = rearrange::regroupIfBeneficial(
          graph, tSrc, preferredGrouping, prog, {dnai, "groupIfBeneficial-4"});
    } else {
      logging::popops::warn("sequence slice NOT regrouping src as sliced "
                            "dimension not a multiple of {} elements",
                            preferredGrouping);
    }
  }
  auto &base = baseIsSrc ? tSrc : tDst;
  logging::popops::debug("baseIsSrc {} ({} > 2*{})", baseIsSrc,
                         tSrc.numElements(), tDst.numElements());
  logging::popops::debug("base shape {}", base.shape());

  // Build vertices assuming all sliced dimensions have the same mapping as
  // the first one and the non-sliced dimension is contiguous. If this is
  // not honoured gathering internal exchange/copies will be generated
  auto baseSlice0 = base.slice(0, 1, 0);
  auto mappingSlice0 = graph.getTileMapping(baseSlice0);

  // Check the spread of the base tensor over tiles against an available
  // memory proportion to determine if we will use excessive temporary memory
  // for the base tensor and result or not.
  //
  boost::optional<Tensor> originalDst;
  {
    const auto maxUnslicedElemsPerTile = [&] {
      std::size_t maxElems = 0;
      for (unsigned tile = 0; tile < mappingSlice0.size(); ++tile) {
        const auto elemsThisTile = std::accumulate(
            mappingSlice0[tile].begin(), mappingSlice0[tile].end(),
            std::size_t(0),
            [](std::size_t t, const Interval &i) { return t + i.size(); });
        if (elemsThisTile != 0)
          logging::popops::trace("tile {} has elements: {}", tile,
                                 elemsThisTile);
        maxElems = std::max(maxElems, elemsThisTile);
      }
      return maxElems;
    }();

    // Report balance.
    if (logging::shouldLog(logging::Module::popops, logging::Level::Info)) {
      const auto maxTUnslicedElemsPerTile = [&] {
        std::size_t maxElems = 0;
        auto mappingT = graph.getTileMapping(base.transpose().slice(0, 1, 0));
        for (unsigned tile = 0; tile < mappingT.size(); ++tile) {
          const auto elemsThisTile = std::accumulate(
              mappingT[tile].begin(), mappingT[tile].end(), std::size_t(0),
              [](std::size_t t, const Interval &i) { return t + i.size(); });
          if (elemsThisTile != 0)
            logging::popops::trace("mappingT {} has elements: {}", tile,
                                   elemsThisTile);
          maxElems = std::max(maxElems, elemsThisTile);
        }
        return maxElems;
      }();
      logging::popops::info("seqSlice: mappingT has max: {}",
                            maxTUnslicedElemsPerTile);
    }

    const auto balancedUnslicedElemsPerTile =
        gccs::ceildiv(base.dim(1), numTiles);

    // If we are already as well balanced as we can be then we can't do
    // anything about this without a planned multi-stage or even a serialized
    // slice which we won't try for the timebeing.
    logging::popops::info("sequenceSlice balance check {} > {}",
                          maxUnslicedElemsPerTile,
                          balancedUnslicedElemsPerTile);
    if (maxUnslicedElemsPerTile > balancedUnslicedElemsPerTile) {
      const auto bytesPerElem = target.getTypeSize(type);
      const auto maxBaseBytesPerTile =
          maxUnslicedElemsPerTile * base.dim(0) * bytesPerElem;
      const unsigned availableBytesPerTile = std::ceil(
          target.getBytesPerTile() * options.availableMemoryProportion);

      // We first check if having to rearrange the base slice would cause us to
      // exceed our temporary memory limit to avoid introspecting again if we
      // don't need to.
      logging::popops::info("sequenceSlice bytesPerTile {} vs {}",
                            maxBaseBytesPerTile, availableBytesPerTile);
      bool alwaysRearrange = true;
      if (alwaysRearrange)
        logging::popops::info("sequenceSlice forcing rearrangement");
      if (alwaysRearrange || maxBaseBytesPerTile > availableBytesPerTile) {
        // Do a cheap but imprecise approximation of whether or not all the
        // slices of the base tensor have the same tile mapping as the first by
        // checking just one other slice, chosen to be the last heuristically.
        //
        // If the mapping of all slices does not match we will have to
        // rearrange and hence we will know that our temporary memory budget
        // will be exceeded by rearranging this base tensor (we already checked
        // the max size on a tile above).
        auto n = base.dim(0);
        auto baseSliceN = base.slice(n - 1, n, 0);
        auto mappingSliceN = graph.getTileMapping(baseSliceN);

        if (mappingSlice0 != mappingSliceN) {
          // Rearrange the base tensor to be better spread
          if (!baseIsSrc)
            originalDst = base;
          auto minGrainSize = type == HALF ? 2 : 1;
          Tensor newBase =
              createSliceableTensor(graph, type, base.shape(), {0}, {1},
                                    minGrainSize, {dnai, "baseRearranged"});
          logging::popops::info("sequenceSlice rearranging (shape {})",
                                newBase.shape());
          prog.add(Copy(base, newBase, false, {dnai}));
          baseSlice0 = newBase.slice(0, 1, 0);
          mappingSlice0 = graph.getTileMapping(baseSlice0);
          if (baseIsSrc)
            tSrc = newBase;
          else
            tDst = newBase;
        }
      }
    }
  }

  // instantiate vertices following the mapping of the base's first slice
  std::size_t nActiveTiles{0}, maxTcrs{0};
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    if (mappingSlice0[tile].empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(baseSlice0, mappingSlice0[tile]);
    if (tileContiguousRegions.size() == 0) {
      // do nothing on this tile
      continue;
    }
    ++nActiveTiles;
    maxTcrs = std::max(maxTcrs, tileContiguousRegions.size());

    Tensor tileSrc = concat(tSrc.slices(tileContiguousRegions, 1), 1);
    Tensor tileDst = concat(tDst.slices(tileContiguousRegions, 1), 1);

    std::string vertexName{"popops::SequenceSlice"};
    generateSequenceSliceVertexOnTile(graph, cs, tile, tileSrc, tileDst,
                                      tSrcOffsets, tDstOffsets, tN, vertexName,
                                      srcOffset, {dnai});
  }
  logging::popops::info(
      "sequenceSlice added for up to {} contiguous regions on {} tiles",
      maxTcrs, nActiveTiles);

  prog.add(Execute(cs, {dnai}));

  // If we rearranged the output copy back to the original
  if (originalDst) {
    prog.add(Copy(tDst, *originalDst, false, {dnai}));
  }
}

void sequenceSlice(poplar::Graph &graph, const poplar::Tensor &tSrc,
                   const poplar::Tensor &tDst, const poplar::Tensor &tN,
                   const poplar::Tensor &tSrcOffset,
                   const poplar::Tensor &tDstOffset, bool zeroUnused,
                   poplar::program::Sequence &prog,
                   const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(tSrc, tDst, tN, tSrcOffset, tDstOffset));

  logging::popops::info("sequenceSlice {} -> {}, name={}, {} slices, zero={}",
                        tSrc.shape(), tDst.shape(), debugContext.getPathName(),
                        tN.numElements(), zeroUnused);

  if (tN.rank() != 1)
    throw graph_connection_error("sequenceSlice: tN must be a 1d tensor");
  if (tSrcOffset.shape() != tN.shape())
    throw graph_connection_error("sequenceSlice: tSrcOffset must be a 1d "
                                 "tensor with the same length as tN");
  if (tDstOffset.shape() != tN.shape())
    throw graph_connection_error("sequenceSlice: tDstOffset must be a 1d "
                                 "tensor with the same length as tN");
  if (tSrc.rank() < 2)
    throw graph_connection_error(
        "sequenceSlice: tSrc must have a rank of at least 2");
  if (tDst.rank() < 2)
    throw graph_connection_error(
        "sequenceSlice: tDst must have a rank of at least 2");
  if (tDst.containsAliases() || tDst.containsConstant())
    throw graph_connection_error(
        "sequenceSlice: tDst must not contain aliases or constants");
  if (tSrc.dim(0) == 0 || tDst.dim(0) == 0) {
    // No slicing to be done.
    if (zeroUnused)
      popops::zero(graph, tDst, prog, {debugContext, "AlwaysZero"});
    return;
  }
  if (tSrc.numElements() / tSrc.dim(0) != tDst.numElements() / tDst.dim(0))
    throw graph_connection_error(
        "sequenceSlice: tSrc and tDst (inner) dimensions " +
        tSrc.shapeToString() + " / " + tDst.shapeToString() +
        " are incompatible");
  // Zero the output if required.
  // If we encounter signficiant output rearrangement costs this could be
  // pushed into a different vertex with an Output rather than InOut tOut.
  if (zeroUnused)
    popops::zero(graph, tDst, prog, {di, "InitialZero"});

  // Expand to get an index for each element in the output
  auto tSrc2d = tSrc.flatten(1, tSrc.rank());
  auto tDst2d = tDst.flatten(1, tDst.rank());
  generateSequenceSliceVertices(graph, prog, tSrc2d, tDst2d, tN, tSrcOffset,
                                tDstOffset, {}, {}, {di, "Slicing"});
}

} // end namespace popops
