#include "ReductionStages.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>

#include <boost/icl/split_interval_map.hpp>
#include <boost/optional.hpp>

#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_support/Algorithms.hpp>
#include <poplibs_support/ContiguousRegionsByTile.hpp>
#include <poplibs_support/vv_iterator.hpp>
#include <poplibs_support/IclUtil.hpp>
#include <poplibs_support/print.hpp>

#include <popops/Cast.hpp>

#include "ReductionConnection.hpp"
#include "RegionWrapping.hpp"
#include "IntermediatePartialsUtil.hpp"

using namespace poplar;
using namespace poputil;
using namespace poplibs;

namespace popops {

void inputToOutputNoExchange(Graph &graph,
    const Tensor &in,
    const Graph::TileToTensorMapping &mapping,
    const Tensor &finalOutput,
    Type inVertexType,
    ReduceParams params,
    ComputeSetList &css,
    const std::string &debugPrefix,
    ReductionDebug *debug) {

  // If we're doing an update, things get really complicated if we have to do
  // casts too, so for now just use the same type for accumulation as the
  // output type.
  if (params.update)
    inVertexType = finalOutput.elementType();

  // The inVertex type is also the type that the vertex outputs (for simplicity
  // and to avoid having a million template specialisations). If it is
  // different from the output type we just add an explicit cast.

  Tensor out;
  bool castRequired = inVertexType != finalOutput.elementType();
  if (castRequired) {
    out = graph.clone(inVertexType, finalOutput);
    graph.setTileMapping(out, graph.getTileMapping(finalOutput, false));
  } else {
    out = finalOutput;
  }

  assert(in.rank() == 2);

  // Number of output values of the reduction.
  auto outputSize = in.dim(1);

  assert(out.rank() == 1);
  assert(out.numElements() == outputSize);

  auto inType = in.elementType();

  // If the output isn't mapped yet, map it exactly the same as the first
  // row of the input which ensures no exchange will happen.
  try {
    graph.getTileMapping(out);
  } catch (invalid_tile_mapping&) {
    auto mapping = graph.getTileMapping(in.slice(0, 1, 0));
    graph.setTileMapping(out, mapping);
  }

  // Get the set of contiguous regions on each tile (splitting them if
  // necessary at tile mapping boundaries). The region indices here are in
  // the flattened input tensor.
  auto contiguousRegionsByTile = getSortedContiguousRegionsByTile(graph,
                                                                  in,
                                                                  mapping);

  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Input to Output (No Exchange)";
  }

  std::size_t csPos = css.pos();

  // Loop through the tiles. We can process each tile independently.
  for (unsigned tile = 0; tile < contiguousRegionsByTile.size(); ++tile) {
    const auto &contiguousRegionsThisTile = contiguousRegionsByTile[tile];

    // Ignore empty tiles.
    if (contiguousRegionsThisTile.empty())
      continue;

    // Wrap the regions to the output size, and add them all to a
    // split_interval_set.
    auto outputRegionsSplitIcl
        = getSplitWrappedRegions(contiguousRegionsThisTile, outputSize);

    // Convert it to poplar format.
    std::vector<Interval> outputRegionsSplit
        = splitIntervalSetToPoplar(outputRegionsSplitIcl);

    // Split them if it would make it faster by processing them separately
    // with different vertices.
    outputRegionsSplit = splitOutputRegionsForWorkers(
                           graph.getTarget(),
                           graph.getTarget().getNumWorkerContexts(),
                           params.op,
                           inType,
                           outputRegionsSplit
                         );

    // Store partials and output tensors that we will reduce.
    std::vector<RegionReduction> reductions;

    reductions.reserve(outputRegionsSplit.size());

    for (const auto &re : outputRegionsSplit) {
      RegionReduction rt;

      // Get the output slice.
      Tensor outputSlice = out.slice(re.begin(), re.end());
      rt.output = outputSlice;

      // Get the input slice for this output region.
      Tensor partialSlice = in.slice(re.begin(), re.end(), 1);

      // It should be the case that every element of partialSlice is mapped
      // to this tile.

      Tensor partialSliceFlat = partialSlice.flatten();

      // Optimisation: We could get the sorted contiguous regions for this slice
      // and then merge rows as is done below.

      // Get the contiguous regions for the slice.
      auto contiguousRegions = partialSliceFlat.getContiguousRegions();

      for (auto cRe : contiguousRegions) {
        // It should be the case that the contiguous region is a whole number
        // of rows of partialSlice.
        assert(cRe.begin() % re.size() == 0);
        assert(cRe.end() % re.size() == 0);

        rt.partials.push_back(partialSliceFlat.slice(cRe));

        // Debug information
        ReductionDebug::Partial di;
        // Note that this is the region of the sliced tensor.
        di.sourceCols = re;
        di.sourceRows = {cRe.begin() / re.size(),
                        cRe.end() / re.size()};
        di.sourceTile = tile;
        rt.partialsDebugInfo.push_back(di);
      }

      // Debugging info about the output..
      rt.outputDebugInfo.outputRegion = re;
      rt.outputDebugInfo.dataRegion = re;

      reductions.push_back(rt);
    }

    ReductionDebug::TileReduction *tileDebug = nullptr;
    if (stageDebug != nullptr) {
      stageDebug->tiles.emplace_back();
      tileDebug = &stageDebug->tiles.back();
    }

    // Start from our current position in the compute set list.
    ComputeSetList cssFork = css;
    connectReductions(graph, cssFork, params, inType, inVertexType,
                      tile, reductions,
                      debugPrefix + "/InToOutNoExchange", tileDebug);
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos)
      csPos = cssFork.pos();
  }

  css.setPos(csPos);

  if (castRequired) {
    // If the mapping of finalOutput was incomplete we need to set it.
    graph.setTileMapping(finalOutput, graph.getTileMapping(out));
    auto cs = css.add(graph, debugPrefix + "/Cast");
    cast(graph, out, finalOutput, cs);
  }
}

struct WrappedSplitContiguousSortedRegions {
  struct ColumnRegion {
    // A list of contiguous memory regions, and for each one
    // the rows that make it up (in order!)
    std::vector<std::vector<std::size_t>> contiguousRows;
  };

  // A set of regions of columns, for example column [0,3), etc.
  std::vector<ColumnRegion> cols;
};

// Given a set of sorted contiguous regions, this wraps them to
// a 2D matrix (number of columns is given by wrapSize) and then
// splits them all based on splitRegions. splitRegions should
// be arranged so that after the splits every splitRegion has only whole
// regions in it.
//
// For each column region the sorted contiguous regions are found
// (indexed by row, since every row is a contiguous region).
WrappedSplitContiguousSortedRegions
wrapAndSplitContiguousSortedRegions(
    const std::vector<std::vector<Interval>> &contiguousRegionSets,
    const boost::icl::split_interval_set<size_t> &splitRegions,
    size_t wrapSize) {

  // Convert the splitRegions into a map from index to splitRegion index.
  boost::icl::interval_map<size_t, size_t,
      boost::icl::partial_enricher> splitRegionsMap;
  size_t n = 0;
  for (const auto &re : splitRegions) {
    splitRegionsMap.set(std::make_pair(re, n));
    ++n;
  }

  WrappedSplitContiguousSortedRegions out;
  out.cols.resize(n);

  // For each set of regions that are contiguous.
  for (const auto &contiguousSet : contiguousRegionSets) {

    boost::optional<size_t> lastOutputRegion;

    wrapRegionsToRows(contiguousSet.begin(), contiguousSet.end(), wrapSize,
                [&](size_t begin, size_t end, size_t row) {

      for (auto col = splitRegionsMap(begin);
           col <= splitRegionsMap(end-1); ++col) {
        assert(col < n);

        if (!lastOutputRegion || col != lastOutputRegion.get()) {
          // Add a new one.
          out.cols[col].contiguousRows.emplace_back();
          lastOutputRegion = col;
        }
        // Append it to that.
        out.cols[col].contiguousRows.back().push_back(row);
      }

    });

  }
  return out;
}

IntermediatePartials inputToIntermediateNoExchange(Graph &graph,
    const Tensor &in,
    const Graph::TileToTensorMapping &mapping,
    Operation op,
    const Type &inVertexType,
    const Type &outType,
    ComputeSetList &css,
    const std::string &debugPrefix,
    ReductionDebug *debug) {

  // TODO: inVertexType is currently unused.

  // Number of output values of the reduction.
  auto outputSize = in.dim(1);

  auto inType = in.elementType();

  // Add a new tensor for each tile to output its partials to. These tensors
  // and the meta-info needed are stored in an IntermediatePartials.
  IntermediatePartials ir;
  ir.setDataType(outType);
  ir.setOutputSize(outputSize);

  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Input to Intermediate (No Exchange)";
  }

  std::size_t csPos = css.pos();

  // Get the set of contiguous regions on each tile (splitting them if
  // necessary at tile mapping boundaries). The region indices here are in
  // the flattened input tensor.
  auto contiguousRegionsByTile = getSortedContiguousRegionsByTile(graph,
                                                                  in,
                                                                  mapping);

  // Loop through the tiles. We can process each tile independently.
  for (unsigned tile = 0; tile < contiguousRegionsByTile.size(); ++tile) {
    const auto &contiguousRegionsThisTile = contiguousRegionsByTile[tile];

    // Ignore empty tiles.
    if (contiguousRegionsThisTile.empty())
      continue;

    // Get the set of output regions for this tile, but separated by contiguous
    // region. For example if we had these contiguous regions:
    //
    //  [##  #][###][###]
    //  [##  #]     [###]
    //      [#]
    //              [###  ###
    //   ##]
    //
    // The output would be
    //
    //  [##][#][###][###][###]
    //
    auto outputRegionsSplitIcl
        = getSplitWrappedRegions(contiguousRegionsThisTile, outputSize);

    // Convert to poplar format.
    auto outputRegionsSplit = splitIntervalSetToPoplar(outputRegionsSplitIcl);

    // Split them if it would make it faster by processing them separately
    // with different vertices. This never merges regions.
    outputRegionsSplit = splitOutputRegionsForWorkers(
                           graph.getTarget(),
                           graph.getTarget().getNumWorkerContexts(),
                           op,
                           inType,
                           outputRegionsSplit
                         );

    // Convert back.
    outputRegionsSplitIcl = poplarToSplitIntervalSet(outputRegionsSplit);

    // Add a tensor for this tile.
    Tensor data = graph.addVariable(outType,
                                    {outputRegionsSplitIcl.size()},
                                    debugPrefix + "/tile_data");
    // Map it to this tile.
    graph.setTileMapping(data, tile);

    // Record the tensor in the IR, and the merged regions.
    ir.setTensor(tile,
                 data,
                 boost::icl::interval_set<std::size_t>(outputRegionsSplitIcl));


    // Now get the contiguous sorted regions for each output region.
    auto regions = wrapAndSplitContiguousSortedRegions(
                     contiguousRegionsThisTile,
                     outputRegionsSplitIcl,
                     outputSize);

    assert(regions.cols.size() == outputRegionsSplit.size());

    // Store the tensors that we will connect up.
    std::vector<RegionReduction> reductions;
    reductions.reserve(outputRegionsSplit.size());

    for (size_t i = 0; i < outputRegionsSplit.size(); ++i) {
      auto re = outputRegionsSplit[i];

      RegionReduction rt;

      for (const auto &partialRows : regions.cols[i].contiguousRows) {
        // Convert the rows in partialRows to a tensor by concatenating slices
        // of it.

        // Most of the time it'll be 0, 1, 2, 3 etc. and in that case
        // we can merge them. Meow.
        std::vector<Tensor> cats;

        size_t rowBegin = 0;
        size_t rowEnd = 0;
        for (size_t p = 0; p < partialRows.size(); ++p) {
          if (partialRows[p] == rowEnd) {
            // If we can just continue this slice, do so.
            ++rowEnd;
          } else {
            // Otherwise append the previous slice and start a new one.
            if (rowBegin != rowEnd)
              cats.push_back(in.slice({rowBegin, re.begin()},
                                     {rowEnd, re.end()}));
            rowBegin = partialRows[p];
            rowEnd = rowBegin + 1;
          }
        }
        // The final one.
        if (rowBegin != rowEnd)
          cats.push_back(in.slice({rowBegin, re.begin()}, {rowEnd, re.end()}));

        rt.partials.push_back(concat(cats, 0).flatten());

        ReductionDebug::Partial di;
        di.sourceCols = re;
        di.sourceRows = {rowBegin, rowEnd};
        di.sourceTile = tile;
        rt.partialsDebugInfo.push_back(di);
      }

      // Connect the output region. The region in the final output is
      // it.first, we need to convert it to a region in ir.data(tile).
      size_t len = re.size();
      size_t dataIdx = ir.dataElement(tile, re.begin());

      // At this point it should be true that [dataIdx, dataIdx+len) is one
      // region in the output.
      assert(ir.dataElement(tile, re.begin() + len - 1) == dataIdx + len - 1);

      rt.output = ir.data(tile).slice(dataIdx, dataIdx+len);

      // Debugging info about the output..
      rt.outputDebugInfo.outputRegion = re;
      rt.outputDebugInfo.dataRegion = {dataIdx, dataIdx+len};

      reductions.push_back(rt);
    }

    ReductionDebug::TileReduction *tileDebug = nullptr;
    if (stageDebug != nullptr) {
      stageDebug->tiles.emplace_back();
      tileDebug = &stageDebug->tiles.back();
    }

    // Start from our current position in the compute set list.
    ComputeSetList cssFork = css;
    connectReductions(graph, cssFork, op, inType, outType,
                      tile, reductions,
                      debugPrefix + "/InToIntermediateNoExchange", tileDebug);
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos)
      csPos = cssFork.pos();
  }

  css.setPos(csPos);

  return ir;
}

IntermediatePartials intermediateToIntermediate(Graph &graph,
    const IntermediatePartials &ipIn,
    Operation op,
    const Type &inVertexType,
    const Type &outType,
    ComputeSetList &css,
    const std::string &debugPrefix,
    ReductionDebug *debug) {

  // TODO: inVertexType is currently unused.

  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Intermediate to Intermediate";
  }

  IntermediatePartials ir;

  ir.setOutputSize(ipIn.outputSize());
  ir.setDataType(outType);

  auto inType = ipIn.dataType();

  const auto &target = graph.getTarget();

  unsigned grainSize = target.getVectorWidth(inType);

  if (grainSize == 0)
    throw poputil::poplibs_error("Zero vector width for type " +
                                inType.toString());

  // The grain size is doubled for ADD (and ABS_ADD and SQUARE_ADD) because
  // these operations have dedicated instructions on Colossus that can operate
  // on twice as much data as all the other operations (MUL etc).
  if (op == popops::Operation::ADD ||
      op == popops::Operation::SQUARE_ADD) // Or ABS_ADD.
    grainSize *= 2;

  // If each piece is really small the overhead of having extra reduction
  // stages, and exchange and everything outweighs the savings.
  //
  // Optimisation: This was found empirically and not tested a lot.
  std::size_t minPieceSize = 64;

  auto splitMapIcl = calculateSplit(ipIn,
                                    grainSize,
                                    grainSize, 2,
                                    minPieceSize,
                                    target.getNumTiles());

  std::vector<boost::icl::interval<std::size_t>::type> allOutputRegionsSplit;
  allOutputRegionsSplit.reserve(splitMapIcl.iterative_size());
  for (const auto &it : splitMapIcl)
    allOutputRegionsSplit.push_back(it.first);

  // 1. Find all the partials for each output region.
  // 2. Split them up into N pieces.
  // 3. Assign them to tiles in a round-robin way.

  const auto &tilesForOutput = ipIn.getTilesForOutput();

  // Just do a round-robin assignment for now.

  // If we assign two blocks of the same interval to one tile then they will
  // be merged.

  struct ReductionBlock {
    std::vector<unsigned> sourceTiles;
  };

  // The reductions for each tile.
  struct TileReductions {
    // Map from the interval number (index into allOutputRegionsSplit)
    // to a list of source tiles to reduce on this tile.
    std::map<unsigned, std::vector<unsigned>> sourceTilesForInterval;
  };

  std::vector<TileReductions> tileReductions(target.getNumTiles());

  // Divide a by b, rounding up.
  auto udiv = [](unsigned a, unsigned b) { return (a + b - 1) / b; };

  unsigned t = 0;
  unsigned ival = 0;
  for (const auto &it : splitMapIcl) {
    const auto &sourceTiles = tilesForOutput(it.first.lower());

    auto numPartials = sourceTiles.size();
    auto splitCount = it.second;

    assert(splitCount > 0);

    // N is the number of rows to take for each reduction. This should be at
    // least 2 so we actually do some reducing.
    std::size_t N = udiv(numPartials, splitCount);
    if (N < 2)
      N = 2;

    for (unsigned i = 0; i < numPartials; i += N) {
      auto &st = tileReductions[t].sourceTilesForInterval[ival];

      unsigned Nclip = std::min(N, numPartials - i);

      st.reserve(st.size() + Nclip);

      st.insert(st.end(),
                sourceTiles.nth(i),
                sourceTiles.nth(i + Nclip));

      t = (t + 1) % target.getNumTiles();
    }

    ++ival;
  }

  std::size_t csPos = css.pos();

  // For each output tile...
  for (unsigned tile = 0; tile < tileReductions.size(); ++tile) {
    auto &tr = tileReductions[tile];

    if (tileReductions[tile].sourceTilesForInterval.empty())
      continue; // TODO: This could be break; if you're feeling confident.

    // Work out the set of all output regions for this tile.
    boost::icl::interval_set<std::size_t> outputRegionsMergedIcl;
    for (auto it : tr.sourceTilesForInterval) {
      outputRegionsMergedIcl.insert(allOutputRegionsSplit[it.first]);
    }

    // Add a variable to receive the results.
    Tensor data = graph.addVariable(outType,
                                    {outputRegionsMergedIcl.size()},
                                    debugPrefix + "/tile_data");

    graph.setTileMapping(data, tile);

    // Add it to the output.
    ir.setTensor(tile, data, outputRegionsMergedIcl);

    // Store the tensors that we will connect up.
    std::vector<RegionReduction> reductions;
    reductions.reserve(tr.sourceTilesForInterval.size());

    // For each of the regions.
    for (const auto &it : tr.sourceTilesForInterval) {
      auto re = allOutputRegionsSplit[it.first];

      // The corresponding region in the data
      RegionReduction rt;

      size_t outputDataIdx = ir.dataElement(tile, re.lower());
      size_t len = boost::icl::size(re);

      // Check it is contiguous.
      assert(ir.dataElement(tile, re.lower() + len - 1) ==
             outputDataIdx + len - 1);

      // Loop through the source tiles for this region...
      for (auto partialTile : it.second) {
        size_t sourceDataIdx = ipIn.dataElement(partialTile, re.lower());

        assert(ipIn.dataElement(partialTile, re.upper() - 1) ==
               sourceDataIdx + boost::icl::size(re) - 1);

        rt.partials.push_back(
              ipIn.data(partialTile).slice(sourceDataIdx, sourceDataIdx + len));

        // Debugging info about the partial.
        ReductionDebug::Partial di;
        di.sourceCols = {sourceDataIdx, sourceDataIdx + len};
        di.sourceTile = partialTile;
        rt.partialsDebugInfo.push_back(di);
      }

      // Connect the output region.

      rt.output = ir.data(tile).slice(outputDataIdx, outputDataIdx + len);

      // Debugging infor about the output...
      rt.outputDebugInfo.outputRegion = {re.lower(), re.upper()};
      rt.outputDebugInfo.dataRegion = {outputDataIdx, outputDataIdx + len};

      reductions.push_back(rt);
    }

    ReductionDebug::TileReduction *tileDebug = nullptr;
    if (stageDebug != nullptr) {
      stageDebug->tiles.emplace_back();
      tileDebug = &stageDebug->tiles.back();
    }

    // Start from our current position in the compute set list.
    ComputeSetList cssFork = css;
    connectReductions(graph, cssFork, op, inType, outType,
                      tile, reductions,
                      debugPrefix + "/IntermediateToIntermediate", tileDebug);
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos)
      csPos = cssFork.pos();
  }

  css.setPos(csPos);

  return ir;
}

void intermediateToOutput(Graph &graph,
    const IntermediatePartials &ipIn,
    const Tensor &finalOutput,
    ReduceParams params,
    Type inVertexType,
    ComputeSetList &css,
    const std::string &debugPrefix,
    ReductionDebug *debug) {

  // If we're doing an update, things get really complicated if we have to do
  // casts too, so for now just use the same type for accumulation as the
  // output type.
  if (params.update)
    inVertexType = finalOutput.elementType();

  // The inVertex type is also the type that the vertex outputs (for simplicity
  // and to avoid having a million template specialisations). If it is
  // different from the output type we just add an explicit cast.

  Tensor out;
  bool castRequired = inVertexType != finalOutput.elementType();
  if (castRequired) {
    out = graph.clone(inVertexType, finalOutput);
    graph.setTileMapping(out, graph.getTileMapping(finalOutput, false));

  } else {
    out = finalOutput;
  }

  // This is assumed below.
  assert(out.rank() == 1);

  auto inType = ipIn.dataType();

  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Intermediate To Output";
  }

  // If the output isn't already mapped, map it linearly and do the reduction
  // there, otherwise decide whether it is better to do the reduction at the
  // destination or not.
  Graph::TileToTensorMapping mapping;
  try {
    mapping = graph.getTileMapping(out);
    if (!shouldReduceAtDestination(graph.getTarget(),
                                   ipIn,
                                   mapping,
                                   inVertexType,
                                   out.numElements())) {
      mapping = poputil::calcLinearTileMapping(graph, out);
    }
  } catch (invalid_tile_mapping&) {
    mapping = poputil::calcLinearTileMapping(graph, out);
    graph.setTileMapping(out, mapping);
  }

  // An interval_map from output element to the set of tiles that have
  // partials for it.
  const auto &tilesForOutput = ipIn.getTilesForOutput();

  // An interval_map from output element to the tile it is mapped to.
  auto mappingIcl = tileMappingToIntervalMap(mapping);

  assert(tilesForOutput.size() == ipIn.outputSize());
  assert(mappingIcl.size() == ipIn.outputSize());

  // We've got something like:
  //
  //   [0, 12) has partials on tiles {1, 4, 6}
  //   [12, 40) has partials on tiles {5, 6, 7}
  //   [40, 100) has partials on tiles {1, 2}
  //
  //         and
  //
  //   [0, 2) is mapped to tile 1
  //   [2, 5) is mapped to tile 4
  //   [5, 35) is mapped to tiles 3
  //   [35, 100) is mapped to tile 1
  //
  // And I want an interval_map<size_t, set<unsigned>> for each tile:
  //
  //   [
  //       {} // Tile 0
  //       {  // Tile 1
  //           [0, 2) has partials on {1, 4, 6}
  //           [35, 40) has partials on {5, 6, 7}
  //           [40, 100) has partials on tiles {1, 2}
  //       }
  //       {} // Tile 2
  //       {  // Tile 3
  //           [5, 12) has partials on {1, 4, 6}
  //           [12, 35) has partials on {5, 6, 7}
  //       }
  //       {  // Tile 4
  //           [2, 5) has partials on {1, 4, 6}
  //       }
  //   ]

  std::vector<boost::icl::interval_map<std::size_t,
                                       boost::container::flat_set<unsigned>>>
      tilesForOutputPerTile(mapping.size());

  // Iterate through both maps together.
  for_each_zipped_region(mappingIcl.begin(), mappingIcl.end(),
                         tilesForOutput.begin(), tilesForOutput.end(),
    [&](std::size_t begin, std::size_t end,
        unsigned mappedToTile,
        const boost::container::flat_set<unsigned> &partialTiles) {
      tilesForOutputPerTile[mappedToTile].set(
          std::make_pair(
            boost::icl::interval<std::size_t>::right_open(begin, end),
            partialTiles
            )
          );
    });

  std::size_t csPos = css.pos();

  // Partition tilesForOutput based on mappingIcl.

  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    if (mapping[tile].empty())
      continue;

    // Get the regions that are mapped to this tile.
    auto outputRegionsSplitIcl = poplarToSplitIntervalSet(mapping[tile]);

    // Take the subset of the map from output element to partial tiles
    // for the output regions that are mapped to this tile.
    const auto &thisTilesForOutput = tilesForOutputPerTile[tile];

    // Convert the output element indices to poplar interval format.

    std::vector<Interval> outputRegionsSplit;
    outputRegionsSplit.reserve(thisTilesForOutput.size());

    for (const auto &ival : thisTilesForOutput)
      outputRegionsSplit.emplace_back(ival.first.lower(), ival.first.upper());

    // Split them if it would make it faster by processing them separately
    // with different vertices.
    outputRegionsSplit = splitOutputRegionsForWorkers(
                           graph.getTarget(),
                           graph.getTarget().getNumWorkerContexts(),
                           params.op,
                           inVertexType,
                           outputRegionsSplit
                         );

    // Store the tensors that we will connect up. Have to do this
    // here so we can resize the Vectors in the vertex.
    std::vector<RegionReduction> reductions;

    reductions.reserve(outputRegionsSplit.size());

    // Finally we repeat the above but this time record the actual connections.
    for (const auto &re : outputRegionsSplit) {
      RegionReduction rt;

      // Connect the output. This is fine because output is 1D.
      rt.output = out.slice(re);

      rt.partials.reserve(32); // This speeds things up a bit.

      // Get the list of partials to use.
      auto partialTiles = thisTilesForOutput(re.begin());

      for (auto partialTile : partialTiles) {
        size_t sourceDataIdx = ipIn.dataElement(partialTile, re.begin());
        size_t len = re.size();

        assert(ipIn.dataElement(partialTile, re.begin() + len - 1) ==
               sourceDataIdx + len - 1);

        rt.partials.emplace_back(
              ipIn.data(partialTile).slice(sourceDataIdx, sourceDataIdx + len));

        // Debugging info about the partial.
        ReductionDebug::Partial di;
        di.sourceCols = {sourceDataIdx, sourceDataIdx + len};
        di.sourceTile = partialTile;
        rt.partialsDebugInfo.push_back(di);
      }

      // Debugging infor about the output...
      rt.outputDebugInfo.outputRegion = re;
      rt.outputDebugInfo.dataRegion = re;

      reductions.push_back(rt);
    }

    ReductionDebug::TileReduction *tileDebug = nullptr;
    if (stageDebug != nullptr) {
      stageDebug->tiles.emplace_back();
      tileDebug = &stageDebug->tiles.back();
    }

    // Start from our current position in the compute set list.
    ComputeSetList cssFork = css;
    connectReductions(graph, cssFork, params, inType, inVertexType,
                      tile, reductions,
                      debugPrefix + "/IntermediateToOutput", tileDebug);
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos)
      csPos = cssFork.pos();
  }

  css.setPos(csPos);

  if (castRequired) {
    // If the mapping of finalOutput was incomplete we need to
    // set it.
    graph.setTileMapping(finalOutput, graph.getTileMapping(out));
    auto cs = css.add(graph, debugPrefix + "/Cast");
    cast(graph, out, finalOutput, cs);
  }
}

}
