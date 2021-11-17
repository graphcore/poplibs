// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "ReductionStages.hpp"
#include "ReductionIntrospection.hpp"
#include "ReductionPlan.hpp"

#include <algorithm>
#include <cassert>

#include <fstream>
#include <numeric>

#include <boost/container/flat_map.hpp>
#include <boost/functional/hash.hpp>
#include <boost/icl/split_interval_map.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include "poplibs_support/logging.hpp"
#include <poplibs_support/Algorithms.hpp>
#include <poplibs_support/ContiguousRegionsByTile.hpp>
#include <poplibs_support/IclUtil.hpp>
#include <poplibs_support/print.hpp>
#include <poplibs_support/vv_iterator.hpp>

#include <popops/Cast.hpp>

#include "IntermediatePartialsUtil.hpp"
#include "ReductionConnection.hpp"
#include "RegionWrapping.hpp"

using namespace poplar;
using namespace poputil;
using namespace poplibs;
using namespace poplibs_support;

namespace popops {

// Determine grainSize.  Logically this would be per reduction, we would want
// to split with a different grainSize based on the operation and partials
// widths.  However we catch many cases by checking that all reductions have
// appropriate partials widths.  This means we can pass one grainSize
// parameter to splitOutputRegionsForWorkers which is far simpler than a per
// reduction array.

unsigned findGrainSizeForOp(Graph &graph, Type partialType,
                            Operation &operation) {
  // NOTE - The efficient (not specialisation 0,1) vertices available don't
  // benefit from a vector width of 2 * grainSize. In fact splitting to
  // grainSize produces a speed increase.  However it comes at a small cost in
  // memory which in some models was pushing it beyond the tile limit.  This
  // should be revisited. See T19945 for details.

  const unsigned grainSize = graph.getTarget().getVectorWidth(partialType);
  if (operation == popops::Operation::ADD ||
      operation == popops::Operation::SQUARE_ADD) {
    return grainSize * 2;
  }
  // Other reductions (including MAX and MIN) can generally be split between
  // workers with the basic grain size
  return grainSize;
}

// Store reduction results for a later writeUndef - store based on type.
void storeReductionResultTensors(ResultTensors &results, const Tensor &data) {
  if (results.typeA.size() == 0) {
    results.typeA.push_back(data);
  } else {
    if (results.typeA.back().elementType() == data.elementType()) {
      results.typeA.push_back(data);
    } else {
      results.typeB.push_back(data);
    }
  }
}

// Create reductions for the cases: Input to Output and Input to Intermediate
void createInputReductions(
    Graph &graph, const Tensor &in,
    boost::variant<Tensor &, IntermediatePartials &> out, bool createOutput,
    const StridedRegionsByTile &contiguousRegionsByTile,
    const TilePartialsDescription &groupedPartials, ReduceParams params,
    Type inputType, Type inVertexType, Type outputType, ComputeSetList &css,
    ResultTensors &reductionResultTensors, const DebugNameAndId &dnai) {

  logging::popops::debug("DebugStr: {}", dnai.getPathName());
  const bool isInputToOutput = out.type() == typeid(Tensor);

  // Number of columns in the reduction.
  const auto columns = in.dim(1);
  // Store the output tensors for each reduction vertex, one per column
  std::vector<Tensor> outputs(isInputToOutput ? columns : 0);
  std::size_t csPos = css.pos();
  auto inType = in.elementType();
  // Loop through the tiles. We can process each tile independently.
  for (unsigned tile = 0; tile < groupedPartials.size(); ++tile) {
    const auto &contiguousRegionsThisTile = contiguousRegionsByTile[tile];

    // Ignore empty tiles.
    if (groupedPartials[tile].empty()) {
      continue;
    }

    // Divide the patterns to split work between workers and cope with
    // other limitations
    auto splitGroupedPartials =
        dividePartials(groupedPartials[tile], graph, in.elementType(), params);

    // logging begin
    if (logging::popops::shouldLog(logging::Level::Trace)) {
      // Use to select which to view at compile time...
      auto &debugPartials = splitGroupedPartials;
      logging::popops::trace(" Tile:{} Reduction Patterns:{}", tile,
                             debugPartials.size());
      for (auto &pats : debugPartials) {
        std::stringstream colStr;
        for (auto col : pats.columns) {
          colStr << " " << col;
        }
        logging::popops::trace("  Patterns:{} Column list[{}]:{}",
                               pats.patterns.size(), pats.columns.size(),
                               colStr.str());
        for (auto &pat : pats.patterns) {
          logging::popops::trace(
              "    Pattern Inner factor:{} Start:{} Stride:{} Outer "
              "factor:{} Region:{}",
              pat.innerFactor, pat.regionOffset, pat.stride, pat.outerFactor,
              pat.regionIdx);
        }
      }
    }
    // logging end

    // Create the regionReductions with partials populated from patterns
    auto reductions = listPartialsUsingPatterns(
        splitGroupedPartials, in, contiguousRegionsThisTile, tile);
    // Record the tensor in the IR, and the merged regions.
    std::vector<Interval> outputRegionsSplit;
    for (unsigned i = 0; i < splitGroupedPartials.size(); i++) {
      for (unsigned j = 0; j < splitGroupedPartials[i].columns.size(); j++) {
        outputRegionsSplit.push_back({splitGroupedPartials[i].columns[j],
                                      splitGroupedPartials[i].columns[j] + 1});
      }
    }
    // Create a 2D array of Intervals, each referencing a single column of the
    // whole reduction - So all columns should be referenced once, when
    // we aggregate over all tiles.  This is maintained as intervals rather
    // than individual columns as it is used below (required to be intervals).
    // Dimensions: [reduction][output columns in reduction]
    // For example, 2 reductions with regions/columns
    // {[0,3)} and {[4,5), [7,8), [6,7)]}
    // Gives [0] = [0,1), [1,2), [2,3)
    //       [1] = [4,5), [7,8), [6,7)
    std::vector<std::vector<Interval>> outputRegionsSplit2D(
        splitGroupedPartials.size());
    for (unsigned i = 0; i < splitGroupedPartials.size(); i++) {
      auto columns = splitGroupedPartials[i].columns.size();
      outputRegionsSplit2D[i].reserve(columns);
      for (unsigned j = 0; j < columns; j++) {
        outputRegionsSplit2D[i].push_back(
            {splitGroupedPartials[i].columns[j],
             splitGroupedPartials[i].columns[j] + 1});
      }
    }
    if (!isInputToOutput) {
      // Add a tensor for this tile.
      const auto thisTileColumns = std::accumulate(
          groupedPartials[tile].begin(), groupedPartials[tile].end(), 0u,
          [](unsigned total, const PartialsDescription &in) {
            return total + in.columns.size();
          });
      auto data = graph.addVariable(outputType, {thisTileColumns},
                                    {dnai, "tile_data1"});
      storeReductionResultTensors(reductionResultTensors, data);

      // Map it to this tile.
      graph.setTileMapping(data, tile);
      auto outputRegionsSplitIcl = poplarToSplitIntervalSet(outputRegionsSplit);

      boost::get<IntermediatePartials &>(out).setTensor(
          tile, data,
          boost::icl::interval_set<std::size_t>(outputRegionsSplitIcl));
      // Converting this back provides a sorted list of output columns
      // which tells us the order in which to connect the 2D column intervals
      auto outputRegionsSplit = splitIntervalSetToPoplar(outputRegionsSplitIcl);
      // Create a revised mapping so that the references are wrt to the partial
      // outputs. Ie - each is in the numerical order of their original column
      // number but have an index range equal to the number of individual
      // columns found on tile.
      //
      // {[1,3)} and {[4,5), [7,8), [6,7)]}
      // Gives [0] = [1,2), [2,3)                (5 elements with gaps, start=1)
      //       [1] = [4,5), [7,8), [6,7)
      // So, columns 1, 2, 4, 7, 6 appear in that order.
      // We want to maintain order but represent 5 columns, zero based:
      //             0, 1, 2, 4, 3
      // Now   [0] = [0,1), [1,2),               (5 elements, start=0, no gaps)
      //       [1] = [2,3), [4,5), [3,4)

      for (unsigned i = 0; i < reductions.size(); i++) {
        for (unsigned j = 0; j < outputRegionsSplit2D[i].size(); j++) {
          const auto match = std::lower_bound(outputRegionsSplit.begin(),
                                              outputRegionsSplit.end(),
                                              outputRegionsSplit2D[i][j]);
          const unsigned offset = match - outputRegionsSplit.begin();
          outputRegionsSplit2D[i][j] = {offset, offset + 1};
        }
      }
    }
    for (unsigned i = 0; i < reductions.size(); i++) {
      if (isInputToOutput) {
        if (!createOutput) {
          // Get the output slice, mapping each to the required slices
          // of the output tensor to ensure correct ordering: column 0...N
          reductions[i].output =
              concat(boost::get<Tensor &>(out).slices(outputRegionsSplit2D[i]));
        } else {
          // Get the output slice.
          reductions[i].output = graph.addVariable(
              outputType, {splitGroupedPartials[i].columns.size()},
              {dnai, "output"});
          graph.setTileMapping(reductions[i].output, tile);
          // Record the outputs from the reduction ready to make the output
          // tensor, created in this function, to avoid re ordering
          for (unsigned j = 0; j < splitGroupedPartials[i].columns.size();
               j++) {
            outputs[splitGroupedPartials[i].columns[j]] =
                reductions[i].output[j].reshape({1});
          }
        }
      } else {
        auto &ir = boost::get<IntermediatePartials &>(out);
        // TODO: InputToIntermediate only: This:
        // size_t dataIdx = outputRegionsSplit2D[i][0].begin();
        // reductions[i].output = ir.data(tile).slice(dataIdx,
        //                        dataIdx + outputRegionsSplit2D[i].size());
        // With the re-arranged outputRegionsSplit2D will result in a correct
        // output but a rearrangedTensor being created at the end of the first
        // stage.  Although better than re-arranging the input it could be left
        // until the last reduction stage.  However the IR information contains
        // sorted columns, meaning that the information required is lost.
        reductions[i].output =
            concat(ir.data(tile).slices(outputRegionsSplit2D[i]));
      }
    }

    // Start from our current position in the compute set list.
    ComputeSetList cssFork = css;
    connectReductions(graph, cssFork, params, inputType, inVertexType,
                      outputType, tile, reductions, true, {dnai});
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos) {
      csPos = cssFork.pos();
    }
  }
  css.setPos(csPos);

  if (createOutput) {
    boost::get<Tensor>(out) = concat(outputs);
  }
}

void inputToOutputNoExchange(
    Graph &graph, const Tensor &in,
    const StridedRegionsByTile &contiguousRegionsByTile,
    const TilePartialsDescription &groupedPartials,
    boost::optional<Tensor> &finalOutput,
    boost::optional<Tensor> &originalOutput,
    const std::vector<std::size_t> outputShape, Type inVertexType,
    Type outputType, ReduceParams params, ComputeSetList &css,
    ResultTensors &reductionResultTensors, const DebugNameAndId &dnai) {
  // If we have an output, create the output Tensor for the
  // createInputReductions function. If we don't have an output,
  // createInputReductions will create its own output
  Tensor out;
  if (finalOutput) {
    out = finalOutput.get().flatten();
    if (!params.update) {
      storeReductionResultTensors(
          reductionResultTensors,
          originalOutput ? originalOutput.get().flatten() : out);
    }
    // If the output isn't mapped yet, map it exactly the same as the first
    // row of the input which ensures no exchange will happen.
    bool mappingComplete;
    graph.getTileMapping(out, &mappingComplete);
    if (!mappingComplete) {
      auto mapping = graph.getTileMapping(in.slice(0, 1, 0));
      graph.setTileMapping(out, mapping);
    }
  }
  assert(in.rank() == 2);

  createInputReductions(graph, in, out, !static_cast<bool>(finalOutput),
                        contiguousRegionsByTile, groupedPartials, params,
                        in.elementType(), inVertexType, outputType, css,
                        reductionResultTensors, {dnai, "InToOutNoExchange"});
  if (!finalOutput) {
    finalOutput = out;
  }
  finalOutput = finalOutput.get().reshape(outputShape);
}

IntermediatePartials inputToIntermediateNoExchange(
    Graph &graph, const Tensor &in,
    const StridedRegionsByTile &contiguousRegionsByTile,
    const TilePartialsDescription &groupedPartials, Operation op,
    const Type &inVertexType, const Type &outputType, ComputeSetList &css,
    ResultTensors &reductionResultTensors, const poplar::DebugNameAndId &dnai) {

  // Number of output values of the reduction.
  auto outputSize = in.dim(1);

  auto inType = in.elementType();

  // Add a new tensor for each tile to output its partials to. These tensors
  // and the meta-info needed are stored in an IntermediatePartials.
  IntermediatePartials ir;
  ir.setDataType(inVertexType);
  ir.setOutputSize(outputSize);

  createInputReductions(graph, in, ir, false, contiguousRegionsByTile,
                        groupedPartials, op, in.elementType(), inVertexType,
                        inVertexType, css, reductionResultTensors,
                        {dnai, "InToIntermediateNoExchange"});
  return ir;
}

template <typename T> struct DebugRange {
  T min;
  T max;
  void update(T value) {
    min = std::min(min, value);
    max = std::max(max, value);
  }
};

// Cases where we exchange a single half to reduce cause a problem can be
// implemented efficiently using continuousReduce, however inefficient copies
// would be required on the destination tile. Casting to float on the
// source tile solves this problem
bool reductionBenefitsFromPreExchangeCast(const IntermediatePartials &ipIn) {
  if (ipIn.dataType() != poplar::HALF) {
    return false;
  }
  auto tilesUsed = ipIn.tiles();
  bool atLeastOnePartialHasWidthOne =
      std::any_of(tilesUsed.begin(), tilesUsed.end(), [&](const unsigned tile) {
        return ipIn.data(tile).numElements() == 1;
      });
  return atLeastOnePartialHasWidthOne;
}

// We use a slightly different intermediate representation for the partials
// in this stage from IntermediatePartials to allow us to merge output regions
// which have partials coming from different sets of tiles. The
// IntermediatePartials type doesn't allow us to represent this because the
// number of partials for an output region is tied to the tiles on which
// those partials are mapped.
//
// TODO: Use the same type for all Intermediate representations of partials.
//
class IntermediateIntermediatePartials {
public:
  class Partials {
    std::size_t outputOffset_;
    Tensor t;

  public:
    Partials(std::size_t outputOffset, Tensor t)
        : outputOffset_(outputOffset), t(std::move(t)) {
      assert(this->t.rank() == 2);
    }
    std::size_t outputOffset() const { return outputOffset_; }
    std::size_t depth() const { return t.dim(0); }
    std::size_t width() const { return t.dim(1); }
    std::size_t numElements() const { return t.numElements(); }
    Tensor get(std::size_t depthIdx,
               boost::icl::interval<std::size_t>::type outputRegion) const {
      assert(outputRegion.lower() >= outputOffset_ &&
             outputRegion.upper() - outputOffset_ <= width());
      return t.slice(outputRegion.lower() - outputOffset_,
                     outputRegion.upper() - outputOffset_, 1)[depthIdx];
    }
    Tensor &&extract() { return std::move(t); }
    void set(Tensor tt) {
      assert(tt.rank() == 2);
      t = std::move(tt);
    }
  };

private:
  std::vector<Partials> p;

public:
  void reserve(std::size_t n) { p.reserve(n); }
  // Add a new partial to this map. Partials must be added in order
  // of ascending output offset and must form a continuous range starting
  // from zero. Asserts are added to verify this.
  void push(std::size_t outputOffset, Tensor t) {
    assert((p.empty() && outputOffset == 0) ||
           (p.back().outputOffset() + p.back().width() == outputOffset));
    p.emplace_back(Partials{outputOffset, std::move(t)});
  }
  decltype(p)::const_iterator begin() const { return p.begin(); }
  decltype(p)::const_iterator end() const { return p.end(); }
  std::size_t size() const { return p.size(); }

  const Partials &getPartialsAtOutputOffset(std::size_t offset) const {
    assert(!p.empty());
    return *std::prev(std::upper_bound(p.begin(), p.end(), offset,
                                       [](const auto val, const auto &entry) {
                                         return val < entry.outputOffset();
                                       }));
  }

  std::vector<Partials> &&extract() { return std::move(p); }
  void set(std::vector<Partials> v) {
    // Validate that our conditions hold for the new partials.
    if (v.empty()) {
      std::vector<Partials> tmp;
      std::swap(p, tmp);
      return;
    }
    assert(v[0].outputOffset() == 0);
    for (std::size_t i = 1; i != v.size(); ++i) {
      assert(v[i].outputOffset() == v[i - 1].outputOffset() + v[i - 1].width());
    }
    p = std::move(v);
  }
};

/** This pass takes the given partials and merges splits of the output region
 *  that split the given grain size (set to be the minimum grain size for
 *  implementing the reduction on-tile efficiently).
 *
 *  e.g. If we have some input partials like:
 *
 *                    |--------|
 *       |------|     |        |--------|
 *       |      |---|-|        |        |
 *       |      |   | |        |        |
 *       |------|---|-|--------|--------|
 *
 *  Here the y-axis gives the depth of the partials for each output region
 *  and the x-axis gives the output region to which those partials correspond.
 *
 *  This pass groups together partials when it means the resulting output
 *  region will be a multiple of the given grain size as these splits are
 *  respected when dividing up the reductions to be done on each tile. Grouping
 *  together partials can only be done when their depth matches.
 *
 *  The idea is that the on-tile reductions resulting from the split of these
 *  output regions will be more efficient if each region can always be a
 *  multiple of the grain size. Because each partial for each output region
 *  must be gathered via exchange to a single tile, we reason that the cost
 *  in terms of exchange code of gathering output regions together should
 *  be no more than it would have been when separate reduction vertices were
 *  processing each output region separately.
 *
 *  There are some exceptions for merging output regions when the size of
 *  the partial elements divides exchange atom size to potentially avoid
 *  problems gathering together odd sized regions via the exchange but this is
 *  largely speculative.
 *
 *  For the example above grain size that would be split by the divide between
 *  the 2nd and 3rd output regions, the resulting partials would be a series of
 *  partials tensors like:
 *
 *                   |--------|
 *       |------|    |        |--------|
 *       |      |----|        |        |
 *       |      |    |        |        |
 *       |------|----|--------|--------|
 *
 */
static void mergeOutputRegionsThatSplitGrainSize(
    IntermediateIntermediatePartials &partials_, const Target &target,
    const Type &partialsType, std::size_t grainSize) {
  auto partials = partials_.extract();
  assert(!partials.empty());
  auto outIt = partials.begin();
  auto inIt = std::next(outIt);
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();
  const auto partialsTypeSize = target.getTypeSize(partialsType);
  while (inIt != partials.end()) {
    const auto splitOffset = inIt->outputOffset();
    const auto lastNumPartials = outIt->depth();
    const auto numPartials = inIt->depth();
    const bool gathersSubwordRegions =
        splitOffset * partialsTypeSize % exchangeBytesPerCycle != 0;
    if (splitOffset % grainSize != 0 && !gathersSubwordRegions &&
        lastNumPartials == numPartials) {
      // Concat together the partials for the 2 output intervals,
      // erase the 2 entries in the list and replace them with
      // one new entry combining the 2.
      outIt->set(concat(outIt->extract(), inIt->extract(), 1));
      ++inIt;
    } else {
      ++outIt;
      *outIt = *inIt;
      ++inIt;
    }
  }
  partials.erase(std::next(outIt), partials.end());
  partials_.set(std::move(partials));
}

static std::vector<OutputPartialsDesc>
getOutputPartialsDescs(const IntermediateIntermediatePartials &partials) {
  std::vector<OutputPartialsDesc> descs;
  descs.reserve(partials.size());
  for (const auto &entry : partials) {
    descs.emplace_back(entry.outputOffset(), OutputPartialsDesc::PartialsDesc{
                                                 entry.depth(), entry.width()});
  }
  return descs;
}

IntermediatePartials intermediateToIntermediate(
    Graph &graph, const IntermediatePartials &ipIn, Operation op,
    const Type &outType, ComputeSetList &css,
    ResultTensors &reductionResultTensors, const unsigned startTile,
    const poplar::DebugNameAndId &dnai) {

  logging::popops::debug("DebugStr: {}", dnai.getPathName());

  // TODO: temporarily only for ADD, SQUARE ADD as if applied to other types
  //       we produce a mix of partials types.  This can be dealt with when
  //       D20584 lands, but implies that a final cast stage would be needed
  //       for types other than ADD, SQUARE_ADD
  const bool opIsAddOrSquareAdd =
      op == popops::Operation::ADD || op == popops::Operation::SQUARE_ADD;

  boost::optional<ComputeSet> castComputeSet;
  if (reductionBenefitsFromPreExchangeCast(ipIn) && opIsAddOrSquareAdd) {
    logging::popops::debug("Inserting pre-exchange cast half to float");
    castComputeSet = css.add(graph, {dnai, "PreExchangeCast"});
  }
  auto resultType = castComputeSet ? poplar::FLOAT : outType;

  IntermediatePartials ir;
  ir.setOutputSize(ipIn.outputSize());
  ir.setDataType(resultType);

  const auto inType = castComputeSet ? poplar::FLOAT : ipIn.dataType();
  const auto &target = graph.getTarget();

  const unsigned grainSize = findGrainSizeForOp(graph, inType, op);
  if (grainSize == 0)
    throw poputil::poplibs_error("Zero vector width for type " +
                                 inType.toString());

  // Divide a by b, rounding up.
  auto udiv = [](unsigned a, unsigned b) { return (a + b - 1) / b; };

  const auto &tilesForOutput = ipIn.getTilesForOutput();

  // Calculate the pieces we should split this into. Basically, first we get
  // the entire set of output regions from every tile (preserving splits).
  //
  // For example if we have the following regions on each tile:
  //
  // Tile                Output regions on tile
  //
  //  0    |-------|      |-------------------|  |------|
  //  1    |-------|    |---------------------|  |------------|
  //  2    |--------------------------------------------------|
  //  3            |----------------|
  //
  // Then splitOutputRegions is
  //
  //       |-------|----|-|---------|---------|--|------|-----|
  //
  boost::icl::split_interval_set<std::size_t> splitOutputRegions;
  for (const auto tile : ipIn.tiles()) {
    splitOutputRegions += ipIn.outputRegions(tile);
  }

  // Create a map that gives the partials for a particular output
  // region.
  IntermediateIntermediatePartials partialsMap;
  partialsMap.reserve(splitOutputRegions.size());
  for (const auto &interval : splitOutputRegions) {
    const auto lower = interval.lower();
    const auto regionSize = boost::icl::length(interval);
    const auto &sourceTiles = tilesForOutput(lower);
    std::vector<Tensor> toConcat;
    for (const auto tile : sourceTiles) {
      const auto dataIdx = ipIn.dataElement(tile, lower);
      toConcat.emplace_back(
          ipIn.data(tile).slice(dataIdx, dataIdx + regionSize).expand({0}));
    }
    partialsMap.push(lower, concat(toConcat));
  }

  // Apply optimisations to the partials map.
  mergeOutputRegionsThatSplitGrainSize(partialsMap, target, resultType,
                                       grainSize);

  // If each piece is really small the overhead of having extra reduction
  // stages, and exchange and everything outweighs the savings.
  //
  // Optimisation: reductionFactorThresholdToAddMoreStages was found
  // empirically and not tested a lot.

  auto splits = calculateSplit(
      getOutputPartialsDescs(partialsMap), grainSize, grainSize, 2,
      reductionFactorThresholdToAddMoreStages, target.getNumTiles());

  // A pair of output offset & interval of the partials depth.
  // The output offset will exactly match an entry in the splits map.
  struct TileReduction {
    boost::icl::interval<std::size_t>::type outputInterval;
    poplar::Interval partialInterval;
  };
  using TileReductions = std::vector<TileReduction>;

  DebugRange<std::size_t> debugNumPartials = {UINT_MAX, 0};
  DebugRange<unsigned> debugNclip = {UINT_MAX, 0};
  DebugRange<std::size_t> debugPartialsWidths = {UINT_MAX, 0};

  std::vector<TileReductions> tileReductions(target.getNumTiles());
  unsigned debugTiles = 0;
  {
    unsigned tile = startTile;
    auto partialIt = partialsMap.begin();
    for (auto it = splits.begin(); it != splits.end(); ++it) {
      const auto &[outputInterval, nSplits] = *it;
      if (outputInterval.lower() >=
          partialIt->outputOffset() + partialIt->width()) {
        ++partialIt;
      }
      const auto numPartials = partialIt->depth();

      debugNumPartials.update(numPartials);
      debugPartialsWidths.update(partialIt->width());

      const auto reductionFactor = std::max(2u, udiv(numPartials, nSplits));
      // Number of reductions for this output region.
      const auto numPartialReductions = udiv(numPartials, reductionFactor);
      for (std::size_t i = 0; i != numPartialReductions; ++i) {
        const auto begin = i * reductionFactor;
        const auto end = std::min((i + 1) * reductionFactor, numPartials);
        debugNclip.update(end - begin);
        ++debugTiles;
        tileReductions[tile].emplace_back(
            TileReduction{outputInterval, poplar::Interval{begin, end}});
        tile = (tile + 1) % target.getNumTiles();
      }
    }
  }

  logging::popops::debug(debugNumPartials.min == debugNumPartials.max
                             ? "  Remaining reduction of {} partials"
                             : "  Remaining reduction of {} to {} partials",
                         debugNumPartials.min, debugNumPartials.max);
  logging::popops::debug(
      debugNclip.min == debugNclip.max
          ? "  This stage uses {} tiles, which all reduce {} partials"
          : "  This stage uses {} tiles reducing between {} and {} partials",
      debugTiles, debugNclip.min, debugNclip.max);
  logging::popops::debug(debugPartialsWidths.min == debugPartialsWidths.max
                             ? "  Partial width {}"
                             : " With widths between {} and {}",
                         debugPartialsWidths.min, debugPartialsWidths.max);

  // If we intend to cast partials before exchange then produce a tensor
  // per tile, which is already cast to mirror the partials found on that
  // tile.
  std::map<unsigned, Tensor> castIpIn;
  if (castComputeSet) {
    auto v = partialsMap.extract();
    for (auto &entry : v) {
      entry.set(
          cast(graph, entry.extract(), poplar::FLOAT, castComputeSet.get()));
    }
    partialsMap.set(std::move(v));
  }

  unsigned debugUsedTiles = 0;
  std::size_t csPos = css.pos();
  for (unsigned tile = 0; tile < tileReductions.size(); ++tile) {
    auto &tr = tileReductions[tile];
    if (tr.empty()) {
      continue;
    }
    ++debugUsedTiles;
    logging::popops::trace("Tile {} reductions:", tile);

    boost::icl::interval_set<std::size_t> outputRegionsMergedIcl;
    for (const auto &entry : tr) {
      outputRegionsMergedIcl += entry.outputInterval;
    }

    // Add a variable to receive the results.
    Tensor data = graph.addVariable(
        resultType, {boost::icl::length(outputRegionsMergedIcl)},
        {dnai, "tile_data2"});
    storeReductionResultTensors(reductionResultTensors, data);

    graph.setTileMapping(data, tile);

    // Add it to the output.
    ir.setTensor(tile, data, outputRegionsMergedIcl);

    // Store the tensors that we will connect up.
    std::vector<RegionReduction> reductions;
    reductions.reserve(tr.size());

    // For each of the regions.
    for (const auto [outputInterval, partialInterval] : tr) {
      const auto outputRegionSize = boost::icl::length(outputInterval);
      const auto &partials =
          partialsMap.getPartialsAtOutputOffset(outputInterval.lower());

      // The corresponding region in the data
      RegionReduction rt;

      // Loop through the partials for this region...
      IrregularPartials iPartials;
      iPartials.data.reserve(partialInterval.size());

      // Calculate the current output interval relative to the start of the
      // entry in the partials map so we can retrieve the correct partials
      // tensor.
      for (auto i = partialInterval.begin(); i != partialInterval.end(); ++i) {
        iPartials.data.emplace_back(partials.get(i, outputInterval));
      }
      rt.partials = iPartials;
      logging::popops::trace("  Partials:{} Width:{} Output region:{})",
                             partialInterval.size(), outputRegionSize,
                             outputInterval);

      // Connect the output region.
      const auto outputDataIdx = ir.dataElement(tile, outputInterval.lower());
      rt.output =
          ir.data(tile).slice(outputDataIdx, outputDataIdx + outputRegionSize);
      reductions.push_back(rt);
    }

    // Start from our current position in the compute set list.
    ComputeSetList cssFork = css;

    connectReductions(graph, cssFork, op, inType, inType, resultType, tile,
                      reductions, false, {dnai, "IntermediateToIntermediate"});
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos)
      csPos = cssFork.pos();
  }
  css.setPos(csPos);
  return ir;
}

void intermediateToOutput(Graph &graph, const IntermediatePartials &ipIn,
                          boost::optional<Tensor> &finalOutput,
                          boost::optional<Tensor> &originalOutput,
                          const std::vector<std::size_t> outputShape,
                          Type outputType, ReduceParams params,
                          Type inVertexType, ComputeSetList &css,
                          ResultTensors &reductionResultTensors,
                          const Tensor &in, unsigned reductionId,
                          const DebugNameAndId &dnai) {
  logging::popops::debug("DebugStr: {}", dnai.getPathName());

  const auto numOutElements = in.dim(1);
  bool mappingComplete = false;
  Graph::TileToTensorMapping mapping;
  if (finalOutput) {
    mapping = graph.getTileMapping(finalOutput.get(), &mappingComplete);
  }
  Tensor out;
  if (mappingComplete) {
    // If we have an output then use that as the output from the reduction
    out = finalOutput.get().flatten();
  } else {
    // Otherwise create the output here
    out = graph.addVariable(outputType, {numOutElements}, {dnai});
    graph.setTileMapping(out, graph.getTileMapping(in.slice(0, 1, 0), false));
  }
  if (!params.update) {
    storeReductionResultTensors(reductionResultTensors,
                                originalOutput ? originalOutput.get().flatten()
                                               : out);
  }
  // If the data type is half AND
  // If any tile contains a single output, or any tile contains a single
  // partial then we will be be exchanging single partials and getting
  // inefficient reductions.  This can be avoided by casting before exchange.
  //
  // TODO: temporarily only for ADD, SQUARE ADD as if applied to other types
  //       we produce a mix of partials types.  This can be dealt with when
  //       D20584 lands, but implies that a final cast stage would be needed
  //       for types other than ADD, SQUARE_ADD
  boost::optional<ComputeSet> castComputeSet;

  if ((params.op == popops::Operation::ADD ||
       params.op == popops::Operation::SQUARE_ADD) &&
      ipIn.dataType() == poplar::HALF) {
    auto singleOutputOnAnyTile = std::any_of(
        mapping.begin(), mapping.end(),
        [](const std::vector<Interval> &tileOutputs) {
          return tileOutputs.size() == 1 && tileOutputs[0].size() == 1;
        });
    if (singleOutputOnAnyTile || reductionBenefitsFromPreExchangeCast(ipIn)) {
      logging::popops::debug("Inserting pre-exchange cast half to float");
      castComputeSet = css.add(graph, {dnai, "PreExchangeCast"});
      inVertexType = poplar::FLOAT;
    }
  }

  // This is assumed below.
  assert(out.rank() == 1);

  const auto inType = castComputeSet ? poplar::FLOAT : ipIn.dataType();

  // If the output isn't already mapped, map it linearly and do the reduction
  // there, otherwise decide whether it is better to do the reduction at the
  // destination or not.  Choose grainSize and minElementsPerTile based on the
  // reduction input type not the final output, and to influence the number of
  // vertices per tile.
  // Choosing the more correct sounding numWorkers == 6 can result in 7
  // vertices being generated as calcLinearTileMapping tries to balance the
  // data over a minimal number of tiles, without a maximum amount of data per
  // tile being enforced.
  const auto target = graph.getTarget();
  const unsigned minElementsPerTile = 4 * target.getVectorWidth(inVertexType);
  if (mappingComplete) {
    if (!shouldReduceAtDestination(target, ipIn, mapping, inVertexType,
                                   out.numElements())) {
      std::size_t seed = 0x9e3779b9UL;
      boost::hash_combine(seed, reductionId);
      auto offset =
          chooseMappingOffset(target.getNumTiles(), out.shape(), seed);
      mapping = poputil::calcLinearTileMapping(
          graph, out.shape(), minElementsPerTile,
          target.getVectorWidth(inVertexType), offset);

      logging::popops::debug("Reducing using a temporary result, offset:{}",
                             offset);
    }
  } else {
    logging::popops::debug("Mapping output");
    mapping =
        poputil::calcLinearTileMapping(graph, out.shape(), minElementsPerTile,
                                       target.getVectorWidth(inVertexType));
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
  for_each_zipped_region(
      mappingIcl.begin(), mappingIcl.end(), tilesForOutput.begin(),
      tilesForOutput.end(),
      [&](std::size_t begin, std::size_t end, unsigned mappedToTile,
          const boost::container::flat_set<unsigned> &partialTiles) {
        tilesForOutputPerTile[mappedToTile].set(std::make_pair(
            boost::icl::interval<std::size_t>::right_open(begin, end),
            partialTiles));
      });

  std::size_t csPos = css.pos();

  unsigned debugTiles = 0;
  if (logging::popops::shouldLog(logging::Level::Debug)) {
    for (unsigned tile = 0; tile < mapping.size(); ++tile) {
      if (!mapping[tile].empty()) {
        debugTiles++;
      }
    }
    logging::popops::debug("  Using {} tiles", debugTiles);
  }
  DebugRange<std::size_t> debugNumPartials = {UINT_MAX, 0};
  DebugRange<std::size_t> debugPartialsWidths = {UINT_MAX, 0};

  // If we intend to cast partials before exchange then produce a tensor
  // per tile, which is already cast to mirror the partials found on that tile.
  std::map<unsigned, Tensor> castIpIn;
  if (castComputeSet) {
    auto partialsTiles = ipIn.tiles();
    for (const auto tile : partialsTiles) {
      auto floatPartials =
          cast(graph, ipIn.data(tile), poplar::FLOAT, castComputeSet.get());
      graph.setTileMapping(floatPartials, tile);
      castIpIn[tile] = floatPartials;
    }
  }
  // Partition tilesForOutput based on mappingIcl.
  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    if (mapping[tile].empty()) {
      continue;
    }
    logging::popops::trace("Tile {} reductions:", tile);

    // Get the regions that are mapped to this tile.
    auto outputRegionsSplitIcl = poplarToSplitIntervalSet(mapping[tile]);

    // Take the subset of the map from output element to partial tiles
    // for the output regions that are mapped to this tile.
    const auto &thisTilesForOutput = tilesForOutputPerTile[tile];

    // Convert the output element indices to poplar interval format.

    std::vector<Interval> outputRegionsSplit;
    outputRegionsSplit.reserve(thisTilesForOutput.size());

    for (const auto &ival : thisTilesForOutput) {
      outputRegionsSplit.emplace_back(ival.first.lower(), ival.first.upper());
    }

    // Determine the grainSize based on partials type, we have no easy way to
    // compare partials widths so don't assume anything about them
    const unsigned grainSize =
        findGrainSizeForOp(graph, inVertexType, params.op);
    // Split them if it would make it faster by processing them separately
    // with different vertices.
    outputRegionsSplit = splitOutputRegionsForWorkers(
        grainSize, graph.getTarget().getNumWorkerContexts(), inVertexType,
        outputRegionsSplit);

    // Store the tensors that we will connect up. Have to do this
    // here so we can resize the Vectors in the vertex.
    std::vector<RegionReduction> reductions;

    reductions.reserve(outputRegionsSplit.size());

    // Finally we repeat the above but this time record the actual connections.

    // Store all the partials here and at the end we can decide if we want to
    // make a single RegularPartials Tensor or IrregularPartials vector of
    // Tensors to describe them.
    std::vector<std::vector<poplar::Tensor>> rtPartials;

    rtPartials.resize(outputRegionsSplit.size());

    std::size_t idx = 0;
    for (const auto &re : outputRegionsSplit) {
      RegionReduction rt;
      auto &rtPartial = rtPartials[idx++];

      // Connect the output. This is fine because output is 1D.
      rt.output = out.slice(re);

      // Get the list of partials to use.
      auto partialTiles = thisTilesForOutput(re.begin());
      debugNumPartials = {std::min(debugNumPartials.min, partialTiles.size()),
                          std::max(debugNumPartials.max, partialTiles.size())};

      rtPartial.reserve(partialTiles.size());
      for (auto partialTile : partialTiles) {
        size_t sourceDataIdx = ipIn.dataElement(partialTile, re.begin());
        size_t len = re.size();

        assert(ipIn.dataElement(partialTile, re.begin() + len - 1) ==
               sourceDataIdx + len - 1);

        if (castComputeSet) {
          rtPartial.emplace_back(
              castIpIn[partialTile].slice(sourceDataIdx, sourceDataIdx + len));
        } else {
          rtPartial.emplace_back(
              ipIn.data(partialTile).slice(sourceDataIdx, sourceDataIdx + len));
        }
      }
      // As the partials are all the same size we can do this to create a
      // valid outerFactor.
      rt.outerFactor = rtPartial.back().numElements() * rtPartial.size() /
                       rt.output.numElements();
      logging::popops::trace(
          "  Partials:{} Width:{} Output data index:[{}, {})",
          partialTiles.size(), rtPartial.back().numElements(), re.begin(),
          re.end());
      debugPartialsWidths = {
          std::min(debugPartialsWidths.min, rtPartial.back().numElements()),
          std::max(debugPartialsWidths.max, rtPartial.back().numElements())};

      reductions.push_back(std::move(rt));
    }

    // Only now we have gathered all the reductions for this tile, and all the
    // partials, we can build an allPartials tensor and add striding information
    // which should allow for better work splitting using strided reduce with
    // no copies being introduced. This only makes sense if the lots of the
    // reductions have partials that can be arranged in a regular way to form
    // one region.  The advantage of doing this is that it allows the partials
    // regions to be split/joined later which helps with work division.
    // We choose to only do it if all the regions can be formed into one
    // contiguous regular block.
    const bool regularReductions =
        std::all_of(rtPartials.begin(), rtPartials.end(),
                    [=](std::vector<poplar::Tensor> &partials) {
                      return partials.size() == rtPartials[0].size();
                    });
    // Given that we use the map of source tiles to determine the reductions to
    // implement we may find that some/all of the reductions have width 1 (and
    // are therefore candidates for continuous reduce) or another non-grainsize
    // width (And so will fall back to a slow default vertex).  Including these
    // in a single regular partials tensor is a bad idea as they would lie in a
    // column/columns - so not contiguous in memory. If any of these are
    // detected, fall back on irregular partials.

    const bool reductionsWithNonGrainsizeWidth =
        std::any_of(rtPartials.begin(), rtPartials.end(),
                    [=](std::vector<poplar::Tensor> &partials) {
                      return partials[0].numElements() % grainSize;
                    });

    if (regularReductions && !reductionsWithNonGrainsizeWidth) {
      std::vector<std::vector<Tensor>> partialsByRow(rtPartials[0].size());
      for (unsigned i = 0; i < reductions.size(); i++) {
        for (unsigned j = 0; j < rtPartials[i].size(); j++) {
          partialsByRow[j].push_back(rtPartials[i][j]);
        }
      }
      std::vector<Tensor> gatheredPartials(partialsByRow.size());
      for (unsigned i = 0; i < partialsByRow.size(); i++) {
        gatheredPartials[i] = concat(partialsByRow[i]);
      }
      auto allPartials = concat(gatheredPartials);

      unsigned cumulativeColumn = 0;
      const auto stride =
          std::accumulate(reductions.begin(), reductions.end(), 0u,
                          [](unsigned total, RegionReduction &r) {
                            return total + r.output.numElements();
                          });

      for (unsigned i = 0; i < reductions.size(); i++) {
        reductions[i].getPartials().push_back(allPartials);
        reductions[i].getStride() = stride;
        reductions[i].getOffset() = cumulativeColumn;
        cumulativeColumn += reductions[i].output.numElements();
      }
    } else {
      for (unsigned i = 0; i < reductions.size(); i++) {
        IrregularPartials iPartials({rtPartials[i]});
        reductions[i].partials = iPartials;
      }
    }
    // Start from our current position in the compute set list.
    ComputeSetList cssFork = css;
    connectReductions(graph, cssFork, params, inType, inVertexType, outputType,
                      tile, reductions, false, {dnai, "IntermediateToOutput"});
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos)
      csPos = cssFork.pos();
  }
  logging::popops::debug(debugNumPartials.min == debugNumPartials.max
                             ? "  Tiles all reduce {} partials"
                             : "  Tiles have between {} and {} partials",
                         debugNumPartials.min, debugNumPartials.max);
  logging::popops::debug(debugPartialsWidths.min == debugPartialsWidths.max
                             ? "  Partial width {}"
                             : " With widths between {} and {}",
                         debugPartialsWidths.min, debugPartialsWidths.max);

  css.setPos(csPos);

  if (!finalOutput) {
    finalOutput = out;
  }
  finalOutput = finalOutput.get().reshape(outputShape);
}

} // namespace popops
