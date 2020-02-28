// Copyright (c) Graphcore Ltd, All rights reserved.
#include "ReductionStages.hpp"
#include "ReductionPlan.hpp"

#include <algorithm>
#include <cassert>

#include <fstream>
#include <numeric>

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

// createElementRefsFromRegions: List a reference containing (region, offset)
// for each tensor element found in the intervals for each region.  These are
// arranged into a vector per column - ie output element to reduce into.
struct ElementRef {
  unsigned region;
  unsigned offset;
};
std::vector<std::vector<ElementRef>> createElementRefsFromRegions(
    const std::vector<std::vector<Interval>> &regions,
    std::vector<PartialsDescription> &partialsDescription,
    const unsigned columns, bool detectColumns) {
  std::vector<std::vector<ElementRef>> elementRefs(columns);

  for (unsigned r = 0; r < regions.size(); r++) {
    unsigned regionStartOffset = 0;
    for (unsigned i = 0; i < regions[r].size(); i++) {
      for (unsigned e = 0; e < regions[r][i].size(); e++) {
        // Examine the column number of every element on tile.  Append it to the
        // vector of elements for that column
        const unsigned column = (regions[r][i].begin() + e) % columns;
        elementRefs[column].push_back({r, regionStartOffset});
        regionStartOffset++;
      }
    }
  }
  // Largely to make test cases simple to understand, we may only be interested
  // in certain columns.  Omit those that are not of interest.
  if (!detectColumns) {
    for (unsigned i = 0; i < columns; i++) {
      bool useColumn = false;
      for (unsigned j = 0; j < partialsDescription.size(); j++) {
        if (partialsDescription[j].columns[0] == i) {
          useColumn = true;
        }
      }
      if (!useColumn) {
        elementRefs[i].resize(0);
      }
    }
  }
  return elementRefs;
}

// updatePartialsDescription: Given a "signal" indicating that the column
// of interest is/is not detected in the region, update the partialsDescription
// structure.

struct PatternBuildState {
  bool patternColumnEnd;
  unsigned patternColumnRef;
  bool buildingPattern = false;
};

void updatePartialsDescription(PatternBuildState &pbs, PartialsDescription &rt,
                               bool thisColumnFound, unsigned currentRegion,
                               unsigned priorRegion, unsigned elementOffset,
                               bool isRegionEnd) {

  if (thisColumnFound && !pbs.buildingPattern) {
    // The first pattern in this region
    pbs.patternColumnRef = elementOffset;
    rt.patterns.push_back({0, pbs.patternColumnRef, 0, 0, currentRegion});
    pbs.patternColumnEnd = false;
    pbs.buildingPattern = true;
  }
  if (pbs.buildingPattern) {
    auto length = elementOffset - pbs.patternColumnRef;
    if (!pbs.patternColumnEnd && !thisColumnFound) {
      // Like a falling edge of the signal
      // "column == this reduction column"
      // Means the innerFactor can be created or checked
      if (rt.patterns.back().innerFactor) {
        if (rt.patterns.back().innerFactor != length) {
          // A new pattern as the "column == this reduction column"
          // signal was too long compared to the current pattern
          // Begin a fresh pattern as if the signal pulse was all part
          // of it
          // OR A new pattern as the signal was too short
          if (isRegionEnd) {
            rt.patterns.push_back(
                {length, pbs.patternColumnRef, 0, 0, priorRegion});
          } else {
            rt.patterns.push_back(
                {length, pbs.patternColumnRef, 0, 0, currentRegion});
          }
        }
      } else {
        // Initialise the innerFactor of a new pattern
        rt.patterns.back().innerFactor = length;
      }
      pbs.patternColumnEnd = true;
      rt.patterns.back().outerFactor++;
    }
    if (thisColumnFound && pbs.patternColumnEnd) {
      // Like a rising edge of the signal
      // "column == this reduction column"
      // Means the stride can be created or checked
      pbs.patternColumnEnd = false;
      if (rt.patterns.back().stride) {
        if (rt.patterns.back().stride != length) {
          // The stride is inconsistent with the current pattern so
          // start a new pattern
          rt.patterns.push_back({0, elementOffset, 0, 0, currentRegion});
          pbs.buildingPattern = true;
        }
      } else {
        rt.patterns.back().stride = length;
      }
      pbs.patternColumnRef = elementOffset;
      // Update length to assist with end of region condition
      length = 0;
    }
    if (isRegionEnd) {
      if (pbs.buildingPattern && !pbs.patternColumnEnd) {
        if (rt.patterns.back().innerFactor) {
          if (rt.patterns.back().innerFactor == length + 1) {
            // Region ends nicely truncating the pattern at the
            // point of a "column == this reduction column" signal
            // "falling edge"
            rt.patterns.back().outerFactor++;
          } else {
            // Truncated early - add a fresh pattern to describe it
            rt.patterns.push_back(
                {length + 1, pbs.patternColumnRef, 0, 1, currentRegion});
          }
        }
        if (rt.patterns.back().innerFactor == 0) {
          // Pattern innerFactor not yet been found:
          // "column == this reduction column" signal was = 1 throughout
          // the region or for a last separate pattern
          rt.patterns.back().innerFactor = length + 1;
          rt.patterns.back().outerFactor = 1;
        }
      }
      // Fresh region will begin if there is one
      pbs.buildingPattern = false;
    }
  }
}

unsigned
initialisePatternStructs(PatternBuildState &patternBuildState,
                         std::vector<PartialsDescription> &partialsDescription,
                         const std::vector<ElementRef> &elementRefs,
                         bool detectColumns, const unsigned column) {
  // Is this the end of the region, if so complete the pattern accordingly
  bool regionEnd = false;
  if (elementRefs.size() == 1) {
    regionEnd = true;
  } else if (elementRefs[0].region != elementRefs[1].region) {
    regionEnd = true;
  }
  const unsigned lastOne = regionEnd ? 1u : 0u;
  // Create a pattern and complete a struct to look after updating it
  unsigned preDetIndex;
  if (detectColumns) {
    partialsDescription.push_back({{column},
                                   {{lastOne, elementRefs[0].offset, 0, lastOne,
                                     elementRefs[0].region}}});
  } else {
    for (preDetIndex = 0; preDetIndex < partialsDescription.size();
         preDetIndex++) {
      if (partialsDescription[preDetIndex].columns[0] == column) {
        break;
      }
    }
    partialsDescription[preDetIndex].patterns.push_back(
        {lastOne, elementRefs[0].offset, 0, lastOne, elementRefs[0].region});
  }

  patternBuildState = {false, elementRefs[0].offset, !regionEnd};
  return detectColumns ? partialsDescription.size() - 1 : preDetIndex;
}

// Reduction patterns describe the part of a contiguous region of data that is
// required by a given reduction.  See the definition of PartialsPattern and
// PartialsDescription for an explanation.
//
// In the description below we talk about a "signal" where "column == this
// reduction column".  In other words 1 = signal true, 0 = signal false in
// the examples.
//
//  Examples:
//  00111001110011100 1 pattern :
//    innerFactor=3, start=2, stride=5, outerFactor=3, region=0
//
//  011100111010100 2 patterns :
//     innerFactor=3, start=1, stride=5, outerFactor=2, region=0
//     innerFactor=1, start=10, stride=2, outerFactor=2, region=0
//
//
// gatherReductionPatterns will scan the regions on tile and determine what data
// is required to reduce each column. It will create a PartialsDescription
// containing as many patterns as are required to describe that columns's data.
//
// If the partialsDescription vector is empty on entry it will automatically
// determine what columns have data on tile, otherwise it will look to the
// 'columns' entry within the partialsDescription and create patterns for those
// columns only. Either way each PartialsDescription will describe all
// of the elements for a particular column in the given regions.
//
// Note: the purpose of only finding selected column's data is for test, as the
//       results are clearer.

void gatherReductionPatterns(
    std::vector<PartialsDescription> &partialsDescription,
    const std::vector<std::vector<Interval>> &regions, unsigned columns) {

  // First list all references to each column in a vector of vectors: One outer
  // vector per column (ie output element from the reduction)
  const bool detectColumns = (partialsDescription.size() == 0);
  auto elementRefs = createElementRefsFromRegions(regions, partialsDescription,
                                                  columns, detectColumns);

  // Looking at each vector in turn, build a pattern.
  for (unsigned i = 0; i < elementRefs.size(); i++) {
    // elements belonging to this column were detected on tile
    if (elementRefs[i].size()) {

      PatternBuildState patternBuildState;
      // Create a pattern structure to deal with this, and return a reference
      // to it.  Initialise the pattern build state.
      auto currentPatternIdx =
          initialisePatternStructs(patternBuildState, partialsDescription,
                                   elementRefs[i], detectColumns, i);
      PartialsDescription &currentPattern =
          partialsDescription[currentPatternIdx];

      // Add the rest of the elements belonging to this column to the pattern
      for (unsigned j = 1; j < elementRefs[i].size(); j++) {

        const bool isNewRegion =
            elementRefs[i][j].region != elementRefs[i][j - 1].region;

        const bool nonColumnElementsExist =
            isNewRegion ||
            elementRefs[i][j].offset != elementRefs[i][j - 1].offset + 1;
        // Update the pattern for the presence of memory that isn't in its
        // column
        if (nonColumnElementsExist) {
          // Mark the end of the "column detected" signal with a single
          // element where column detect = false.  This could be because a
          // new region was found - in which case it updates due to the gap
          // between regions
          updatePartialsDescription(
              patternBuildState, currentPattern, false,
              elementRefs[i][j].region, elementRefs[i][j - 1].region,
              elementRefs[i][j - 1].offset + 1, isNewRegion);
          if (!isNewRegion) {
            // If that didn't happen due to a region Change, then update the
            // pattern with the information that there were potentially many
            // elements with a "column detected" signal = 0
            updatePartialsDescription(patternBuildState, currentPattern, false,
                                      elementRefs[i][j].region,
                                      elementRefs[i][j - 1].region,
                                      elementRefs[i][j].offset - 1, false);
          }
        }
        // Update the pattern for its own column, taking note of special case of
        // the end of the data on tile for this column
        const bool isLastElement = j == elementRefs[i].size() - 1;
        updatePartialsDescription(patternBuildState, currentPattern, true,
                                  elementRefs[i][j].region,
                                  elementRefs[i][j - 1].region,
                                  elementRefs[i][j].offset, isLastElement);
      }
    }
  }
}

// Cleaner function for use below, which returns a PartialsDescription vector
// and therefore will always automatically determine all columns referenced in
// the  "regions".  The function above is mostly useful for test.
std::vector<PartialsDescription>
gatherReductionPatterns(const std::vector<std::vector<Interval>> &regions,
                        unsigned columns) {
  std::vector<PartialsDescription> partialsDescription;
  gatherReductionPatterns(partialsDescription, regions, columns);
  return partialsDescription;
}

void addPartialDebug(const PartialsDescription &partialsDescription,
                     RegionReduction &reduction, unsigned tile, unsigned start,
                     unsigned end, unsigned columns) {
  ReductionDebug::Partial di;
  di.sourceCols = {partialsDescription.columns[0],
                   partialsDescription.columns[0] +
                       partialsDescription.columns.size()};
  di.sourceRows = {start / columns, end / columns};
  di.sourceTile = tile;
  reduction.partialsDebugInfo.push_back(di);
}

// A function which accepts a vector of patterns which each describe
// a reduction of one or more columns. Each pattern references a region /
// regions and describes a number of tensor elements (partials) found within
// that region.
// The function adds references to the partials for each reduction into the
// "reductions" structure.
std::vector<RegionReduction> listPartialsUsingPatterns(
    const std::vector<PartialsDescription> &partialsDescription,
    const Tensor &input, const std::vector<std::vector<Interval>> &inputRegions,
    unsigned tile, unsigned columns) {
  // For speed, prepare a vector of tensors for each on tile region, each of
  // which will be referenced many times in the loop below.
  std::vector<Tensor> regionTensors(inputRegions.size());
  for (unsigned i = 0; i < inputRegions.size(); i++) {
    regionTensors[i] = concat(input.flatten().slices(inputRegions[i]));
  }

  std::vector<RegionReduction> reductions(partialsDescription.size());
  for (unsigned i = 0; i < reductions.size(); i++) {
    for (auto &pat : partialsDescription[i].patterns) {
      auto &partials = reductions[i].partials;
      reductions[i].innerFactor = pat.innerFactor;
      reductions[i].outerFactor = pat.outerFactor;
      auto &in = regionTensors[pat.regionIdx];
      if (pat.outerFactor > 1) {
        if (pat.stride == partialsDescription[i].columns.size() &&
            pat.innerFactor == 1) {
          // If the sequence of columns repeats end to end with no gap in
          // memory we can create partials with a single slice.
          // (Note that this expression could be simplified as stride == no of
          // columns.  However the expression below is clearer)
          const auto end = pat.regionOffset +
                           pat.stride * (pat.outerFactor - 1) +
                           partialsDescription[i].columns.size();
          partials.push_back(in.slice(pat.regionOffset, end));
          addPartialDebug(partialsDescription[i], reductions[i], tile,
                          pat.regionOffset, end, columns);
        } else {
          // If the patterns repeats and has "gaps"
          // (i.e. stride != no of columns) we need multiple slices to create
          // the partials.
          for (unsigned k = 0; k < pat.outerFactor; k++) {
            const auto start = pat.regionOffset + k * pat.stride;
            const auto end =
                pat.regionOffset + k * pat.stride +
                pat.innerFactor * partialsDescription[i].columns.size();
            partials.push_back(in.slice(start, end));
            addPartialDebug(partialsDescription[i], reductions[i], tile, start,
                            end, columns);
          }
        }
      } else {
        // If there are no pattern repetitions we can create partials with a
        // single silce.
        const auto end =
            pat.regionOffset +
            pat.innerFactor * partialsDescription[i].columns.size();
        partials.push_back(in.slice(pat.regionOffset, end));
        addPartialDebug(partialsDescription[i], reductions[i], tile,
                        pat.regionOffset, end, columns);
      }
    }
  }
  return reductions;
}
// Function defining the criteria for two patterns to be adjacent - that is,
// they can be grouped together.  The two patterns need to be next to each other
// memory consistently each time the pattern repeats, and in every region the
// pattern appears in.  The actual column number is not important, so we can
// end up with a grouping of patterns from columns 3, 4, 6, 7 which lie
// sequentially in memory but are not numbered sequentially.
// We are always keeping complete columns together, never grouping parts of
// columns, even over separate regions.
bool isAdjacent(const PartialsDescription &a, const PartialsDescription &b,
                unsigned columns) {
  if (a.patterns.size() != b.patterns.size()) {
    return false;
  }
  for (unsigned i = 0; i < a.patterns.size(); i++) {
    if (a.patterns[i].regionOffset + a.patterns[i].innerFactor !=
            b.patterns[i].regionOffset ||
        a.patterns[i].innerFactor != b.patterns[i].innerFactor ||
        a.patterns[i].stride != b.patterns[i].stride ||
        a.patterns[i].outerFactor != b.patterns[i].outerFactor ||
        a.patterns[i].regionIdx != b.patterns[i].regionIdx) {
      return false;
    }
  }
  return true;
}
// Group partials operates on PartialsDescriptions, each of which contains
// information about the layout of a single column's data on tile.  It attempts
// to group any structures that describe columns which are "adjacent" - ie
// next to each other in memory and of consistent shape.  The "isAdjacent"
// function defines this.
std::vector<PartialsDescription>
groupPartials(std::vector<PartialsDescription> &partialsDescription,
              unsigned columns) {
  std::vector<PartialsDescription> groupedPartials;
  // Keep track of which patterns have been added to grouped patterns
  std::vector<bool> partialsDescriptionIsGrouped(partialsDescription.size(),
                                                 false);
  unsigned partialsDescriptionsToGroup = partialsDescription.size();
  for (unsigned i = 0;
       i < partialsDescription.size() && partialsDescriptionsToGroup; i++) {
    // If the next one hasn't been grouped already, put it into the
    // groupedPartials structure.
    if (partialsDescriptionIsGrouped[i] == false) {
      groupedPartials.push_back(partialsDescription[i]);
      groupedPartials.back().columns.resize(1);
      partialsDescriptionIsGrouped[i] = true;
      partialsDescriptionsToGroup--;

      // Scan the remaining ones for adjacent, matching patterns, append their
      // column to the column list and mark them as grouped
      for (unsigned j = i + 1; j < partialsDescription.size(); j++) {
        if (partialsDescriptionIsGrouped[j] == false) {
          if (isAdjacent(partialsDescription[i], partialsDescription[j],
                         columns)) {
            groupedPartials.back().columns.push_back(
                partialsDescription[j].columns[0]);
            partialsDescriptionIsGrouped[j] = true;
            partialsDescriptionsToGroup--;
            // Update offsets into the patterns so that we can continue to group
            // Overwrites the structure, but it's not needed anymore
            for (unsigned k = 0; k < partialsDescription[i].patterns.size();
                 k++) {
              partialsDescription[i].patterns[k].regionOffset +=
                  partialsDescription[i].patterns[k].innerFactor;
            }
          }
        }
      }
    }
  }
  return groupedPartials;
}

// Determine grainSize.  Logically this would be per reduction, we would want
// to split with a different grainSize based on the operation and partials
// widths.  However we catch many cases by checking that all reductions have
// appropriate partials widths.  This means we can pass one grainSize
// parameter to splitOutputRegionsForWorkers which is far simpler than a per
// reduction array.
unsigned
findGrainSizeForOp(Graph &graph, Type partialType, ReduceParams &params,
                   const boost::optional<std::vector<Interval>> &regions) {

  const unsigned grainSize = graph.getTarget().getVectorWidth(partialType);
  if (params.op == popops::Operation::ADD ||
      params.op == popops::Operation::SQUARE_ADD) {
    // NOTE - depending on the eventual vertex targeted we could split using
    // grainSize instead of grainSize*2 here with no speed decrease.
    return grainSize * 2;
  }
  if (regions && (params.op == popops::Operation::MAX ||
                  params.op == popops::Operation::MIN)) {
    // If partials are all equal size and a multiple of grainsize*2 then we can
    // benefit by using the partialsEqualSize vertex providing we don't split
    // them below grainSize*2.
    bool partialsEqualSize = std::all_of(
        regions.get().begin(), regions.get().end(),
        [=](Interval r) { return (r.size() % (grainSize * 2)) == 0; });
    if (partialsEqualSize) {
      return grainSize * 2;
    }
  }
  // Other reductions (including MAX and MIN) can generally be split between
  // workers with the basic grain size
  return grainSize;
}

// dividePartials: Accepts a number of groupedPartials structures, each of which
// can contain pattern layout information about a number of columns to be
// reduced.  These are divided up into smaller groups of columns so that:
// a) There are no multi column groups of patterns where the innerFactor
//    parameter is not the same for all patterns
// b) To divide work between available workers

std::vector<PartialsDescription>
dividePartials(std::vector<PartialsDescription> &groupedPartials, Graph &graph,
               Type inType, ReduceParams params) {

  std::vector<PartialsDescription> splitGroupedPartials;
  // Split up patterns that have > 1 column and an inconsistent innerFactor, as
  // these don't fit the description of a reduction
  for (unsigned i = 0; i < groupedPartials.size(); i++) {
    // Check the characteristics of each pattern within the group of partials
    bool patternsAreSimple = true;
    if (groupedPartials[i].columns.size() != 1) {
      for (unsigned j = 0; j < groupedPartials[i].patterns.size(); j++) {
        if (groupedPartials[i].patterns[j].innerFactor !=
            groupedPartials[i].patterns[0].innerFactor) {
          // If the innerFactor of different patterns is inconsistent, split
          // this up.
          patternsAreSimple = false;
          break;
        }
      }
    }
    // Copy or split up patterns accordingly
    if (patternsAreSimple) {
      splitGroupedPartials.push_back(groupedPartials[i]);
    } else {
      // Split all the patterns so that we have a pattern per column,
      // maintaining the innerFactor
      splitGroupedPartials.reserve(groupedPartials[i].columns.size());
      for (unsigned j = 0; j < groupedPartials[i].columns.size(); j++) {
        // The split partials have the same patterns but only one column
        splitGroupedPartials.emplace_back();
        splitGroupedPartials.back().patterns = groupedPartials[i].patterns;
        splitGroupedPartials.back().columns.push_back(
            groupedPartials[i].columns[j]);

        // Adjust the start of the new patterns to match the new starting
        // column
        for (unsigned k = 0; k < groupedPartials[i].patterns.size(); k++) {
          splitGroupedPartials.back().patterns[k].regionOffset =
              groupedPartials[i].patterns[k].regionOffset +
              j * groupedPartials[i].patterns[k].innerFactor;
        }
      }
    }
  }

  // Split up patterns to divide work between workers by column.  Later on
  // reductions can be split by row as well/instead. Both have a potential
  // downside: Splitting by row requires a second reduction stage. Splitting
  // by column could introduce copies.
  //
  // The method here is based on splitting output regions, which we
  // temporarily create just for splitting of work purposes.  Each output
  // region relates to the size of a group of columns represented by a pattern.
  // It also has a unique interval, based on accumulating the number of
  // columns over all the columns found on this tile.

  // Final result from this function.
  std::vector<PartialsDescription> partialsResult;

  std::vector<Interval> outRegions;
  // Index the start of a region based on a cumulative column number - in effect
  // the Nth column found on this tile
  unsigned columnAccumulate = 0;
  for (auto &partials : splitGroupedPartials) {
    if (partials.patterns[0].innerFactor == 1) {
      outRegions.push_back(
          {columnAccumulate, columnAccumulate + partials.columns.size()});
      columnAccumulate += partials.columns.size();
    } else {
      // Don't consider those with innerFactor != 1 for
      // splitting here as they can be split differently later.
      // Instead, push them into the output untouched.
      partialsResult.push_back(partials);
      // Having no columns will mean that this is not included in the column
      // search for the next loop, and avoids removing it from
      // splitGroupedPartials
      partials.columns.clear();
    }
  }
  if (outRegions.size() == 0) {
    return partialsResult;
  }
  const unsigned numWorkers = graph.getTarget().getNumWorkerContexts();
  const unsigned remainingWorkers = numWorkers > partialsResult.size()
                                        ? numWorkers - partialsResult.size()
                                        : 1;
  // outRegions represents intervals which are equivalent to the width of the
  // partials, so use it to check partial width in determining grainSize.
  const unsigned grainSize =
      findGrainSizeForOp(graph, inType, params, outRegions);
  outRegions = splitOutputRegionsForWorkers(grainSize, remainingWorkers, inType,
                                            outRegions);

  // Having divided the temporary output regions, copy from splitGroupedPartials
  // to partialsResult so that each set of columns represents an outRegion
  for (auto &region : outRegions) {
    unsigned columnAccumulate = 0;
    for (auto &partial : splitGroupedPartials) {
      if (region.begin() >= columnAccumulate &&
          region.begin() < columnAccumulate + partial.columns.size() &&
          partial.columns.size() != 0) {
        // We have found the splitGroupedPartial that contains the
        // region.begin() in question.  Regions don't span splitGroupedPartials
        // so it does contain the whole region. So create a partial that
        // describes the same pattern, but with only the group of columns
        // required to match the region size.

        partialsResult.emplace_back(partial);
        partialsResult.back().columns.resize(region.size());
        const auto columnIter =
            partial.columns.begin() + region.begin() - columnAccumulate;
        std::copy(columnIter, columnIter + region.size(),
                  partialsResult.back().columns.begin());
        // Adjust the regionOffset as the columns copied into this
        // partialsResult may not be the 1st set of columns listed.
        for (auto &pat : partialsResult.back().patterns) {
          pat.regionOffset += (region.begin() - columnAccumulate) *
                              partial.patterns[0].innerFactor;
        }
        break;
      }
      columnAccumulate += partial.columns.size();
    }
  }
  return partialsResult;
}
// Create reductions for the cases: Input to Output and Input to Intermediate
void createInputReductions(
    Graph &graph, const Tensor &in,
    boost::variant<Tensor &, IntermediatePartials &> out, bool createOutput,
    const Graph::TileToTensorMapping mapping, ReduceParams params,
    Type inputType, Type inVertexType, Type outputType, ComputeSetList &css,
    ResultTensors &reductionResultTensors, const std::string &debugPrefix,
    ReductionDebug::ReductionStage *stageDebug) {

  logging::debug("DebugStr: {}", debugPrefix);
  const bool isInputToOutput = out.type() == typeid(Tensor);

  // Store the output tensors for each reduction vertex, one per column
  std::vector<Tensor> outputs(isInputToOutput ? in.dim(1) : 0);
  std::size_t csPos = css.pos();
  // Get the set of contiguous regions on each tile (splitting them if
  // necessary at tile mapping boundaries). The region indices here are in
  // the flattened input tensor.
  auto contiguousRegionsByTile =
      getSortedContiguousRegionsByTile(graph, in, mapping);
  // Number of columns in the reduction.
  const auto columns = in.dim(1);
  auto inType = in.elementType();
  // Loop through the tiles. We can process each tile independently.
  for (unsigned tile = 0; tile < contiguousRegionsByTile.size(); ++tile) {
    const auto &contiguousRegionsThisTile = contiguousRegionsByTile[tile];

    // Ignore empty tiles.
    if (contiguousRegionsThisTile.empty()) {
      continue;
    }
    // Make a pattern for each column that is detected in the regions on tile
    auto partialsDescription =
        gatherReductionPatterns(contiguousRegionsThisTile, columns);

    // Grouping works by identifying a compatible patterns that follow a base
    // pattern in memory.  This requires them to be in memory order.
    std::sort(partialsDescription.begin(), partialsDescription.end(),
              [](const PartialsDescription &a, const PartialsDescription &b) {
                return (a.patterns[0].regionOffset <
                        b.patterns[0].regionOffset);
              });

    // Group the patterns according to columns with identical patterns and
    // adjacent in memory
    auto groupedPartials = groupPartials(partialsDescription, columns);

    // Divide the patterns to split work between workers and cope with
    // other limitations
    auto splitGroupedPartials =
        dividePartials(groupedPartials, graph, in.elementType(), params);

    // logging begin
    if (logging::shouldLog(logging::Level::Trace)) {
      // Use to select which to view at compile time...
      auto &debugPartials = splitGroupedPartials;
      logging::trace(" Tile:{} Reduction Patterns:{}", tile,
                     debugPartials.size());
      for (auto &pats : debugPartials) {
        std::stringstream colStr;
        for (auto col : pats.columns) {
          colStr << " " << col;
        }
        logging::trace("  Patterns:{} Column list[{}]:{}", pats.patterns.size(),
                       pats.columns.size(), colStr.str());
        for (auto &pat : pats.patterns) {
          logging::trace("    Pattern Inner factor:{} Start:{} Stride:{} Outer "
                         "factor:{} Region:{}",
                         pat.innerFactor, pat.regionOffset, pat.stride,
                         pat.outerFactor, pat.regionIdx);
        }
      }
    }
    // logging end

    // Create the regionReductions with partials populated from patterns
    auto reductions = listPartialsUsingPatterns(
        splitGroupedPartials, in, contiguousRegionsThisTile, tile, columns);
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
      for (unsigned j = 0; j < splitGroupedPartials[i].columns.size(); j++) {
        outputRegionsSplit2D[i].push_back(
            {splitGroupedPartials[i].columns[j],
             splitGroupedPartials[i].columns[j] + 1});
      }
    }
    if (!isInputToOutput) {
      // Add a tensor for this tile.
      auto data = graph.addVariable(outputType, {partialsDescription.size()},
                                    debugPrefix + "/tile_data1");
      reductionResultTensors.partials.push_back(data);
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
              outputType, {splitGroupedPartials[i].columns.size()});
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
      // Debugging info about the output..
      reductions[i].outputDebugInfo.outputRegion = outputRegionsSplit[i];
      reductions[i].outputDebugInfo.dataRegion = outputRegionsSplit[i];
    }

    ReductionDebug::TileReduction *tileDebug = nullptr;
    if (stageDebug != nullptr) {
      stageDebug->tiles.emplace_back();
      tileDebug = &stageDebug->tiles.back();
    }

    // Start from our current position in the compute set list.
    ComputeSetList cssFork = css;
    connectReductions(graph, cssFork, params, inputType, inVertexType,
                      outputType, tile, reductions, true, debugPrefix,
                      tileDebug);
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

void inputToOutputNoExchange(Graph &graph, const Tensor &in,
                             const Graph::TileToTensorMapping &mapping,
                             boost::optional<Tensor> &finalOutput,
                             const std::vector<std::size_t> outputShape,
                             Type inVertexType, Type outputType,
                             ReduceParams params, ComputeSetList &css,
                             ResultTensors &reductionResultTensors,
                             const std::string &debugPrefix,
                             ReductionDebug *debug) {
  // If we have an output, create the output Tensor for the
  // createInputReductions function. If we don't have an output,
  // createInputReductions will create its own output
  Tensor out;
  if (finalOutput) {
    out = finalOutput.get().flatten();
    if (!params.update) {
      reductionResultTensors.results.push_back(out);
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

  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Input to Output (No Exchange)";
  }
  createInputReductions(graph, in, out, !static_cast<bool>(finalOutput),
                        mapping, params, in.elementType(), inVertexType,
                        outputType, css, reductionResultTensors,
                        debugPrefix + "/InToOutNoExchange", stageDebug);
  if (!finalOutput) {
    finalOutput = out;
  }
  finalOutput = finalOutput.get().reshape(outputShape);
}

IntermediatePartials inputToIntermediateNoExchange(
    Graph &graph, const Tensor &in, const Graph::TileToTensorMapping &mapping,
    Operation op, const Type &inVertexType, const Type &outputType,
    ComputeSetList &css, ResultTensors &reductionResultTensors,
    const std::string &debugPrefix, ReductionDebug *debug) {

  // Number of output values of the reduction.
  auto outputSize = in.dim(1);

  auto inType = in.elementType();

  // Add a new tensor for each tile to output its partials to. These tensors
  // and the meta-info needed are stored in an IntermediatePartials.
  IntermediatePartials ir;
  ir.setDataType(inVertexType);
  ir.setOutputSize(outputSize);

  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Input to Intermediate (No Exchange)";
  }
  createInputReductions(graph, in, ir, false, mapping, op, in.elementType(),
                        inVertexType, inVertexType, css, reductionResultTensors,
                        debugPrefix + "/InToIntermediateNoExchange",
                        stageDebug);
  return ir;
}

template <typename T> struct DebugRange {
  T min;
  T max;
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

IntermediatePartials intermediateToIntermediate(
    Graph &graph, const IntermediatePartials &ipIn, Operation op,
    const Type &outType, ComputeSetList &css,
    ResultTensors &reductionResultTensors, const std::string &debugPrefix,
    ReductionDebug *debug) {

  logging::debug("DebugStr: {}", debugPrefix);
  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Intermediate to Intermediate";
  }

  // TODO: temoprarily only for ADD, SQUARE ADD as if applied to other types
  //       we produce a mix of partials types.  This can be dealt with when
  //       D20584 lands, but implies that a final cast stage would be needed
  //       for types other than ADD, SQUARE_ADD
  const bool opIsAddOrSquareAdd =
      op == popops::Operation::ADD || op == popops::Operation::SQUARE_ADD;

  boost::optional<ComputeSet> castComputeSet;
  if (reductionBenefitsFromPreExchangeCast(ipIn) && opIsAddOrSquareAdd) {
    logging::debug("Inserting pre-exchange cast half to float");
    castComputeSet = css.add(graph, debugPrefix + "/PreExchangeCast");
  }
  auto resultType = castComputeSet ? poplar::FLOAT : outType;

  IntermediatePartials ir;
  ir.setOutputSize(ipIn.outputSize());
  ir.setDataType(resultType);

  const auto inType = castComputeSet ? poplar::FLOAT : ipIn.dataType();

  const auto &target = graph.getTarget();

  // The grain size is doubled for ADD (and SQUARE_ADD) because
  // these operations have dedicated instructions on Colossus that can operate
  // on twice as much data as all the other operations (MUL etc).
  const unsigned grainSize = opIsAddOrSquareAdd
                                 ? 2 * target.getVectorWidth(inType)
                                 : target.getVectorWidth(inType);
  if (grainSize == 0)
    throw poputil::poplibs_error("Zero vector width for type " +
                                 inType.toString());

  // If each piece is really small the overhead of having extra reduction
  // stages, and exchange and everything outweighs the savings.
  //
  // Optimisation: reductionFactorThresholdToAddMoreStages was found empirically
  // and not tested a lot.

  auto splitMapIcl = calculateSplit(ipIn, grainSize, grainSize, 2,
                                    reductionFactorThresholdToAddMoreStages,
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
  // Debug variables
  unsigned debugTiles = 0;
  DebugRange<std::size_t> debugNumPartials = {UINT_MAX, 0};
  DebugRange<unsigned> debugNclip = {UINT_MAX, 0};
  DebugRange<std::size_t> debugPartialsWidths = {UINT_MAX, 0};
  // If we have only one reduction to do then use the tile containing the first
  // partial to avoid overloading tile 0 when doing multiple individual
  // reductions
  const bool useFirstOutputTile = splitMapIcl.size() == 1;
  for (const auto &it : splitMapIcl) {
    const auto &sourceTiles = tilesForOutput(it.first.lower());

    auto numPartials = sourceTiles.size();

    debugNumPartials = {std::min(debugNumPartials.min, numPartials),
                        std::max(debugNumPartials.max, numPartials)};

    auto splitCount = it.second;
    if (useFirstOutputTile) {
      t = *sourceTiles.begin();
    }
    assert(splitCount > 0);

    // N is the number of rows to take for each reduction. This should be at
    // least 2 so we actually do some reducing.
    std::size_t N = udiv(numPartials, splitCount);
    if (N < 2) {
      N = 2;
    }

    for (unsigned i = 0; i < numPartials; i += N) {
      auto &st = tileReductions[t].sourceTilesForInterval[ival];

      unsigned Nclip = std::min(N, numPartials - i);
      debugNclip = {std::min(debugNclip.min, Nclip),
                    std::max(debugNclip.max, Nclip)};

      debugTiles++;
      st.reserve(st.size() + Nclip);

      st.insert(st.end(), sourceTiles.nth(i), sourceTiles.nth(i + Nclip));

      t = (t + 1) % target.getNumTiles();
    }

    ++ival;
  }
  logging::debug(debugNumPartials.min == debugNumPartials.max
                     ? "  Remaining reduction of {} partials"
                     : "  Remaining reduction of {} to {} partials",
                 debugNumPartials.min, debugNumPartials.max);

  logging::debug(
      debugNclip.min == debugNclip.max
          ? "  This stage uses {} tiles, which all reduce {} partials"
          : "  This stage uses {} tiles reducing between {} and {} partials",
      debugTiles, debugNclip.min, debugNclip.max);

  std::size_t csPos = css.pos();
  unsigned debugTileCount = 0;

  // For each output tile...
  for (unsigned tile = 0; tile < tileReductions.size(); ++tile) {
    auto &tr = tileReductions[tile];

    if (tileReductions[tile].sourceTilesForInterval.empty()) {
      continue;
    }
    logging::trace("Tile {} reductions:", tile);
    debugTileCount++;

    // Work out the set of all output regions for this tile.
    boost::icl::interval_set<std::size_t> outputRegionsMergedIcl;
    for (auto it : tr.sourceTilesForInterval) {
      outputRegionsMergedIcl.insert(allOutputRegionsSplit[it.first]);
    }

    // Add a variable to receive the results.
    Tensor data = graph.addVariable(resultType, {outputRegionsMergedIcl.size()},
                                    debugPrefix + "/tile_data2");
    reductionResultTensors.partials.push_back(data);

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
        if (castComputeSet) {
          auto floatPartials = cast(
              graph,
              ipIn.data(partialTile).slice(sourceDataIdx, sourceDataIdx + len),
              poplar::FLOAT, castComputeSet.get());

          graph.setTileMapping(floatPartials, partialTile);
          rt.partials.push_back(floatPartials);
        } else {
          rt.partials.push_back(
              ipIn.data(partialTile).slice(sourceDataIdx, sourceDataIdx + len));
        }
        // Debugging info about the partial.
        ReductionDebug::Partial di;
        di.sourceCols = {sourceDataIdx, sourceDataIdx + len};
        di.sourceTile = partialTile;
        rt.partialsDebugInfo.push_back(di);
      }
      logging::trace("  Partials:{} Width:{} Output data index:[{}, {})",
                     it.second.size(), rt.partials.back().numElements(),
                     re.lower(), re.upper());
      debugPartialsWidths = {
          std::min(debugPartialsWidths.min, rt.partials.back().numElements()),
          std::max(debugPartialsWidths.max, rt.partials.back().numElements())};

      // Connect the output region.
      rt.output = ir.data(tile).slice(outputDataIdx, outputDataIdx + len);

      // Debugging info about the output...
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
    connectReductions(graph, cssFork, op, inType, inType, resultType, tile,
                      reductions, false,
                      debugPrefix + "/IntermediateToIntermediate", tileDebug);
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos)
      csPos = cssFork.pos();
  }
  logging::debug(debugPartialsWidths.min == debugPartialsWidths.max
                     ? "  Partial width {}"
                     : " With widths between {} and {}",
                 debugPartialsWidths.min, debugPartialsWidths.max);

  css.setPos(csPos);

  return ir;
}

void intermediateToOutput(Graph &graph, const IntermediatePartials &ipIn,
                          boost::optional<Tensor> &finalOutput,
                          const std::vector<std::size_t> outputShape,
                          Type outputType, ReduceParams params,
                          Type inVertexType, ComputeSetList &css,
                          ResultTensors &reductionResultTensors,
                          const Tensor &in, const std::string &debugPrefix,
                          ReductionDebug *debug) {
  logging::debug("DebugStr: {}", debugPrefix);

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
    out = graph.addVariable(outputType, {numOutElements}, debugPrefix);
    graph.setTileMapping(out, graph.getTileMapping(in.slice(0, 1, 0), false));
  }
  if (!params.update) {
    reductionResultTensors.results.push_back(out);
  }
  // If the data type is half AND
  // If any tile contains a single output, or any tile contains a single
  // partial then we will be be exchanging single partials and getting
  // inefficient reductions.  This can be avoided by casting before exchange.
  //
  // TODO: temprarily only for ADD, SQUARE ADD as if applied to other types
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
      logging::debug("Inserting pre-exchange cast half to float");
      castComputeSet = css.add(graph, debugPrefix + "/PreExchangeCast");
      inVertexType = poplar::FLOAT;
    }
  }

  // This is assumed below.
  assert(out.rank() == 1);

  const auto inType = castComputeSet ? poplar::FLOAT : ipIn.dataType();

  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Intermediate To Output";
  }

  // If the output isn't already mapped, map it linearly and do the reduction
  // there, otherwise decide whether it is better to do the reduction at the
  // destination or not.  Choose grainSize and minElementsPerTile based on the
  // reduction input type not the final output.
  if (mappingComplete) {
    if (!shouldReduceAtDestination(graph.getTarget(), ipIn, mapping,
                                   inVertexType, out.numElements())) {
      mapping =
          poputil::calcLinearTileMapping(graph, graph.clone(inVertexType, out));
    }
  } else {
    mapping =
        poputil::calcLinearTileMapping(graph, graph.clone(inVertexType, out));
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
  if (logging::shouldLog(logging::Level::Debug)) {
    for (unsigned tile = 0; tile < mapping.size(); ++tile) {
      if (!mapping[tile].empty()) {
        debugTiles++;
      }
    }
    logging::debug("  Using {} tiles", debugTiles);
  }
  DebugRange<std::size_t> debugNumPartials = {UINT_MAX, 0};
  DebugRange<std::size_t> debugPartialsWidths = {UINT_MAX, 0};

  // Partition tilesForOutput based on mappingIcl.
  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    if (mapping[tile].empty()) {
      continue;
    }
    logging::trace("Tile {} reductions:", tile);

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

    // Determine the grainSize based on operation, we have no easy way to
    // compare partials widths so don't assume anything about them
    const unsigned grainSize =
        findGrainSizeForOp(graph, inVertexType, params, boost::none);

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
    for (const auto &re : outputRegionsSplit) {
      RegionReduction rt;

      // Connect the output. This is fine because output is 1D.
      rt.output = out.slice(re);

      rt.partials.reserve(32); // This speeds things up a bit.

      // Get the list of partials to use.
      auto partialTiles = thisTilesForOutput(re.begin());
      debugNumPartials = {std::min(debugNumPartials.min, partialTiles.size()),
                          std::max(debugNumPartials.max, partialTiles.size())};

      for (auto partialTile : partialTiles) {
        size_t sourceDataIdx = ipIn.dataElement(partialTile, re.begin());
        size_t len = re.size();

        assert(ipIn.dataElement(partialTile, re.begin() + len - 1) ==
               sourceDataIdx + len - 1);

        if (castComputeSet) {
          auto floatPartials = cast(
              graph,
              ipIn.data(partialTile).slice(sourceDataIdx, sourceDataIdx + len),
              poplar::FLOAT, castComputeSet.get());

          graph.setTileMapping(floatPartials, partialTile);
          rt.partials.emplace_back(floatPartials);
        } else {
          rt.partials.emplace_back(
              ipIn.data(partialTile).slice(sourceDataIdx, sourceDataIdx + len));
        }
        // Debugging info about the partial.
        ReductionDebug::Partial di;
        di.sourceCols = {sourceDataIdx, sourceDataIdx + len};
        di.sourceTile = partialTile;
        rt.partialsDebugInfo.push_back(di);
      }
      logging::trace("  Partials:{} Width:{} Output data index:[{}, {})",
                     partialTiles.size(), rt.partials.back().numElements(),
                     re.begin(), re.end());
      debugPartialsWidths = {
          std::min(debugPartialsWidths.min, rt.partials.back().numElements()),
          std::max(debugPartialsWidths.max, rt.partials.back().numElements())};

      // Debugging info about the output...
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
    connectReductions(graph, cssFork, params, inVertexType, inVertexType,
                      outputType, tile, reductions, false,
                      debugPrefix + "/IntermediateToOutput", tileDebug);
    // Record the maximum number of compute sets we've used.
    if (cssFork.pos() > csPos)
      csPos = cssFork.pos();
  }
  logging::debug(debugNumPartials.min == debugNumPartials.max
                     ? "  Tiles all reduce {} partials"
                     : "  Tiles have between {} and {} partials",
                 debugNumPartials.min, debugNumPartials.max);
  logging::debug(debugPartialsWidths.min == debugPartialsWidths.max
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
