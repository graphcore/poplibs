// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ReductionIntrospection.hpp"
#include "ReductionConnection.hpp"
#include "ReductionStages.hpp"
#include "RegionWrapping.hpp"

#include "poplibs_support/ContiguousRegionsByTile.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_support/print.hpp"

#include <tbb/parallel_for.h>

using namespace poplar;
using namespace poputil;
using namespace poplibs;
using namespace popops;
using namespace poplibs_support;

namespace {

// createElementRefsFromRegions: List a reference of the column and position
// for each tensor element found in the intervals for each region.  These are
// just concatenated and sorted later.  Making a vector per column is another
// way to achieve this, but with a large number of columns (and potentially)
// relatively few on any 1 time this can be slow.
struct ElementRef {
  unsigned column;
  unsigned region;
  unsigned offset;
};

std::pair<std::vector<ElementRef>, std::vector<StridedRegionList>>
preprocessRegions(const std::vector<std::vector<Interval>> &regions,
                  std::vector<PartialsDescription> &partialsDescription,
                  const unsigned columns, bool detectColumns) {

  std::vector<StridedRegionList> stridedRegions(regions.size());
  std::vector<ElementRef> elementRefs;
  for (unsigned r = 0; r < regions.size(); r++) {
    unsigned regionStartOffset = 0;
    stridedRegions[r].reserve(regions[r].size());
    for (const auto &region : regions[r]) {
      stridedRegions[r].emplace_back(region);
      for (unsigned e = region.begin(); e < region.end(); e++) {
        // Examine the column number of every element on tile, and record it.
        const unsigned column = e % columns;
        elementRefs.emplace_back(ElementRef{column, r, regionStartOffset});
        regionStartOffset++;
      }
    }
  }

  for (auto &contiguousRegion : stridedRegions) {
    mergeStridedRegions(contiguousRegion);
  }

  // Sort so that columns are together and then then elements are in the order
  // found on tile
  std::sort(elementRefs.begin(), elementRefs.end(),
            [](ElementRef &lhs, ElementRef &rhs) {
              return std::tie(lhs.column, lhs.region, lhs.offset) <
                     std::tie(rhs.column, rhs.region, rhs.offset);
            });
  // To make test cases simple to understand, we may only be interested
  // in certain columns.  Discard those that are not of interest, they are small
  // test cases so speed isn't an issue, and this is intentionally separate
  // from the loop above
  if (!detectColumns) {
    for (unsigned i = 0; i < columns; i++) {
      bool useColumn = false;
      for (unsigned j = 0; j < partialsDescription.size(); j++) {
        if (partialsDescription[j].columns[0] == i) {
          useColumn = true;
        }
      }
      if (!useColumn) {
        unsigned index = 0;
        while (index < elementRefs.size()) {
          if (elementRefs[index].column == i) {
            elementRefs.erase(elementRefs.begin() + index);
          } else {
            index++;
          }
        }
      }
    }
  }
  return std::make_pair(elementRefs, stridedRegions);
}

// updatePartialsDescription: Given a "signal" indicating that the column
// of interest is/is not detected in the region, update the partialsDescription
// structure.

struct PatternBuildState {
  bool patternColumnEnd;
  unsigned patternColumnRef;
  bool buildingPattern = false;
};

static inline void
updatePartialsDescription(PatternBuildState &pbs, PartialsDescription &rt,
                          bool thisColumnFound, unsigned currentRegion,
                          unsigned priorRegion, unsigned elementOffset,
                          bool isRegionEnd) {

  if (thisColumnFound && !pbs.buildingPattern) {
    // The first pattern in this region
    pbs.patternColumnRef = elementOffset;
    rt.patterns.emplace_back(
        PartialsPattern{0, pbs.patternColumnRef, 0, 0, currentRegion});
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
            rt.patterns.emplace_back(PartialsPattern{
                length, pbs.patternColumnRef, 0, 0, priorRegion});
          } else {
            rt.patterns.emplace_back(PartialsPattern{
                length, pbs.patternColumnRef, 0, 0, currentRegion});
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
          rt.patterns.emplace_back(
              PartialsPattern{0, elementOffset, 0, 0, currentRegion});
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
            rt.patterns.emplace_back(PartialsPattern{
                length + 1, pbs.patternColumnRef, 0, 1, currentRegion});
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
                         unsigned index, bool detectColumns,
                         const unsigned column) {
  // Is this the end of the region, if so complete the pattern accordingly
  bool regionEnd = false;
  if (elementRefs.size() == index + 1) {
    regionEnd = true;
  } else if (elementRefs[index].column != elementRefs[index + 1].column) {
    regionEnd = true;
  } else

      if (elementRefs[index].region != elementRefs[index + 1].region) {
    regionEnd = true;
  }
  const unsigned lastOne = regionEnd ? 1u : 0u;
  // Create a pattern and complete a struct to look after updating it
  unsigned preDetIndex;
  if (detectColumns) {
    partialsDescription.emplace_back(
        PartialsDescription{{column},
                            {{lastOne, elementRefs[index].offset, 0, lastOne,
                              elementRefs[index].region}}});
  } else {
    for (preDetIndex = 0; preDetIndex < partialsDescription.size();
         preDetIndex++) {
      if (partialsDescription[preDetIndex].columns[0] == column) {
        break;
      }
    }
    partialsDescription[preDetIndex].patterns.emplace_back(
        PartialsPattern{lastOne, elementRefs[index].offset, 0, lastOne,
                        elementRefs[index].region});
  }

  patternBuildState = {false, elementRefs[index].offset, !regionEnd};
  return detectColumns ? partialsDescription.size() - 1 : preDetIndex;
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
} // anonymous namespace

namespace popops {

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

std::vector<StridedRegionList>
gatherReductionPatterns(std::vector<PartialsDescription> &partialsDescription,
                        const std::vector<std::vector<Interval>> &regions,
                        unsigned columns) {

  // First list all references to each column in a vector of vectors: One outer
  // vector per column (ie output element from the reduction)
  const bool detectColumns = (partialsDescription.size() == 0);
  auto [elementRefs, stridedRegions] =
      preprocessRegions(regions, partialsDescription, columns, detectColumns);

  // Looking at each vector in turn, build a pattern for each column we find.
  unsigned index = 0;
  while (index < elementRefs.size()) {

    PatternBuildState patternBuildState;
    // Create a pattern structure to deal with this, and return a reference
    // to it.  Initialise the pattern build state.
    auto currentPatternIdx = initialisePatternStructs(
        patternBuildState, partialsDescription, elementRefs, index,
        detectColumns, elementRefs[index].column);
    PartialsDescription &currentPattern =
        partialsDescription[currentPatternIdx];
    // Add the rest of the elements belonging to this column to the pattern
    const unsigned currentColumn = elementRefs[index].column;
    index++;
    while (index < elementRefs.size() &&
           currentColumn == elementRefs[index].column) {
      const bool isNewRegion =
          elementRefs[index].region != elementRefs[index - 1].region;

      const bool nonColumnElementsExist =
          isNewRegion ||
          elementRefs[index].offset != elementRefs[index - 1].offset + 1;
      // Update the pattern for the presence of memory that isn't in its
      // column
      if (nonColumnElementsExist) {
        // Mark the end of the "column detected" signal with a single
        // element where column detect = false.  This could be because a
        // new region was found - in which case it updates due to the gap
        // between regions
        updatePartialsDescription(
            patternBuildState, currentPattern, false, elementRefs[index].region,
            elementRefs[index - 1].region, elementRefs[index - 1].offset + 1,
            isNewRegion);
        if (!isNewRegion) {
          // If that didn't happen due to a region Change, then update the
          // pattern with the information that there were potentially many
          // elements with a "column detected" signal = 0
          updatePartialsDescription(patternBuildState, currentPattern, false,
                                    elementRefs[index].region,
                                    elementRefs[index - 1].region,
                                    elementRefs[index].offset - 1, false);
        }
      }
      // Update the pattern for its own column, taking note of special case of
      // the end of the data on tile for this column
      const bool isLastElement =
          (index == elementRefs.size() - 1) ||
          (currentColumn != elementRefs[index + 1].column);
      updatePartialsDescription(patternBuildState, currentPattern, true,
                                elementRefs[index].region,
                                elementRefs[index - 1].region,
                                elementRefs[index].offset, isLastElement);
      index++;
    }
  }
  // Where stride hadn't been set in the above, set it
  for (auto &parDesc : partialsDescription) {
    for (auto &pat : parDesc.patterns) {
      pat.stride = pat.stride == 0 ? 1 : pat.stride;
    }
  }
  return stridedRegions;
}

// Cleaner function for use below, which returns a PartialsDescription vector
// and therefore will always automatically determine all columns referenced in
// the  "regions".  The function above is mostly useful for test.
static std::pair<std::vector<PartialsDescription>,
                 std::vector<StridedRegionList>>
gatherReductionPatterns(const std::vector<std::vector<Interval>> &regions,
                        unsigned columns) {
  std::vector<PartialsDescription> partialsDescription;
  auto stridedRegions =
      gatherReductionPatterns(partialsDescription, regions, columns);
  return std::make_pair(std::move(partialsDescription),
                        std::move(stridedRegions));
}

// Consider several patterns, which need to describe the same data layout
// with a consistent stride to be seen as regular
static bool patternsAreRegular(const std::vector<PartialsPattern> &patterns) {
  if (patterns.size() == 1) {
    return true;
  }
  const auto firstRegionOffsetDelta =
      patterns[1].regionOffset - patterns[0].regionOffset;
  for (unsigned i = 1; i < patterns.size(); i++) {
    const auto regionOffsetDelta =
        patterns[i].regionOffset - patterns[i - 1].regionOffset;
    if (regionOffsetDelta != firstRegionOffsetDelta ||
        patterns[i].innerFactor != patterns.front().innerFactor ||
        patterns[i].stride != patterns.front().stride ||
        patterns[i].outerFactor != patterns.front().outerFactor ||
        patterns[i].regionIdx != patterns.front().regionIdx) {
      return false;
    }
  }
  return true;
}

// A function which accepts a vector of patterns which each describe
// a reduction of one or more columns. Each pattern references a region /
// regions and describes a number of tensor elements (partials) found within
// that region.
// The function adds references to the partials for each reduction into the
// "reductions" structure.
std::vector<RegionReduction> listPartialsUsingPatterns(
    const std::vector<PartialsDescription> &partialsDescription,
    const Tensor &input, const std::vector<StridedRegionList> &inputRegions,
    unsigned tile) {
  // For speed, prepare a vector of tensors for each on tile region, each of
  // which will be referenced many times in the loop below.
  std::vector<Tensor> regionTensors(inputRegions.size());
  const auto inputFlat = input.flatten();
  for (unsigned i = 0; i < inputRegions.size(); i++) {
    regionTensors[i] = sliceStridedRegions(inputFlat, inputRegions[i]);
  }

  std::vector<RegionReduction> reductions(partialsDescription.size());
  for (unsigned i = 0; i < reductions.size(); i++) {
    if (patternsAreRegular(partialsDescription[i].patterns)) {
      // This reduction's partials can be described by a single tensor
      // and offset, stride pattern which can make for a more efficient
      // implementation.
      auto &pat = partialsDescription[i].patterns[0];
      auto &patterns = partialsDescription[i].patterns;
      reductions[i].getPartials().emplace_back(regionTensors[pat.regionIdx]);

      reductions[i].getOffset() = pat.regionOffset;
      reductions[i].getStride() = pat.stride;
      reductions[i].innerFactor = pat.innerFactor;
      reductions[i].outerFactor = pat.outerFactor;
      reductions[i].getNumOuterStrides() = patterns.size();
      if (patterns.size() > 1) {
        reductions[i].getOuterStride() =
            patterns[1].regionOffset - patterns[0].regionOffset;
      } else {
        reductions[i].getOuterStride() = 0;
      }
    } else {
      // Store all the partials in iPartials  and assign them at the end
      IrregularPartials iPartials;
      for (auto &pat : partialsDescription[i].patterns) {
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
            iPartials.data.emplace_back(in.slice(pat.regionOffset, end));
          } else {
            // If the patterns repeats and has "gaps"
            // (i.e. stride != no of columns) we need multiple slices to create
            // the partials.
            for (unsigned k = 0; k < pat.outerFactor; k++) {
              const auto start = pat.regionOffset + k * pat.stride;
              const auto end =
                  pat.regionOffset + k * pat.stride +
                  pat.innerFactor * partialsDescription[i].columns.size();
              iPartials.data.emplace_back(in.slice(start, end));
            }
          }
        } else {
          // If there are no pattern repetitions we can create partials with a
          // single slice.
          const auto end =
              pat.regionOffset +
              pat.innerFactor * partialsDescription[i].columns.size();
          iPartials.data.emplace_back(in.slice(pat.regionOffset, end));
        }
      }
      reductions[i].partials = iPartials;
    }
  }
  return reductions;
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
  groupedPartials.reserve(partialsDescription.size());
  // Keep track of which patterns have been added to grouped patterns
  std::vector<bool> partialsDescriptionIsGrouped(partialsDescription.size(),
                                                 false);
  unsigned partialsDescriptionsToGroup = partialsDescription.size();
  for (unsigned i = 0;
       i < partialsDescription.size() && partialsDescriptionsToGroup; i++) {
    // If the next one hasn't been grouped already, put it into the
    // groupedPartials structure.
    if (partialsDescriptionIsGrouped[i] == false) {
      groupedPartials.emplace_back(partialsDescription[i]);
      groupedPartials.back().columns.resize(1);
      partialsDescriptionIsGrouped[i] = true;
      partialsDescriptionsToGroup--;

      // Scan the remaining ones for adjacent, matching patterns, append their
      // column to the column list and mark them as grouped
      for (unsigned j = i + 1; j < partialsDescription.size(); j++) {
        if (partialsDescriptionIsGrouped[j] == false) {
          if (isAdjacent(partialsDescription[i], partialsDescription[j],
                         columns)) {
            groupedPartials.back().columns.emplace_back(
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

// dividePartials: Accepts a number of groupedPartials structures, each of which
// can contain pattern layout information about a number of columns to be
// reduced.  These are divided up into smaller groups of columns so that:
// a) There are no multi column groups of patterns where the innerFactor
//    parameter is not the same for all patterns
// b) To divide work between available workers

std::vector<PartialsDescription>
dividePartials(const std::vector<PartialsDescription> &groupedPartials,
               Graph &graph, Type inType, ReduceParams params) {

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
      splitGroupedPartials.emplace_back(groupedPartials[i]);
    } else {
      // Split all the patterns so that we have a pattern per column,
      // maintaining the innerFactor
      splitGroupedPartials.reserve(groupedPartials[i].columns.size());
      for (unsigned j = 0; j < groupedPartials[i].columns.size(); j++) {
        // The split partials have the same patterns but only one column
        splitGroupedPartials.emplace_back();
        splitGroupedPartials.back().patterns = groupedPartials[i].patterns;
        splitGroupedPartials.back().columns.emplace_back(
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
    // Don't consider those with a column size that can be optimised later
    // by reducing in 2 stages. (This specifically includes 3 columns as with
    // partials type float we get a grainsize of 2. So our 3 column reduction
    // would result in 2 reductions with 2 and 1 columns, which prevents the
    // later problemColumn count optimisation from happening)
    if (partials.patterns[0].innerFactor == 1 && partials.columns.size() != 3) {
      outRegions.emplace_back(Interval{
          columnAccumulate, columnAccumulate + partials.columns.size()});
      columnAccumulate += partials.columns.size();
    } else {
      // Push into the output untouched.
      partialsResult.emplace_back(partials);
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
  const unsigned grainSize = findGrainSizeForOp(graph, inType, params.op);
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

// Analyse the contiguousRegionsPer tile, gather reduction patterns and produce
// groups of partials for all tiles
std::pair<TilePartialsDescription, StridedRegionsByTile>
allTileGroupedPartials(const Graph &graph, const Tensor &t,
                       const std::vector<std::vector<Interval>> &mapping,
                       unsigned columns, bool showLogging) {

  TilePartialsDescription result(mapping.size());
  StridedRegionsByTile stridedRegionsByTile(mapping.size());
  // Loop through the tiles. We can process each tile independently.
  const unsigned numTiles = mapping.size();
  tbb::parallel_for(unsigned(0), numTiles, [&](unsigned tile) {
    const auto contiguousRegionsThisTile =
        graph.getSortedContiguousRegions(t, mapping[tile]);
    // Ignore empty tiles.
    if (!contiguousRegionsThisTile.empty()) {
      // Make a pattern for each column that is detected in the regions on tile
      auto [partialsDescription, stridedRegions] =
          gatherReductionPatterns(contiguousRegionsThisTile, columns);

      stridedRegionsByTile[tile] = std::move(stridedRegions);

      // Grouping works by identifying a compatible patterns that follow a base
      // pattern in memory.  This requires them to be in memory order.
      std::sort(partialsDescription.begin(), partialsDescription.end(),
                [](const PartialsDescription &a, const PartialsDescription &b) {
                  return (a.patterns[0].regionOffset <
                          b.patterns[0].regionOffset);
                });

      // Group the patterns according to columns with identical patterns and
      // adjacent in memory
      result[tile] = groupPartials(partialsDescription, columns);
    }
  });

  // logging begin
  // Normally disabled to avoid too much information but as the column
  // re-arrangement hides the actual memory output this can be useful for debug
  if (showLogging && logging::popops::shouldLog(logging::Level::Trace)) {
    for (unsigned tile = 0; tile < numTiles; tile++) {
      auto &debugPartials = result[tile];
      if (debugPartials.empty()) {
        continue;
      }
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
  }
  // logging end
  return std::make_pair(std::move(result), std::move(stridedRegionsByTile));
}

// Struct to record the columns seen before or after a given column.
// The before and after columns are used to extract column sequences. After
// building a vector of these, we break circular lists of columns.
// This makes sequence extraction simpler
struct ColumnLink {
  bool needsCircularLinkCheck = true;
  bool hasBefore = false;
  bool hasAfter = false;
  unsigned before;
  unsigned after;
};

struct ColumnLinks {
  std::vector<ColumnLink> links;
  ColumnLinks(std::size_t numColumns) : links(numColumns) {}

  void breakCircularLinks(void) {
    for (unsigned start = 0; start < links.size(); start++) {
      auto curr = start;
      if (links[curr].needsCircularLinkCheck) {
        while (links[curr].hasAfter) {
          // Note that the current link is being verified to not be a part of a
          // circular chain.  This avoids repeatedly checking the same chain
          // and speeds the process of checking/breaking links greatly
          links[curr].needsCircularLinkCheck = false;
          if (links[curr].after == start) {
            // The sequence of links led us back to the same column, so break
            // that link
            links[curr].hasAfter = false;
            links[start].hasBefore = false;
            break;
          }
          curr = links[curr].after;
        }
      }
    }
  }

  void addNewLink(unsigned column, unsigned nextColumn) {
    // Log the information about what column follows another column.
    // If there is already a column noted after `column` or before `nextColumn`
    // we don't revise that information (stick with the first link detected).
    // This can produce circular links which aren't desirable but checking for
    // them here results in checking many times which is time consuming.
    if (!links[column].hasAfter && !links[nextColumn].hasBefore) {
      links[column].after = nextColumn;
      links[column].hasAfter = true;
      links[nextColumn].before = column;
      links[nextColumn].hasBefore = true;
    }
  }
};

// Analyse over all tiles to extract the layout of columns in memory.
// There are 3 possibilities
// a) Sequential column ordering: 0,1,2... on all tiles
//    - Nothing to do
// b) Inconsistent ordering: 0,1,2 on some tiles, 2,1,0... on others
//    - It should still help to re-order but the result may not be optimal
// c) Non-sequential ordering: 2,1,0 on all tiles.
//    - We can rearrange inputs, outputs to cope with this and reduce/eliminate
//      copies depending on if we reduce with output or not.

boost::optional<std::vector<unsigned>>
findCommonColumnOrder(const TilePartialsDescription &groupedPartials,
                      unsigned columns) {

  ColumnLinks columnLinks(columns);

  // Analyse all the regions provided.  Each column has an entry in the
  // columnLinks vector.  Populate each with the column that is found after
  // it in memory and that before it.  Only record the first column
  // association found between columns - there is no record made of columns
  // being after > 1 other columns
  for (const auto tilePartials : groupedPartials) {
    for (const auto regionPartial : tilePartials) {
      for (unsigned i = 0; i < regionPartial.columns.size() - 1; i++) {
        auto column = regionPartial.columns[i];
        auto nextColumn = regionPartial.columns[i + 1];
        columnLinks.addNewLink(column, nextColumn);
      }
    }
  }
  // Break circular links, which simplifies column order extraction.  This is
  // because if there are no circular links, starting from the beginning of each
  // chain is a good way to extract all columns only once.
  columnLinks.breakCircularLinks();

  // Extract the column ordering
  // There can be many independent groups (eg 2 columns on 1 tile, 2 on another)
  // So gather each and append them.
  std::vector<unsigned> commonColumnOrder;
  commonColumnOrder.reserve(columns);
  bool linearOrder = true;
  for (unsigned c = 0; c < columns; ++c) {
    // Always start from the beginning of a chain
    if (columnLinks.links[c].hasBefore) {
      continue;
    }
    commonColumnOrder.emplace_back(c);
    unsigned curr = c;
    while (columnLinks.links[curr].hasAfter) {
      if (curr + 1 != columnLinks.links[curr].after) {
        // Check for linear ordering within a chain only
        linearOrder = false;
      }
      commonColumnOrder.emplace_back(columnLinks.links[curr].after);
      curr = columnLinks.links[curr].after;
    }
  }

  if (linearOrder) {
    // Don't return a sequential order as we can avoid recalculating mapping etc
    logging::popops::debug("Sequential column ordering detected");
    return boost::none;
  }
  return commonColumnOrder;
}

} // namespace popops
