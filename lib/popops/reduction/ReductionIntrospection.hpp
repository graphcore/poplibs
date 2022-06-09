// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef ReductionIntrospection_hpp
#define ReductionIntrospection_hpp

#include "ReductionConnection.hpp"
#include "popops/Reduce.hpp"

#include <boost/optional.hpp>
#include <poplar/Interval.hpp>
#include <poplibs_support/StridedRegions.hpp>

namespace popops {

// Initially each reduction is referenced as a series of patterns which
// describe the part of a contiguous region/regions of data that are
// required by a given reduction.  A pattern takes the form:
//
// innerFactor: number of contiguous elements of any single column in the
//              pattern
// regionOffset:index of the 1st element that is required relative to the
//              region start
// stride:      number of elements within the whole pattern before it repeats
// outerFactor: number of times the pattern repeats
// regionIdx:   The index into the contiguous regions list that the column data
//              is found in. (Ie which of the contiguous regions on tile is it
//              in)
struct PartialsPattern {
  unsigned innerFactor;
  unsigned regionOffset;
  unsigned stride;
  unsigned outerFactor;
  unsigned regionIdx;
};
// The PartialsDescription structure is used in 2 ways.
// Initially a single column is identified and recorded in 'columns'.  All
// elements of that column that are found on tile are recorded in 'patterns'.
// Where the regularity of the layout of the column elements is broken,
// (either by a change in 'stride' or similar or a new region) a fresh pattern
// is started.
//
// Later, PartialsDescriptions can be grouped. Those with compatible patterns
// can be combined. The patterns listed are unchanged and describe the layout
// of the 1st column in the 'columns' vector.  Further columns are added to the
// columns vector, which have the same layout in memory but a 'start' parameter
// which increments with position in the 'columns' vector.
struct PartialsDescription {
  // List of the columns which these patterns describe
  std::vector<unsigned> columns;
  // The partials described in the form of patterns
  std::vector<PartialsPattern> patterns;
};

// A function which accepts a vector of patterns which each describe
// a reduction of one or more columns. Each pattern references a region /
// regions and describes a number of tensor elements (partials) found within
// that region.
// The function adds references to the partials for each reduction into the
// "reductions" structure.
std::vector<popops::RegionReduction> listPartialsUsingPatterns(
    const std::vector<PartialsDescription> &partialsDescription,
    const poplar::Tensor &input,
    const std::vector<poplibs_support::StridedRegionList> &inputRegions,
    unsigned tile);

// Given a set of contiguous regions for a tensor with shape {rows, columns},
// fills partialsDescription with patterns describing each column individually.
std::vector<poplibs_support::StridedRegionList> gatherReductionPatterns(
    std::vector<PartialsDescription> &partialsDescription,
    const std::vector<std::vector<poplar::Interval>> &regions,
    unsigned columns);

// Given a set of PartialsDescriptions, group together those that sit next to
// each other in memory and which can be described with the same
// PartialsPattern.
std::vector<PartialsDescription>
groupPartials(std::vector<PartialsDescription> &partialsDescription,
              unsigned columns);

// Divide partials up either to ditribute work or to break down patterns that
// can't be translated into a RegionReduction
std::vector<PartialsDescription>
dividePartials(const std::vector<PartialsDescription> &groupedPartials,
               poplar::Graph &graph, poplar::Type inType, unsigned numWorkers,
               popops::ReduceParams params);

using TilePartialsDescription = std::vector<std::vector<PartialsDescription>>;
using RegionsByTile = std::vector<std::vector<std::vector<poplar::Interval>>>;
using StridedRegionsByTile =
    std::vector<std::vector<poplibs_support::StridedRegionList>>;

// Analyse the contiguousRegionsPer tile, gather reduction patterns and produce
// groups of partials for all tiles
std::pair<TilePartialsDescription, StridedRegionsByTile> allTileGroupedPartials(
    const poplar::Graph &graph, const poplar::Tensor &in,
    const std::vector<std::vector<poplar::Interval>> &tileMapping,
    unsigned columns, bool showLogging = false);

// Analyse over all tiles to see if the layout of columns in memory is
// consistent and not a simple consecutive column number order.
// If so return the ordering detected
boost::optional<std::vector<unsigned>>
findCommonColumnOrder(const TilePartialsDescription &groupedPartials,
                      unsigned columns);
} // namespace popops

#endif // ReductionStages_hpp
