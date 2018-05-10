#include "IntermediatePartialsUtil.hpp"

#include <poputil/exceptions.hpp>

#include "ReductionDebug.hpp"
#include "RegionWrapping.hpp"

namespace popops {

bool mappingHasMultipleValuesFromOneColumnOnTheSameTile(
    const poplar::Graph::TileToTensorMapping &mapping, std::size_t wrapSize) {

  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    if (mapping[tile].empty())
      continue;

    // The set of output indices for which we have partials on this tile.
    boost::icl::interval_set<std::size_t> outputIndices;

    // Loop through the regions mapped to this tile.
    for (const auto &ival : mapping[tile]) {
      if (ival.size() == 0)
        continue;

      for (auto i = ival.begin(); i < ival.end();) {

        // Index on this row.
        auto x = i % wrapSize;

        // Length of the rest of the region.
        auto len = ival.end() - i;

        // If it goes past the end of the row truncate it.
        if (x + len > wrapSize) {
          len = wrapSize - x;
        }

        // Convert to boost format.
        auto re = boost::icl::discrete_interval<size_t>::right_open(x, x + len);

        // If it intersects, then we have multiple values from one column
        // on the same tile.
        if (boost::icl::intersects(outputIndices, re))
          return true;

        // Otherwise note that we have these values on the tile.
        outputIndices.add(re);

        i += len;
      }
    }
  }
  return false;
}

IntermediatePartials
tensorToIntermediatePartials(const poplar::Tensor &A,
                             const poplar::Graph::TileToTensorMapping &mapping,
                             ReductionDebug *debug) {

  // Debug information.
  ReductionDebug::ReductionStage *stageDebug = nullptr;
  if (debug != nullptr) {
    debug->stages.emplace_back();
    stageDebug = &debug->stages.back();
    stageDebug->label = "Input to Intermediate (Zero Cost)";
  }

  if (A.rank() != 2)
    throw poputil::poplib_error("tensorToIntermediatePartials called with "
                                "tensor of rank " + std::to_string(A.rank())
                                + " (should be 2)");

  IntermediatePartials ir;
  ir.setDataType(A.elementType());
  ir.setOutputSize(A.dim(1));

  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    if (mapping[tile].empty())
      continue;

    // The "reduction" on this tile. Pretend it is a nop reduction
    // so we can see how the concat'd tensor is constructed.
    ReductionDebug::TileReduction *tileDebug = nullptr;
    if (stageDebug != nullptr) {
      stageDebug->tiles.emplace_back();
      tileDebug = &stageDebug->tiles.back();
      tileDebug->tileIndex = tile;
    }

    // Map from output index to the row and end of a region in the
    // input tensor.

    struct SourceRegion {
      SourceRegion(std::size_t begin, std::size_t end, std::size_t row)
        : begin(begin), end(end), row(row) {}
      std::size_t begin;
      std::size_t end;
      std::size_t row;
    };

    std::vector<SourceRegion> sortedRegions;
    sortedRegions.reserve(mapping[tile].size());

    wrapRegionsToRows(mapping[tile].begin(),
                      mapping[tile].end(),
                      A.dim(1),
                      [&](std::size_t begin, std::size_t end, std::size_t row) {
                        sortedRegions.emplace_back(begin, end, row);
                      });

    // Sort them based on begin (the first output index).
    std::sort(sortedRegions.begin(), sortedRegions.end(),
              [](const SourceRegion &a, const SourceRegion &b) {
                return a.begin < b.begin;
              });

    // Verify there is no overlap.
    for (std::size_t i = 1; i < sortedRegions.size(); ++i)
      if (sortedRegions[i].begin < sortedRegions[i-1].end)
        throw poputil::poplib_error("tensorToIntermediatePartials called but "
                                    "tile " + std::to_string(tile) + " has "
                                    "multiple partials from the same output");

    // The list of tensors to concatenate to get the data tensor.
    std::vector<poplar::Tensor> toConcat;

    // The output indices (i.e. columns of the 2D input tensor) for the regions
    // mapped to this tile.
    boost::icl::interval_set<std::size_t> outputIndices;

    // Iterate through output regions in order.
    for (const auto &re : sortedRegions) {

      // Add it to regions to concatenate together.
      toConcat.push_back(A.slice({re.row, re.begin},
                                 {re.row + 1, re.end}).flatten());

      outputIndices.add(
          boost::icl::discrete_interval<size_t>::right_open(re.begin, re.end));

      if (tileDebug != nullptr) {
        // Add debug information to show the flow of data, even though this
        // doesn't actually involve any computation.
        ReductionDebug::RegionReduction regionDebug;
        regionDebug.vertex = 0;
        regionDebug.output.outputRegion = {re.begin, re.end};
        regionDebug.output.dataRegion = {outputIndices.size(),
                                         outputIndices.size()
                                         + re.end - re.begin};


        ReductionDebug::Partial partialDebug;
        partialDebug.sourceTile = tile;
        partialDebug.sourceCols = {re.begin, re.end};
        partialDebug.sourceRows = {re.row, re.row + 1};
        regionDebug.partials.push_back(partialDebug);

        tileDebug->firstStage.firstStageRegions.push_back(regionDebug);
      }
    }

    auto var = poplar::concat(toConcat);

    // If there are duplicates then the size of outputIndices and var
    // will be different and this will throw an exception.
    ir.setTensor(tile, var, outputIndices);
  }

  return ir;
}

}
