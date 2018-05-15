#include "poplibs_support/ContiguousRegionsByTile.hpp"

#if defined(ALTERNATE_IMPLEMENTATION__)
#include <boost/icl/interval_map.hpp>
#include <boost/optional.hpp>
#endif

namespace poplibs {

#if defined(ALTERNATE_IMPLEMENTATION__)

std::vector<std::vector<std::vector<poplar::Interval>>>
getSortedContiguousRegionsByTile(
    const poplar::Graph &graph,
    const poplar::Tensor &A,
    const poplar::Graph::TileToTensorMapping &mapping) {

  // Get the sorted contiguous regions up-front.
  const auto scrs = graph.getSortedContiguousRegions(A, {{0, A.numElements()}});

  // Convert the mapping to boost ICL format.
  auto mappingIcl = tileMappingToIntervalMap(mapping);

  // Make sure that the mapping is complete. This should always be the case
  // since getTileMapping() throws an exception if it isn't (by default).
  assert(mappingIcl.size() == A.numElements());

  // The result - a set of sorted contiguous regions for each tile.
  std::vector<
      std::vector<
        std::vector<
          poplar::Interval
        >
      >
    > out(mapping.size());

  // For each sorted contiguous region:
  for (const auto &scr : scrs) {
    if (scr.empty())
      continue;

    // The tile that the last bit of this contiguous region was mapped to.
    boost::optional<unsigned> lastTile;

    // For each tensor region in this contiguous variable region:
    for (const auto &re : scr) {
      if (re.size() == 0)
        continue;

      // Find the parts of this region that are mapped to different tiles.
      // It is possible to iterate through `mappingIcl & re` but that is slow.

      auto begin = mappingIcl.find(re.begin());
      auto end = mappingIcl.find(re.end());

      // For each part on a different tile:
      for (auto it = begin; it != end; ++it) {
        // Get the region.
        auto lower = std::max(it->first.lower(), re.begin());
        auto upper = std::min(it->first.upper(), re.end());

        // The tile that this part is mapped to.
        const auto &tile = it->second;

        // If it is the first part of this contiguous region, or if the
        // last part was on a different tile, start a new contiguous region
        // on this tile.
        if (!lastTile || tile != lastTile) {
          lastTile = tile;
          out[tile].emplace_back();
        }

        // And append it to the current contiguous region.
        out[tile].back().emplace_back(lower, upper);
      }
    }
  }

  return out;
}

#else

std::vector<std::vector<std::vector<poplar::Interval>>>
getSortedContiguousRegionsByTile(
    const poplar::Graph &graph,
    const poplar::Tensor &A,
    const poplar::Graph::TileToTensorMapping &mapping) {

  std::vector<
    std::vector<
      std::vector<poplar::Interval>
      >
    > contiguousRegionsByTile;

  for (const auto &m : mapping) {
    contiguousRegionsByTile.emplace_back(
      graph.getSortedContiguousRegions(A, m)
    );
  }

  return contiguousRegionsByTile;
}

#endif

} // namespace poplibs
