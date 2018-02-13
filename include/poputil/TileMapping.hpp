#ifndef __popstd_TileMapping_hpp__
#define __popstd_TileMapping_hpp__
#include <vector>
#include "poplar/Graph.hpp"

namespace poputil {

/* Calculate a tile mapping that spreads the tensor
 * evenly over the tiles in a linear manner (i.e. with the
 * indices of the flatenned tensor mapped across from low -> high tile
 * numbers).
 */
std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph,
                      std::vector<std::size_t> shape,
                      unsigned minElementsPerTile,
                      unsigned grainSize);

/* Calculate a tile mapping that spreads the tensor
 * evenly over the tiles in a linear manner (i.e. with the
 * indices of the flatenned tensor mapped across from low -> high tile
 * numbers).
 *
 * In this case the elements are split so as not to split vectors of elements
 * for the devices natural vector widths and to try and keep at least 128 bytes
 * of data on each tile to avoid high exchange costs.
 */
std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph,
                      const poplar::Tensor &t);

void
mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t,
                  unsigned minElementsPerTile ,
                  unsigned grainSize);


void
mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t);

}

#endif // __popstd_TileMapping_hpp__
