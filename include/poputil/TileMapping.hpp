// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poputil_TileMapping_hpp
#define poputil_TileMapping_hpp
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

class TensorUseTrackerState;

/** Class that tracks the usage of data on different tiles.
 *
 *  If data is broadcast to many tiles, it is sometimes efficient to
 *  map the data to be spread evenly amongst the tiles that use it.
 *
 *  This class can collect uses of data and then calculate such a tile
 *  mapping.
 */
class TensorUseTracker {
  std::unique_ptr<TensorUseTrackerState> st;
public:
  TensorUseTracker(unsigned numTiles);
  ~TensorUseTracker();
  /** Add a data use case.
   *
   *  \param graph  The poplar graph
   *  \param tile   The tile that the use occurs on.
   *  \param t      The tensor representing the data being used.
   */
  void add(const poplar::Graph &graph, unsigned tile, const poplar::Tensor &t);

  /** Map data according to use.
   *
   *  This function will set the tile mapping of all the variables references
   *  by the use() method to be spread over the tiles that use them.
   *
   *  \param graph                The poplar graph
   *  \param grainSize            The number of elements that cannot be split
   *                              amongst tiles.
   *  \param minElemntsPerTile    The minimum number of elements that must be
   *                              mapped to a tile.
   *  \param optimizeHaloRegions  Map "halo regions" to single tiles. Halo
   *                              regions that are used by multiple tiles but
   *                              have neighbouring regions used by subsets of
   *                              those tiles.
   */
  void mapTensorsByUse(poplar::Graph &graph,
                       unsigned grainSize,
                       unsigned minElementsPerTile,
                       bool optimizeHaloRegions = false);

  /** Have any use cases by registered. */
  bool empty() const;
};

}

#endif // poputil_TileMapping_hpp
