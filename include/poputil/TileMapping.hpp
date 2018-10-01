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


/** Determine how unbalanced a tensor is mapped over tiles
 *
 *  \param mapping The tile mapping of the tensor
 *  \param minElementsPerTile The expected minimum number of elements per tile.
 *  \param grainSize The expected "grain size" i.e. atomic grains used to
 *                   split of elements over tiles
 *
 *  \returns The maximum number of elements over expected on any tile.
 */
unsigned
getTileImbalance(const poplar::Graph::TileToTensorMapping &mapping,
                 unsigned minElementsPerTile = 0, unsigned grainSize = 1);


/** Determine how unbalanced a tensor is mapped over tiles
 *
 *  \param graph The graph.
 *  \param t The tensor to be inspected.
 *  \param minElementsPerTile The expected minimum number of elements per tile.
 *  \param grainSize The expected "grain size" i.e. atomic grains used to
 *                   split of elements over tiles
 *
 *  \returns The maximum number of elements over expected on any tile.
 */
unsigned
getTileImbalance(const poplar::Graph &graph, const poplar::Tensor &t,
                 unsigned minElementsPerTile = 0, unsigned grainSize = 1);

/** Update a tensor's tile mapping to be balanced over tiles
 *
 *  \param graph The graph to which the tensor belongs.
 *  \param t The tensor to rebalance.
 *  \param minElementsPerTile The minimum number of elements per tile.
 *  \param grainSize The "grain size" i.e. atomic grains used to
 *                   split of elements over tiles.
 *  \param imbalanceThreshold This value is checked against the current
 *                            tensor tile imbalance and if the imbalance
 *                            is less than this value, the tile mapping
 *                            will not be altered.
 */
void
rebalanceTensor(poplar::Graph &graph, const poplar::Tensor &t,
                unsigned minElementsPerTile, unsigned grainSize,
                unsigned imbalanceThreshold);


/** Update a tensor's tile mapping to be balanced over tiles
 *
 *  \param graph The graph to which the tensor belongs.
 *  \param t The tensor to rebalance.
 */
void rebalanceTensor(poplar::Graph &graph, const poplar::Tensor &t);


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
