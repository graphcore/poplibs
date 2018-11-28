// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poputil_TileMapping_hpp
#define poputil_TileMapping_hpp
#include <vector>
#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"

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

/** Update a tensor's tile mapping such that when it is used as the
 *  output of an element-wise operation (operation with no dependency
 *  between more than one element of the output and any given element
 *  of any input tensor).
 *
 *  Use the resulting tensor to map element-wise operations to tiles
 *  to produce an operation that is computationally balanced across tiles
 *  and which minimises exchange.
 *
 *  \param graph            A graph which the given inputs/output belong to.
 *  \param inputs           List of input tensors for the operation.
 *  \param output           Output tensor for the operation.
 *  \param grainSize        Grain-size for elements mapped to each tile.
 *  \param minGrainsPerTile Minimum no. of grains mapped to a tile.
 */
void mapOutputForElementWiseOp(
    poplar::Graph &graph,
    const std::vector<poplar::Tensor> &inputs,
    const poplar::Tensor &output,
    unsigned grainSize = 1,
    unsigned minGrainsPerTile = 0);

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

/**
 * Create a clone of the specified tensor. Elements of the cloned tensor
 * are mapped to the specified IPU such the index of the tile an element is
 * mapped to within an IPU is preserved.
 *
 * \param graph   The graph representing the entire multi-IPU device.
 * \param t       The tensor to clone.
 * \param dstIPU  The index of the IPU to clone the tensor onto.
 * \param name    A debug name to give to any new tensors allocated in the graph
 *                during the clone. If this is empty then the debug names will
 *                be derived from existing tensor debug names.
 * \param method  The method to use for the cloning.
 * \return The cloned tensor.
 */
poplar::Tensor
cloneToIpu(poplar::Graph &graph, const poplar::Tensor &t, unsigned dstIPU,
           poplar::StringRef name = "",
           poplar::TensorCloneMethod method =
               poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/** Move a tensor from one IPU to another by duplicating it, mapping the clone
 *  onto another IPU, and copying the original to the new one.
 *
 * \param masterGraph The graph representing the entire multi-IPU device.
 * \param t The tensor to move from one IPU to another.
 * \param prog A program sequence to which the Copy will be added.
 * \param dstIPU The index of the IPU onto which the Tensor will be moved.
 * \return The new tensor on the specified IPU.
 */
poplar::Tensor copyToIpu(poplar::Graph& masterGraph, const poplar::Tensor &t,
                         poplar::program::Sequence &prog, unsigned dstIPU);

}

#endif // poputil_TileMapping_hpp
