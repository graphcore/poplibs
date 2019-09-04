// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poputil_TileMapping_hpp
#define poputil_TileMapping_hpp
#include <vector>
#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"

namespace poputil {

/* Calculate a tile mapping that spreads the tensor
 * evenly over the tiles in a linear manner (i.e. with the
 * indices of the flattened tensor mapped across from low -> high tile
 * numbers).
 */
std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph,
                      std::vector<std::size_t> shape,
                      unsigned minElementsPerTile,
                      unsigned grainSize);

/* Calculate a tile mapping that spreads the tensor
 * evenly over the tiles in a linear manner (i.e. with the
 * indices of the flattened tensor mapped across from low -> high tile
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
  TensorUseTracker(const TensorUseTracker &other);
  TensorUseTracker(TensorUseTracker &&other);
  TensorUseTracker &operator=(const TensorUseTracker &other);
  TensorUseTracker &operator=(TensorUseTracker &&other);
  ~TensorUseTracker();
  /** Add a data use case.
   *
   *  \param graph  The Poplar graph
   *  \param tile   The tile that the use occurs on.
   *  \param t      The tensor representing the data being used.
   */
  void add(const poplar::Graph &graph, unsigned tile, const poplar::Tensor &t);

  /** Add data use cases from another tracker.
   *
   *  \param other The TensorUseTracker from which to merge data uses.
   */
  void add(TensorUseTracker other);

  /** Resolve data uses for mapping. Data used on multiple tiles
   *  will have their uses spread across those tiles.
   *
   *  \param grainSize            The number of elements that cannot be split
   *                              amongst tiles.
   *  \param minElementsPerTile   The minimum number of elements that must be
   *                              mapped to a tile.
   *  \param optimizeHaloRegions  Map "halo regions" to single tiles. These are
   *                              regions that are used by multiple tiles but
   *                              have neighbouring regions used by subsets of
   *                              those tiles.
   *  \param extendPartialUsage   When set, partial uses of tensors will be
   *                              extended to cover the entire tensor, based
   *                              on the usage of neighbouring regions.
   */
  void resolve(const poplar::Graph &graph,
               unsigned grainSize,
               unsigned minElementsPerTile,
               bool optimizeHaloRegions = false,
               bool extendPartialUsage = false);

  /** Map data according to use.
   *
   *  This function will set the tile mapping of variable regions based on
   *  tracked data uses. Variable regions with uses on multiple tiles will have
   *  their elements spread across those tiles.
   *
   *  \param graph                The Poplar graph
   *  \param grainSize            The number of elements that cannot be split
   *                              amongst tiles.
   *  \param minElementsPerTile   The minimum number of elements that must be
   *                              mapped to a tile.
   *  \param optimizeHaloRegions  Map "halo regions" to single tiles. These are
   *                              regions that are used by multiple tiles but
   *                              have neighbouring regions used by subsets of
   *                              those tiles.
   *  \param extendPartialUsage   When set, partial uses of tensors will be
   *                              extended to cover the entire tensor, based
   *                              on the usage of neighbouring regions before
   *                              mapping.
   */
  void mapTensorsByUse(poplar::Graph &graph,
                       unsigned grainSize,
                       unsigned minElementsPerTile,
                       bool optimizeHaloRegions = false,
                       bool extendPartialUsage = false);

  /** Have any use cases been registered.
   * \return True if no data use cases, false otherwise
   */
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
 * \param name A debug name to give to the tensor created on dstIPU.
 *             If this is empty then the debug names will be derived from
 *             existing tensor debug names.
 * \param method The method to use for cloning of the tensor on the destination
 *               IPU.
 * \return The new tensor on the specified IPU.
 */
poplar::Tensor
copyToIpu(poplar::Graph& masterGraph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, unsigned dstIPU,
          poplar::StringRef name = "",
          poplar::TensorCloneMethod method =
                      poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/** Move a tensor from one IPU to another by duplicating it, mapping the clone
 *  onto another IPU, and provide the src/dsts tensors of an inter-IPU copy
 * (but to not add that copy to a program at this point).
 *
 * \param masterGraph The graph representing the entire multi-IPU device.
 * \param t The tensor to move from one IPU to another.
 * \param dstIPU The index of the IPU onto which the Tensor will be moved.
 * \param copySrc A tensor that can be used as the source to do the copy
 * \param copyDst A tensor that can be used as the dest to do the copy
 * \param name A debug name to give to the tensor created on dstIPU.
 *             If this is empty then the debug names will be derived from
 *             existing tensor debug names.
 * \param method The method to use for cloning of the tensor on the destination
 *               IPU.
 * \return The new tensor on the specified IPU.
 */
poplar::Tensor
createIpuCopy(poplar::Graph &graph,
              const poplar::Tensor &t,
              unsigned dstIpu,
              poplar::Tensor &copySrc,
              poplar::Tensor &copyDst,
              poplar::StringRef name = "",
              poplar::TensorCloneMethod method =
                poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);


/** Check if the tile mapping of the given tensor is or isn't such that
 *  the given dimension is split over more than 1 IPU.
 *
 * \param graph     The graph to introspect.
 * \param t         The tensor to introspect.
 * \param dimension The dimension to check.
 *
 * \returns true if any slice of the given dimension is spread over more than
 *          one IPU.
 */
bool
dimIsSplitOverIPUs(const poplar::Graph &graph,
                   const poplar::Tensor &t,
                   unsigned dimension);

// Returns a list with the innermost grouped dimension first
// moving outwards, with groupings for each. The same dimension may appear
// more than once. This uses detectInnermostGrouping iteratively.
using GroupingInfo = std::pair<unsigned, unsigned>;
std::vector<GroupingInfo>
detectDimGroupings(const poplar::Graph &graph, const poplar::Tensor &t);

unsigned detectInnermostGrouping(const poplar::Graph &graph,
                               const poplar::Tensor &t0);

}

#endif // poputil_TileMapping_hpp
