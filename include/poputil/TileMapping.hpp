// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file TileMapping.hpp
 *
 * Functions for handling the mapping of tensors to tiles.
 *
 */

#ifndef poputil_TileMapping_hpp
#define poputil_TileMapping_hpp
#include "poplar/DebugContext.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"
#include <vector>

namespace poputil {

/** Calculate a tile mapping that spreads the tensor evenly over the tiles
 * in a graph.
 *
 * The indices of the flattened tensor are mapped from low to high tile
 * numbers.
 *
 * \param graph     The graph to calculate the mapping for.
 * \param shape     The shape of the tensor to be mapped: a vector containing
 *                  the size of each dimension of the tensor.
 * \param minElementsPerTile
 *                  The minimum number of tensor elements to be allocated to a
 *                  tile.
 * \param grainSize The number of elements mapped to each tile will be an
 *                  integer multiple of the grain size.
 *
 * \returns A vector containing
 */
std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph,
                      std::vector<std::size_t> shape,
                      unsigned minElementsPerTile, unsigned grainSize);

/** Calculate a tile mapping that spreads the tensor evenly over the tiles in a
 * graph.
 *
 * The indices of the flattened tensor are mapped from low to high tile numbers.
 *
 * In this case the elements are distributed so that groups of elements of the
 * device's natural vector width will not be split. It effectively sets the
 * grain size to the natural vector width for the data type. This means the
 * number of elements on each tile will be a multiple of the natural vector
 * width and the index of the first element is aligned to the natural vector
 * width.
 *
 * The natural vector width is the largest vector width supported in hardware
 * for arithmetic operations on that data type.
 *
 * It will also try to keep at least 128 bytes of data on each tile to avoid
 * high exchange costs.
 *
 * \param graph     The graph to add the operation to.
 * \param shape     The tensor to be mapped.
 */
std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph, const poplar::Tensor &t);

void mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t,
                       unsigned minElementsPerTile, unsigned grainSize);

void mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t);

/** Determine how unbalanced a tensor is when mapped over tiles in a graph.
 *
 * This reports how well a tensor mapping compares with the mapping
 * based on a given number of elements per tile.
 *
 *  \param mapping   The current tile mapping of the tensor.
 *  \param minElementsPerTile
 *                   The suggested minimum number of elements per tile.
 * \param grainSize  The number of elements mapped to each tile would be an
 *                   integer multiple of the suggested grain size.
 *
 *  \returns The maximum number of elements greater than expected on any tile.
 */
unsigned getTileImbalance(const poplar::Graph::TileToTensorMapping &mapping,
                          unsigned minElementsPerTile = 0,
                          unsigned grainSize = 1);

/** Determine how unbalanced a tensor is mapped over tiles.
 *
 * This compares the way a tensor is mapped to a set of tiles to the mapping
 * based on a given number of elements per tile.
 *
 *  \param graph     The graph containing the mapped tensor.
 *  \param mapping   The tensor currently mapped to tiles in the graph.
 *  \param minElementsPerTile
 *                   The suggested minimum number of elements per tile.
 * \param grainSize  The number of elements mapped to each tile would be an
 *                   integer multiple of the suggested grain size.
 *
 *  \returns The maximum number of elements greater than expected on any tile.
 */
unsigned getTileImbalance(const poplar::Graph &graph, const poplar::Tensor &t,
                          unsigned minElementsPerTile = 0,
                          unsigned grainSize = 1);

class TensorUseTrackerState;

/** Class that tracks the usage of data on different tiles.
 *
 *  If data is broadcast to many tiles, it is sometimes efficient to map the
 *  data so that it is spread evenly amongst the tiles that use it.
 *
 *  This class can collect information about the use of data and then calculate
 *  a suitable tile mapping.
 */
class TensorUseTracker {
  std::unique_ptr<TensorUseTrackerState> st;

public:
  TensorUseTracker(unsigned numTiles);
  TensorUseTracker(const TensorUseTracker &other);
  TensorUseTracker(TensorUseTracker &&other);
  TensorUseTracker &operator=(const TensorUseTracker &other);
  TensorUseTracker &operator=(TensorUseTracker &&other);
  enum class MappingMethod {
    /// Map "halo regions" to single tiles. These are regions that are used by
    /// multiple tiles but have neighbouring regions used by subsets of those
    /// tiles.
    OptimizeHaloRegions,

    /// Mapping of elements is constrained to be only on tiles that use them.
    /// Otherwise, to meet grain size constraints, elements may be mapped to
    /// tiles which do not use them.
    ConstrainMappingToUsedTiles,

    /// No mapping method used.
    None
  };
  ~TensorUseTracker();

  /** Add a data use case.
   *
   *  \param graph  The Poplar graph being tracked.
   *  \param tile   The tile that the use occurs on.
   *  \param t      The tensor representing the data being used.
   */
  void add(const poplar::Graph &graph, unsigned tile, const poplar::Tensor &t);

  /** Add data use cases from another tracker.
   *
   *  \param other The \c TensorUseTracker to merge data use information from.
   */
  void add(TensorUseTracker other);

  /** Resolve data uses for mapping. Data used on multiple tiles
   *  will have their uses spread across those tiles.
   *
   *  \param graph                The Poplar graph being tracked.
   *  \param grainSize            The number of elements mapped to each tile
   *                              will be an integer multiple of the grain size.
   *  \param minElementsPerTile   The minimum number of elements that must be
   *                              mapped to a tile.
   *  \param extendPartialUsage   When set, partial uses of tensors will be
   *                              extended to cover the entire tensor, based
   *                              on the usage of neighbouring regions.
   *  \param mappingMethod        Method used for mapping elements.
   */
  void resolve(const poplar::Graph &graph, unsigned grainSize,
               unsigned minElementsPerTile, bool extendPartialUsage = false,
               TensorUseTracker::MappingMethod mappingMethod =
                   TensorUseTracker::MappingMethod::None);

  /** Map data according to use.
   *
   *  This function will set the tile mapping of variable regions based on
   *  tracked data uses. Variable regions with uses on multiple tiles will have
   *  their elements spread across those tiles.
   *
   *  \param graph                The Poplar graph being tracked.
   *  \param grainSize            The number of elements mapped to each tile
   *                              will be an integer multiple of the grain size.
   *  \param minElementsPerTile   The minimum number of elements that must be
   *                              mapped to a tile.
   *  \param extendPartialUsage   When set, partial uses of tensors will be
   *                              extended to cover the entire tensor, based
   *                              on the usage of neighbouring regions before
   *                              mapping.
   *  \param mappingMethod        Method used for mapping elements.
   */
  void mapTensorsByUse(poplar::Graph &graph, unsigned grainSize,
                       unsigned minElementsPerTile,
                       bool extendPartialUsage = false,
                       TensorUseTracker::MappingMethod mappingMethod =
                           TensorUseTracker::MappingMethod::None);

  /** Have any use cases been registered.
   * \return True if no data use cases, false otherwise
   */
  bool empty() const;
};

/** Create a clone of the specified tensor on the specified IPU.
 *
 * The cloned tensor is mapped to the IPU in such a way that the mapping of
 * tensor elements to tiles is preserved.
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
           const poplar::DebugContext &debugContext = {},
           poplar::TensorCloneMethod method =
               poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/** Move a tensor from one IPU to another.
 *
 * The tensor is moved by duplicating it, mapping the clone onto another IPU,
 *  and copying the original tensor values to the new one.
 *
 * \param masterGraph The graph representing the entire multi-IPU device.
 * \param t           The tensor to move from one IPU to another.
 * \param prog        A program sequence to add the Copy to.
 * \param dstIPU      The index of the IPU onto which the tensor will be moved.
 * \param debugContext  A debug name to give to the tensor created on dstIPU.
 *                    If this is empty then the debug names will be derived from
 *                    existing tensor debug names.
 * \param method      The method to use for cloning of the tensor on the
 *                    destination IPU.
 * \return The new tensor on the specified IPU.
 */
poplar::Tensor
copyToIpu(poplar::Graph &masterGraph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, unsigned dstIPU,
          const poplar::DebugContext &debugContext = {},
          poplar::TensorCloneMethod method =
              poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/** Prepare to move a tensor from one IPU to another.
 *
 * The tensor is duplicated and the clone is mapped onto another IPU. References
 * to source and destination tensors are provided for use by an inter-IPU copy.
 *
 * The necessary copy operation is **not** added to the program.
 *
 * \param masterGraph The graph representing the entire multi-IPU device.
 * \param t           The tensor to move from one IPU to another.
 * \param dstIPU      The index of the IPU onto which the tensor will be moved.
 * \param copySrc     A tensor that can be used as the source to do the copy.
 * \param copyDst     A tensor that can be used as the destination of the copy.
 * \param debugContext  A debug name to give to the tensor created on dstIPU.
 *                    If this is empty then the debug names will be derived from
 *                    existing tensor debug names.
 * \param method      The method to use for cloning of the tensor on the
 *                    destination IPU.
 * \return The new tensor on the specified IPU.
 */
poplar::Tensor
createIpuCopy(poplar::Graph &graph, const poplar::Tensor &t, unsigned dstIpu,
              poplar::Tensor &copySrc, poplar::Tensor &copyDst,
              const poplar::DebugContext &debugContext = {},
              poplar::TensorCloneMethod method =
                  poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

/** Check if a dimension of a tensor is split over more than one tile.
 *
 *  Examines the mapping of the specified tensor to see if the specified
 *  dimension is split over more than one tile.
 *
 * \param graph     The graph to examine.
 * \param t         The tensor to check.
 * \param dimension The dimension to check.
 *
 * \returns True if elements of the given dimension are spread over more than
 *          one tile.
 */
bool dimIsSplitOverTiles(const poplar::Graph &graph, const poplar::Tensor &t,
                         unsigned dimension);

/** Check if a dimension of a tensor is split over more than one IPU.
 *
 *  Examines the mapping of the specified tensor to see if the specified
 *  dimension is split over more than one IPU.
 *
 * \param graph     The graph to examine.
 * \param t         The tensor to check.
 * \param dimension The dimension to check.
 *
 * \returns True if elements of the given dimension are spread over more than
 *          one IPU.
 */
bool dimIsSplitOverIPUs(const poplar::Graph &graph, const poplar::Tensor &t,
                        unsigned dimension);

/** Create a simpler tensor that is mapped in the same way as another, full,
 * tensor.
 *
 * The full tensor is typically a left hand side operand of an operation while
 * the created tensor is the right hand side. The created tensor has one
 * dimension, which is the same size as the specified dimension of the full
 * tensor.
 *
 * Because the created tensor has the same mapping as the full tensor, it
 * reduces the amount of data exchange or copies that are required for an
 * operation using the two tensors.
 *
 * \param graph       The graph which the output tensor is added to.
 * \param fullTensor  The tensor mapping for the output tensor is copied from
 *                    this tensor.
 * \param type        The type of the output tensor.
 * \param dim         The dimension of the input tensor which is the
 *                    size of the created tensor.
 * \param ditherMapping Enable dithering to be applied to the mapping of the
 *                    output tensor.
 * \param debugContext Optional debug information.
 *
 * \returns           The created output tensor.
 */
poplar::Tensor
createBroadcastOperand(poplar::Graph &graph, const poplar::Tensor &fullTensor,
                       const poplar::Type &type, unsigned dim,
                       bool ditherMapping = false,
                       const poplar::DebugContext &debugContext = {});

} // namespace poputil

#endif // poputil_TileMapping_hpp
