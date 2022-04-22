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
 * By default the indices of the resulting mapping go from from low to high
 * tile numbers, however offset and direction can be specified.
 *
 * \param graph     The graph to calculate the mapping for.
 * \param shape     The shape of the tensor to be mapped: a vector containing
 *                  the size of each dimension of the tensor.
 * \param minElementsPerTile
 *                  The minimum number of tensor elements to be allocated to a
 *                  tile.
 * \param grainSize The number of elements mapped to each tile will be an
 *                  integer multiple of the grain size.
 * \param offset    The offset to the first tile used for mapping
 * \param ascendingOrder
 *                  If true, the first tile used = offset and tiles are
 *                  allocated in increasing order
 *                  If false, the
 *                  first tile used = (number of device tiles -1 - offset) and
 *                  tiles are allocated in decreasing order
 *
 * \returns A vector specifying the mapping
 */
std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph,
                      std::vector<std::size_t> shape,
                      unsigned minElementsPerTile, unsigned grainSize,
                      unsigned offset = 0, bool ascendingOrder = true);

/** Calculate a tile mapping that spreads the tensor evenly over the tiles in a
 * graph.
 *
 * By default the indices of the resulting mapping go from from low to high
 * tile numbers, however offset and direction can be specified.
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
 * \param t         The tensor to be mapped
 * \param offset    The offset to the first tile used for mapping
 * \param ascendingOrder
 *                  If true, the first tile used = offset and tiles are
 *                  allocated in increasing order
 *                  If false, the
 *                  first tile used = (number of device tiles - 1 - offset) and
 *                  tiles are allocated in decreasing order
 *
 * \returns A vector specifying the mapping
 */
std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph, const poplar::Tensor &t,
                      unsigned offset = 0, bool ascendingOrder = true);

/** Calculate a tile mapping that spreads the tensor evenly over the tiles in a
 * graph.
 *
 * This function is similar to `poputil::calcLinearTileMapping` but with an
 * additional "new offset" output equal to the last plus one tile used for the
 * mapping. For example, consider a target with 8 tiles and a resulting mapping
 * over 4 tiles. The value of the returned offset will be:
 *   - 6 if `offset = 2`.
 *   - 2 if `offset = 6`.
 *
 * \param graph     The graph to add the operation to.
 * \param t         The tensor to be mapped
 * \param offset    The offset to the first tile used for mapping
 *
 * \returns         A pair consisting of a vector specifying the mapping and the
 *                  new advanced offset.
 */
std::pair<poplar::Graph::TileToTensorMapping, unsigned>
calcLinearTileMappingAndNewOffset(const poplar::Graph &graph,
                                  const poplar::Tensor &t, unsigned offset = 0);

/** Map the specified tensor, spreading the tensor evenly over the tiles
 * in a graph.
 *
 * The indices of the flattened tensor are mapped from low to high
 * tile numbers.
 *
 * \param graph     The graph to calculate the mapping for.
 * \param t         The tensor to be mapped.
 * \param minElementsPerTile
 *                  The minimum number of tensor elements to be allocated to a
 *                  tile.
 * \param grainSize The number of elements mapped to each tile will be an
 *                  integer multiple of the grain size.
 */
void mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t,
                       unsigned minElementsPerTile, unsigned grainSize);

/** Map the specified tensor, spreading the tensor evenly over the tiles
 * in a graph.
 *
 * The indices of the flattened tensor are mapped from low to high
 * tile numbers, however offset and direction can be specified.
 *
 * \param graph     The graph to calculate the mapping for.
 * \param t         The tensor to be mapped.
 * \param minElementsPerTile
 *                  The minimum number of tensor elements to be allocated to a
 *                  tile.
 * \param grainSize The number of elements mapped to each tile will be an
 *                  integer multiple of the grain size.
 * \param offset    The offset to the first tile used for mapping
 * \param ascendingOrder
 *                  If true, the first tile used = offset and tiles are
 *                  allocated in increasing order.
 *                  If false, the
 *                  first tile used = (number of device tiles -1 - offset) and
 *                  tiles are allocated in decreasing order.
 */
void mapTensorLinearlyWithOffset(poplar::Graph &graph, const poplar::Tensor &t,
                                 unsigned minElementsPerTile,
                                 unsigned grainSize, unsigned offset,
                                 bool ascendingOrder = true);

/** Map the specified tensor, spreading the tensor evenly over the tiles
 * in a graph.
 *
 * The indices of the flattened tensor are mapped from low to high
 * tile numbers.
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
 * \param t         The tensor to be mapped.
 */
void mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t);

/** Map the specified tensor, spreading the tensor evenly over the tiles
 * in a graph.
 *
 * The indices of the flattened tensor are mapped from low to high
 * tile numbers, however offset and direction can be specified.
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
 * \param graph     The graph to calculate the mapping for.
 * \param t         The tensor to be mapped.
 * \param offset    The offset to the first tile used for mapping.
 * \param ascendingOrder
 *                  If true, the first tile used = offset and tiles are
 *                  allocated in increasing order.
 *                  If false, the
 *                  first tile used = (number of device tiles -1 - offset) and
 *                  tiles are allocated in decreasing order.
 */
void mapTensorLinearlyWithOffset(poplar::Graph &graph, const poplar::Tensor &t,
                                 unsigned offset = 0,
                                 bool ascendingOrder = true);

/** Choose an offset for use with tensor mapping functions using a hash of the
 * shape provided.
 *
 *  \param numTiles      The number of tiles of the intended target device.
 *  \param shape         The shape to produce a hash of.
 *
 *  \returns             The selected offset in the range 0 to numTiles - 1
 **/
std::size_t chooseMappingOffset(std::size_t numTiles,
                                const std::vector<std::size_t> &shape);

/** Choose an offset for use with tensor mapping functions using a hash of the
 * shape, and a seed.
 *
 *  \param numTiles      The number of tiles of the intended target device.
 *  \param shape         The shape to produce a hash of.
 *  \param seed          Optional seed to use in producing the hash.
 *
 *  \returns             The selected offset in the range 0 to numTiles - 1
 **/
std::size_t chooseMappingOffset(std::size_t numTiles,
                                const std::vector<std::size_t> &shape,
                                std::size_t seed);

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
  /** Constructor for the TensorUseTracker class.
   *
   *  \param numTiles  The number of tiles to track data use of.
   *  \param startTile The tile to start tracking data use on.
   *  \param ascendingMappingOrder
   *                   If true, the first tile used = startTile and tiles are
   *                   allocated in increasing order.
   *                   If false, the first tile used = (number of device tiles
   * -1
   *                   - startTile) and tiles are allocated in decreasing order.
   */
  TensorUseTracker(unsigned numTiles, unsigned startTile = 0,
                   bool ascendingMappingOrder = true);

  /** Constructor for the TensorUseTracker class.
   */
  TensorUseTracker(const TensorUseTracker &other);

  /** Default constructor for the TensorUseTracker class.
   */
  TensorUseTracker(TensorUseTracker &&other);

  /** Assignment operator for the TensorUseTracker class.
   */
  TensorUseTracker &operator=(const TensorUseTracker &other);

  /** Default assignment operator for the TensorUseTracker class.
   */
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

  /** Destructor for the TensorUserTracker class
   */
  ~TensorUseTracker();

  /** Add a case of data usage.
   *
   *  \param graph  The Poplar graph being tracked.
   *  \param tile   The tile that the use occurs on.
   *  \param t      The tensor representing the data being used.
   */
  void add(const poplar::Graph &graph, unsigned tile, const poplar::Tensor &t);

  /** Add cases of data usage from another tracker.
   *
   *  \param other The \c TensorUseTracker to merge data usage information from.
   */
  void add(TensorUseTracker other);

  /** Resolve data usage for mapping.
   *
   *  Data used on multiple tiles will have their usage spread across those
   *  tiles.
   *
   *  \param graph                The Poplar graph being tracked.
   *  \param grainSize            The number of elements mapped to each tile
   *                              will be an integer multiple of the grain size.
   *  \param minElementsPerTile   The minimum number of elements that must be
   *                              mapped to a tile.
   *  \param extendPartialUsage   When set, partial usage of tensors will be
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
   *  tracked data use. Variable regions with usage on multiple tiles will have
   *  their elements spread across those tiles.
   *
   *  \param graph                The Poplar graph being tracked.
   *  \param grainSize            The number of elements mapped to each tile
   *                              will be an integer multiple of the grain size.
   *  \param minElementsPerTile   The minimum number of elements that must be
   *                              mapped to a tile.
   *  \param extendPartialUsage   When set, partial use of tensors will be
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

  /** Check if any cases of data usage have been registered.
  *
   * \return If true, no cases have been registered. If false,
             cases have been registered.
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

/** Create a clone of the specified tensor on the specified graph.
 *
 * The cloned tensor is mapped to the destination graph in such a way that the
 * mapping of tensor elements to tiles is preserved.
 *
 * \param srcGraph     The graph representing the source tiles.
 * \param dstGraph     The graph representing the destination tiles.
 * \param t            The tensor to clone.
 * \param debugContext Optional debug information
 * \param method       The method to use for the cloning.
 * \return The cloned tensor.
 *
 * \note It is assumed that the destination graph has enough tiles to clone the
 * input tensor. This includes any gaps in the tile mapping. This means the
 * maximum mapped tile of `t` in the source graph must be less than
 * `dstGraph.getTarget().getNumTiles()`.
 */
poplar::Tensor
cloneToGraph(poplar::Graph &srcGraph, poplar::Graph &dstGraph,
             const poplar::Tensor &t,
             const poplar::DebugContext &debugContext = {},
             poplar::TensorCloneMethod method =
                 poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/** Create a clone of the specified tensor on the specified graph.
 *
 * The cloned tensor is mapped to the graph in such a way that the mapping of
 * tensor elements to tiles is preserved. If the source tensor consists of
 * aliasing intervals, these will be made non-aliasing in the cloned tensor and
 * mapped linearly accross the tiles with the specified tile offset. The
 * remapping is done as a precautionary measure to reduce the chance of getting
 * out of memory issues on a tile which has many aliasing elements.
 *
 * In addition to the cloned tensor, this function returns "new offset" output
 * equal to the last plus one tile used for the mapping of the expanded aliasing
 * elements. See `poputil::calcLinearTileMappingAndNewOffset` for more details.
 *
 * \param graph        The graph to add the operation to.
 * \param t            The tensor to clone.
 * \param offset       The offset to the first tile used for mapping the
 *                     elements of the resulting tensor corresponding to
 *                     aliasing elements of the source tensor.
 * \param debugContext Optional debug information
 *
 * \returns            A pair consisting of the cloned tensor and the new
 *                     advanced offset.
 */
std::pair<poplar::Tensor, unsigned>
cloneAndExpandAliasing(poplar::Graph &graph, const poplar::Tensor &t,
                       unsigned offset = 0,
                       const poplar::DebugContext &debugContext = {});

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

/** Transform a tile index such that the result begins at zero and increments
 *
 * \param tile      The tile number to transform.
 * \param numTiles  The number of tiles on the target device.
 * \param offset    The offset to the first tile used for the mapping before the
 *                  transform takes place.
 * \param ascendingOrder
 *                  Mapping order before the transform takes place:
 *                  If true, the first tile used = offset and tiles are
 *                  allocated in increasing order.
 *                  If false, the
 *                  first tile used = (number of device tiles -1 - offset) and
 *                  tiles are allocated in decreasing order.
 *
 * \returns Transformed tile number.
 */

unsigned transformTileIndex(unsigned tile, unsigned numTiles, unsigned offset,
                            bool ascending);

/** Transform a tile index such that the result begins at an offset and
 *  increments or decrements.
 *
 * \param tile      The tile number to transform.
 * \param numTiles  The number of tiles on the target device.
 * \param offset    The offset to the first tile used for the mapping after the
 *                  transform takes place.
 * \param ascendingOrder
 *                  Mapping order after the transform takes place:
 *                  If true, the first tile used = offset and tiles are
 *                  allocated in increasing order.
 *                  If false, the
 *                  first tile used = (number of device tiles -1 - offset) and
 *                  tiles are allocated in decreasing order.
 *
 * \returns Transformed tile number
 */
unsigned invTransformTileIndex(unsigned tile, unsigned numTiles,
                               unsigned offset, bool ascending);

} // namespace poputil

#endif // poputil_TileMapping_hpp
