// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Set elements of tensor to zero.
 *
 */

#ifndef popops_Zero_hpp
#define popops_Zero_hpp

#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include <vector>

namespace popops {

/** Append vertices to compute set \p zeroCS which zeroes elements in \p
 * tileRegions of tensor \p t which reside on \p tile.
 *
 * \param graph         The graph that the operation will be added to.
 * \param t             The tensor whose elements are to be set to zero.
 * \param tileRegions   The region mapping of the tensor on \p tile.
 * \param tile          The tile which the regions relate to.
 * \param zeroCS        The compute set to add the operation to.
 */
void zero(poplar::Graph &graph, poplar::Tensor t,
          const std::vector<poplar::Interval> &tileRegions, unsigned tile,
          poplar::ComputeSet zeroCS);

/** Append vertices to compute set \p zeroCS which zeroes all elements of
 * tensor \p t which reside on \p tile.
 *
 * \param graph         The graph that the operation will be added to.
 * \param t             The tensor whose elements are to be set to zero.
 * \param tile          The tile which the tensor is mapped to.
 * \param zeroCS        The compute set to add the operation to.
 */
void zero(poplar::Graph &graph, const poplar::Tensor &t, unsigned tile,
          poplar::ComputeSet zeroCS);

/** Append vertices to compute set \p zeroCS which zeroes elements in
 * \p mapping of tensor \p t which reside on tiles represented with \p mapping.
 *
 * \param graph         The graph that the operation will be added to.
 * \param t             The tensor whose elements are to be set to zero.
 * \param mapping       The tensor's region mapping per tile. Each element
 *                      describes a region mapping of a tile (ordered).
 *                      For example, mapping[0] is tile 0's region mapping for
 *                      tensor \p t.
 * \param zeroCS        The compute set to add the operation to.
 */
void zero(poplar::Graph &graph, const poplar::Tensor &t,
          const std::vector<std::vector<poplar::Interval>> &mapping,
          poplar::ComputeSet zeroCS);

/** Append programs to program sequence \p prog which zeroes all elements of
 * tensor \p t
 *
 * \param graph         The graph that the operation will be added to.
 * \param t             The tensor whose elements are to be set to zero.
 * \param prog          The Poplar program sequence to append the operation to.
 * \param debugContext  Optional debug information.
 */
void zero(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog,
          const poplar::DebugContext &debugContext = {});

} // namespace popops

#endif // popops_Zero_hpp
