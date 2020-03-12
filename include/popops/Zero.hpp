// Copyright (c) 2016 Graphcore Ltd. All rights reserved.

#ifndef popops_Zero_hpp
#define popops_Zero_hpp

#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include <vector>

namespace popops {

/** Appends vertices to \p zeroCS which zeroes elements in \p tileRegions of
 * \p t which reside on tile \p tile.
 *
 * \param graph         The graph that the operation will be added to.
 * \param t             The tensor whose elements are to be set to zero.
 * \param tileRegions   Region mapping of the tensor on \p tile.
 * \param tile          Tile which the regions relate to.
 * \param zeroCS        Compute set to add the operation into.
 */
void zero(poplar::Graph &graph, poplar::Tensor t,
          const std::vector<poplar::Interval> &tileRegions, unsigned tile,
          poplar::ComputeSet zeroCS);

/** Appends vertices to \p zeroCS which zeroes all elements of \p t
 * which reside on tile \p tile.
 *
 * \param graph         The graph that the operation will be added to.
 * \param t             The tensor whose elements are to be set to zero.
 * \param tile          Tile on which the tensor is mapped to.
 * \param zeroCS        Compute set to add the operation into.
 */
void zero(poplar::Graph &graph, const poplar::Tensor &t, unsigned tile,
          poplar::ComputeSet zeroCS);

/** Appends vertices to \p zeroCS which zeroes elements in \p mapping of \p t
 * which reside on tiles represented with \p mapping.
 *
 * \param graph         The graph that the operation will be added to.
 * \param t             The tensor whose elements are to be set to zero.
 * \param mapping       The tensor's region mapping per tile. Each element
 *                      describes a region mapping of a tile (ordered).
 *                      i.e. mapping[0] -> tile 0's region mapping for \p t.
 * \param zeroCS        Compute set to add the operation into.
 */
void zero(poplar::Graph &graph, const poplar::Tensor &t,
          const std::vector<std::vector<poplar::Interval>> &mapping,
          poplar::ComputeSet zeroCS);

/** Appends programs to \p prog which zeroes all elements of the Tensor \p t
 *
 * \param graph         The graph that the operation will be added to.
 * \param t             The tensor whose elements are to be set to zero.
 * \param prog          Poplar program sequence to append the operation onto.
 * \param debugPrefix   Name of the operation, for debugging.
 */
void zero(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_Zero_hpp
