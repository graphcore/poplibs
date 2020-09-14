// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions to fill tensors with values.
 *
 */

#ifndef popops_Fill_hpp
#define popops_Fill_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/TypeTraits.hpp>

namespace popops {

/** Appends vertices to \p fillCS which fills elements in \p tileRegions of
 *  \p t which reside on tile \p tile.
 *
 *  \param graph         The graph that the operation will be added to.
 *  \param t             The tensor whose elements are to be set to zero.
 *  \param tileRegions   Region mapping of the tensor on \p tile.
 *  \param tile          Tile which the regions relate to.
 *  \param fillCS        Compute set to add the operation into.
 *  \param fillValue     The value to fill \p t with.
 */
template <typename FillValueType>
void fill(poplar::Graph &graph, poplar::Tensor t,
          const std::vector<poplar::Interval> &tileRegions, unsigned tile,
          poplar::ComputeSet fillCS, FillValueType fillValue);

/** Appends vertices to \p fillCS which fills all elements of \p t
 *  which reside on tile \p tile.
 *
 *  \param graph         The graph that the operation will be added to.
 *  \param t             The tensor whose elements are to be set to zero.
 *  \param tile          Tile on which the tensor is mapped to.
 *  \param fillCS        Compute set to add the operation into.
 *  \param fillValue     The value to fill \p t with.
 */
template <typename FillValueType>
void fill(poplar::Graph &graph, const poplar::Tensor &t, unsigned tile,
          poplar::ComputeSet fillCS, FillValueType fillValue);

/** Appends vertices to \p fillCS which fills elements in \p mapping of \p t
 *  which reside on tiles represented with \p mapping.
 *
 *  \param graph         The graph that the operation will be added to.
 *  \param t             The tensor whose elements are to be set to zero.
 *  \param mapping       The tensor's region mapping per tile. Each element
 *                       describes a region mapping of a tile (ordered).
 *                       That is, \c mapping[0] is the region of \p t mapped
 *                       onto tile 0.
 *  \param fillCS        Compute set to add the operation into.
 *  \param fillValue     The value to fill \p t with.
 */
template <typename FillValueType>
void fill(poplar::Graph &graph, const poplar::Tensor &t,
          const std::vector<std::vector<poplar::Interval>> &mapping,
          poplar::ComputeSet fillCS, FillValueType fillValue);

/** Appends programs to \p prog which fills all elements of the tensor \p t with
 *  a value of \p fillValue.
 *
 *  \note The type of \p fillValue must be compatible with the element type of
 *  \p t.
 *
 *  \param graph         The graph that the operation will be added to.
 *  \param t             The tensor whose elements are to be filled.
 *  \param prog          Poplar program sequence to append the operation onto.
 *  \param fillValue     The value to fill \p t with.
 *  \param debugPrefix   Name of the operation, for debugging.
 */
template <typename FillValueType>
void fill(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, FillValueType fillValue,
          const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_Fill_hpp
