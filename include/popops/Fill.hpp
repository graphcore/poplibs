// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_Fill_hpp
#define popops_Fill_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/TypeTraits.hpp>

namespace popops {

void fill(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, const void *fillValue,
          const poplar::TypeTraits &traits,
          const std::string &debugPrefix = "");

/** Appends programs to \p prog which fills all elements of the Tensor \p t with
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
          const std::string &debugPrefix = "") {
  static_assert(poplar::TypeTraits::isSimpleType<FillValueType>(),
                "FillValueType must be an integral or floating point type.");
  fill(graph, t, prog, reinterpret_cast<const void *>(&fillValue),
       poplar::TypeTraits::make<FillValueType>(), debugPrefix);
}

} // namespace popops

#endif // popops_Fill_hpp
