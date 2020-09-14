// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions for padding a tensor.
 *
 */

#ifndef popops_Pad_hpp
#define popops_Pad_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <vector>

namespace popops {

// Some general comments:
// Constant padding requires a Graph, as a Variable is created. This is
// the reason for separate API functions for constant padding and other
// paddings.

namespace padding {

/// Padding types as per numpy.pad
enum class Type {
  /// Also known as nearest-neighbour padding, each new pad element has
  /// its value set to that of the pre-padded element nearest to it. Any
  /// such nearest neighbour lies on the edge of the pre-padded tensor,
  /// hence the name.
  EDGE,
  /// The tensor is reflected outwards. Specifically, a new pad element
  /// has its value set to that of the element which is an equal
  /// distance to the pad element's nearest neighbour as the pad
  /// element, but in the opposite direction.
  REFLECT
};

/// Methods to map added padding elements to tiles.
enum class MappingMethod {
  /// Padding won't be mapped.
  NONE,
  /// Set tile mapping of padding element to tile 0 for the graph.
  ZERO,
  /// Set tile mapping of padding elements to match the nearest-neighbour
  /// element which lies on the edge of the tensor prior to padding.
  /// Requires a non-empty tensor to be padded with a complete tile
  /// mapping.
  EDGE

};

} // namespace padding

/// Return a tensor with constant padding added.
/// \param graph          The graph containing the tensor.
/// \param t              The tensor to pad.
/// \param paddingLower   A vector specifying the amount of padding to add at
///                       the start of each dimension. Negative padding
///                        truncates.
/// \param paddingUpper   A vector specifying the amount of padding to add at
///                       the end of each dimension. Negative padding
///                        truncates.
/// \param val            The input tensor will be padded with this value.
/// \param mappingMethod  The method that should be used to map added padding
///                        elements.
/// \return The tensor with padding added.
/// @{
poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t,
    const std::vector<std::ptrdiff_t> &paddingLower,
    const std::vector<std::ptrdiff_t> &paddingUpper, float val = 0.0f,
    padding::MappingMethod mappingMethod = padding::MappingMethod::ZERO);

poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t,
    const std::vector<std::ptrdiff_t> &paddingLower,
    const std::vector<std::ptrdiff_t> &paddingUpper, int val,
    padding::MappingMethod mappingMethod = padding::MappingMethod::ZERO);

poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t,
    const std::vector<std::ptrdiff_t> &paddingLower,
    const std::vector<std::ptrdiff_t> &paddingUpper, const poplar::Tensor &val,
    padding::MappingMethod mappingMethod = padding::MappingMethod::ZERO);
/// @}

/// Return a tensor with constant padding added to one dimension.
/// \param t              The tensor to pad.
/// \param paddingLower   The amount of padding to add at the start of the
///                       dimension. Negative padding truncates.
/// \param paddingUpper   The amount of padding to add at the end of the
///                       dimension. Negative padding truncates.
/// \param dim            The dimension to pad.
/// \param val            The input tensor will be padded with this value.
/// \param mappingMethod  The method that should be used to map added padding
///                        elements.
/// \return The tensor with padding added.
///
/// @{
poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t, std::ptrdiff_t paddingLower,
    std::ptrdiff_t paddingUpper, unsigned dim, float val = 0.0f,
    padding::MappingMethod mappingMethod = padding::MappingMethod::ZERO);

poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t, std::ptrdiff_t paddingLower,
    std::ptrdiff_t paddingUpper, unsigned dim, int val,
    padding::MappingMethod mappingMethod = padding::MappingMethod::ZERO);

poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t, std::ptrdiff_t paddingLower,
    std::ptrdiff_t paddingUpper, unsigned dim, const poplar::Tensor &val,
    padding::MappingMethod mappingMethod = padding::MappingMethod::ZERO);
/// @}

/// Return a tensor with numpy-style padding added.
/// \param t            The tensor to pad.
/// \param paddingLower A vector specifying the amount of padding to add at the
///                     start of each dimension. Negative padding truncates.
/// \param paddingUpper A vector specifying the amount of padding to add at the
///                     end of each dimension. Negative padding truncates.
/// \param type         The type of padding.
/// \return The tensor with padding added.
poplar::Tensor pad(const poplar::Tensor &t,
                   const std::vector<std::ptrdiff_t> &paddingLower,
                   const std::vector<std::ptrdiff_t> &paddingUpper,
                   padding::Type type);

/// Return a tensor with numpy-style padding added to one dimension.
/// \param t            The tensor to pad.
/// \param paddingLower The amount of padding to add at the start of the
///                     dimension. Negative padding truncates.
/// \param paddingUpper The amount of padding to add at the end of the
///                     dimension. Negative padding truncates.
/// \param dim          The dimension to pad.
/// \return The tensor with padding added.
poplar::Tensor pad(const poplar::Tensor &t, std::ptrdiff_t paddingLower,
                   std::ptrdiff_t paddingUpper, unsigned dim,
                   padding::Type type);

} // namespace popops

#endif // popops_Pad_hpp
