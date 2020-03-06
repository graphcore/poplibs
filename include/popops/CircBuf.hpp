// Copyright (c) 2017 Graphcore Ltd, All rights reserved.

#ifndef popops_CircBuf_hpp
#define popops_CircBuf_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <string>

namespace popops {

class CircBuf {
public:
  /** CircBuf represents a circular buffer of tensors which can be indexed using
   * prev(). Each call to add() will add the given tensor to the circular buffer
   * with the potential to overwrite a previous element if the buffer is full.
   *
   * \param graph             Graph to add the circular buffer to.
   * \param dataType          Datatype of the tensor elements in buffer.
   * \param size              Size of the circular buffer.
   * \param shape             Shape of the tensor elements in buffer.
   * \param debugPrefix       Prefix of the circular buffer tensor,
   *                          for debugging.
   */
  CircBuf(poplar::Graph &graph, const poplar::Type &dataType, unsigned size,
          const std::vector<std::size_t> &shape,
          const std::string &debugPrefix = "");

  /** Return elements \p i entries old. \p i must be < size
   *
   * \param i             Index into the circular buffer.
   * \param seq           Program to add the operation to.
   * \param debugPrefix   Name of the operation, for debugging.
   * \return              Tensor returned from the circular buffer.
   */
  poplar::Tensor prev(unsigned i, poplar::program::Sequence &seq,
                      const std::string &debugPrefix = "");

  /** Append an element to the end of the circular buffer.
   *
   * \param t             Tensor to append to the circular buffer
   * \param seq           Program to add the operation to.
   * \param debugPrefix   Name of the operation, for debugging.
   */
  void add(poplar::Tensor t, poplar::program::Sequence &seq,
           const std::string &debugPrefix = "");

  /// Tensor representing the index into the circular buffer
  poplar::Tensor getIndex() const;

  /// Size of the circular buffer
  unsigned size() const;

  /// Return tensor mapping of the tensor returned by indexing into a circular
  /// buffer
  poplar::Graph::TileToTensorMapping getTileMapping();

private:
  poplar::Graph &graph;
  unsigned size_;
  poplar::Tensor index;
  std::vector<std::size_t> shape;
  // The history buffer may be padded to ensure an integral number of grains
  unsigned padElements;
  poplar::Tensor hist;
};

} // namespace popops

#endif // popops_CircBuf_hpp
