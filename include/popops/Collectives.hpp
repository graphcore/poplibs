// Copyright (c) 2018,2019 Graphcore Ltd, All rights reserved.

#ifndef popops_Collectives_hpp
#define popops_Collectives_hpp

#include "popops/Operation.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

namespace popops {

struct Chunk {
  poplar::Tensor tensor;
  unsigned index;  // ring index (data parallel index)
  unsigned offset;  // offset with in rank (model parallel index)
  Chunk() = default;
  Chunk(poplar::Tensor tensor, unsigned index, unsigned offset) :
      tensor(tensor), index(index), offset(offset) {}
};

struct Chunks {
  poplar::Tensor originalInput;  // used to undo shuffles introduced in scatter
  std::vector<Chunk> chunks;  // chunks produced by the scatter step
  Chunks() = default;
  Chunks(unsigned size) : chunks(std::vector<Chunk>(size)) {}
};

/// Given a tensor of rank 2 reduce across the outermost dimension using the
/// specified reduction operator. This function assumes index i in the
/// outermost dimension is mapped to IPU i. The result is distrubuted over IPUs
/// such that each IPU has a slice of the final result. The return value is a
/// vector of chunks where chunk i resides on IPU i. The chunks may have
/// different number of elements (e.g. when the number of IPUs does not exactly
/// divide the number of elements).
/// \param graph The graph.
/// \param toReduce The tensor to reduce. Each partial should be mapped
///                 identically to the others across the ipus with in the rank.
/// \param op The reduction operator (e.g. Operation::ADD).
/// \param prog The program sequence to add operations to.
/// \param debugPrefix String used as a prefix for compute sets.
/// \param options Collective options.
Chunks
reduceScatter(poplar::Graph &graph, const poplar::Tensor &toReduce,
              popops::Operation op,
              poplar::program::Sequence &prog,
              const std::string &debugPrefix = "",
              const poplar::OptionFlags &options = {});

/// Broadcast data distributed over IPUs to all IPUs. This function assumes
/// chunk i is mapped to IPU i. The result is 2-d tensor that contains a copy of
/// the data for each IPU. Index i in the outermost dimension of the result
/// is mapped to IPU i.
/// \param graph The graph.
/// \param toGather The chunks to gather.
/// \param prog The program sequence to add operations to.
/// \param debugPrefix String used as a prefix for compute sets.
/// \param options Collective options.
poplar::Tensor
allGather(poplar::Graph &graph, const Chunks &toGather,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "",
          const poplar::OptionFlags &options = {});

/// Perform an all-reduce operation on the specified tensor. This operation
/// reduces across the outermost dimension of input and produces a tensor with
/// the same shape where the innermost dimension is the result of the reduction
/// and the outermost dimension is a number of copies of the result.
/// This function assumes index i in the outermost dimension of the input is
/// mapped to IPU i. Index i in the outermost dimension of the result is mapped
/// to IPU i.
/// \param graph The graph.
/// \param toReduce The tensor to reduce. Each partial should be mapped
///                 identically to the others across the ipus with in the rank.
/// \param op The reduction operator (e.g. Operation::ADD).
/// \param prog The program sequence to add operations to.
/// \param debugPrefix String used as a prefix for compute sets.
/// \param options Collective options.
poplar::Tensor
allReduce(poplar::Graph &graph, const poplar::Tensor &toReduce,
          popops::Operation op,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "",
          const poplar::OptionFlags &options = {});

/// Perform an all-reduce operation on the specified replicated tensor.
/// This operation reduces across the tensors the replicated tensor is a handle
/// for. The result returned as a replicated tensor.
/// \param graph The replicated graph the input tensor belongs to.
/// \param data The replicated tensor to reduce.
/// \param op The reduction operator (e.g. Operation::ADD)
/// \param prog The program sequence to add operations to.
/// \param debugPrefix String used as a prefix for compute sets.
/// \param options Collective options
poplar::Tensor
replicatedAllReduce(poplar::Graph &graph,
                    const poplar::Tensor &data,
                    popops::Operation op,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {});

/// Same as above but writes the result to the output tensor instead of
/// creating a new one
void
replicatedAllReduceWithOutput(poplar::Graph &graph,
                              const poplar::Tensor &data,
                              poplar::Tensor &output,
                              popops::Operation op,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {});

/// Perform an all-reduce operation on the specified replicated tensor.
/// This variant of replicatedAllReduce() is deprecated and may be removed
/// in future.
poplar::Tensor
replicatedAllReduce(poplar::Graph &graph,
                    poplar::Graph &parentGraph,
                    const poplar::Tensor &data,
                    popops::Operation op,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {});

} // End namespace popops

#endif //popops_Collectives_hpp
