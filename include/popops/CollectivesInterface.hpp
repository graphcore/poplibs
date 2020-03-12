// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef popops_CollectivesInterface_hpp
#define popops_CollectivesInterface_hpp

#include "popops/Operation.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

#include <memory>
#include <string>
#include <vector>

namespace popops {

class VersionedInterface {
public:
  /// Version of the api
  virtual std::string version() = 0;
};

class ReplicatedCollectivesInterface : public VersionedInterface {

public:
  virtual ~ReplicatedCollectivesInterface() {}

  /// Perform an all-reduce operation on the specified replicated tensor.
  /// This operation reduces across the tensors the replicated tensor is a
  /// handle for. The result returned as a replicated tensor.
  ///
  /// \param graph  The replicated graph the input tensor belongs to.
  /// \param data   The replicated tensor to reduce.
  /// \param op     The reduction operator (for example, Operation::ADD)
  /// \param prog   The program sequence to add operations to.
  /// \param debugPrefix String used as a prefix for compute sets.
  /// \param options Collective options
  virtual poplar::Tensor
  replicatedAllReduce(poplar::Graph &graph, const poplar::Tensor &data,
                      popops::Operation op, poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {}) = 0;

  /// Same as above but writes the result to the output tensor instead of
  /// creating a new one
  virtual void replicatedAllReduceWithOutput(
      poplar::Graph &graph, const poplar::Tensor &data, poplar::Tensor &output,
      popops::Operation op, poplar::program::Sequence &prog,
      const std::string &debugPrefix = "",
      const poplar::OptionFlags &options = {}) = 0;

  /// Perform an all-reduce operation on the specified replicated tensor.
  /// This variant of replicatedAllReduce() is deprecated and may be removed
  /// in future.
  virtual poplar::Tensor
  replicatedAllReduce(poplar::Graph &graph, poplar::Graph &parentGraph,
                      const poplar::Tensor &data, popops::Operation op,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {}) = 0;

  static std::shared_ptr<ReplicatedCollectivesInterface> defaultImpl;
};

} // End namespace popops

#endif // popops_CollectivesInterface_hpp
