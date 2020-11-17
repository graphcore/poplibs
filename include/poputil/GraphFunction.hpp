// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file GraphFunctions.hpp
 *
 * Definitions for reusing graph structures.
 *
 * Since the graph structure takes up memory it is sometimes useful to
 * re-apply the same graph structure for multiple different data items.
 * The functions in this namespace provide a way to do this by treating
 * graphs as reusable functions.
 *
 */

#ifndef poputil_GraphFunction_hpp
#define poputil_GraphFunction_hpp
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace poputil {

/// Support for using poplar::Program objects like function calls.
namespace graphfn {

/// Type of argument to function program.
enum ArgType { InputArg, OutputArg, InOutArg, CreatedArg };

struct ArgSig {
  ArgType type;
  poplar::Tensor similarTensor;
  std::string debugName;
  ArgSig(ArgType type, poplar::Tensor tensor, std::string debugName)
      : type(type), similarTensor(std::move(tensor)),
        debugName(std::move(debugName)) {}
};

inline ArgSig input(poplar::Tensor similar, std::string debugName = "") {
  return ArgSig(InputArg, std::move(similar), std::move(debugName));
}

inline ArgSig inout(poplar::Tensor similar, std::string debugName = "") {
  return ArgSig(InOutArg, std::move(similar), std::move(debugName));
}

inline ArgSig output(poplar::Tensor similar, std::string debugName = "") {
  return ArgSig(OutputArg, std::move(similar), std::move(debugName));
}

inline ArgSig created(std::string debugName = "") {
  return ArgSig(CreatedArg, poplar::Tensor(), std::move(debugName));
}

using Signature = std::vector<ArgSig>;

class VoidFunction {
  poplar::Graph &graph;
  Signature sig;
  bool inlined;
  poplar::program::Sequence prog;
  poplar::Function func;
  std::vector<poplar::Tensor> params;

public:
  VoidFunction(poplar::Graph &graph, Signature sig,
               std::function<void(std::vector<poplar::Tensor> &,
                                  poplar::program::Sequence &)>
                   f,
               bool inlined = false,
               const poplar::DebugContext &debugContext = {});

  VoidFunction(poplar::Graph &graph, Signature sig,
               std::function<void(std::vector<poplar::Tensor> &,
                                  poplar::program::Sequence &,
                                  const poplar::DebugNameAndId &)>
                   f,
               bool inlined = false,
               const poplar::DebugContext &debugContext = {});

  void operator()(std::vector<poplar::Tensor> &args,
                  poplar::program::Sequence &seq,
                  const poplar::DebugContext &dc = {});
};

class ProgramFunction {
  VoidFunction voidFunc;

public:
  ProgramFunction(
      poplar::Graph &graph, Signature sig,
      std::function<poplar::program::Program(std::vector<poplar::Tensor> &)> f,
      bool inlined = false, const poplar::DebugContext &debugContext = {});

  ProgramFunction(
      poplar::Graph &graph, Signature sig,
      std::function<poplar::program::Program(std::vector<poplar::Tensor> &,
                                             const poplar::DebugNameAndId &)>
          f,
      bool inlined = false, const poplar::DebugContext &debugContext = {});

  poplar::program::Program
  operator()(std::vector<poplar::Tensor> &args,
             const poplar::DebugContext &debugContext = {});
};

class TensorFunction {
  VoidFunction voidFunc;

public:
  TensorFunction(poplar::Graph &graph, Signature sig,
                 std::function<poplar::Tensor(std::vector<poplar::Tensor> &,
                                              poplar::program::Sequence &)>
                     f,
                 bool inlined = false,
                 const poplar::DebugContext &debugContext = {});

  TensorFunction(poplar::Graph &graph, Signature sig,
                 std::function<poplar::Tensor(std::vector<poplar::Tensor> &,
                                              poplar::program::Sequence &,
                                              const poplar::DebugNameAndId &)>
                     f,
                 bool inlined = false,
                 const poplar::DebugContext &debugContext = {});

  poplar::Tensor operator()(std::vector<poplar::Tensor> &args,
                            poplar::program::Sequence &prog,
                            const poplar::DebugContext &debugContext = {});
};

} // namespace graphfn
} // namespace poputil

#endif // poputil_GraphFunction_hpp
