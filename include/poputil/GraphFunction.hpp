// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poputil_GraphFunction_hpp
#define poputil_GraphFunction_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace poputil {
namespace graphfn {

/**********************************************************************
 * Graph "functions" for reusing graph structure on multiple data.
 *
 * Since graph structure takes up memory it is sometimes useful to
 * re-apply the same graph structure on multiple different data items.
 * The functions in this header provide a way to do this.
 *
 **********************************************************************/

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
               bool inlined = true);
  void operator()(std::vector<poplar::Tensor> &args,
                  poplar::program::Sequence &seq);
};

class ProgramFunction {
  VoidFunction voidFunc;

public:
  ProgramFunction(
      poplar::Graph &graph, Signature sig,
      std::function<poplar::program::Program(std::vector<poplar::Tensor> &)> f,
      bool inlined = true);
  poplar::program::Program operator()(std::vector<poplar::Tensor> &args);
};

class TensorFunction {
  VoidFunction voidFunc;

public:
  TensorFunction(poplar::Graph &graph, Signature sig,
                 std::function<poplar::Tensor(std::vector<poplar::Tensor> &,
                                              poplar::program::Sequence &)>
                     f,
                 bool inlined = true);
  poplar::Tensor operator()(std::vector<poplar::Tensor> &args,
                            poplar::program::Sequence &prog);
};

} // namespace graphfn
} // namespace poputil

#endif // poputil_GraphFunction_hpp
