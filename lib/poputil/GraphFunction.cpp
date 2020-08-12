// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <poputil/GraphFunction.hpp>

using namespace poplar;
using namespace poplar::program;

namespace poputil {
namespace graphfn {

VoidFunction::VoidFunction(
    Graph &graph, Signature sig_,
    std::function<void(std::vector<Tensor> &, Sequence &)> f, bool inlined)
    : graph(graph), sig(std::move(sig_)), inlined(inlined) {
  for (const auto &s : sig) {
    if (s.type == CreatedArg) {
      params.push_back(Tensor());
      continue;
    }
    auto t = graph.clone(s.similarTensor, s.debugName);
    params.push_back(std::move(t));
  }
  f(params, prog);
  if (!inlined) {
    func = graph.addFunction(prog);
  }
}

void VoidFunction::operator()(std::vector<poplar::Tensor> &args,
                              poplar::program::Sequence &seq) {
  for (unsigned i = 0; i < sig.size(); ++i) {
    if (sig[i].type == InputArg || sig[i].type == InOutArg) {
      seq.add(Copy(args[i], params[i]));
    }
  }
  if (inlined) {
    seq.add(prog);
  } else {
    seq.add(poplar::program::Call(func));
  }
  for (unsigned i = 0; i < sig.size(); ++i) {
    if (sig[i].type == CreatedArg) {
      args[i] = graph.clone(params[i], sig[i].debugName);
    }
    if (sig[i].type == OutputArg || sig[i].type == InOutArg ||
        sig[i].type == CreatedArg) {
      seq.add(Copy(params[i], args[i]));
    }
  }
}

ProgramFunction::ProgramFunction(
    Graph &graph, Signature sig,
    std::function<Program(std::vector<Tensor> &)> f, bool inlined)
    : voidFunc(
          graph, std::move(sig),
          [&](std::vector<Tensor> &args, Sequence &seq) { seq.add(f(args)); },
          inlined) {}

Program ProgramFunction::operator()(std::vector<poplar::Tensor> &args) {
  Sequence seq;
  voidFunc(args, seq);
  return std::move(seq);
}

static inline Signature extendWithCreated(Signature s) {
  s.push_back(created());
  return s;
}

TensorFunction::TensorFunction(
    Graph &graph, Signature sig,
    std::function<Tensor(std::vector<Tensor> &, Sequence &seq)> f, bool inlined)
    : voidFunc(
          graph, extendWithCreated(std::move(sig)),
          [&](std::vector<Tensor> &args, Sequence &seq) {
            args.back() = f(args, seq);
          },
          inlined) {}

Tensor TensorFunction::operator()(std::vector<poplar::Tensor> &args,
                                  Sequence &seq) {
  args.push_back(Tensor());
  voidFunc(args, seq);
  auto t = args.back();
  args.resize(args.size() - 1);
  return t;
}

} // namespace graphfn
} // namespace poputil
