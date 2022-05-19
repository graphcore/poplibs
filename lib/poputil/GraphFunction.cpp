// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplibs_support/Tracepoint.hpp"
#include <poputil/DebugInfo.hpp>
#include <poputil/GraphFunction.hpp>

using namespace poplar;
using namespace poplar::program;

namespace poputil {
namespace graphfn {

VoidFunction::VoidFunction(VoidFunction &&) = default;

VoidFunction::VoidFunction(
    Graph &graph, Signature sig_,
    std::function<void(std::vector<Tensor> &, Sequence &)> f, bool inlined,
    const poplar::DebugContext &debugContext)
    : graph(graph), sig(std::move(sig_)), inlined(inlined), prog(debugContext) {

  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(/*sig_,*/ inlined));

  for (const auto &s : sig) {
    if (s.type == CreatedArg) {
      params.push_back(Tensor());
      continue;
    }
    auto t = graph.clone(s.similarTensor, {di, s.debugName});
    params.push_back(std::move(t));
  }
  f(params, prog);
  if (!inlined) {
    func = graph.addFunction(prog);
  }
}

VoidFunction::VoidFunction(Graph &graph, Signature sig_,
                           std::function<void(std::vector<Tensor> &, Sequence &,
                                              const poplar::DebugNameAndId &)>
                               f,
                           bool inlined,
                           const poplar::DebugContext &debugContext)
    : graph(graph), sig(std::move(sig_)), inlined(inlined), prog(debugContext) {

  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(/*sig_,*/ inlined));

  for (const auto &s : sig) {
    if (s.type == CreatedArg) {
      params.push_back(Tensor());
      continue;
    }
    auto t = graph.clone(s.similarTensor, {di, s.debugName});
    params.push_back(std::move(t));
  }
  f(params, prog, {di});
  if (!inlined) {
    func = graph.addFunction(prog);
  }
}

void VoidFunction::operator()(std::vector<poplar::Tensor> &args,
                              poplar::program::Sequence &seq,
                              const poplar::DebugContext &debugContext) {

  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(args));
  for (unsigned i = 0; i < sig.size(); ++i) {
    if (sig[i].type == InputArg || sig[i].type == InOutArg) {
      seq.add(Copy(args[i], params[i], false, {di}));
    }
  }
  if (inlined) {
    seq.add(prog);
  } else {
    seq.add(poplar::program::Call(func, {di}));
  }
  for (unsigned i = 0; i < sig.size(); ++i) {
    if (sig[i].type == CreatedArg) {
      args[i] = graph.clone(params[i], {di, sig[i].debugName});
    }
    if (sig[i].type == OutputArg || sig[i].type == InOutArg ||
        sig[i].type == CreatedArg) {
      seq.add(Copy(params[i], args[i], false, {di}));
    }
  }
}

ProgramFunction::ProgramFunction(
    Graph &graph, Signature sig,
    std::function<Program(std::vector<Tensor> &)> f, bool inlined,
    const poplar::DebugContext &debugContext)
    : voidFunc(
          graph, std::move(sig),
          [&](std::vector<Tensor> &args, Sequence &seq) { seq.add(f(args)); },
          inlined, debugContext) {}

ProgramFunction::ProgramFunction(
    Graph &graph, Signature sig,
    std::function<Program(std::vector<Tensor> &,
                          const poplar::DebugNameAndId &)>
        f,
    bool inlined, const poplar::DebugContext &debugContext)
    : voidFunc(
          graph, std::move(sig),
          [&](std::vector<Tensor> &args, Sequence &seq,
              const poplar::DebugNameAndId &dnai) { seq.add(f(args, dnai)); },
          inlined, debugContext) {}

Program ProgramFunction::operator()(std::vector<poplar::Tensor> &args,
                                    const poplar::DebugContext &debugContext) {

  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(args));

  Sequence seq({}, {di});
  voidFunc(args, seq, {di});
  return std::move(seq);
}

static inline Signature extendWithCreated(Signature s) {
  s.push_back(created());
  return s;
}

TensorFunction::TensorFunction(
    Graph &graph, Signature sig,
    std::function<Tensor(std::vector<Tensor> &, Sequence &seq)> f, bool inlined,
    const poplar::DebugContext &debugContext)
    : voidFunc(
          graph, extendWithCreated(std::move(sig)),
          [&](std::vector<Tensor> &args, Sequence &seq) {
            args.back() = f(args, seq);
          },
          inlined, debugContext) {}

TensorFunction::TensorFunction(
    Graph &graph, Signature sig,
    std::function<Tensor(std::vector<Tensor> &, Sequence &,
                         const poplar::DebugNameAndId &)>
        f,
    bool inlined, const poplar::DebugContext &debugContext)
    : voidFunc(
          graph, extendWithCreated(std::move(sig)),
          [&](std::vector<Tensor> &args, Sequence &seq,
              const poplar::DebugNameAndId &dnai) {
            args.back() = f(args, seq, dnai);
          },
          inlined, debugContext) {}

Tensor TensorFunction::operator()(std::vector<poplar::Tensor> &args,
                                  Sequence &seq,
                                  const poplar::DebugContext &debugContext) {

  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(args));

  args.push_back(Tensor());
  voidFunc(args, seq, {di});
  auto t = args.back();
  args.resize(args.size() - 1);

  di.addOutput(t);
  return t;
}

} // namespace graphfn
} // namespace poputil
