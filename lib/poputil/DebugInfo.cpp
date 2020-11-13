// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poputil/DebugInfo.hpp>

#include <sstream>

namespace poputil {

template <> poplar::ProfileValue toProfileValue(const poplar::ComputeSet &t) {
  return poplar::ProfileValue(t.getId());
}

template <>
poplar::ProfileValue toProfileValue(const poplar::program::Copy &t) {
  return poplar::ProfileValue("program::Copy");
}

template <> poplar::ProfileValue toProfileValue(const poplar::OptionFlags &t) {
  return getAsProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const poplar::Tensor &t) {
  if (t.valid()) {
    poplar::ProfileValue::Map v;
    std::stringstream ss;
    ss << "[";
    const auto &shape = t.shape();
    for (size_t i = 0; i < shape.size(); ++i) {

      if (i != 0) {
        ss << ", ";
      }

      ss << shape[i];
    }
    ss << "]";

    v.insert({"shape", poplar::ProfileValue(ss.str())});
    v.insert({"type", poplar::ProfileValue(t.elementType().toString())});
    return v;
  } else {
    return poplar::ProfileValue("<poplar::Tensor> - uninitialized");
  }
}

template <> poplar::ProfileValue toProfileValue(const poplar::Type &t) {
  return poplar::ProfileValue(t.toString());
}

template <> poplar::ProfileValue toProfileValue(const bool &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const float &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const int &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const unsigned int &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const unsigned long &t) {
  return poplar::ProfileValue(t);
}

// Need a specialization for vector<bool> on mac as vec[b] is a
// __bit_const_reference
template <> poplar::ProfileValue toProfileValue(const std::vector<bool> &vec) {
  poplar::ProfileValue::Map v;
  for (size_t i = 0; i < vec.size(); ++i) {
    bool b = vec[i];
    v.insert({std::to_string(i), poplar::ProfileValue(b)});
  }
  return v;
}

OpDebugInfo::OpDebugInfo(const poplar::DebugContext &debugContext,
                         std::string api)
    : poplar::DebugInfo(debugContext, "poplibs") {
  setValue("api", api);
}

void OpDebugInfo::add(const std::string &name,
                      const std::vector<ArgType> &args) {
  if (args.size() > 0) {
    poplar::ProfileValue::Map argsPV;
    for (auto &a : args) {
      argsPV.insert({a.n, a.pv});
    }
    setValue(name, argsPV);
  }
}

PoplibsOpDebugInfo::PoplibsOpDebugInfo(const poplar::DebugContext &debugContext,
                                       const std::vector<ArgType> &args,
                                       const std::string &api)
    : OpDebugInfo(debugContext, api) {
  add("args", args);
}

void PoplibsOpDebugInfo::addOutputs(const std::vector<ArgType> &outputs) {
  add("outputs", outputs);
}

void PoplibsOpDebugInfo::addOutput(const poplar::Tensor &output) {
  setValue("output", toProfileValue(output));
}

} // namespace poputil
