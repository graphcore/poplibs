// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "poplar/TensorCloneMethod.hpp"
#include <poputil/DebugInfo.hpp>

#include <sstream>

namespace poputil {

template <> poplar::ProfileValue toProfileValue(const poplar::ComputeSet &t) {
  return poplar::ProfileValue(t.getId());
}

template <> poplar::ProfileValue toProfileValue(const poplar::OptionFlags &t) {
  return getAsProfileValue(t);
}

template <>
poplar::ProfileValue toProfileValue(const poplar::program::Copy &t) {
  return poplar::ProfileValue("<poplar::program::Copy>");
}

template <>
poplar::ProfileValue toProfileValue(const poplar::program::Sequence &t) {
  return poplar::ProfileValue("<poplar::program::Sequence>");
}

template <>
poplar::ProfileValue toProfileValue(const poplar::TensorCloneMethod &t) {
  switch (t) {
  case poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES:
    return poplar::ProfileValue("PRESERVE_ORDER_AND_ALIASES");
  case poplar::TensorCloneMethod::CREATE_NEW_ORDER:
    return poplar::ProfileValue("CREATE_NEW_ORDER");
  case poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES:
    return poplar::ProfileValue("PRESERVE_ORDER_UNLESS_ALIASES");
  case poplar::TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES:
    return poplar::ProfileValue("GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
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

template <> poplar::ProfileValue toProfileValue(const int &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const unsigned long &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const unsigned int &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const unsigned char &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const double &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const float &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const bool &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const std::string &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const long long &t) {
  return poplar::ProfileValue(t);
}

template <> poplar::ProfileValue toProfileValue(const unsigned long long &t) {
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

void OpDebugInfo::add(std::string name, const std::vector<ArgType> &args) {
  if (args.size() > 0) {
    poplar::ProfileValue::Map argsPV;
    for (auto &a : args) {
      argsPV.insert({a.n, a.pv});
    }
    setValue(std::move(name), argsPV);
  }
}

void OpDebugInfo::add(std::string name, poplar::ProfileValue pv) {
  setValue(std::move(name), std::move(pv));
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
