
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poputil_DebugInfo_hpp
#define poputil_DebugInfo_hpp

#include <poplar/DebugContext.hpp>

#include <poplar/GraphElements.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>

#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#if defined(__clang__)
#define SUPPORTS_FUNCTION_BUILTINS __has_builtin(__builtin_FUNCTION)
#elif __GNUC__ >= 7
#define SUPPORTS_FUNCTION_BUILTINS 1
#else
#define SUPPORTS_FUNCTION_BUILTINS 0
#endif

namespace poputil {

// template definitions of the toProfileValue that needs to be specialized
// for types passed to the DebugInfo
template <typename T> poplar::ProfileValue toProfileValue(const T &t);

// template specializations for basic types
template <> poplar::ProfileValue toProfileValue(const unsigned int &v);
template <> poplar::ProfileValue toProfileValue(const bool &v);
template <> poplar::ProfileValue toProfileValue(const float &v);

// template specializations for Poplar types
template <> poplar::ProfileValue toProfileValue(const poplar::ComputeSet &t);
template <> poplar::ProfileValue toProfileValue(const poplar::Tensor &t);
template <> poplar::ProfileValue toProfileValue(const poplar::Type &t);

// Generic case for a pointer
template <typename T> poplar::ProfileValue toProfileValue(const T *t) {
  if (t == nullptr) {
    return poplar::ProfileValue("<nullptr>");
  } else {
    return toProfileValue(*t);
  }
}

// Generic case for a vector
template <typename T>
poplar::ProfileValue toProfileValue(const std::vector<T> &vec) {
  poplar::ProfileValue::Map v;
  for (size_t i = 0; i < vec.size(); ++i) {
    v.insert({std::to_string(i), toProfileValue(vec[i])});
  }
  return v;
}

class ArgType {

public:
  std::string n;
  poplar::ProfileValue pv;

  ArgType(const std::string &name, const poplar::ProfileValue &value)
      : n(name), pv(value) {}
  ArgType(const std::pair<std::string, const poplar::ProfileValue> &p)
      : n(p.first), pv(p.second) {}
};

class OpDebugInfo : public poplar::DebugInfo {
public:
  OpDebugInfo(const poplar::DebugContext &debugContext, std::string api);
  OpDebugInfo &operator=(const OpDebugInfo &) = delete;
  OpDebugInfo(const OpDebugInfo &) = delete;
  virtual ~OpDebugInfo() = default;

  void add(const std::string &name, const std::vector<ArgType> &args);
};
class PoplibsOpDebugInfo : public OpDebugInfo {

public:
#if SUPPORTS_FUNCTION_BUILTINS
  PoplibsOpDebugInfo(const poplar::DebugContext &debugContext,
                     const std::vector<ArgType> &args = {},
                     const std::string &api = __builtin_FUNCTION());
#else
  PoplibsOpDebugInfo(const poplar::DebugContext &debugContext,
                     const std::vector<ArgType> &args = {},
                     const std::string &api = "");
#endif
  PoplibsOpDebugInfo &operator=(const PoplibsOpDebugInfo &) = delete;
  PoplibsOpDebugInfo(const PoplibsOpDebugInfo &) = delete;
  virtual ~PoplibsOpDebugInfo() = default;

  void addOutputs(const std::vector<ArgType> &outputs);

  // Convience method when there is only a single output
  void addOutput(const poplar::Tensor &output);
};

} // namespace poputil

#define DI_PROCESS_ONE_ELEMENT(r, data, i, elem)                               \
  BOOST_PP_COMMA_IF(i) {                                                       \
    BOOST_PP_STRINGIZE(elem), poputil::toProfileValue(elem)                    \
  }

// Convience macro to turn
//   DI_ARGS(a, b, c)
// into
//   {{"a", a}, {"b", b}, {"c"}, c}
// Which can then be passed to the std::vector<ArgType> constructor

#define DI_ARGS(...)                                                           \
  {                                                                            \
    BOOST_PP_SEQ_FOR_EACH_I(DI_PROCESS_ONE_ELEMENT, _,                         \
                            BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))             \
  }

#endif // poputil_DebugInfo_hpp
