// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file DebugInfo.hpp
 *
 * Poplibs generic debug info structure
 *
 */

#ifndef poputil_DebugInfo_hpp
#define poputil_DebugInfo_hpp

#include <poplar/DebugContext.hpp>

#include <poplar/GraphElements.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>

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

#define DI_STR(S) #S
#define DI_XSTR(S) DI_STR(S)

#define DI_NUM_ARGS(...)                                                       \
  DI_NUM_ARGS_IMPL(__VA_ARGS__, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define DI_NUM_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, N, ...) N

#define DI_CONCAT_IMPL(x, y) x##y
#define DI_MACRO_CONCAT(x, y) DI_CONCAT_IMPL(x, y)

#define DI_KEY_VALUE_ARG_0()                                                   \
  {}
#define DI_KEY_VALUE_ARG_1(T)                                                  \
  { DI_XSTR(T), poputil::toProfileValue(T) }
#define DI_KEY_VALUE_ARG_2(T, ...)                                             \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_1(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_3(T, ...)                                             \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_2(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_4(T, ...)                                             \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_3(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_5(T, ...)                                             \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_4(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_6(T, ...)                                             \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_5(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_7(T, ...)                                             \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_6(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_8(T, ...)                                             \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_7(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_9(T, ...)                                             \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_8(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_10(T, ...)                                            \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_9(__VA_ARGS__)
#define DI_KEY_VALUE_ARG_11(T, ...)                                            \
  DI_KEY_VALUE_ARG_1(T), DI_KEY_VALUE_ARG_10(__VA_ARGS__)
#define DI_KEY_VALUE_ARG(...)                                                  \
  DI_MACRO_CONCAT(DI_KEY_VALUE_ARG_, DI_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

// Convience macro to turn
//   DI_ARGS(a, b, c)
// into
//   {{"a", a}, {"b", b}, {"c"}, c}
// Which can then be passed to the std::vector<ArgType> constructor

#define DI_ARGS(...)                                                           \
  { DI_KEY_VALUE_ARG(__VA_ARGS__) }

#endif // poputil_DebugInfo_hpp
