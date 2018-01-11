#ifndef __popstd_vertex_templates_hpp__
#define __popstd_vertex_templates_hpp__
#include <string>
#include <poplar/Type.hpp>

namespace popstd {

inline std::string templateVertexParams(bool first) {
  if (first)
    return "";
  else
    return ">";
}

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const std::string &val,
                                        Args&&... args);

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const char *val,
                                        Args&&... args);

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const poplar::Type &type,
                                        Args&&... args);

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        bool b,
                                        Args&&... args);

template <typename T, typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const T&val, Args&&... args) {
  std::string p = first ? "<" : ",";
  p += std::to_string(val) + templateVertexParams(false,
                                                  std::forward<Args>(args)...);
  return p;
}

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const poplar::Type &type,
                                        Args&&... args) {
  std::string p = first ? "<" : ",";
  p += type.toString() + templateVertexParams(false,
                                              std::forward<Args>(args)...);
  return p;
}

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const std::string &val,
                                        Args&&... args) {
  std::string p = first ? "<" : ",";
  p += val + templateVertexParams(false, std::forward<Args>(args)...);
  return p;
}

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        const char *val,
                                        Args&&... args) {
  std::string p = first ? "<" : ",";
  p += val + templateVertexParams(false, std::forward<Args>(args)...);
  return p;
}

template <typename ...Args>
inline std::string templateVertexParams(bool first,
                                        bool b,
                                        Args&&... args) {
  std::string p = first ? "<" : ",";
  auto bstr = (b ? "true" : "false");
  p += bstr + templateVertexParams(false, std::forward<Args>(args)...);
  return p;
}

template <typename ...Args>
inline std::string templateVertex(const std::string &name,
                                  Args&&... args) {
  return name + templateVertexParams(true, std::forward<Args>(args)...);
}

} // end namespace popstd

#endif // __popstd_vertex_templates_hpp__
