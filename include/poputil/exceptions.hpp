// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poputil_exceptions_hpp
#define poputil_exceptions_hpp

#include <stdexcept>
#include <string>

namespace poputil {

struct poplib_error : std::logic_error {
  std::string type;
  explicit poplib_error(const std::string &s);
  explicit poplib_error(const char *s);
};

} // namespace poputil

#endif // poputil_exceptions_hpp
