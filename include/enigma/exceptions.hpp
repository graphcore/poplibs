// Copyright (c) 2016, Graphcore Ltd, All rights reserved.

#ifndef enigma_exceptions_hpp_
#define enigma_exceptions_hpp_

#include <stdexcept>
#include <string>

namespace enigma {

struct enigma_error : std::logic_error {
  std::string type;
  explicit enigma_error(const std::string &s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
  explicit enigma_error(const char *s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
};

} // End namespace enigma.

#endif // enigma_exceptions_hpp_
