#include "poputil/exceptions.hpp"

namespace poputil {

poplib_error::poplib_error(const std::string &s) : std::runtime_error(s) {
  type = __FUNCTION__;
}
poplib_error::poplib_error(const char *s) : std::runtime_error(s) {
  type = __FUNCTION__;
}

} // namespace poputil
