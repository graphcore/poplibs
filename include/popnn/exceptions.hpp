#ifndef __neural_net_exceptions_hpp__
#define __neural_net_exceptions_hpp__
#include <string>

struct net_creation_error : std::logic_error {
  std::string type;
  explicit net_creation_error(const std::string &s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
  explicit net_creation_error(const char *s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
};

#endif //__neural_net_exceptions_hpp__
