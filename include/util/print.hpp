#ifndef _print_hpp_
#define _print_hpp_

#include <ostream>

// Print elements in a container in the form {a,b,c,d}
template <class T>
static void printContainer(const T &container, std::ostream &os) {
  os << "{";
  bool needComma = false;
  for (auto &x : container) {
    if (needComma)
      os << ',';
    os << x;
    needComma = true;
  }
  os << '}';
}

#endif // _print_hpp_
