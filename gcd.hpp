#ifndef _gcd_hpp_
#define _gcd_hpp_

// Greatest common divisor
template <typename T>
static T gcd(T a, T b) {
  while (b != 0) {
    T tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

#endif
