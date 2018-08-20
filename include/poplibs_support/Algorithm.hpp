#ifndef poplibs_support_Algorithm_hpp
#define poplibs_support_Algorithm_hpp

namespace poplibs_support {

// Unsigned integer version of log2 rounded up
// Single-line constexpr form added to allow compile-time calculation.
// Could be nicer if using multi-line constexpr function (needs C++14).
constexpr static unsigned ceilLog2Aux(unsigned n) {
  return (n ? 1 + ceilLog2Aux(n >> 1) : 0);
}
// Check if power of 2 and then call to count up to most significant bit
constexpr static unsigned ceilLog2(unsigned n) {
  return ((n & (n - 1)) ? 1 : 0) + ceilLog2Aux(n >> 1);
}

} // end namespace poplibs_support

#endif // poplibs_support_Algorithm_hpp
