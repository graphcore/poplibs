#include "poplibs_support/Algorithms.hpp"

namespace poplibs {

template<>
std::size_t ival_begin<boost::icl::discrete_interval<std::size_t>>(
    const boost::icl::discrete_interval<std::size_t>& ival) {
   return ival.lower();
}
template<>
std::size_t ival_end<boost::icl::discrete_interval<std::size_t>>(
    const boost::icl::discrete_interval<std::size_t>& ival) {
   return ival.upper();
}

} // namespace poplibs
