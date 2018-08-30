#include "poplibs_support/Algorithms.hpp"

namespace poplibs {

template<>
std::size_t ival_begin<boost::icl::interval<std::size_t>::type>(
    const boost::icl::interval<std::size_t>::type& ival) {
   return ival.lower();
}
template<>
std::size_t ival_end<boost::icl::interval<std::size_t>::type>(
    const boost::icl::interval<std::size_t>::type& ival) {
   return ival.upper();
}

} // namespace poplibs
