// Copyright (c) 2017 Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Pass_hpp
#define poplibs_test_Pass_hpp

#include <iosfwd>

namespace poplibs_test {

// class to allow the training pass to be specified
enum class Pass { FWD, BWD, WU, ALL };

const char *asString(const Pass &pass);

std::istream &operator>>(std::istream &is, Pass &pass);

std::ostream &operator<<(std::ostream &os, const Pass &pass);

} // End namespace poplibs_test.

#endif // poplibs_test_Pass_hpp
