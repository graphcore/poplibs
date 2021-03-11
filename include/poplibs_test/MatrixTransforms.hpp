// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_MatrixTransforms_hpp
#define poplibs_test_MatrixTransforms_hpp

#include <boost/multi_array.hpp>

namespace poplibs_test {
namespace matrix {

template <typename FPType>
boost::multi_array<FPType, 2>
transpose(const boost::multi_array<FPType, 2> &in) {
  const auto inRows = in.shape()[0];
  const auto inColumns = in.shape()[1];
  boost::multi_array<FPType, 2> out(boost::extents[inColumns][inRows]);
  for (unsigned inRow = 0; inRow < inRows; inRow++) {
    for (unsigned inColumn = 0; inColumn < inColumns; inColumn++) {
      out[inColumn][inRow] = in[inRow][inColumn];
    }
  }
  return out;
}

} // End namespace matrix
} // End namespace poplibs_test

#endif // poplibs_test_MatrixTransforms_hpp
