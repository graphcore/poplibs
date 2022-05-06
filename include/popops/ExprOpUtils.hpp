// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef _popops_ExprOpUtils_hpp_
#define _popops_ExprOpUtils_hpp_

#include <iosfwd>

#include <popops/ExprOp.hpp>

namespace popops::expr {
namespace ostream_ext {

std::ostream &operator<<(std::ostream &os, const UnaryOpType &t);
std::ostream &operator<<(std::ostream &os, const BinaryOpType &t);
std::ostream &operator<<(std::ostream &os, const TernaryOpType &t);

} // end namespace ostream_ext
} // end namespace popops::expr

#endif // _popops_ExprOpUtils_hpp_
