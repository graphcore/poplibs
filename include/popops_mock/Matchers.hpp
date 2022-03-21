// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef popops_Matchers_hpp
#define popops_Matchers_hpp

#include <gmock/gmock.h>

#include <popops/Expr.hpp>

namespace popops_mock {

MATCHER_P(IsExpr, e, "") { return arg.deepEquals(e); }

} // end namespace popops_mock

#endif // popops_Matchers_hpp
