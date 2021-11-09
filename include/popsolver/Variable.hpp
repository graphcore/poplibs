// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popsolver_Variable_hpp
#define popsolver_Variable_hpp

#include <gccs/generic_id.hpp>

#include <cstdint>

namespace popsolver {

using Variable = gccs::generic_id<class VariableIdTag, std::uint32_t>;

} // namespace popsolver

#endif // popsolver_Variable_hpp
