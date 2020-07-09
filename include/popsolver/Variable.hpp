// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popsolver_Variable_hpp
#define popsolver_Variable_hpp

#include <cstdint>

namespace popsolver {

class Variable {
public:
  using IndexType = std::uint32_t;

  Variable() = default;
  explicit Variable(IndexType id) : id(id) {}
  IndexType id;
};

} // namespace popsolver

#endif // popsolver_Variable_hpp
