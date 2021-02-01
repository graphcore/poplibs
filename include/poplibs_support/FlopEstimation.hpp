// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef _popops_Flop_Estimation_h_
#define _popops_Flop_Estimation_h_

#include <cstdint>
#include <poplar/Type.hpp>

namespace poplibs_support {

// Floating point type
// \param type Poplar type
// \return true if poplar type is a floating point type
static inline bool isFPType(const poplar::Type &type) {
  return type == poplar::FLOAT || type == poplar::HALF;
}

// Data dependent rules for determining FLOPS
// \param flops Data independent flops
// \param type The  type to which to convert the flops to
// \return type dependent flops
static inline std::uint64_t convertToTypeFlops(std::uint64_t flops,
                                               const poplar::Type &type) {
  return isFPType(type) ? flops : 0;
}

// Flops for Add operation
static inline unsigned flopsForAdd() { return 1u; }

// Flops for multiply operation
static inline unsigned flopsForMultiply() { return 1u; }

// Flops for a multiply-and-accumulate operation
static inline unsigned flopsForMAC() {
  return flopsForAdd() + flopsForMultiply();
}

} // namespace poplibs_support

#endif // _popops_Flop_Estimation_h_