// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "ReduceCodelets.hpp"

#ifdef __IPU__
// For real implementation
using ShortType = unsigned short;
#else
// To avoid size overflow on CPU implementation
using ShortType = unsigned;
#endif

namespace popops {

template <typename OutType, bool isUpdate>
using ROT = typename std::conditional<
    isUpdate, poplar::InOut<poplar::Vector<OutType, PTR_ALIGN32, 4>>,
    poplar::Output<poplar::Vector<OutType, PTR_ALIGN32, 4>>>::type;

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
static constexpr bool useExternal() {
  bool opIsAddOrSquareAdd = std::is_same<ReduceOp, ReduceAdd>::value ||
                            std::is_same<ReduceOp, ReduceSquareAdd>::value;

  bool partialsAndOutputAreFloatsOrHalfs =
      (std::is_same<OutType, float>::value ||
       std::is_same<OutType, half>::value) &&
      (std::is_same<PartialsType, float>::value ||
       std::is_same<PartialsType, half>::value);

  bool opIsMinOrMax = std::is_same<ReduceOp, ReduceMax>::value ||
                      std::is_same<ReduceOp, ReduceMin>::value;

  bool partialsAndOutputAreTheSameType =
      std::is_same<PartialsType, OutType>::value;

  return (opIsAddOrSquareAdd && partialsAndOutputAreFloatsOrHalfs) ||
         (opIsMinOrMax && partialsAndOutputAreFloatsOrHalfs &&
          partialsAndOutputAreTheSameType);
}

} // namespace popops
