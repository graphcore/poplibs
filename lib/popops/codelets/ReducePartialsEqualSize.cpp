// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "ReducePartialsEqualSize.hpp"

using namespace poplar;

namespace popops {

// Specialisation of reduction vertices where all partials are of equal size.
// Additional constraints are that the partials must be a multiple of 128 bits
// in size, and the partials size is a multiple of the output size.
// The approach is to reduce each column of all partials first, as the inner
// loop which given the constraints above is an efficient implementation.

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
class ReducePartialsEqualSize : public Vertex {
  IS_EXTERNAL_CODELET((IsExternal<ReduceOp, PartialsType>()()));

  ReduceOutput<OutType, isUpdate> out;
  const ShortType outCount;
  ReducePartials<PartialsType> partials;
  const ShortType partialsSizeM1;

public:
  ReducePartialsEqualSize();

  bool compute() {
    const auto fn = computePartialsEqualSizeReduction<ReduceOp, OutType,
                                                      PartialsType, isUpdate>;
    return fn(out, outCount, partialsSizeM1 + 1, partials, 1.0f);
  }
};

template class ReducePartialsEqualSize<popops::ReduceAdd, float, float, true>;
template class ReducePartialsEqualSize<popops::ReduceAdd, half, float, true>;
template class ReducePartialsEqualSize<popops::ReduceAdd, float, half, true>;
template class ReducePartialsEqualSize<popops::ReduceAdd, half, half, true>;
template class ReducePartialsEqualSize<popops::ReduceAdd, int, int, true>;

template class ReducePartialsEqualSize<popops::ReduceAdd, float, float, false>;
template class ReducePartialsEqualSize<popops::ReduceAdd, half, float, false>;
template class ReducePartialsEqualSize<popops::ReduceAdd, float, half, false>;
template class ReducePartialsEqualSize<popops::ReduceAdd, half, half, false>;
template class ReducePartialsEqualSize<popops::ReduceAdd, int, int, false>;

template class ReducePartialsEqualSize<popops::ReduceSquareAdd, float, float,
                                       true>;
template class ReducePartialsEqualSize<popops::ReduceSquareAdd, half, float,
                                       true>;
template class ReducePartialsEqualSize<popops::ReduceSquareAdd, float, half,
                                       true>;
template class ReducePartialsEqualSize<popops::ReduceSquareAdd, half, half,
                                       true>;
template class ReducePartialsEqualSize<popops::ReduceSquareAdd, int, int, true>;

template class ReducePartialsEqualSize<popops::ReduceSquareAdd, float, float,
                                       false>;
template class ReducePartialsEqualSize<popops::ReduceSquareAdd, half, float,
                                       false>;
template class ReducePartialsEqualSize<popops::ReduceSquareAdd, float, half,
                                       false>;
template class ReducePartialsEqualSize<popops::ReduceSquareAdd, half, half,
                                       false>;
template class ReducePartialsEqualSize<popops::ReduceSquareAdd, int, int,
                                       false>;

template class ReducePartialsEqualSize<popops::ReduceMul, float, float, true>;
template class ReducePartialsEqualSize<popops::ReduceMul, half, float, true>;
template class ReducePartialsEqualSize<popops::ReduceMul, float, half, true>;
template class ReducePartialsEqualSize<popops::ReduceMul, half, half, true>;
template class ReducePartialsEqualSize<popops::ReduceMul, int, int, true>;

template class ReducePartialsEqualSize<popops::ReduceMul, float, float, false>;
template class ReducePartialsEqualSize<popops::ReduceMul, half, float, false>;
template class ReducePartialsEqualSize<popops::ReduceMul, float, half, false>;
template class ReducePartialsEqualSize<popops::ReduceMul, half, half, false>;
template class ReducePartialsEqualSize<popops::ReduceMul, int, int, false>;

template class ReducePartialsEqualSize<popops::ReduceMax, float, float, true>;
template class ReducePartialsEqualSize<popops::ReduceMax, half, half, true>;
template class ReducePartialsEqualSize<popops::ReduceMax, int, int, true>;

template class ReducePartialsEqualSize<popops::ReduceMax, float, float, false>;
template class ReducePartialsEqualSize<popops::ReduceMax, half, half, false>;
template class ReducePartialsEqualSize<popops::ReduceMax, int, int, false>;

template class ReducePartialsEqualSize<popops::ReduceMin, float, float, true>;
template class ReducePartialsEqualSize<popops::ReduceMin, half, half, true>;
template class ReducePartialsEqualSize<popops::ReduceMin, int, int, true>;

template class ReducePartialsEqualSize<popops::ReduceMin, float, float, false>;
template class ReducePartialsEqualSize<popops::ReduceMin, half, half, false>;
template class ReducePartialsEqualSize<popops::ReduceMin, int, int, false>;

template class ReducePartialsEqualSize<popops::ReduceAnd, bool, bool, true>;
template class ReducePartialsEqualSize<popops::ReduceAnd, bool, bool, false>;

template class ReducePartialsEqualSize<popops::ReduceOr, bool, bool, true>;
template class ReducePartialsEqualSize<popops::ReduceOr, bool, bool, false>;

} // namespace popops
