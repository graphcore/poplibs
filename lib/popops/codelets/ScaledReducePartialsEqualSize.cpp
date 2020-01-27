// Copyright (c) Graphcore Ltd, All rights reserved.
#include "ReducePartialsEqualSize.hpp"

using namespace poplar;

namespace popops {

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
class ScaledReducePartialsEqualSize : public Vertex {
  IS_EXTERNAL_CODELET((IsExternal<ReduceOp, PartialsType>()()));

  ReduceOutput<OutType, isUpdate> out;
  const ShortType outCount;
  ReducePartials<PartialsType> partials;
  const ShortType partialsSizeM1;
  /* Multiplication factor.*/
  /* Actually we just need a scalar here, but creating a vector allows use of a
     SCALED_PTR32, which packs into the rest of the vertex state efficiently
     and saves space (although at the cost of 3 instructions to unpack) */
  Input<Vector<float, PTR_ALIGN32>> k;

public:
  ScaledReducePartialsEqualSize();

  bool compute() {
    const auto fn = computePartialsEqualSizeReduction<ReduceOp, OutType,
                                                      PartialsType, isUpdate>;
    return fn(out, outCount, partialsSizeM1 + 1, partials, k[0]);
  }
};

template class ScaledReducePartialsEqualSize<popops::ReduceAdd, float, float,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceAdd, half, float,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceAdd, float, half,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceAdd, half, half,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceAdd, int, int, true>;

template class ScaledReducePartialsEqualSize<popops::ReduceAdd, float, float,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceAdd, half, float,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceAdd, float, half,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceAdd, half, half,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceAdd, int, int,
                                             false>;

template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, float,
                                             float, true>;
template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, half,
                                             float, true>;
template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, float,
                                             half, true>;
template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, half,
                                             half, true>;
template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, int, int,
                                             true>;

template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, float,
                                             float, false>;
template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, half,
                                             float, false>;
template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, float,
                                             half, false>;
template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, half,
                                             half, false>;
template class ScaledReducePartialsEqualSize<popops::ReduceSquareAdd, int, int,
                                             false>;

template class ScaledReducePartialsEqualSize<popops::ReduceMul, float, float,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceMul, half, float,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceMul, float, half,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceMul, half, half,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceMul, int, int, true>;

template class ScaledReducePartialsEqualSize<popops::ReduceMul, float, float,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceMul, half, float,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceMul, float, half,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceMul, half, half,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceMul, int, int,
                                             false>;

template class ScaledReducePartialsEqualSize<popops::ReduceMax, float, float,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceMax, half, half,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceMax, int, int, true>;

template class ScaledReducePartialsEqualSize<popops::ReduceMax, float, float,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceMax, half, half,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceMax, int, int,
                                             false>;

template class ScaledReducePartialsEqualSize<popops::ReduceMin, float, float,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceMin, half, half,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceMin, int, int, true>;

template class ScaledReducePartialsEqualSize<popops::ReduceMin, float, float,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceMin, half, half,
                                             false>;
template class ScaledReducePartialsEqualSize<popops::ReduceMin, int, int,
                                             false>;

template class ScaledReducePartialsEqualSize<popops::ReduceAnd, bool, bool,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceAnd, bool, bool,
                                             false>;

template class ScaledReducePartialsEqualSize<popops::ReduceOr, bool, bool,
                                             true>;
template class ScaledReducePartialsEqualSize<popops::ReduceOr, bool, bool,
                                             false>;

} // namespace popops
