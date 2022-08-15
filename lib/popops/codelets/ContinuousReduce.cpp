// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "ContinuousReduce.hpp"

using namespace poplar;

namespace popops {

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
class ContinuousReduce : public Vertex {
public:
  ContinuousReduce();
  using AccType = AccType<PartialsType, ReduceOp>;

  IS_EXTERNAL_CODELET(
      (useExternal<ReduceOp, PartialsType, OutType, isUpdate>()));

  Input<Vector<PartialsType, PTR_ALIGN32, 8, false>> partials;
  ROT<OutType, isUpdate> out;
  const ShortType numOutputsM1;
  const ShortType numPartials;

  void compute() {
    for (unsigned o = 0; o < numOutputsM1 + 1; ++o) {
      AccType acc = ReduceOp::template init<AccType>();
      for (unsigned p = 0; p < numPartials; ++p) {
        const auto index = (o * numPartials) + p;
        ReduceOp::update(acc, static_cast<AccType>(partials[index]));
      }
      if constexpr (isUpdate) {
        if constexpr (std::is_same<ReduceOp, ReduceLogAdd>::value) {
          ReduceOp::update(out[o], acc);
        } else {
          out[o] += static_cast<OutType>(acc);
        }
      } else {
        out[o] = static_cast<OutType>(acc);
      }
    }
  }
};

// Operation: ReduceAdd
template class ContinuousReduce<popops::ReduceAdd, float, float, true>;
template class ContinuousReduce<popops::ReduceAdd, half, float, true>;
template class ContinuousReduce<popops::ReduceAdd, float, half, true>;
template class ContinuousReduce<popops::ReduceAdd, half, half, true>;
template class ContinuousReduce<popops::ReduceAdd, int, int, true>;

template class ContinuousReduce<popops::ReduceAdd, float, float, false>;
template class ContinuousReduce<popops::ReduceAdd, half, float, false>;
template class ContinuousReduce<popops::ReduceAdd, float, half, false>;
template class ContinuousReduce<popops::ReduceAdd, half, half, false>;
template class ContinuousReduce<popops::ReduceAdd, int, int, false>;

// Operation: ReduceSquareAdd
template class ContinuousReduce<popops::ReduceSquareAdd, float, float, true>;
template class ContinuousReduce<popops::ReduceSquareAdd, half, float, true>;
template class ContinuousReduce<popops::ReduceSquareAdd, float, half, true>;
template class ContinuousReduce<popops::ReduceSquareAdd, half, half, true>;
template class ContinuousReduce<popops::ReduceSquareAdd, int, int, true>;

template class ContinuousReduce<popops::ReduceSquareAdd, float, float, false>;
template class ContinuousReduce<popops::ReduceSquareAdd, half, float, false>;
template class ContinuousReduce<popops::ReduceSquareAdd, float, half, false>;
template class ContinuousReduce<popops::ReduceSquareAdd, half, half, false>;
template class ContinuousReduce<popops::ReduceSquareAdd, int, int, false>;

// Operation: ReduceLogAdd
template class ContinuousReduce<popops::ReduceLogAdd, float, float, true>;
template class ContinuousReduce<popops::ReduceLogAdd, half, float, true>;
template class ContinuousReduce<popops::ReduceLogAdd, float, half, true>;
template class ContinuousReduce<popops::ReduceLogAdd, half, half, true>;

template class ContinuousReduce<popops::ReduceLogAdd, float, float, false>;
template class ContinuousReduce<popops::ReduceLogAdd, half, float, false>;
template class ContinuousReduce<popops::ReduceLogAdd, float, half, false>;
template class ContinuousReduce<popops::ReduceLogAdd, half, half, false>;

template class ContinuousReduce<popops::ReduceMul, float, float, true>;
template class ContinuousReduce<popops::ReduceMul, half, float, true>;
template class ContinuousReduce<popops::ReduceMul, float, half, true>;
template class ContinuousReduce<popops::ReduceMul, half, half, true>;
template class ContinuousReduce<popops::ReduceMul, int, int, true>;

// Operation: ReduceMul
template class ContinuousReduce<popops::ReduceMul, float, float, false>;
template class ContinuousReduce<popops::ReduceMul, half, float, false>;
template class ContinuousReduce<popops::ReduceMul, float, half, false>;
template class ContinuousReduce<popops::ReduceMul, half, half, false>;
template class ContinuousReduce<popops::ReduceMul, int, int, false>;

// Operation: ReduceMax
template class ContinuousReduce<popops::ReduceMax, float, float, true>;
template class ContinuousReduce<popops::ReduceMax, half, half, true>;
template class ContinuousReduce<popops::ReduceMax, int, int, true>;
template class ContinuousReduce<popops::ReduceMax, unsigned int, unsigned int,
                                true>;

template class ContinuousReduce<popops::ReduceMax, float, float, false>;
template class ContinuousReduce<popops::ReduceMax, half, half, false>;
template class ContinuousReduce<popops::ReduceMax, int, int, false>;
template class ContinuousReduce<popops::ReduceMax, unsigned int, unsigned int,
                                false>;

// Operation: ReduceMin
template class ContinuousReduce<popops::ReduceMin, float, float, true>;
template class ContinuousReduce<popops::ReduceMin, half, half, true>;
template class ContinuousReduce<popops::ReduceMin, int, int, true>;
template class ContinuousReduce<popops::ReduceMin, unsigned int, unsigned int,
                                true>;

template class ContinuousReduce<popops::ReduceMin, float, float, false>;
template class ContinuousReduce<popops::ReduceMin, half, half, false>;
template class ContinuousReduce<popops::ReduceMin, int, int, false>;
template class ContinuousReduce<popops::ReduceMin, unsigned int, unsigned int,
                                false>;

// Operation: ReduceAnd
template class ContinuousReduce<popops::ReduceAnd, bool, bool, true>;
template class ContinuousReduce<popops::ReduceAnd, bool, bool, false>;

// Operation: ReduceOr
template class ContinuousReduce<popops::ReduceOr, bool, bool, true>;
template class ContinuousReduce<popops::ReduceOr, bool, bool, false>;

} // namespace popops
