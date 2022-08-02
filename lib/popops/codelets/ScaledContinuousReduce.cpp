// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "ContinuousReduce.hpp"

using namespace poplar;

namespace popops {

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
class ScaledContinuousReduce : public Vertex {
private:
  constexpr static bool opIsLogAdd =
      std::is_same<ReduceOp, ReduceLogAdd>::value;

public:
  ScaledContinuousReduce();
  using AccType = AccType<PartialsType, ReduceOp>;

  IS_EXTERNAL_CODELET(
      (useExternal<ReduceOp, PartialsType, OutType, isUpdate>()));

  Input<Vector<PartialsType, PTR_ALIGN32, 8, false>> partials;
  ROT<OutType, isUpdate> out;
  const ShortType numOutputsM1;
  const ShortType numPartials;
  // Leave this as a vector so as to be consistent with the other reduction
  // codelets, even though we are not using a SCALED_PTR32 in this case
  Input<Vector<float, ONE_PTR>> k; // Scale factor

  void compute() {
    for (unsigned o = 0; o < numOutputsM1 + 1; ++o) {
      AccType acc = ReduceOp::template init<AccType>();
      for (unsigned p = 0; p < numPartials; ++p) {
        const auto index = (o * numPartials) + p;
        ReduceOp::update(acc, partials[index]);
      }
      // Apply scale.  For log-probability arithmetic this is an add.
      const auto scaledAcc =
          opIsLogAdd ? static_cast<OutType>(acc + static_cast<AccType>(k[0]))
                     : static_cast<OutType>(acc * static_cast<AccType>(k[0]));
      if constexpr (isUpdate) {
        if constexpr (opIsLogAdd) {
          ReduceOp::update(out[o], scaledAcc);
        } else {
          out[o] += scaledAcc;
        }
      } else {
        out[o] = scaledAcc;
      }
    }
  }
};
// Operation: ReduceAdd
template class ScaledContinuousReduce<popops::ReduceAdd, float, float, true>;
template class ScaledContinuousReduce<popops::ReduceAdd, half, float, true>;
template class ScaledContinuousReduce<popops::ReduceAdd, float, half, true>;
template class ScaledContinuousReduce<popops::ReduceAdd, half, half, true>;
template class ScaledContinuousReduce<popops::ReduceAdd, int, int, true>;

template class ScaledContinuousReduce<popops::ReduceAdd, float, float, false>;
template class ScaledContinuousReduce<popops::ReduceAdd, half, float, false>;
template class ScaledContinuousReduce<popops::ReduceAdd, float, half, false>;
template class ScaledContinuousReduce<popops::ReduceAdd, half, half, false>;
template class ScaledContinuousReduce<popops::ReduceAdd, int, int, false>;

// Operation: ReduceSquareAdd
template class ScaledContinuousReduce<popops::ReduceSquareAdd, float, float,
                                      true>;
template class ScaledContinuousReduce<popops::ReduceSquareAdd, half, float,
                                      true>;
template class ScaledContinuousReduce<popops::ReduceSquareAdd, float, half,
                                      true>;
template class ScaledContinuousReduce<popops::ReduceSquareAdd, half, half,
                                      true>;
template class ScaledContinuousReduce<popops::ReduceSquareAdd, int, int, true>;

template class ScaledContinuousReduce<popops::ReduceSquareAdd, float, float,
                                      false>;
template class ScaledContinuousReduce<popops::ReduceSquareAdd, half, float,
                                      false>;
template class ScaledContinuousReduce<popops::ReduceSquareAdd, float, half,
                                      false>;
template class ScaledContinuousReduce<popops::ReduceSquareAdd, half, half,
                                      false>;
template class ScaledContinuousReduce<popops::ReduceSquareAdd, int, int, false>;

// Operation: ReduceLogAdd
template class ScaledContinuousReduce<popops::ReduceLogAdd, float, float, true>;
template class ScaledContinuousReduce<popops::ReduceLogAdd, half, float, true>;
template class ScaledContinuousReduce<popops::ReduceLogAdd, float, half, true>;
template class ScaledContinuousReduce<popops::ReduceLogAdd, half, half, true>;

template class ScaledContinuousReduce<popops::ReduceLogAdd, float, float,
                                      false>;
template class ScaledContinuousReduce<popops::ReduceLogAdd, half, float, false>;
template class ScaledContinuousReduce<popops::ReduceLogAdd, float, half, false>;
template class ScaledContinuousReduce<popops::ReduceLogAdd, half, half, false>;

// Operation: ReduceMul
template class ScaledContinuousReduce<popops::ReduceMul, float, float, true>;
template class ScaledContinuousReduce<popops::ReduceMul, half, float, true>;
template class ScaledContinuousReduce<popops::ReduceMul, float, half, true>;
template class ScaledContinuousReduce<popops::ReduceMul, half, half, true>;
template class ScaledContinuousReduce<popops::ReduceMul, int, int, true>;

template class ScaledContinuousReduce<popops::ReduceMul, float, float, false>;
template class ScaledContinuousReduce<popops::ReduceMul, half, float, false>;
template class ScaledContinuousReduce<popops::ReduceMul, float, half, false>;
template class ScaledContinuousReduce<popops::ReduceMul, half, half, false>;
template class ScaledContinuousReduce<popops::ReduceMul, int, int, false>;

// Operation: ReduceMax
template class ScaledContinuousReduce<popops::ReduceMax, float, float, true>;
template class ScaledContinuousReduce<popops::ReduceMax, half, half, true>;
template class ScaledContinuousReduce<popops::ReduceMax, int, int, true>;

template class ScaledContinuousReduce<popops::ReduceMax, float, float, false>;
template class ScaledContinuousReduce<popops::ReduceMax, half, half, false>;
template class ScaledContinuousReduce<popops::ReduceMax, int, int, false>;

// Operation: ReduceMin
template class ScaledContinuousReduce<popops::ReduceMin, float, float, true>;
template class ScaledContinuousReduce<popops::ReduceMin, half, half, true>;
template class ScaledContinuousReduce<popops::ReduceMin, int, int, true>;

template class ScaledContinuousReduce<popops::ReduceMin, float, float, false>;
template class ScaledContinuousReduce<popops::ReduceMin, half, half, false>;
template class ScaledContinuousReduce<popops::ReduceMin, int, int, false>;

// Operation: ReduceAnd
template class ScaledContinuousReduce<popops::ReduceAnd, bool, bool, true>;
template class ScaledContinuousReduce<popops::ReduceAnd, bool, bool, false>;

// Operation: ReduceOr
template class ScaledContinuousReduce<popops::ReduceOr, bool, bool, true>;
template class ScaledContinuousReduce<popops::ReduceOr, bool, bool, false>;

} // namespace popops
