// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "ReduceCodelets.hpp"

using namespace poplar;

namespace popops {

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate, ReductionSpecialisation specialisation>
class ScaledReduce : public Vertex {
  // This template handles the first two specialisations
  static_assert(specialisation == ReductionSpecialisation::DEFAULT ||
                    specialisation ==
                        ReductionSpecialisation::SCALAR_OUTPUT_REGIONS,
                "unsupported specialisation");

private:
  constexpr static bool vectorised_8() {
    return std::is_same<ReduceOp, ReduceAdd>::value ||
           std::is_same<ReduceOp, ReduceSquareAdd>::value;
  }
  constexpr static bool vectorised_4() {
    return ((std::is_same<ReduceOp, ReduceMul>::value ||
             std::is_same<ReduceOp, ReduceMax>::value ||
             std::is_same<ReduceOp, ReduceMin>::value) &&
            std::is_same<PartialsType, OutType>::value && !isUpdate);
  }

public:
  ScaledReduce();

  IS_EXTERNAL_CODELET((!std::is_same<PartialsType, int>::value &&
                       ((vectorised_8() || vectorised_4()))));

  /* Vector of regions to output. */
  ReduceOutputAlign<OutType, isUpdate, specialisation> out;

  /* The number of input regions (partials) for each output region. */
  /* This should sum to `partials.size()`. */
  Input<Vector<unsigned short, PTR_ALIGN32, 4>> numPartials;

  /* Vector of regions to use as input. */
  Input<VectorList<PartialsType, DELTAN_TYPE, 8, false>> partials;

  /* Multiplication factor.*/
  /* Actually we just need a scalar here, but creating a vector allows use of a
     PTR_ALIGN32, which packs into the rest of the vertex state efficiently
     and saves space (although at the cost of 3 instructions to unpack) */
  Input<Vector<float, PTR_ALIGN32>> k;

  bool compute() {
    const auto function = computeReduce<ReduceOp, PartialsType, OutType,
                                        isUpdate, specialisation>;
    return function(out, numPartials, partials, k[0]);
  }
};

// Specialised reduce to one output region from a single edge
template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
class ScaledReduce<ReduceOp, PartialsType, OutType, isUpdate,
                   ReductionSpecialisation::SINGLE_OUTPUT_REGION>
    : public Vertex {
private:
  constexpr static bool opIsMaxMinWithAssembler() {
    return (std::is_same<ReduceOp, ReduceMax>::value ||
            std::is_same<ReduceOp, ReduceMin>::value) &&
           (std::is_same<PartialsType, float>::value ||
            std::is_same<PartialsType, half>::value);
  }
  constexpr static bool opIsAddSquareAddWithAssembler() {
    return (std::is_same<ReduceOp, ReduceAdd>::value ||
            std::is_same<ReduceOp, ReduceSquareAdd>::value) &&
           (std::is_same<OutType, float>::value ||
            std::is_same<OutType, half>::value);
  }

public:
  ScaledReduce();
  using AccType = AccType<PartialsType, ReduceOp>;

  constexpr static bool isExternal() {
    return (opIsMaxMinWithAssembler() || opIsAddSquareAddWithAssembler());
  }
  // External codelets require the partials and outputs to be a multiple of
  // 64bits to give aligned memory accesses
  IS_EXTERNAL_CODELET(isExternal());
  template <typename T>
  using ReduceOutput =
      typename std::conditional<isUpdate, InOut<T>, Output<T>>::type;
  ReduceOutput<Vector<OutType, PTR_ALIGN32, 4>> out;
  Input<Vector<PartialsType, PTR_ALIGN32, 8>> partials;
  ShortType numOutputs;
  ShortType numPartials;
  /* Multiplication factor.*/
  /* Actually we just need a scalar here, but creating a vector allows use of a
     PTR_ALIGN32, which packs into the rest of the vertex state efficiently
     and saves space (although at the cost of 3 instructions to unpack) */
  Input<Vector<float, PTR_ALIGN32>> k;

  bool compute() {
    for (unsigned o = 0; o < numOutputs; ++o) {
      const PartialsType *pPtr = &partials[o];
      AccType acc = ReduceOp::template init<AccType>();
      for (unsigned p = 0; p < numPartials; ++p) {
        ReduceOp::update(acc, static_cast<AccType>(*pPtr));
        pPtr += numOutputs;
      }
      const auto scaledOut =
          static_cast<OutType>(static_cast<AccType>(k[0]) * acc);
      if (isUpdate) {
        out[o] += scaledOut;
      } else {
        out[o] = scaledOut;
      }
    }
    return true;
  }
};

template class ScaledReduce<popops::ReduceAdd, float, float, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, float, float, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, float, float, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, half, float, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, half, float, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, half, float, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, float, half, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, float, half, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, float, half, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, half, half, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, half, half, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, half, half, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, int, int, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, int, int, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, int, int, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, float, float, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, float, float, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, float, float, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, half, float, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, half, float, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, half, float, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, float, half, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, float, half, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, float, half, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, half, half, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, half, half, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, half, half, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAdd, int, int, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAdd, int, int, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAdd, int, int, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, float, float, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, float, float, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, float, float, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, half, float, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, half, float, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, half, float, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, float, half, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, float, half, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, float, half, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, half, half, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, half, half, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, half, half, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, int, int, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, int, int, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, int, int, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, float, float, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, float, float, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, float, float, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, half, float, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, half, float, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, half, float, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, float, half, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, float, half, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, float, half, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, half, half, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, half, half, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, half, half, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceSquareAdd, int, int, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceSquareAdd, int, int, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceSquareAdd, int, int, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, float, float, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, float, float, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, float, float, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, half, float, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, half, float, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, half, float, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, float, half, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, float, half, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, float, half, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, half, half, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, half, half, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, half, half, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, int, int, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, int, int, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, int, int, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, float, float, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, float, float, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, float, float, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, half, float, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, half, float, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, half, float, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, float, half, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, float, half, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, float, half, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, half, half, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, half, half, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, half, half, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMul, int, int, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMul, int, int, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMul, int, int, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMax, float, float, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMax, float, float, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMax, float, float, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMax, half, half, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMax, half, half, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMax, half, half, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMax, int, int, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMax, int, int, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMax, int, int, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMax, float, float, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMax, float, float, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMax, float, float, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMax, half, half, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMax, half, half, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMax, half, half, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMax, int, int, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMax, int, int, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMax, int, int, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMin, float, float, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMin, float, float, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMin, float, float, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMin, half, half, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMin, half, half, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMin, half, half, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMin, int, int, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMin, int, int, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMin, int, int, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMin, float, float, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMin, float, float, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMin, float, float, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMin, half, half, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMin, half, half, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMin, half, half, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceMin, int, int, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceMin, int, int, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceMin, int, int, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAnd, bool, bool, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAnd, bool, bool, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAnd, bool, bool, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceAnd, bool, bool, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceAnd, bool, bool, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceAnd, bool, bool, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceOr, bool, bool, true,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceOr, bool, bool, true,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceOr, bool, bool, true,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

template class ScaledReduce<popops::ReduceOr, bool, bool, false,
                            ReductionSpecialisation::DEFAULT>;
template class ScaledReduce<popops::ReduceOr, bool, bool, false,
                            ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class ScaledReduce<popops::ReduceOr, bool, bool, false,
                            ReductionSpecialisation::SINGLE_OUTPUT_REGION>;

} // namespace popops
