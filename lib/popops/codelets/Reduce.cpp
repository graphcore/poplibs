// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "ReduceCodelets.hpp"

using namespace poplar;

namespace popops {

/* If `out` were:                                        */

/*   [                                                   */
/*       [ a, b, c, d ],                                 */
/*       [ i, k, l, m, n],                               */
/*       [ x, y, z ],                                    */
/*       [ q ],                                          */
/*   ]                                                   */

/* And `partials` were                                   */

/*   [                                                   */
/*       [ a1, b1, c1, d1 ],                             */
/*       [ a2, b2, c2, d2 ],                             */
/*       [ a3, b3, c3, d3 ],                             */
/*       [ a4, b4, c4, d4, a5, b5, c5, d5 ],             */
/*       [ i1, k1, l1, m1, n1],                          */
/*       [ i2, k2, l2, m2, n2],                          */
/*       [ i3, k3, l3, m3, n3],                          */
/*       [ x1, y1, z1, x2, y2, z2, x3, y3, z3 ],         */
/*       [ x4, y4, z4 ],                                 */
/*       [ q1, q2, q4, q5, q6, q7, q8, q9, q10 ],        */
/*       [ q11, q12 ],                                   */
/*   ]                                                   */

/* Then all the a's from `partials` would be summed to form the a in `out` */
/* and so on.*/

/* `numReductions` is 4 and `numPartials` is {4, 3, 2, 2} in this case.    */

/* Ragedy ends are not allowed. Each partial's size must be an integer     */
/* multiple of its output size.                                            */

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate, ReductionSpecialisation specialisation>
class Reduce : public Vertex {
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
  Reduce();

  IS_EXTERNAL_CODELET((!std::is_same<PartialsType, int>::value &&
                       ((vectorised_8() || vectorised_4()))));

  /* Vector of regions to output. */
  ReduceOutputAlign<OutType, isUpdate, specialisation> out;

  /* The number of input regions (partials) for each output region. */
  /* This should sum to `partials.size()`. */
  Input<Vector<unsigned short, PTR_ALIGN32, 4>> numPartials;

  /* Vector of regions to use as input. */
  Input<VectorList<PartialsType, DELTAN_TYPE, 8, false>> partials;

  bool compute() {
    const auto function = computeReduce<ReduceOp, PartialsType, OutType,
                                        isUpdate, specialisation>;
    return function(out, numPartials, partials, 1.0f);
  }
};

// Specialised reduce to a single output from a single edge
template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
class Reduce<ReduceOp, PartialsType, OutType, isUpdate,
             ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>
    : public Vertex {
private:
  constexpr static bool opHasAssembler() {
    return std::is_same<ReduceOp, ReduceAdd>::value ||
           std::is_same<ReduceOp, ReduceSquareAdd>::value;
  }

public:
  Reduce();
  using AccType = AccType<PartialsType, ReduceOp>;

  constexpr static bool isExternal() {
    return opHasAssembler() && std::is_same<PartialsType, float>::value &&
           !isUpdate;
  }
  IS_EXTERNAL_CODELET(isExternal());
  template <typename T>
  using ReduceOutput =
      typename std::conditional<isUpdate, InOut<T>, Output<T>>::type;
  ReduceOutput<Vector<OutType, ONE_PTR>> out;
  Input<Vector<PartialsType, PTR_ALIGN64, 8>> partials;
  const ShortType numPartials;
  bool compute() {
    AccType acc = ReduceOp::template init<AccType>();
    for (unsigned p = 0; p < numPartials; ++p)
      ReduceOp::update(acc, static_cast<AccType>(partials[p]));

    if (isUpdate) {
      out[0] += static_cast<OutType>(acc);
    } else {
      out[0] = static_cast<OutType>(acc);
    }
    return true;
  }
};

// Specialised reduce to one output region from part of a single edge,
// using independent partialsWidth (address stride) and numOutputs parameters
template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
class Reduce<ReduceOp, PartialsType, OutType, isUpdate,
             ReductionSpecialisation::STRIDED_REDUCE> : public Vertex {
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
  constexpr static bool opIsLogAddWithAssembler() {
    return std::is_same<ReduceOp, ReduceLogAdd>::value &&
           (std::is_same<OutType, half>::value ||
            std::is_same<OutType, float>::value);
  }

public:
  Reduce();
  using AccType = AccType<PartialsType, ReduceOp>;

  constexpr static bool isExternal() {
    return (opIsMaxMinWithAssembler() || opIsAddSquareAddWithAssembler() ||
            opIsLogAddWithAssembler());
  }
  // External codelets require the partials to be a multiple of
  // 64bits to give aligned memory accesses, outputs must be 32 bit aligned
  IS_EXTERNAL_CODELET(isExternal());
  template <typename T>
  using ReduceOutput =
      typename std::conditional<isUpdate, InOut<T>, Output<T>>::type;
  ReduceOutput<Vector<OutType, PTR_ALIGN32, 4>> out;
  Input<Vector<PartialsType, PTR_ALIGN32, 8>> partials;
  ShortType numOutputs;
  ShortType numPartialsM1;
  ShortType partialsWidth;

  bool compute() {
    for (unsigned o = 0; o < numOutputs; ++o) {
      const PartialsType *pPtr = &partials[o];
      AccType acc = ReduceOp::template init<AccType>();
      for (unsigned p = 0; p < numPartialsM1 + 1; ++p) {
        ReduceOp::update(acc, static_cast<AccType>(*pPtr));
        pPtr += partialsWidth;
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
    return true;
  }
};

// Operation: ReduceAdd
template class Reduce<popops::ReduceAdd, float, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, float, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, half, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, half, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, half, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, half, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, float, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, float, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, float, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, float, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, half, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, half, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, int, int, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, int, int, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, float, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, float, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, half, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, half, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, half, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, half, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, float, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, float, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, float, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, float, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, half, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, half, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAdd, int, int, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAdd, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAdd, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAdd, int, int, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

// Operation: ReduceSquareAdd
template class Reduce<popops::ReduceSquareAdd, float, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, float, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, half, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, half, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, half, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, half, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, float, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, float, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, float, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, float, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, half, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, half, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, int, int, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, int, int, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, float, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, float, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, half, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, half, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, half, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, half, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, float, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, float, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, float, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, float, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, half, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, half, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceSquareAdd, int, int, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceSquareAdd, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceSquareAdd, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceSquareAdd, int, int, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

// Operation: ReduceLogAdd
template class Reduce<popops::ReduceLogAdd, float, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceLogAdd, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceLogAdd, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceLogAdd, float, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceLogAdd, float, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceLogAdd, float, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceLogAdd, float, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceLogAdd, float, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceLogAdd, half, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceLogAdd, half, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceLogAdd, half, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceLogAdd, half, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceLogAdd, half, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceLogAdd, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceLogAdd, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceLogAdd, half, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceLogAdd, float, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceLogAdd, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceLogAdd, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceLogAdd, float, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceLogAdd, float, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceLogAdd, float, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceLogAdd, float, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceLogAdd, float, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceLogAdd, half, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceLogAdd, half, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceLogAdd, half, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceLogAdd, half, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceLogAdd, half, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceLogAdd, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceLogAdd, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceLogAdd, half, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

// Operation: ReduceMul
template class Reduce<popops::ReduceMul, float, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, float, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, half, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, half, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, half, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, half, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, float, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, float, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, float, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, float, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, half, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, half, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, int, int, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, int, int, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, float, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, float, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, half, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, half, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, half, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, half, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, float, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, float, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, float, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, float, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, half, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, half, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMul, int, int, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMul, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMul, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMul, int, int, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

// Operation: ReduceMax
template class Reduce<popops::ReduceMax, float, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMax, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMax, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMax, float, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMax, half, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMax, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMax, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMax, half, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMax, int, int, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMax, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMax, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMax, int, int, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMax, float, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMax, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMax, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMax, float, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMax, half, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMax, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMax, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMax, half, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMax, int, int, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMax, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMax, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMax, int, int, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

// Operation: ReduceMin
template class Reduce<popops::ReduceMin, float, float, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMin, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMin, float, float, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMin, float, float, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMin, half, half, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMin, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMin, half, half, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMin, half, half, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMin, int, int, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMin, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMin, int, int, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMin, int, int, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMin, float, float, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMin, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMin, float, float, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMin, float, float, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMin, half, half, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMin, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMin, half, half, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMin, half, half, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceMin, int, int, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceMin, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceMin, int, int, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceMin, int, int, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

// Operation: ReduceAnd
template class Reduce<popops::ReduceAnd, bool, bool, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAnd, bool, bool, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAnd, bool, bool, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAnd, bool, bool, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceAnd, bool, bool, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceAnd, bool, bool, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceAnd, bool, bool, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceAnd, bool, bool, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

// Operation: ReduceOr
template class Reduce<popops::ReduceOr, bool, bool, true,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceOr, bool, bool, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceOr, bool, bool, true,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceOr, bool, bool, true,
                      ReductionSpecialisation::STRIDED_REDUCE>;

template class Reduce<popops::ReduceOr, bool, bool, false,
                      ReductionSpecialisation::DEFAULT>;
template class Reduce<popops::ReduceOr, bool, bool, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_REGIONS>;
template class Reduce<popops::ReduceOr, bool, bool, false,
                      ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT>;
template class Reduce<popops::ReduceOr, bool, bool, false,
                      ReductionSpecialisation::STRIDED_REDUCE>;

} // namespace popops
