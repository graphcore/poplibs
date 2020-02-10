// Copyright (c) Graphcore Ltd, All rights reserved.
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
          bool isUpdate, unsigned specialisation>
class Reduce : public Vertex {
  // This template handles the first two specialisations
  static_assert(specialisation == 0 || specialisation == 1,
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
class Reduce<ReduceOp, PartialsType, OutType, isUpdate, 2u> : public Vertex {
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

// Specialised reduce to one output region from a single edge
template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
class Reduce<ReduceOp, PartialsType, OutType, isUpdate, 3u> : public Vertex {
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
           std::is_same<OutType, float>::value;
  }

public:
  Reduce();
  using AccType = AccType<PartialsType, ReduceOp>;

  constexpr static bool isExternal() {
    return (opIsMaxMinWithAssembler() || opIsAddSquareAddWithAssembler()) &&
           !isUpdate;
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
  bool compute() {
    for (unsigned o = 0; o < numOutputs; ++o) {
      const PartialsType *pPtr = &partials[o];
      AccType acc = ReduceOp::template init<AccType>();
      for (unsigned p = 0; p < numPartials; ++p) {
        ReduceOp::update(acc, static_cast<AccType>(*pPtr));
        pPtr += numOutputs;
      }

      if (isUpdate) {
        out[o] += static_cast<OutType>(acc);
      } else {
        out[o] = static_cast<OutType>(acc);
      }
    }
    return true;
  }
};

template class Reduce<popops::ReduceAdd, float, float, true, 0u>;
template class Reduce<popops::ReduceAdd, float, float, true, 1u>;
template class Reduce<popops::ReduceAdd, float, float, true, 2u>;
template class Reduce<popops::ReduceAdd, float, float, true, 3u>;

template class Reduce<popops::ReduceAdd, half, float, true, 0u>;
template class Reduce<popops::ReduceAdd, half, float, true, 1u>;
template class Reduce<popops::ReduceAdd, half, float, true, 2u>;
template class Reduce<popops::ReduceAdd, half, float, true, 3u>;

template class Reduce<popops::ReduceAdd, float, half, true, 0u>;
template class Reduce<popops::ReduceAdd, float, half, true, 1u>;
template class Reduce<popops::ReduceAdd, float, half, true, 2u>;
template class Reduce<popops::ReduceAdd, float, half, true, 3u>;

template class Reduce<popops::ReduceAdd, half, half, true, 0u>;
template class Reduce<popops::ReduceAdd, half, half, true, 1u>;
template class Reduce<popops::ReduceAdd, half, half, true, 2u>;
template class Reduce<popops::ReduceAdd, half, half, true, 3u>;

template class Reduce<popops::ReduceAdd, int, int, true, 0u>;
template class Reduce<popops::ReduceAdd, int, int, true, 1u>;
template class Reduce<popops::ReduceAdd, int, int, true, 2u>;
template class Reduce<popops::ReduceAdd, int, int, true, 3u>;

template class Reduce<popops::ReduceAdd, float, float, false, 0u>;
template class Reduce<popops::ReduceAdd, float, float, false, 1u>;
template class Reduce<popops::ReduceAdd, float, float, false, 2u>;
template class Reduce<popops::ReduceAdd, float, float, false, 3u>;

template class Reduce<popops::ReduceAdd, half, float, false, 0u>;
template class Reduce<popops::ReduceAdd, half, float, false, 1u>;
template class Reduce<popops::ReduceAdd, half, float, false, 2u>;
template class Reduce<popops::ReduceAdd, half, float, false, 3u>;

template class Reduce<popops::ReduceAdd, float, half, false, 0u>;
template class Reduce<popops::ReduceAdd, float, half, false, 1u>;
template class Reduce<popops::ReduceAdd, float, half, false, 2u>;
template class Reduce<popops::ReduceAdd, float, half, false, 3u>;

template class Reduce<popops::ReduceAdd, half, half, false, 0u>;
template class Reduce<popops::ReduceAdd, half, half, false, 1u>;
template class Reduce<popops::ReduceAdd, half, half, false, 2u>;
template class Reduce<popops::ReduceAdd, half, half, false, 3u>;

template class Reduce<popops::ReduceAdd, int, int, false, 0u>;
template class Reduce<popops::ReduceAdd, int, int, false, 1u>;
template class Reduce<popops::ReduceAdd, int, int, false, 2u>;
template class Reduce<popops::ReduceAdd, int, int, false, 3u>;

template class Reduce<popops::ReduceSquareAdd, float, float, true, 0u>;
template class Reduce<popops::ReduceSquareAdd, float, float, true, 1u>;
template class Reduce<popops::ReduceSquareAdd, float, float, true, 2u>;
template class Reduce<popops::ReduceSquareAdd, float, float, true, 3u>;

template class Reduce<popops::ReduceSquareAdd, half, float, true, 0u>;
template class Reduce<popops::ReduceSquareAdd, half, float, true, 1u>;
template class Reduce<popops::ReduceSquareAdd, half, float, true, 2u>;
template class Reduce<popops::ReduceSquareAdd, half, float, true, 3u>;

template class Reduce<popops::ReduceSquareAdd, float, half, true, 0u>;
template class Reduce<popops::ReduceSquareAdd, float, half, true, 1u>;
template class Reduce<popops::ReduceSquareAdd, float, half, true, 2u>;
template class Reduce<popops::ReduceSquareAdd, float, half, true, 3u>;

template class Reduce<popops::ReduceSquareAdd, half, half, true, 0u>;
template class Reduce<popops::ReduceSquareAdd, half, half, true, 1u>;
template class Reduce<popops::ReduceSquareAdd, half, half, true, 2u>;
template class Reduce<popops::ReduceSquareAdd, half, half, true, 3u>;

template class Reduce<popops::ReduceSquareAdd, int, int, true, 0u>;
template class Reduce<popops::ReduceSquareAdd, int, int, true, 1u>;
template class Reduce<popops::ReduceSquareAdd, int, int, true, 2u>;
template class Reduce<popops::ReduceSquareAdd, int, int, true, 3u>;

template class Reduce<popops::ReduceSquareAdd, float, float, false, 0u>;
template class Reduce<popops::ReduceSquareAdd, float, float, false, 1u>;
template class Reduce<popops::ReduceSquareAdd, float, float, false, 2u>;
template class Reduce<popops::ReduceSquareAdd, float, float, false, 3u>;

template class Reduce<popops::ReduceSquareAdd, half, float, false, 0u>;
template class Reduce<popops::ReduceSquareAdd, half, float, false, 1u>;
template class Reduce<popops::ReduceSquareAdd, half, float, false, 2u>;
template class Reduce<popops::ReduceSquareAdd, half, float, false, 3u>;

template class Reduce<popops::ReduceSquareAdd, float, half, false, 0u>;
template class Reduce<popops::ReduceSquareAdd, float, half, false, 1u>;
template class Reduce<popops::ReduceSquareAdd, float, half, false, 2u>;
template class Reduce<popops::ReduceSquareAdd, float, half, false, 3u>;

template class Reduce<popops::ReduceSquareAdd, half, half, false, 0u>;
template class Reduce<popops::ReduceSquareAdd, half, half, false, 1u>;
template class Reduce<popops::ReduceSquareAdd, half, half, false, 2u>;
template class Reduce<popops::ReduceSquareAdd, half, half, false, 3u>;

template class Reduce<popops::ReduceSquareAdd, int, int, false, 0u>;
template class Reduce<popops::ReduceSquareAdd, int, int, false, 1u>;
template class Reduce<popops::ReduceSquareAdd, int, int, false, 2u>;
template class Reduce<popops::ReduceSquareAdd, int, int, false, 3u>;

template class Reduce<popops::ReduceMul, float, float, true, 0u>;
template class Reduce<popops::ReduceMul, float, float, true, 1u>;
template class Reduce<popops::ReduceMul, float, float, true, 2u>;
template class Reduce<popops::ReduceMul, float, float, true, 3u>;

template class Reduce<popops::ReduceMul, half, float, true, 0u>;
template class Reduce<popops::ReduceMul, half, float, true, 1u>;
template class Reduce<popops::ReduceMul, half, float, true, 2u>;
template class Reduce<popops::ReduceMul, half, float, true, 3u>;

template class Reduce<popops::ReduceMul, float, half, true, 0u>;
template class Reduce<popops::ReduceMul, float, half, true, 1u>;
template class Reduce<popops::ReduceMul, float, half, true, 2u>;
template class Reduce<popops::ReduceMul, float, half, true, 3u>;

template class Reduce<popops::ReduceMul, half, half, true, 0u>;
template class Reduce<popops::ReduceMul, half, half, true, 1u>;
template class Reduce<popops::ReduceMul, half, half, true, 2u>;
template class Reduce<popops::ReduceMul, half, half, true, 3u>;

template class Reduce<popops::ReduceMul, int, int, true, 0u>;
template class Reduce<popops::ReduceMul, int, int, true, 1u>;
template class Reduce<popops::ReduceMul, int, int, true, 2u>;
template class Reduce<popops::ReduceMul, int, int, true, 3u>;

template class Reduce<popops::ReduceMul, float, float, false, 0u>;
template class Reduce<popops::ReduceMul, float, float, false, 1u>;
template class Reduce<popops::ReduceMul, float, float, false, 2u>;
template class Reduce<popops::ReduceMul, float, float, false, 3u>;

template class Reduce<popops::ReduceMul, half, float, false, 0u>;
template class Reduce<popops::ReduceMul, half, float, false, 1u>;
template class Reduce<popops::ReduceMul, half, float, false, 2u>;
template class Reduce<popops::ReduceMul, half, float, false, 3u>;

template class Reduce<popops::ReduceMul, float, half, false, 0u>;
template class Reduce<popops::ReduceMul, float, half, false, 1u>;
template class Reduce<popops::ReduceMul, float, half, false, 2u>;
template class Reduce<popops::ReduceMul, float, half, false, 3u>;

template class Reduce<popops::ReduceMul, half, half, false, 0u>;
template class Reduce<popops::ReduceMul, half, half, false, 1u>;
template class Reduce<popops::ReduceMul, half, half, false, 2u>;
template class Reduce<popops::ReduceMul, half, half, false, 3u>;

template class Reduce<popops::ReduceMul, int, int, false, 0u>;
template class Reduce<popops::ReduceMul, int, int, false, 1u>;
template class Reduce<popops::ReduceMul, int, int, false, 2u>;
template class Reduce<popops::ReduceMul, int, int, false, 3u>;

template class Reduce<popops::ReduceMax, float, float, true, 0u>;
template class Reduce<popops::ReduceMax, float, float, true, 1u>;
template class Reduce<popops::ReduceMax, float, float, true, 2u>;
template class Reduce<popops::ReduceMax, float, float, true, 3u>;

template class Reduce<popops::ReduceMax, half, half, true, 0u>;
template class Reduce<popops::ReduceMax, half, half, true, 1u>;
template class Reduce<popops::ReduceMax, half, half, true, 2u>;
template class Reduce<popops::ReduceMax, half, half, true, 3u>;

template class Reduce<popops::ReduceMax, int, int, true, 0u>;
template class Reduce<popops::ReduceMax, int, int, true, 1u>;
template class Reduce<popops::ReduceMax, int, int, true, 2u>;
template class Reduce<popops::ReduceMax, int, int, true, 3u>;

template class Reduce<popops::ReduceMax, float, float, false, 0u>;
template class Reduce<popops::ReduceMax, float, float, false, 1u>;
template class Reduce<popops::ReduceMax, float, float, false, 2u>;
template class Reduce<popops::ReduceMax, float, float, false, 3u>;

template class Reduce<popops::ReduceMax, half, half, false, 0u>;
template class Reduce<popops::ReduceMax, half, half, false, 1u>;
template class Reduce<popops::ReduceMax, half, half, false, 2u>;
template class Reduce<popops::ReduceMax, half, half, false, 3u>;

template class Reduce<popops::ReduceMax, int, int, false, 0u>;
template class Reduce<popops::ReduceMax, int, int, false, 1u>;
template class Reduce<popops::ReduceMax, int, int, false, 2u>;
template class Reduce<popops::ReduceMax, int, int, false, 3u>;

template class Reduce<popops::ReduceMin, float, float, true, 0u>;
template class Reduce<popops::ReduceMin, float, float, true, 1u>;
template class Reduce<popops::ReduceMin, float, float, true, 2u>;
template class Reduce<popops::ReduceMin, float, float, true, 3u>;

template class Reduce<popops::ReduceMin, half, half, true, 0u>;
template class Reduce<popops::ReduceMin, half, half, true, 1u>;
template class Reduce<popops::ReduceMin, half, half, true, 2u>;
template class Reduce<popops::ReduceMin, half, half, true, 3u>;

template class Reduce<popops::ReduceMin, int, int, true, 0u>;
template class Reduce<popops::ReduceMin, int, int, true, 1u>;
template class Reduce<popops::ReduceMin, int, int, true, 2u>;
template class Reduce<popops::ReduceMin, int, int, true, 3u>;

template class Reduce<popops::ReduceMin, float, float, false, 0u>;
template class Reduce<popops::ReduceMin, float, float, false, 1u>;
template class Reduce<popops::ReduceMin, float, float, false, 2u>;
template class Reduce<popops::ReduceMin, float, float, false, 3u>;

template class Reduce<popops::ReduceMin, half, half, false, 0u>;
template class Reduce<popops::ReduceMin, half, half, false, 1u>;
template class Reduce<popops::ReduceMin, half, half, false, 2u>;
template class Reduce<popops::ReduceMin, half, half, false, 3u>;

template class Reduce<popops::ReduceMin, int, int, false, 0u>;
template class Reduce<popops::ReduceMin, int, int, false, 1u>;
template class Reduce<popops::ReduceMin, int, int, false, 2u>;
template class Reduce<popops::ReduceMin, int, int, false, 3u>;

template class Reduce<popops::ReduceAnd, bool, bool, true, 0u>;
template class Reduce<popops::ReduceAnd, bool, bool, true, 1u>;
template class Reduce<popops::ReduceAnd, bool, bool, true, 2u>;
template class Reduce<popops::ReduceAnd, bool, bool, true, 3u>;

template class Reduce<popops::ReduceAnd, bool, bool, false, 0u>;
template class Reduce<popops::ReduceAnd, bool, bool, false, 1u>;
template class Reduce<popops::ReduceAnd, bool, bool, false, 2u>;
template class Reduce<popops::ReduceAnd, bool, bool, false, 3u>;

template class Reduce<popops::ReduceOr, bool, bool, true, 0u>;
template class Reduce<popops::ReduceOr, bool, bool, true, 1u>;
template class Reduce<popops::ReduceOr, bool, bool, true, 2u>;
template class Reduce<popops::ReduceOr, bool, bool, true, 3u>;

template class Reduce<popops::ReduceOr, bool, bool, false, 0u>;
template class Reduce<popops::ReduceOr, bool, bool, false, 1u>;
template class Reduce<popops::ReduceOr, bool, bool, false, 2u>;
template class Reduce<popops::ReduceOr, bool, bool, false, 3u>;

} // namespace popops
