#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <limits>

#include "util.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;


#ifdef __IPU__
// For real implementation
using ShortType = unsigned short;
#else
// To avoid size overflow on CPU implementation
using ShortType = unsigned;
#endif

namespace popops {

struct ReduceAdd {
  template <typename T>
  static T init() { return 0; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    acc += static_cast<OutType>(val);
  }
};

struct ReduceSquareAdd {
  template <typename T>
  static T init() { return 0; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    auto valOutType = static_cast<OutType>(val);
    acc += valOutType * valOutType;
  }
};

struct ReduceMul {
  template <typename T>
  static T init() { return 1; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    acc *= static_cast<OutType>(val);
  }
};

struct ReduceMax {
  template <typename T>
  static T init() { return std::numeric_limits<T>::lowest(); }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) { acc = max(acc, val); }
};

struct ReduceMin {
  template <typename T>
  static T init() { return std::numeric_limits<T>::max(); }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) { acc = min(acc, val); }
};

struct ReduceAnd {
  template <typename T>
  static T init() { return true; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) { acc = acc && val; }
};

struct ReduceOr {
  template <typename T>
  static T init() { return false; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) { acc = acc || val; }
};

// Reduce has a number of implementations:
// specialisation=0 for general 2D vertices
// specialisation=1 for 2D vertices with a size1 output region
// specialisation=2 for 1D vertices with a single output, a single input edge
//                  and no scaling
// specialisation=3 for 1D vertices with a single output edge, a single input
//                  edge and no scaling. The input and output must be aligned
//                  multiples of 8 bytes.

/** Generic vertex template for all reductions. The template parameters provide
 *  the information on types, what the reduction operator is, whether to
 *  update in place or not etc. */

template <typename OutType, bool isUpdate, unsigned specialisation>
using ReduceOutputAlign =
  typename std::conditional<isUpdate,
        InOut<VectorList<OutType, DELTAN, specialisation == 1 ? 4 : 8>>,
        Output<VectorList<OutType, DELTAN, specialisation == 1 ? 4 : 8>>>::type;


template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate, unsigned specialisation>
static bool computeReduce(
                  ReduceOutputAlign<OutType, isUpdate, specialisation> out,
                  Input<Vector<unsigned short, SCALED_PTR32, 4>> numPartials,
                  Input<VectorList<PartialsType, DELTAN, 8, false>> partials,
                  float k) {
    /* The number of output regions. */
    unsigned numReductions = out.size();

    /* The current offset into the partials vector. */
    unsigned pidx = 0;

    /* Loop through all the output regions. */
    for (unsigned r = 0; r < numReductions; ++r) {
      /* The number of output elements in the region. */
      unsigned numElem = out[r].size();
      /* How many partials input partial regions for this reduction. */
      unsigned numPartials_r = numPartials[r];

      /* Loop through the elements in the region. */
      for (unsigned outIdx = 0; outIdx < numElem; ++outIdx) {

        /* Calculate the sum of this element... */
        OutType acc = ReduceOp::template init<OutType>();

        /* ..by summing the corresponding element in the partials regions. */
        for (unsigned p = 0; p < numPartials_r; ++p) {
          assert(partials[pidx + p].size() % numElem == 0);

          /* Sum them all */
          for (unsigned o = outIdx; o < partials[pidx + p].size();
               o += numElem) {
            ReduceOp::update(acc, partials[pidx + p][o]);
          }
        }

        acc = static_cast<OutType>(k) * acc;

        /* Store it. */
        if (isUpdate) {
          out[r][outIdx] += acc;
        } else {
          out[r][outIdx] = acc;
        }
      }
      /* And skip forward in the partials vector to the next reduction. */
      pidx += numPartials_r;
    }
    return true;
  }

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


template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate, unsigned specialisation>
class Reduce : public Vertex {
  // This template handles the first two specialisations
  static_assert(specialisation == 0 || specialisation == 1,
                "unsupported specialisation");
private:

  constexpr static bool vectorised_8() {
    return std::is_same<ReduceOp, ReduceAdd>::value
                         || std::is_same<ReduceOp, ReduceSquareAdd>::value;
  }
  constexpr static bool vectorised_4() {
    return ((std::is_same<ReduceOp, ReduceMul>::value
            || std::is_same<ReduceOp, ReduceMax>::value
            || std::is_same<ReduceOp, ReduceMin>::value)
            && std::is_same<PartialsType, OutType>::value
            && !isUpdate);
  }
public:
  Reduce();

  IS_EXTERNAL_CODELET((!std::is_same<PartialsType, int>::value
                      && ((vectorised_8() || vectorised_4()))));

  /* Vector of regions to output. */
  ReduceOutputAlign<OutType, isUpdate, specialisation> out;

  /* The number of input regions (partials) for each output region. */
  /* This should sum to `partials.size()`. */
  Input<Vector<unsigned short, SCALED_PTR32, 4>> numPartials;

  /* Vector of regions to use as input. */
  Input<VectorList<PartialsType, VectorListLayout::DELTAN, 8, false>> partials;

  bool compute() {
    const auto function = computeReduce<ReduceOp, PartialsType, OutType,
                                                    isUpdate, specialisation>;
    return function(out, numPartials, partials, 1.0f);
  }

};

template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate, unsigned specialisation>
class ScaledReduce : public Vertex {
  // This template handles the first two specialisations
  static_assert(specialisation == 0 || specialisation == 1,
                "unsupported specialisation");
private:

  constexpr static bool vectorised_8() {
    return std::is_same<ReduceOp, ReduceAdd>::value
                         || std::is_same<ReduceOp, ReduceSquareAdd>::value;
  }
  constexpr static bool vectorised_4() {
    return ((std::is_same<ReduceOp, ReduceMul>::value
            || std::is_same<ReduceOp, ReduceMax>::value
            || std::is_same<ReduceOp, ReduceMin>::value)
            && std::is_same<PartialsType, OutType>::value
            && !isUpdate);
  }
public:
  ScaledReduce();

  IS_EXTERNAL_CODELET((!std::is_same<PartialsType, int>::value
                      && ((vectorised_8() || vectorised_4()))));

  /* Vector of regions to output. */
  ReduceOutputAlign<OutType, isUpdate, specialisation> out;

  /* The number of input regions (partials) for each output region. */
  /* This should sum to `partials.size()`. */
  Input<Vector<unsigned short, SCALED_PTR32, 4>> numPartials;

  /* Vector of regions to use as input. */
  Input<VectorList<PartialsType, VectorListLayout::DELTAN, 8, false>> partials;

  /* Multiplication factor.*/
  /* Actually we just need a scalar here, but creating a vector allows use of a
     SCALED_PTR32, which packs into the rest of the vertex state efficiently
     and saves space (although at the cost of 3 instructions to unpack) */
  Input<Vector<float, SCALED_PTR32>> k;

  bool compute() {
    const auto function = computeReduce<ReduceOp, PartialsType, OutType,
                                                    isUpdate, specialisation>;
    return function(out, numPartials, partials, k[0]);
  }

};

template <typename OutType, bool isUpdate>
using ROT =
  typename std::conditional<isUpdate,
      InOut<Vector<OutType, SCALED_PTR32, 4>>,
      Output<Vector<OutType, SCALED_PTR32, 4>>>::type;


template<typename ReduceOp, typename PartialsType,
         typename OutType, bool isUpdate>
class ContinuousReduce : public Vertex {
public:
  ContinuousReduce();
  static constexpr bool useExternal() {
    bool externalOp =  std::is_same<ReduceOp, ReduceAdd>::value ||
                       std::is_same<ReduceOp, ReduceSquareAdd>::value;
    bool externalTypes = (std::is_same<OutType, float>::value ||
                          std::is_same<OutType, half>::value) &&
                         (std::is_same<PartialsType, float>::value ||
                          std::is_same<PartialsType, half>::value);
    return externalOp && externalTypes;

  }
  IS_EXTERNAL_CODELET(useExternal());

  Input<Vector<PartialsType, SCALED_PTR32, 8, false>> partials;
  ROT<OutType, isUpdate> out;
  const unsigned short numOutputs;
  const unsigned short numPartials;

  bool compute() {
    for (unsigned o = 0; o < numOutputs + 1; ++o) {
      OutType acc = ReduceOp::template init<OutType>();
      for (unsigned p = 0; p < numPartials; ++p) {
        const auto index = (o * numPartials) + p;
        ReduceOp::update(acc, partials[index]);
      }
      if (isUpdate) {
        out[o] += acc;
      } else {
        out[o] = acc;
      }
    }
    return true;
  }
};

template<typename ReduceOp, typename PartialsType,
         typename OutType, bool isUpdate>
class ScaledContinuousReduce : public Vertex {
public:
  ScaledContinuousReduce();

  IS_EXTERNAL_CODELET((ContinuousReduce<ReduceOp, PartialsType,
                                        OutType, isUpdate>::useExternal()));

  Input<Vector<PartialsType, SCALED_PTR32, 8, false>> partials;
  ROT<OutType, isUpdate> out;
  const unsigned short numOutputs;
  const unsigned short numPartials;
  // Leave this as a vector so as to be consistent with the other reduction
  // codelets, even though we are not using a SCALED_PTR32 in this case
  Input<Vector<float, ONE_PTR>> k; // Scale factor

  bool compute() {
    for (unsigned o = 0; o < numOutputs + 1; ++o) {
      OutType acc = ReduceOp::template init<OutType>();
      for (unsigned p = 0; p < numPartials; ++p) {
        const auto index = (o * numPartials) + p;
        ReduceOp::update(acc, partials[index]);
      }
      acc = acc * static_cast<OutType>(k[0]);
      if (isUpdate) {
        out[o] += acc;
      } else {
        out[o] = acc;
      }
    }
    return true;
  }
};

// Specialised reduce to a single output from a single edge
template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate>
class Reduce<ReduceOp, PartialsType, OutType, isUpdate, 2u> : public Vertex {
public:
  Reduce();
  constexpr static bool isExternal() {
    return (std::is_same<ReduceOp, ReduceAdd>::value ||
            std::is_same<ReduceOp, ReduceSquareAdd>::value) &&
           std::is_same<PartialsType, float>::value &&
           !isUpdate;
  }
  IS_EXTERNAL_CODELET(isExternal());
  template <typename T>
  using ReduceOutput =
      typename std::conditional<isUpdate, InOut<T>, Output<T>>::type;
  ReduceOutput<Vector<OutType, ONE_PTR>> out;
  Input<Vector<PartialsType, SCALED_PTR32, 8>> partials;
  const ShortType numPartials;
  bool compute() {
    OutType acc = ReduceOp::template init<OutType>();
    for (unsigned p = 0; p < numPartials; ++p)
      ReduceOp::update(acc, partials[p]);
    if (isUpdate) {
      out[0] += acc;
    } else {
      out[0] = acc;
    }
    return true;
  }
};

// Specialised reduce to one output region from a single edge
template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate>
class Reduce<ReduceOp, PartialsType, OutType, isUpdate, 3u> : public Vertex {
public:
  Reduce();
  constexpr static bool isExternal() {
    return (std::is_same<ReduceOp, ReduceAdd>::value ||
            std::is_same<ReduceOp, ReduceSquareAdd>::value) &&
            (std::is_same<PartialsType, float>::value ||
             std::is_same<PartialsType, half>::value) &&
            std::is_same<OutType, float>::value &&
           !isUpdate;
  }
  // External codelets require the partials and outputs to be a multiple of
  // 64bits to give aligned memory accesses
  IS_EXTERNAL_CODELET(isExternal());
  template <typename T>
  using ReduceOutput =
      typename std::conditional<isUpdate, InOut<T>, Output<T>>::type;
  ReduceOutput<Vector<OutType, SCALED_PTR32, 4>> out;
  Input<Vector<PartialsType, SCALED_PTR32, 8>> partials;
  ShortType numOutputs;
  ShortType numPartials;
  bool compute() {
    for (unsigned o = 0; o < numOutputs; ++o) {
      const PartialsType *pPtr = &partials[o];
      OutType acc = ReduceOp::template init<OutType>();
      for (unsigned p = 0; p < numPartials; ++p) {
        ReduceOp::update(acc, *pPtr);
        pPtr += numOutputs;
      }

      if (isUpdate) {
        out[o] += acc;
      } else {
        out[o] = acc;
      }
    }
    return true;
  }
};
// Specialised reduce to one output region from a single edge
template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate>
class ScaledReduce<ReduceOp, PartialsType, OutType, isUpdate, 3u> :
      public Vertex {
public:
  ScaledReduce();
  constexpr static bool isExternal() {
    return (std::is_same<ReduceOp, ReduceAdd>::value ||
            std::is_same<ReduceOp, ReduceSquareAdd>::value) &&
            (std::is_same<PartialsType, float>::value ||
             std::is_same<PartialsType, half>::value) &&
            std::is_same<OutType, float>::value &&
           !isUpdate;
  }
  // External codelets require the partials and outputs to be a multiple of
  // 64bits to give aligned memory accesses
  IS_EXTERNAL_CODELET(isExternal());
  template <typename T>
  using ReduceOutput =
      typename std::conditional<isUpdate, InOut<T>, Output<T>>::type;
  ReduceOutput<Vector<OutType, SCALED_PTR32, 4>> out;
  Input<Vector<PartialsType, SCALED_PTR32, 8>> partials;
  ShortType numOutputs;
  ShortType numPartials;
  /* Multiplication factor.*/
  /* Actually we just need a scalar here, but creating a vector allows use of a
     SCALED_PTR32, which packs into the rest of the vertex state efficiently
     and saves space (although at the cost of 3 instructions to unpack) */
  Input<Vector<float, SCALED_PTR32>> k;

  bool compute() {
    for (unsigned o = 0; o < numOutputs; ++o) {
      const PartialsType *pPtr = &partials[o];
      OutType acc = ReduceOp::template init<OutType>();
      for (unsigned p = 0; p < numPartials; ++p) {
        ReduceOp::update(acc, *pPtr);
        pPtr += numOutputs;
      }
      acc = static_cast<OutType>(k[0]) * acc;
      if (isUpdate) {
        out[o] += acc;
      } else {
        out[o] = acc;
      }
    }
    return true;
  }
};


// Specialisation of reduction vertices where all partials are of equal size.
// Additional constraints are that the partials must be a multiple of 128 bits
// in size, and the partials size is a multiple of the output size.
// The approach is to reduce each column of all partials first, as the inner
// loop which given the constraints above is an efficient implementation.

template <typename ReduceOp, typename DataType>
struct IsExternal {
private:
  template <typename R>
  constexpr static bool is() {
    return std::is_same<R, ReduceOp>{};
  }

public:
  constexpr bool operator()() const {
    // Current permutations of template parameters that have assembly.
    return (std::is_same<DataType, half>::value ||
            std::is_same<DataType, float>::value) &&
           (is<ReduceAdd>() || is<ReduceSquareAdd>() ||
            is<ReduceMax>() || is<ReduceMin>());
  }
};

template <typename OutType, bool isUpdate>
using ReduceOutput =
  typename std::conditional<isUpdate,
                            InOut<Vector<OutType, SCALED_PTR64, 8>>,
                            Output<Vector<OutType, SCALED_PTR64, 8>>>::type;

template <typename PartialsType>
using ReducePartials =
  Input<VectorList<PartialsType, VectorListLayout::DELTAN, 8>>;


// common compute method for the reduce down the partial variants.
template <typename ReduceOp, typename OutType, typename PartialsType,
          bool isUpdate>
bool computePartialsEqualSizeReduction(ReduceOutput<OutType, isUpdate> &out,
                                       const ShortType outCount,
                                       const ShortType partialsSize,
                                       ReducePartials<PartialsType> &partials,
                                       const float k) {
  // outCount is scaled down by however many partials we can fit in 128-bits.
  constexpr auto grainSize = std::is_same<PartialsType, half>::value ? 8 : 4;
  const auto outSize = outCount * grainSize;

  // we walk down the partials height-first, reducing to output.
  for (unsigned o = 0; o < outCount; ++o) {

    // Initialise our internal result
    OutType result[grainSize];
    for (unsigned i = 0; i < grainSize; ++i) {
      result[i] = ReduceOp::template init<OutType>();
    }
    // Along the partials
    for (unsigned p = 0; p < partialsSize; ++p) {
      // Reduce, down the height of the partials
      for (unsigned pg = 0; pg < partials.size(); ++pg) {
        const auto pidx = o * grainSize + p * outSize;
        for (unsigned i = 0; i < grainSize; ++i) {
          ReduceOp::update(result[i], partials[pg][pidx + i]);
        }
      }
    }
    // scale accordingly.
    for (unsigned i = 0; i < grainSize; ++i) {
      result[i] *= static_cast<OutType>(k);
    }

    // update output.
    const auto oidx = o * grainSize;
    for (unsigned i = 0; i < grainSize; ++i) {
      if(isUpdate) {
        out[oidx + i] += result[i];
      } else {
        out[oidx + i] = result[i];
      }

    }
  }

  return true;
}

template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate>
class ReducePartialsEqualSize : public Vertex {
  IS_EXTERNAL_CODELET((IsExternal<ReduceOp, PartialsType>()()));

  ReduceOutput<OutType, isUpdate> out;
  const ShortType outCount;
  ReducePartials<PartialsType> partials;
  const ShortType partialsSizeM1;

public:
  ReducePartialsEqualSize();

  bool compute() {
    const auto fn = computePartialsEqualSizeReduction<ReduceOp,
                                                      OutType,
                                                      PartialsType,
                                                      isUpdate>;
    return fn(out, outCount, partialsSizeM1 + 1, partials, 1.0f);
  }
};

template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate>
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
  Input<Vector<float, SCALED_PTR32>> k;

public:
  ScaledReducePartialsEqualSize();

  bool compute() {
    const auto fn = computePartialsEqualSizeReduction<ReduceOp,
                                                      OutType,
                                                      PartialsType,
                                                      isUpdate>;
    return fn(out, outCount, partialsSizeM1 + 1, partials, k[0]);
  }
};


/** Macro to declare a templated popops::Reduce vertex for a particular
 *  operator. The nested macros expand delcarations for every combination
 *  of the final three boolean and specialisation template arguments */
#define DECLARE_REDUCTION_AND_SCALED0(NAME, ...) \
    template class ScaledReduce<popops::NAME, __VA_ARGS__>;\
    template class Reduce<popops::NAME, __VA_ARGS__>;

#define DECLARE_REDUCTION0(NAME, ...) \
    template class Reduce<popops::NAME, __VA_ARGS__>;

#define DECLARE_REDUCTION_AND_SCALED_PARTIALS_EQUAL_SIZE0(NAME, ...) \
    template class ReducePartialsEqualSize<popops::NAME, __VA_ARGS__>; \
    template class ScaledReducePartialsEqualSize<popops::NAME, __VA_ARGS__>;

#define DECLARE_CONTINUOUS_REDUCTION(NAME, ...) \
    template class ContinuousReduce<popops::NAME, __VA_ARGS__>; \
    template class ScaledContinuousReduce<popops::NAME, __VA_ARGS__>;

#define DECLARE_REDUCTION1(NAME, ...) \
    DECLARE_REDUCTION_AND_SCALED0(NAME, __VA_ARGS__, false, 0u) \
    DECLARE_REDUCTION_AND_SCALED0(NAME, __VA_ARGS__, true, 0u) \
    DECLARE_REDUCTION_AND_SCALED0(NAME, __VA_ARGS__, false, 1u) \
    DECLARE_REDUCTION_AND_SCALED0(NAME, __VA_ARGS__, true, 1u) \
    DECLARE_REDUCTION0(NAME, __VA_ARGS__, false, 2u) \
    DECLARE_REDUCTION0(NAME, __VA_ARGS__, true, 2u) \
    DECLARE_REDUCTION_AND_SCALED0(NAME, __VA_ARGS__, false, 3u) \
    DECLARE_REDUCTION_AND_SCALED0(NAME, __VA_ARGS__, true, 3u)\
    DECLARE_REDUCTION_AND_SCALED_PARTIALS_EQUAL_SIZE0(NAME, __VA_ARGS__, false)\
    DECLARE_REDUCTION_AND_SCALED_PARTIALS_EQUAL_SIZE0(NAME, __VA_ARGS__, true)\
    DECLARE_CONTINUOUS_REDUCTION(NAME, __VA_ARGS__, false)\
    DECLARE_CONTINUOUS_REDUCTION(NAME, __VA_ARGS__, true)

#define DECLARE_FULL_TYPES_REDUCTION(NAME) \
    DECLARE_REDUCTION1(NAME, float, float) \
    DECLARE_REDUCTION1(NAME, half, float) \
    DECLARE_REDUCTION1(NAME, float, half) \
    DECLARE_REDUCTION1(NAME, half, half) \
    DECLARE_REDUCTION1(NAME, int, int)

#define DECLARE_EQUAL_TYPES_REDUCTION(NAME) \
    DECLARE_REDUCTION1(NAME, float, float) \
    DECLARE_REDUCTION1(NAME, half, half) \
    DECLARE_REDUCTION1(NAME, int, int)

#define DECLARE_BOOL_TYPES_REDUCTION(NAME) \
    DECLARE_REDUCTION1(NAME, bool, bool)

DECLARE_FULL_TYPES_REDUCTION(ReduceAdd)
DECLARE_FULL_TYPES_REDUCTION(ReduceSquareAdd)
DECLARE_FULL_TYPES_REDUCTION(ReduceMul)

DECLARE_EQUAL_TYPES_REDUCTION(ReduceMax)
DECLARE_EQUAL_TYPES_REDUCTION(ReduceMin)

DECLARE_BOOL_TYPES_REDUCTION(ReduceAnd)
DECLARE_BOOL_TYPES_REDUCTION(ReduceOr)

} // namespace popops
