#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <limits>

#include "util.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;

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
    acc += static_cast<OutType>(val * val);
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


/** Generic vertex template for all reductions. The template parameters provide
 *  the information on types, what the reduction operator is, whether to
 *  update in place or not etc. */
template <typename ReduceOp, typename PartialsType,
          typename OutType, bool isUpdate>
class Reduce : public Vertex {
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
  IS_EXTERNAL_CODELET((!std::is_same<PartialsType, int>::value
                      && ((vectorised_8() || vectorised_4()))));

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

  template <typename T>
  using ReduceOutput =
      typename std::conditional<isUpdate, InOut<T>, Output<T>>::type;

  /* Multiplication factor. Might be unused. */
  float k;

  /* Vector of regions to output. */
  ReduceOutput<VectorList<OutType, VectorListLayout::DELTAN, 8>> out;

    /* The number of input regions (partials) for each output region. */
  /* This should sum to `partials.size()`. */
  Input<Vector<unsigned short, SCALED_PTR32, 4>> numPartials;

  /* Vector of regions to use as input. */
  Input<VectorList<PartialsType, VectorListLayout::DELTAN, 8, false>> partials;

  bool compute() {
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
};


/** Macro to declare a templated popops::Reduce vertex for a particular
 *  operator. The nested macros expand delcarations for every combination
 *  of the final three boolean template arguments */
#define DECLARE_REDUCTION0(NAME, ...) \
    template class Reduce<popops::NAME, __VA_ARGS__>;

#define DECLARE_REDUCTION1(NAME, ...) \
    DECLARE_REDUCTION0(NAME, __VA_ARGS__, false) \
    DECLARE_REDUCTION0(NAME, __VA_ARGS__, true)

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

}
