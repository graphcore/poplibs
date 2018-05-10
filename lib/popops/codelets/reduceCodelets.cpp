#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <limits>

#include "util.hpp"

using namespace poplar;

namespace popops {

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto TWO_PTR = poplar::VectorLayout::TWO_PTR;

/** Generic vertex template for all reductions. The template parameters provide
 *  the information on types, what the reduction operator is, whether to
 *  update in place or not etc. */
template <typename ReduceOp,
          typename PartialsType, typename OutType, bool partialsAreOutputSize,
          bool isScale, bool isUpdate>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
Reduce : public Vertex {
public:
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

  /* Vector of regions to output. */
  Vector<ReduceOutput<Vector<OutType>>> out;
  /* Vector of regions to use as input. */
  Vector<Input<Vector<PartialsType, TWO_PTR, 1, true>>> partials;

  /* The number of input regions (partials) for each output region. */
  /* This should sum to `partials.size()`. */
  Vector<unsigned> numPartials;

  /* Multiplication factor. Might be unused. */
  float k;

  bool compute() {
    /* The number of output regions. */
    unsigned numReductions = out.size();

    /* Check that each output vector has a number saying how many */
    /* input vectors. */
    assert(numPartials.size() == numReductions);

    /* Check that the total number of partials equals the actual number */
    /* of input partial vectors (and that they all have at least 1 input). */
    unsigned sum = 0;
    for (auto np : numPartials) {
      assert(np >= 1);
      sum += np;
    }
    assert(sum == partials.size());

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
        PartialsType acc = ReduceOp::template init<PartialsType>();

        /* ..by summing the corresponding element in the partials regions. */
        for (unsigned p = 0; p < numPartials_r; ++p) {
          /* Check that the partial region size is an integer multiple    */
          /* of the number of output elements for this region, or exactly */
          /* equal if partialsAreOutputSize is set.                       */
          if (partialsAreOutputSize)
            assert(partials[pidx + p].size() == numElem);
          else
            assert(partials[pidx + p].size() % numElem == 0);

          /* Sum them all */
          for (unsigned o = outIdx; o < partials[pidx + p].size();
               o += numElem) {
            ReduceOp::update(acc, partials[pidx + p][o]);
          }
        }

        if (isScale)
          acc = k * acc;

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

#define DECLARE_REDUCTION2(NAME, ...) \
    DECLARE_REDUCTION1(NAME, __VA_ARGS__, false) \
    DECLARE_REDUCTION1(NAME, __VA_ARGS__, true)

#define DECLARE_REDUCTION3(NAME, ...) \
    DECLARE_REDUCTION2(NAME, __VA_ARGS__, false) \
    DECLARE_REDUCTION2(NAME, __VA_ARGS__, true)

#define DECLARE_FULL_TYPES_REDUCTION(NAME) \
    DECLARE_REDUCTION3(NAME, float, float) \
    DECLARE_REDUCTION3(NAME, half, float) \
    DECLARE_REDUCTION3(NAME, float, half) \
    DECLARE_REDUCTION3(NAME, half, half) \
    DECLARE_REDUCTION3(NAME, int, int)

#define DECLARE_EQUAL_TYPES_REDUCTION(NAME) \
    DECLARE_REDUCTION3(NAME, float, float) \
    DECLARE_REDUCTION3(NAME, half, half) \
    DECLARE_REDUCTION3(NAME, int, int)

#define DECLARE_BOOL_TYPES_REDUCTION(NAME) \
    DECLARE_REDUCTION3(NAME, bool, bool)

struct ReduceAdd {
  template <typename T>
  static T init() { return 0; }
  template <typename T>
  static void update(T &acc, T val) { acc += val; }
};

DECLARE_FULL_TYPES_REDUCTION(ReduceAdd)

struct ReduceSquareAdd {
  template <typename T>
  static T init() { return 0; }
  template <typename T>
  static void update(T &acc, T val) { acc += val * val; }
};

DECLARE_FULL_TYPES_REDUCTION(ReduceSquareAdd)

struct ReduceMul {
  template <typename T>
  static T init() { return 1; }
  template <typename T>
  static void update(T &acc, T val) { acc *= val; }
};

DECLARE_FULL_TYPES_REDUCTION(ReduceMul)

struct ReduceMax {
  template <typename T>
  static T init() { return std::numeric_limits<T>::lowest(); }
  template <typename T>
  static void update(T &acc, T val) { acc = max(acc, val); }
};

DECLARE_EQUAL_TYPES_REDUCTION(ReduceMax)

struct ReduceMin {
  template <typename T>
  static T init() { return std::numeric_limits<T>::max(); }
  template <typename T>
  static void update(T &acc, T val) { acc = min(acc, val); }
};

DECLARE_EQUAL_TYPES_REDUCTION(ReduceMin)

struct ReduceAnd {
  template <typename T>
  static T init() { return true; }
  template <typename T>
  static void update(T &acc, T val) { acc = acc && val; }
};

DECLARE_BOOL_TYPES_REDUCTION(ReduceAnd)

struct ReduceOr {
  template <typename T>
  static T init() { return true; }
  template <typename T>
  static void update(T &acc, T val) { acc = acc || val; }
};

DECLARE_BOOL_TYPES_REDUCTION(ReduceOr)

}
