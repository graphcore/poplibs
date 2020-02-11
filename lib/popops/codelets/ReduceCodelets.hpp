// Copyright (c) Graphcore Ltd, All rights reserved.
#include <cassert>
#include <limits>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/ExternalCodelet.hpp"
#include "util.hpp"

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
#ifdef VECTOR_AVAIL_SCALED_PTR64
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::SCALED_PTR64;
#else
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::ONE_PTR;
#endif
#ifdef VECTOR_AVAIL_SCALED_PTR32
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::SCALED_PTR32;
#else
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::ONE_PTR;
#endif
#ifdef VECTORLIST_AVAIL_DELTAN
static constexpr auto DELTAN_TYPE = poplar::VectorListLayout::DELTAN;
#else
static constexpr auto DELTAN_TYPE = poplar::VectorListLayout::DELTANELEMENTS;
#endif

#ifdef __IPU__
// For real implementation
using ShortType = unsigned short;
#else
// To avoid size overflow on CPU implementation
using ShortType = unsigned;
#endif

namespace popops {

struct ReduceAdd {
  template <typename T> static T init() { return 0; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    acc += static_cast<OutType>(val);
  }
};

struct ReduceSquareAdd {
  template <typename T> static T init() { return 0; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    auto valOutType = static_cast<OutType>(val);
    acc += valOutType * valOutType;
  }
};

struct ReduceMul {
  template <typename T> static T init() { return 1; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    acc *= static_cast<OutType>(val);
  }
};

struct ReduceMax {
  template <typename T> static T init() {
    return std::numeric_limits<T>::lowest();
  }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    acc = max(acc, val);
  }
};

struct ReduceMin {
  template <typename T> static T init() {
    return std::numeric_limits<T>::max();
  }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    acc = min(acc, val);
  }
};

struct ReduceAnd {
  template <typename T> static T init() { return true; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    acc = acc && val;
  }
};

struct ReduceOr {
  template <typename T> static T init() { return false; }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    acc = acc || val;
  }
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
using ReduceOutputAlign = typename std::conditional<
    isUpdate,
    poplar::InOut<
        poplar::VectorList<OutType, DELTAN_TYPE, specialisation == 1 ? 4 : 8>>,
    poplar::Output<poplar::VectorList<OutType, DELTAN_TYPE,
                                      specialisation == 1 ? 4 : 8>>>::type;

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate, unsigned specialisation>
static bool computeReduce(
    ReduceOutputAlign<OutType, isUpdate, specialisation> out,
    poplar::Input<poplar::Vector<unsigned short, PTR_ALIGN32, 4>> numPartials,
    poplar::Input<poplar::VectorList<PartialsType, DELTAN_TYPE, 8, false>>
        partials,
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
        for (unsigned o = outIdx; o < partials[pidx + p].size(); o += numElem) {
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

} // namespace popops
