// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <limits>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "../reduction/ReductionVertexDefs.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/LogArithmetic.hpp"

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

#ifdef __IPU__
struct ReduceLogAdd {
  template <typename T> static T init() {
    return poplibs_support::log::probabilityZero;
  }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val) {
    // Casting required as exp<half>() undefined
    const auto a = static_cast<float>(val);
    const auto b = static_cast<float>(acc);
    // Compiled code doesn't produce optimal f32max, f32min instructions
    float max, min;
    asm(" f32max %[asm_max], %[asm_a], %[asm_b]"
        : [asm_max] "=r"(max)
        : [asm_a] "r"(a), [asm_b] "r"(b));
    asm(" f32min %[asm_min], %[asm_a], %[asm_b]"
        : [asm_min] "=r"(min)
        : [asm_a] "r"(a), [asm_b] "r"(b));
    acc = static_cast<OutType>(max + std::log(1 + std::exp(min - max)));
  }
};
#else
struct ReduceLogAdd {
  template <typename T> static T init() {
    return poplibs_support::log::probabilityZero;
  }
  template <typename OutType, typename PartialsType>
  static void update(OutType &acc, PartialsType val_) {
    OutType val = static_cast<float>(val_);
    OutType max = val < acc ? acc : val;
    OutType min = val < acc ? val : acc;
    // Casting required as exp<half>() undefined
    acc = max + static_cast<OutType>(
                    std::log(1 + std::exp(static_cast<float>(min - max))));
  }
};
#endif
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
// Choose an accumulator type to support better accuracy when partials are
// halves and the operation benefits from it.
template <typename PartialsType, typename ReduceOp>
using AccType =
    std::conditional_t<std::is_same<PartialsType, half>::value &&
                           (std::is_same<ReduceOp, ReduceAdd>::value ||
                            std::is_same<ReduceOp, ReduceSquareAdd>::value),
                       float, PartialsType>;

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

template <typename OutType, bool isUpdate,
          ReductionSpecialisation specialisation>
using ReduceOutputAlign = typename std::conditional<
    isUpdate,
    poplar::InOut<poplar::VectorList<
        OutType, DELTAN_TYPE,
        specialisation == ReductionSpecialisation::SCALAR_OUTPUT_REGIONS ? 4
                                                                         : 8>>,
    poplar::Output<poplar::VectorList<
        OutType, DELTAN_TYPE,
        specialisation == ReductionSpecialisation::SCALAR_OUTPUT_REGIONS
            ? 4
            : 8>>>::type;

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate, ReductionSpecialisation specialisation>
static void computeReduce(
    ReduceOutputAlign<OutType, isUpdate, specialisation> out,
    poplar::Input<poplar::Vector<unsigned short, PTR_ALIGN32, 4>> numPartials,
    poplar::Input<poplar::VectorList<PartialsType, DELTAN_TYPE, 8, false>>
        partials,
    float k) {
  using AccType = AccType<PartialsType, ReduceOp>;
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
      AccType acc = ReduceOp::template init<AccType>();

      /* ..by summing the corresponding element in the partials regions. */
      for (unsigned p = 0; p < numPartials_r; ++p) {
        assert(partials[pidx + p].size() % numElem == 0);

        /* Sum them all */
        for (unsigned o = outIdx; o < partials[pidx + p].size(); o += numElem) {
          ReduceOp::update(acc, static_cast<AccType>(partials[pidx + p][o]));
        }
      }

      const auto scaledAcc =
          static_cast<OutType>(static_cast<AccType>(k) * acc);

      /* Store it. */
      if (isUpdate) {
        out[r][outIdx] += scaledAcc;
      } else {
        out[r][outIdx] = scaledAcc;
      }
    }
    /* And skip forward in the partials vector to the next reduction. */
    pidx += numPartials_r;
  }
}

template <typename T, bool isUpdate>
using StridedReduceOutput =
    typename std::conditional<isUpdate, poplar::InOut<T>,
                              poplar::Output<T>>::type;

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate, bool opIsLogAdd>
static void computeStridedReduce(
    StridedReduceOutput<poplar::Vector<OutType, PTR_ALIGN32, 4>, isUpdate> out,
    poplar::Input<poplar::Vector<PartialsType, PTR_ALIGN32, 8>> partials,
    unsigned numOutputsM1, unsigned numPartialsM1, unsigned partialsWidth,
    unsigned numOuterStridesM1, unsigned outerStride, float k) {
  using AccType = AccType<PartialsType, ReduceOp>;

  constexpr auto partialsGrainSize =
      std::is_same<PartialsType, half>::value ? 4u : 2u;
  const auto numOutputLoops = (numOutputsM1 + 1) * partialsGrainSize;
  for (unsigned o = 0; o < numOutputLoops; ++o) {
    const PartialsType *pPtr = &partials[o];
    AccType acc = ReduceOp::template init<AccType>();
    // Reduce numPartialsM1 + 1 partials, then take an outer stride, repeat
    for (unsigned os = 0; os < numOuterStridesM1 + 1; os++) {
      for (unsigned p = 0; p < numPartialsM1; ++p) {
        ReduceOp::update(acc, static_cast<AccType>(*pPtr));
        pPtr += partialsWidth * partialsGrainSize;
      }
      // Last one in the inner group. Avoid computing a pointer that is out of
      // the expected range even if it is never used
      ReduceOp::update(acc, static_cast<AccType>(*pPtr));
      if (outerStride != 0) {
        // Take the outer stride
        pPtr += outerStride * partialsGrainSize;
      }
    }

    // Apply scale.  For log-probability arithmetic this is an add.
    const auto scaledOut =
        opIsLogAdd ? static_cast<OutType>(acc + static_cast<AccType>(k))
                   : static_cast<OutType>(acc * static_cast<AccType>(k));

    if constexpr (isUpdate) {
      if constexpr (opIsLogAdd) {
        ReduceOp::update(out[o], scaledOut);
      } else {
        out[o] += scaledOut;
      }
    } else {
      out[o] = scaledOut;
    }
  }
}

} // namespace popops
