// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <tuple>

#include "elemwiseBinaryOps.hpp"
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

#ifdef __IPU__
#include "inlineAssembler.hpp"
#endif

using namespace poplar;
using namespace popops;

using BinaryOpType = popops::expr::BinaryOpType;

// Defines which broadcast codelets have an assembly implementation
template <BinaryOpType op, typename dType> constexpr static bool hasAssembly() {
  const bool isExternalArithmeticOp =
      (std::is_same<dType, float>::value || std::is_same<dType, half>::value) &&
      (op == BinaryOpType::ADD || op == BinaryOpType::SUBTRACT ||
       op == BinaryOpType::MULTIPLY);
  const bool isExternalBitwiseOp =
      (std::is_same<dType, short>::value ||
       std::is_same<dType, unsigned short>::value) &&
      (op == expr::BinaryOpType::BITWISE_AND ||
       op == expr::BinaryOpType::BITWISE_OR);
  return isExternalArithmeticOp || isExternalBitwiseOp;
}

//******************************************************************************
// Dispatch functions for the broadcast codelets
//******************************************************************************

template <BinaryOpType op, typename inT, typename outT,
          bool allowUnaligned, // Allow input/output that isn't 64-bit aligned
          bool allowRemainder>
struct BroadcastOpDispatch {
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(inT) : 8;
  static void compute(unsigned size,
                      const __attribute__((align_value(minAlign))) inT *in,
                      __attribute__((align_value(minAlign))) outT *out,
                      const inT K) {
    for (unsigned j = 0; j != size; j++) {
      out[j] = BinaryOpFn<op, inT, architecture::active>::fn(in[j], K);
    }
  }
};

template <BinaryOpType op, typename inT, typename outT,
          bool allowUnaligned, // Allow input/output that isn't 64-bit aligned.
          bool allowRemainder>
struct BroadcastOpDispatchMultiVertex {
public:
  static void compute(unsigned size, unsigned worker, const inT *in, outT *out,
                      const inT K) {
    // No vectorisation but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = BinaryOpFn<op, inT, architecture::active>::fn(in[j], K);
  }
};

template <BinaryOpType op, typename inT, typename outT, bool allowRemainder>
struct BroadcastRelationalOpDualOutputDispatchMultiVertex {
public:
  static void compute(unsigned size, unsigned worker, const inT *in, outT *out,
                      outT *outInv, const inT K) {
    // No vectorisation but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS) {
      auto result = BinaryOpFn<op, inT, architecture::active>::fn(in[j], K);
      out[j] = static_cast<outT>(result);
      outInv[j] = 1 - out[j];
    }
  }
};

#ifdef __IPU__

/// Performs the bulk of a *broadcast scalar* 'op' that has bool as output
/// type:  T op T => BOOL (i.e. the comparison operator EQUAL, LESS_THAN etc
/// plus LOGICAL_AND/OR).
/// This processes 4 elements of type T in each cycle of the loop, and writes
/// the 4 aggregated boolean results (1 full word) at a time, avoiding calls
/// to __st8/__st16.
/// This is run by both the '2DData' and the 'MultiVertex' vertices.
/// Implemented as as a struct with a static function instead of a templated
/// function because you cannot partially specialise a templated function
/// (also, doesn't use operator() because it cannot be static)
///
///  \tparam     op        Operation to perform. One of the comparison ops
///                        (EQUAL, LESS_THAN, ...)
///
///  \tparam     T         Type of the operands
///
///  \tparam     stride    Stride (in units of 4 elements) to advance
///                        in/out at each cycle of the loop.
///
///  \param[in]  loopCount How many groups of 4 elements to process.
///
///  \param[in]  in        Pointer to an array of T with the input data
///                        (first operand).
///
///  \param[out] out       Pointer to an array of boolean (1 byte each) that
///                        will be populated with the results
///
///  \param[in]  K         The second operand (the one that is broadcasted)
///
template <BinaryOpType op, typename T, unsigned stride>
struct broadcastBoolOpBulk {
  static void compute(unsigned loopCount, const T *in, int *out, const T K) {
    for (unsigned i = 0; i < loopCount; i++) {
      unsigned result4 = 0;
      // Accumulate in 'result4' the 4 bytes for this loop
      for (unsigned j = 0, shifts = 0; j != 4; ++j, shifts += 8) {
        bool res = BinaryOpFn<op, T, architecture::active>::fn(in[j], K);
        result4 |= res << shifts;
      }
      in += 4 * stride;
      *out = result4;
      out += stride;
    }
  }
};

// Optimisations for operators processing 4 boolean stored in a single word.
// Some of them can be done by a 'vectorized' operation on the whole word
// 'OP_FN' is the code that returns the result of the operation (taking 'word1'
// and 'word2' as the two operands).
#define OPTIMIZED_BOOL_OP_BULK(op, OP_FN)                                      \
  template <unsigned stride> struct broadcastBoolOpBulk<op, bool, stride> {    \
    static void compute(unsigned loopCount, const bool *in, int *out,          \
                        const bool K) {                                        \
      const unsigned *b4In = reinterpret_cast<const unsigned *>(in);           \
      const char4 K4 = {K, K, K, K};                                           \
      const unsigned word2 = reinterpret_cast<const unsigned>(K4);             \
      for (unsigned i = 0; i < loopCount; i++) {                               \
        unsigned word1 = *b4In;                                                \
        *out = OP_FN;                                                          \
        b4In += stride;                                                        \
        out += stride;                                                         \
      }                                                                        \
    }                                                                          \
  };

//  EQUAL is XNOR (then AND-away the other bits) of the two words
OPTIMIZED_BOOL_OP_BULK(BinaryOpType::EQUAL, ~(word1 ^ word2) & 0x01010101)

//  NOT_EQUAL is XOR of the two words
OPTIMIZED_BOOL_OP_BULK(BinaryOpType::NOT_EQUAL, word1 ^ word2)

//  LOGICAL_AND is BITWISE_AND of the two words
OPTIMIZED_BOOL_OP_BULK(BinaryOpType::LOGICAL_AND, word1 &word2)

//  LOGICAL_OR is BITWISE_OR of the two words
OPTIMIZED_BOOL_OP_BULK(BinaryOpType::LOGICAL_OR, word1 | word2)

// Partial specialisation of 'broadcastBoolOpBulk' for float, to use vector
// instructions
template <BinaryOpType op, unsigned stride>
struct broadcastBoolOpBulk<op, float, stride> {
  static void compute(unsigned loopCount, const float *in, int *out,
                      const float K) {
    const float2 *in2 = reinterpret_cast<const float2 *>(in);
    const float2 K2 = {K, K};
    for (unsigned j = 0; j < loopCount; j++) {
      long2 calc_lo =
          BinaryOpFn<op, float2, architecture::active>::fn(*in2++, K2);
      long2 calc_hi =
          BinaryOpFn<op, float2, architecture::active>::fn(*in2, K2);
      in2 += 2 * stride - 1;

      // Pack the two pair of results as 4 bytes in a 32 bit word and keep
      // only the least significant bit of each bytes
      *out = copy_cast<int>(tochar4(calc_lo, calc_hi)) & 0x01010101;
      out += stride;
    }
  }
};

// Partial specialisation of 'broadcastBoolOpBulk' for half, to use vector
// instructions
template <BinaryOpType op, unsigned stride>
struct broadcastBoolOpBulk<op, half, stride> {
  static void compute(unsigned loopCount, const half *in, int *out,
                      const half K) {
    const half4 *in4 = reinterpret_cast<const half4 *>(in);
    const half4 K4 = {K, K, K, K};
    for (unsigned j = 0; j < loopCount; j++) {
      short4 calc = BinaryOpFn<op, half4, architecture::active>::fn(*in4, K4);
      in4 += stride;

      // Pack the four results as 4 bytes in a 32 bit word and keep only the
      // least significant bit of each bytes
      *out = copy_cast<int>(tochar4(calc)) & 0x01010101;
      out += stride;
    }
  }
};

/// Broadcast scalar 'op' for bool output type ( T op T => BOOL) for the
/// trailing 0..3 elements of the input data (used in conjunction with
/// 'broadcastBoolOpBulk')
/// This is run by both the '2DData' and the 'MultiVertex' vertices.
///
///  \tparam     op        Operation to perform. One of the comparison ops
///                        (EQUAL, LESS_THAN, ...)
///
///  \tparam     T         data type (float or half)
///
///  \param[in]  size      Total number of elements contained in 'in'
///
///  \param[in]  in        Pointer to an array of floats with all the input data
///                        (first operand)
///
///  \param[out] out       Pointer to an array of boolean (1 byte each) that
///                        is/will be populated with the results
///
///  \param[in]  K         The second operand (the one that is broadcasted)
template <BinaryOpType op, typename T>
void broadcastBoolOpRemainder(unsigned size,
                              const __attribute__((align_value(8))) T *in,
                              __attribute__((align_value(8))) bool *out,
                              const T K) {
  unsigned remainder = size & 3;
  if (remainder) {
    unsigned offs = size - remainder;
    in = &in[offs]; // make it point to the 'remainder'
    // Read the word of the output in memory that will contain the 1-3 bytes
    // of the remainder, so that we can write back the byte(s) we will not be
    // updating.
    unsigned *out4 = reinterpret_cast<unsigned *>(&out[offs]);
    unsigned result4 = *out4;
    // Accumulate in 'result4' the 1-3 bytes of the remainder
    unsigned mask = 0xff;
    for (unsigned j = 0, shifts = 0; j != remainder; ++j, shifts += 8) {
      bool res = BinaryOpFn<op, T, architecture::active>::fn(in[j], K);
      result4 &= ~(mask << shifts);
      result4 |= res << shifts;
    }
    // Do a single word write back to memory
    *out4 = result4;
  }
}

/// Performs the bulk of any operation that can be performed on a 'short2'
/// vector (starting from an array of 16 bit short integers).
///
///  \tparam     op        Operation to perform
///
///  \tparam     stride    Stride (in units of a vector) to advance in/out at
///                        each cycle of the loop.
///
///  \param[in]  loopCount How many vectors to process.
///
///  \param[in]  in        Pointer to an array with the input data
///                        (first operand).
///
///  \param[out] out       Pointer to an array that will be populated with the
///                        results
///
///  \param[in]  K         The second operand (the one that is broadcasted)
///
template <BinaryOpType op, typename T, unsigned stride>
static void broadcastShort2Bulk(unsigned loopCount,
                                const __attribute__((align_value(4))) T *in,
                                __attribute__((align_value(4))) T *out,
                                const T K) {
  const short2 *s2In = reinterpret_cast<const short2 *>(in);
  short2 *s2Out = reinterpret_cast<short2 *>(out);
  // This works only for short, unsigned short and is for use with bitwise
  // operators only
  static_assert(std::is_same<T, short>::value ||
                std::is_same<T, unsigned short>::value);
  const short2 K2 = {static_cast<short>(K), static_cast<short>(K)};

  for (unsigned j = 0; j < loopCount; j++) {
    short2 calc = BinaryOpFn<op, short2, architecture::active>::fn(*s2In, K2);
    s2In += stride;
    *s2Out = calc;
    s2Out += stride;
  }
}

/// Processes the 'trailing' 1-3 elements (if present) for any operation that
/// can be performed on a 'short4'/'short2' vector (starting from an array
/// of 16 bit short integers)
///
/// \tparam op            Operation to perform
///
/// \tparam vectorWidthShifts  1 or 2 to indicate if we have a vector of 2
///                            or 4 elements, i.e.: log2(vectorWidth)
///
/// \param[in]  size      Total number of elements contained in 'in'
///
/// \param[in]  in        Pointer to an array of input data
///
/// \param[out] out       Pointer to an array that will be populated with the
///                       results
///
/// \param[in]  K         The second operand (the one that is broadcasted)
///
template <BinaryOpType op, typename T, unsigned vectorWidthShifts>
static void broadcastShort2Short4Remainder(
    unsigned size, const __attribute__((align_value(4))) T *in,
    __attribute__((align_value(4))) T *out, const T K) {
  constexpr unsigned mask = (1 << vectorWidthShifts) - 1;
  const unsigned rem = size & mask;
  const short2 *s2In = reinterpret_cast<const short2 *>(&in[size - rem]);
  short2 *s2Out = reinterpret_cast<short2 *>(&out[size - rem]);
  if constexpr (mask == 0x3) {
    if (size & 3) {
      short2 K2 = {K, K};
      *s2Out = BinaryOpFn<op, short2, architecture::active>::fn(*s2In, K2);
      s2In++;
      s2Out++;
    }
  }
  if (size & 1) {
    short2 res = {
        BinaryOpFn<op, short, architecture::active>::fn((*s2In)[0], K),
        (*s2Out)[1],
    };
    *s2Out = res;
  }
}

template <BinaryOpType op, typename T>
static void
broadcastShort2_MultiVertex(unsigned size, unsigned worker,
                            const __attribute__((align_value(4))) T *in,
                            __attribute__((align_value(4))) T *out, const T K) {
  const rptsize_t loopCount = divideWork(size, 1, worker);

  broadcastShort2Bulk<op, T, CTXT_WORKERS>(loopCount, in + 2 * worker,
                                           out + 2 * worker, K);

  // To process the trailing elements (if any) use the last worker as it is
  // most likely to have less to do than the others.
  if (worker == (CTXT_WORKERS - 1)) {
    broadcastShort2Short4Remainder<op, T, 1>(size, in, out, K);
  }
}

template <BinaryOpType op, typename T>
static void
broadcastShort2_2D(unsigned size, const __attribute__((align_value(4))) T *in,
                   __attribute__((align_value(4))) T *out, const T K) {
  const rptsize_t loopCount = size / 2u;
  broadcastShort2Bulk<op, T, 1>(loopCount, in, out, K);
  broadcastShort2Short4Remainder<op, T, 1>(size, in, out, K);
}

// Run by the 2D worker for the *broadcast scalar* 'op' that has bool as output
// type:  T op T => BOOL (i.e. the comparison operator EQUAL, LESS_THAN etc
// plus LOGICAL_AND/OR)
template <BinaryOpType op, typename T, bool allowRemainder>
struct BroadcastOpDispatch<op, T, bool, false, allowRemainder> {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) T *in,
                      __attribute__((align_value(8))) bool *out, const T K) {
    if (size >= 4) {
      const rptsize_t loopCount = size / 4u;
      broadcastBoolOpBulk<op, T, 1>::compute(loopCount, in,
                                             reinterpret_cast<int *>(out), K);
    }
    if (allowRemainder) {
      broadcastBoolOpRemainder<op, T>(size, in, out, K);
    }
  }
};

// Run by the 2D worker for the operators that work on half and return half.
// Optimised to perform vector read/writes.
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatch<op, half, half, allowUnaligned, allowRemainder> {
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(half) : 8;
  // Assumes in and out both point to the same memory (or at least have
  // same alignment at function start) and that there is at least one element
  // to process.
  static void compute(unsigned size,
                      const __attribute__((align_value(minAlign))) half *in,
                      __attribute__((align_value(minAlign))) half *out,
                      const half K) {
    if (allowUnaligned) {
      // Handle initial element
      if (reinterpret_cast<std::uintptr_t>(in) & 3) {
        const half2 *h2In = reinterpret_cast<const half2 *>(
            reinterpret_cast<std::uintptr_t>(in) & ~std::uintptr_t(3));
        half2 *h2Out = reinterpret_cast<half2 *>(
            reinterpret_cast<std::uintptr_t>(out) & ~std::uintptr_t(3));
        half2 res = {
            (*h2Out)[0],
            BinaryOpFn<op, half, architecture::active>::fn((*h2In++)[1], K)};
        *h2Out++ = res;
        size -= 1;
        in = reinterpret_cast<const half *>(h2In);
        out = reinterpret_cast<half *>(h2Out);
      }
      if (size >= 2 && reinterpret_cast<std::uintptr_t>(in) & 7) {
        half2 K2 = {K, K};
        const half2 *h2In = reinterpret_cast<const half2 *>(in);
        half2 *h2Out = reinterpret_cast<half2 *>(out);
        *h2Out++ = BinaryOpFn<op, half2, architecture::active>::fn(*h2In++, K2);
        size -= 2;
        in = reinterpret_cast<const half *>(h2In);
        out = reinterpret_cast<half *>(h2Out);
      }
    }

    if (size >= 4) {
      half4 K4 = {K, K, K, K};
      const half4 *h4In = reinterpret_cast<const half4 *>(in);
      half4 *h4Out = reinterpret_cast<half4 *>(out);

      half4 load = ipu::load_postinc(&h4In, 1);
      const rptsize_t loopCount = (size / 4u) - 1u;
      for (unsigned i = 0; i < loopCount; i++) {
        half4 calc = BinaryOpFn<op, half4, architecture::active>::fn(load, K4);
        load = ipu::load_postinc(&h4In, 1);
        *h4Out++ = calc;
      }
      *h4Out++ = BinaryOpFn<op, half4, architecture::active>::fn(load, K4);
      if (allowRemainder) {
        in = reinterpret_cast<const half *>(h4In);
        half *tmp = reinterpret_cast<half *>(h4Out);
        size -= (tmp - out);
        out = tmp;
      }
    }
    if (allowRemainder) {
      const half2 *h2In = reinterpret_cast<const half2 *>(in);
      half2 *h2Out = reinterpret_cast<half2 *>(out);

      if (size >= 2) {
        half2 K2 = {K, K};
        *h2Out++ = BinaryOpFn<op, half2, architecture::active>::fn(*h2In++, K2);
        size -= 2;
      }

      if (size == 1) {
        write16Aligned32(
            BinaryOpFn<op, half, architecture::active>::fn((*h2In)[0], K),
            h2Out);
      }
    }
  }
};

// Run by the 2D worker for the operators that work on floats and return floats.
// Optimised to perform vector read/writes.
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatch<op, float, float, allowUnaligned, allowRemainder> {
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(float) : 8;
  // Assumes in and out both point to the same memory (or at least have
  // same alignment at function start) and that there is at least one element
  // to process.
  static void compute(unsigned size,
                      const __attribute__((align_value(minAlign))) float *in,
                      __attribute__((align_value(minAlign))) float *out,
                      const float K) {
    if (allowUnaligned) {
      if (reinterpret_cast<std::uintptr_t>(in) & 0x7) {
        *out++ = BinaryOpFn<op, float, architecture::active>::fn(*in++, K);
        size -= 1;
      }
    }
    if (size >= 2) {
      float2 K2 = {K, K};
      const float2 *f2In = reinterpret_cast<const float2 *>(in);
      float2 *f2Out = reinterpret_cast<float2 *>(out);
      float2 load = *f2In++;
      const rptsize_t loopCount = (size / 2u) - 1;
      for (unsigned i = 0; i < loopCount; i++) {
        float2 calc =
            BinaryOpFn<op, float2, architecture::active>::fn(load, K2);
        load = ipu::load_postinc(&f2In, 1);
        *f2Out++ = calc;
      }
      *f2Out++ = BinaryOpFn<op, float2, architecture::active>::fn(load, K2);
      if (allowRemainder) {
        in = reinterpret_cast<const float *>(f2In);
        out = reinterpret_cast<float *>(f2Out);
      }
    }
    if (allowRemainder) {
      if (size & 1) {
        *out = BinaryOpFn<op, float, architecture::active>::fn(*in++, K);
      }
    }
  }
};

// This works only (and is instantiated) for BITWISE operators
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatch<op, short, short, allowUnaligned, allowRemainder> {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) short *in,
                      __attribute__((align_value(8))) short *out,
                      const short K) {
    broadcastShort2_2D<op, short>(size, in, out, K);
  }
};

// Works only (and is instantiated) for BITWISE operators for UNSIGNED SHORT
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatch<op, unsigned short, unsigned short, allowUnaligned,
                           allowRemainder> {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) unsigned short *in,
                      __attribute__((align_value(8))) unsigned short *out,
                      const short K) {
    broadcastShort2_2D<op, unsigned short>(size, in, out, K);
  }
};

/// Processing for operators that return a bool (comparison: EQUAL, LESS_THAN,
/// etc plus LOGICAL_AND/OR), for the MultiVertex vertices.
template <BinaryOpType op, typename T, bool allowRemainder>
class BroadcastOpDispatchMultiVertex<op, T, bool, false, allowRemainder> {
public:
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) T *in,
                      __attribute__((align_value(8))) bool *out, const T K) {
    const rptsize_t loopCount = divideWork(size, 2, worker);
    broadcastBoolOpBulk<op, T, CTXT_WORKERS>::compute(
        loopCount, in + 4 * worker, reinterpret_cast<int *>(out) + worker, K);

    // To process the trailing elements (if any) use the last worker as it is
    // most likely to have less to do than the others.
    if (allowRemainder) {
      if (worker == (CTXT_WORKERS - 1)) {
        broadcastBoolOpRemainder<op, T>(size, in, out, K);
      }
    }
  }
};

// Specialisation of 'BroadcastOpDispatchMultiVertex' for float, float => float.
// Optimised to perform vector read/writes.
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
class BroadcastOpDispatchMultiVertex<op, float, float, allowUnaligned,
                                     allowRemainder> {
public:
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(float) : 8;
  // Assumes in and out both point to the same memory (or at least have
  // same alignment at function start) and that there is at least one element
  // to process.
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(minAlign))) float *in,
                      __attribute__((align_value(minAlign))) float *out,
                      const float K) {

    if (allowUnaligned) {
      // TODO: T12920 Investigate whether performance could be improved by using
      // different workers for trailing and leading elements.
      if (reinterpret_cast<std::uintptr_t>(in) & 0x7) {
        if (worker == 0) {
          *out++ = BinaryOpFn<op, float, architecture::active>::fn(*in++, K);
        } else {
          ++in;
          ++out;
        }
        size -= 1;
      }
    }
    const float2 *f2In = reinterpret_cast<const float2 *>(in) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;
    float2 K2 = {K, K};
    const rptsize_t loopCount = divideWork(size, 1, worker);
    for (unsigned j = 0; j < loopCount; j++) {
      *f2Out = BinaryOpFn<op, float2, architecture::active>::fn(*f2In, K2);
      f2In += CTXT_WORKERS;
      f2Out += CTXT_WORKERS;
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if (allowRemainder) {
      if (worker == (CTXT_WORKERS - 1) && (size & 1)) {
        out[size - 1] =
            BinaryOpFn<op, float, architecture::active>::fn(in[size - 1], K);
      }
    }
  }
};

// Specialisation of 'BroadcastOpDispatchMultiVertex' for half, half => half.
// Optimised to perform vector read/writes.
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatchMultiVertex<op, half, half, allowUnaligned,
                                      allowRemainder> {
public:
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(half) : 8;
  // Assumes in and out both point to the same memory (or at least have
  // same alignment at function start) and that there is at least one element
  // to process.
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(minAlign))) half *in,
                      __attribute__((align_value(minAlign))) half *out,
                      const half K) {
    if (allowUnaligned) {
      if (reinterpret_cast<std::uintptr_t>(in) & 0x3) {
        // Use the same worker to deal with the leading elements as deals with
        // the trailing elements to avoid read-modify-write conflicts in
        // dealing with the odd single element. Pick the last worker as it
        // is most likely to have less to do than the others.
        if (worker == CTXT_WORKERS - 1) {
          const half2 *h2In = reinterpret_cast<const half2 *>(
              reinterpret_cast<std::uintptr_t>(in) & ~std::uintptr_t(0x3));
          half2 *h2Out = reinterpret_cast<half2 *>(
              reinterpret_cast<std::uintptr_t>(out) & ~std::uintptr_t(0x3));
          half2 res = {
              (*h2Out)[0],
              BinaryOpFn<op, half, architecture::active>::fn((*h2In++)[1], K)};
          *h2Out++ = res;
          in = reinterpret_cast<const half *>(h2In);
          out = reinterpret_cast<half *>(h2Out);
        } else {
          ++in;
          ++out;
        }
        size -= 1;
      }
      if (size >= 2 && reinterpret_cast<std::uintptr_t>(out) & 0x7) {
        if (worker == CTXT_WORKERS - 1) {
          half2 K2 = {K, K};
          const half2 *h2In = reinterpret_cast<const half2 *>(in);
          half2 *h2Out = reinterpret_cast<half2 *>(out);
          *h2Out++ =
              BinaryOpFn<op, half2, architecture::active>::fn(*h2In++, K2);
          in = reinterpret_cast<const half *>(h2In);
          out = reinterpret_cast<half *>(h2Out);
        } else {
          in += 2;
          out += 2;
        }
        size -= 2;
      }
    }

    const half4 *h4In = reinterpret_cast<const half4 *>(in) + worker;
    half4 *h4Out = reinterpret_cast<half4 *>(out) + worker;
    half4 K4 = {K, K, K, K};

    const rptsize_t loopCount = divideWork(size, 2, worker);
    for (unsigned j = 0; j < loopCount; j++) {
      *h4Out = BinaryOpFn<op, half4, architecture::active>::fn(*h4In, K4);
      h4In += CTXT_WORKERS;
      h4Out += CTXT_WORKERS;
    }
    // Use the same worker to deal with the leading elements as deals with
    // the trailing elements to avoid read-modify-write conflicts in
    // dealing with the odd single element
    if (allowRemainder) {
      if (worker == CTXT_WORKERS - 1) {
        const half2 *h2In =
            reinterpret_cast<const half2 *>(&in[size & ~unsigned(3)]);
        half2 *h2Out = reinterpret_cast<half2 *>(&out[size & ~unsigned(3)]);
        if (size & 2) {
          half2 K2 = {K, K};
          *h2Out++ =
              BinaryOpFn<op, half2, architecture::active>::fn(*h2In++, K2);
        }
        if (size & 1) {
          h2Out = reinterpret_cast<half2 *>(&out[size - 1]);
          write16Aligned32(
              BinaryOpFn<op, half, architecture::active>::fn((*h2In)[0], K),
              h2Out);
        }
      }
    }
  }
};

// This works only (and is instantiated) for BITWISE operators
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatchMultiVertex<op, short, short, allowUnaligned,
                                      allowRemainder> {
public:
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(4))) short *in,
                      __attribute__((align_value(4))) short *out,
                      const short K) {
    broadcastShort2_MultiVertex<op, short>(size, worker, in, out, K);
  }
};

// Works only (and is instantiated) for BITWISE operators for UNSIGNED SHORT
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatchMultiVertex<op, unsigned short, unsigned short,
                                      allowUnaligned, allowRemainder> {
public:
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(4))) unsigned short *in,
                      __attribute__((align_value(4))) unsigned short *out,
                      const short K) {
    broadcastShort2_MultiVertex<op, unsigned short>(size, worker, in, out, K);
  }
};

template <BinaryOpType op, typename inT, bool allowRemainder>
struct BroadcastRelationalOpDualOutputDispatchMultiVertex<op, inT, half,
                                                          allowRemainder> {
public:
  static void compute(unsigned size, unsigned worker, const inT *in, half *out,
                      half *outInv, const inT K) {
    half2 *h2Out1 = reinterpret_cast<half2 *>(out) + worker;
    half2 *h2Out2 = reinterpret_cast<half2 *>(outInv) + worker;
    const rptsize_t loopCount = divideWork(size, 1, worker);
    auto load = in + worker * 2;
    for (unsigned j = 0; j < loopCount; j++) {
      half2 calc1, calc2;
      for (unsigned i = 0; i < 2; ++i) {
        bool val = BinaryOpFn<op, inT, architecture::active>::fn(load[i], K);
        calc1[i] = static_cast<half>(val);
        calc2[i] = 1 - calc1[i];
      }
      *h2Out1 = calc1;
      *h2Out2 = calc2;
      h2Out1 += CTXT_WORKERS;
      h2Out2 += CTXT_WORKERS;
      load += CTXT_WORKERS * 2;
    }
    if (allowRemainder && (size & 1)) {
      if (worker == (CTXT_WORKERS - 1)) {
        unsigned offset = size - 1;
        half2 *h2Out1 = reinterpret_cast<half2 *>(&out[offset]);
        half2 *h2Out2 = reinterpret_cast<half2 *>(&outInv[offset]);
        bool val = BinaryOpFn<op, inT, architecture::active>::fn(in[offset], K);
        half2 output1 = {static_cast<half>(val), (*h2Out1)[1]};
        half2 output2 = {1 - output1[0], (*h2Out2)[1]};
        *h2Out1++ = output1;
        *h2Out2++ = output2;
      }
    }
  }
};

#endif

namespace popops {

#define INSTANTIATE_SCALAR(name)                                               \
  INSTANTIATE_OP(name, BinaryOpType::ADD, float, half, int, unsigned,          \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::ATAN2, float, half)                       \
  INSTANTIATE_OP(name, BinaryOpType::BITWISE_AND, int, unsigned, short,        \
                 unsigned short, unsigned long long, long long)                \
  INSTANTIATE_OP(name, BinaryOpType::BITWISE_OR, int, unsigned, short,         \
                 unsigned short, unsigned long long, long long)                \
  INSTANTIATE_OP(name, BinaryOpType::BITWISE_XOR, int, unsigned, short,        \
                 unsigned short, unsigned long long, long long)                \
  INSTANTIATE_OP(name, BinaryOpType::BITWISE_XNOR, int, unsigned, short,       \
                 unsigned short, unsigned long long, long long)                \
  INSTANTIATE_OP(name, BinaryOpType::DIVIDE, float, half, int, unsigned,       \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::LOGICAL_AND, bool)                        \
  INSTANTIATE_OP(name, BinaryOpType::LOGICAL_OR, bool)                         \
  INSTANTIATE_OP(name, BinaryOpType::MAXIMUM, float, half, int, unsigned,      \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::MINIMUM, float, half, int, unsigned,      \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::MULTIPLY, float, half, int, unsigned,     \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::POWER, float, half, int)                  \
  INSTANTIATE_OP(name, BinaryOpType::REMAINDER, float, half, int, unsigned,    \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::SHIFT_LEFT, int, unsigned,                \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::SHIFT_RIGHT, int, unsigned,               \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int, unsigned,   \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::SUBTRACT, float, half, int, unsigned,     \
                 unsigned long long, long long)                                \
  INSTANTIATE_OP(name, BinaryOpType::VARIANCE_TO_INV_STD_DEV, float, half)

// Relational (comparison) operators are separate because InPlace vertices
// have them only for boolean
#define INSTANTIATE_SCALAR_RELOP(name)                                         \
  INSTANTIATE_OP(name, BinaryOpType::EQUAL, float, half, int, unsigned, bool,  \
                 short, unsigned short, unsigned long long, long long)         \
  INSTANTIATE_OP(name, BinaryOpType::GREATER_THAN, float, half, int, unsigned, \
                 bool, unsigned long long, long long)                          \
  INSTANTIATE_OP(name, BinaryOpType::GREATER_THAN_EQUAL, float, half, int,     \
                 unsigned, bool, unsigned long long, long long)                \
  INSTANTIATE_OP(name, BinaryOpType::LESS_THAN, float, half, int, unsigned,    \
                 bool, unsigned long long, long long)                          \
  INSTANTIATE_OP(name, BinaryOpType::LESS_THAN_EQUAL, float, half, int,        \
                 unsigned, bool, unsigned long long, long long)                \
  INSTANTIATE_OP(name, BinaryOpType::NOT_EQUAL, float, half, int, unsigned,    \
                 bool, short, unsigned short, unsigned long long, long long)

#define INSTANTIATE_SCALAR_RELOP_IN_PLACE(name)                                \
  INSTANTIATE_OP(name, BinaryOpType::EQUAL, bool)                              \
  INSTANTIATE_OP(name, BinaryOpType::GREATER_THAN, bool)                       \
  INSTANTIATE_OP(name, BinaryOpType::GREATER_THAN_EQUAL, bool)                 \
  INSTANTIATE_OP(name, BinaryOpType::LESS_THAN, bool)                          \
  INSTANTIATE_OP(name, BinaryOpType::LESS_THAN_EQUAL, bool)                    \
  INSTANTIATE_OP(name, BinaryOpType::NOT_EQUAL, bool)

#define INSTANTIATE_SCALAR_DUAL_OUT_RELOP_TYPES(name, opType)                  \
  template class name<opType, unsigned, float>;                                \
  template class name<opType, unsigned, half>

#define INSTANTIATE_SCALAR_DUAL_OUT_RELOP(name)                                \
  INSTANTIATE_SCALAR_DUAL_OUT_RELOP_TYPES(name, BinaryOpType::EQUAL);          \
  INSTANTIATE_SCALAR_DUAL_OUT_RELOP_TYPES(name,                                \
                                          BinaryOpType::GREATER_THAN_EQUAL);   \
  INSTANTIATE_SCALAR_DUAL_OUT_RELOP_TYPES(name, BinaryOpType::GREATER_THAN);   \
  INSTANTIATE_SCALAR_DUAL_OUT_RELOP_TYPES(name,                                \
                                          BinaryOpType::LESS_THAN_EQUAL);      \
  INSTANTIATE_SCALAR_DUAL_OUT_RELOP_TYPES(name, BinaryOpType::LESS_THAN);      \
  INSTANTIATE_SCALAR_DUAL_OUT_RELOP_TYPES(name, BinaryOpType::NOT_EQUAL);

#define INSTANTIATE_VECTOR_OUTER_TYPES(name, opType)                           \
  template class name<opType, float, true>;                                    \
  template class name<opType, half, true>;                                     \
  template class name<opType, float, false>;                                   \
  template class name<opType, half, false>

#define INSTANTIATE_VECTOR_OUTER(name)                                         \
  INSTANTIATE_VECTOR_OUTER_TYPES(name, BinaryOpType::ADD);                     \
  INSTANTIATE_VECTOR_OUTER_TYPES(name, BinaryOpType::SUBTRACT);                \
  INSTANTIATE_VECTOR_OUTER_TYPES(name, BinaryOpType::MULTIPLY)

// The not-in-place broadcast vertices have an 'out' member for the output of
// the operation.
// The in-place broadcast vertices instead use the 'data' member for input and
// output.
// We will add the 'out' member, or not, passing this macro, or an empty
// argument, to each DEF_BROADCAST_xxx_VERTEX() macros.
#define OUT_1D_DEF Output<Vector<dType, ONE_PTR, 8>> out;
#define OUT_2D_DEF Vector<Output<Vector<dType, ONE_PTR, 8>>, ONE_PTR> out;

#define OUT_1D_DEF_HALF Output<Vector<half, ONE_PTR, 8>> out;
#define OUT_2D_DEF_HALF Vector<Output<Vector<half, ONE_PTR, 8>>, ONE_PTR> out;

template <BinaryOpType op, typename dType>
class BroadcastScalar2DData : public Vertex {
  using OutputType = typename BinaryOpOutputType<op, dType>::type;

public:
  Vector<Input<Vector<dType, SPAN, 8>>> data;
  Vector<Output<Vector<OutputType, ONE_PTR, 8>>, ONE_PTR> out;
  Input<dType> B;
  IS_EXTERNAL_CODELET((hasAssembly<op, dType>()));
  bool compute() {
    unsigned limI = data.size();
    for (unsigned i = 0; i < limI; i++) {
      BroadcastOpDispatch<op, dType, OutputType, false, true>::compute(
          data[i].size(), &data[i][0], &out[i][0], *B);
    }
    return true;
  }
};

template <BinaryOpType op, typename dType>
class BroadcastScalar2DDataInPlace : public Vertex {
  using OutputType = typename BinaryOpOutputType<op, dType>::type;

public:
  Vector<InOut<Vector<OutputType, SPAN, 8>>> data;
  Input<dType> B;
  IS_EXTERNAL_CODELET((hasAssembly<op, dType>()));
  bool compute() {
    unsigned limI = data.size();
    for (unsigned i = 0; i < limI; i++) {
      BroadcastOpDispatch<op, dType, OutputType, false, true>::compute(
          data[i].size(), &data[i][0], &data[i][0], *B);
    }
    return true;
  }
};

// Ensure that internal arithmetic is always done in full precision for
// INV_STD_DEV_TO_VARIANCE
#define DEF_BROADCAST_2D_DATA_VERTEX_FP(vertexName, inOutType, outDef,         \
                                        outName)                               \
  template <>                                                                  \
  class vertexName<BinaryOpType::INV_STD_DEV_TO_VARIANCE, half>                \
      : public Vertex {                                                        \
  public:                                                                      \
    Vector<inOutType<Vector<half, SPAN, 8>>> data;                             \
    outDef Input<half> B;                                                      \
    IS_EXTERNAL_CODELET(true);                                                 \
    bool compute() {                                                           \
      unsigned limI = data.size();                                             \
      for (unsigned i = 0; i < limI; i++) {                                    \
        for (unsigned j = 0; j < data[i].size(); j++) {                        \
          outName[i][j] = static_cast<half>(                                   \
              BinaryOpFn<                                                      \
                  BinaryOpType::INV_STD_DEV_TO_VARIANCE, float,                \
                  architecture::active>::fn(static_cast<float>(data[i][j]),    \
                                            static_cast<float>(*B)));          \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_2D_DATA_VERTEX_FP(BroadcastScalar2DData, Input, OUT_2D_DEF_HALF,
                                out)
DEF_BROADCAST_2D_DATA_VERTEX_FP(BroadcastScalar2DDataInPlace, InOut, , data)

template class BroadcastScalar2DDataInPlace<
    BinaryOpType::INV_STD_DEV_TO_VARIANCE, float>;
template class BroadcastScalar2DData<BinaryOpType::INV_STD_DEV_TO_VARIANCE,
                                     float>;

INSTANTIATE_SCALAR(BroadcastScalar2DData);
INSTANTIATE_SCALAR_RELOP(BroadcastScalar2DData);
INSTANTIATE_SCALAR(BroadcastScalar2DDataInPlace);
INSTANTIATE_SCALAR_RELOP_IN_PLACE(BroadcastScalar2DDataInPlace);

template <BinaryOpType op, typename dType>
class BroadcastScalar2D : public Vertex {
  using OutType = typename BinaryOpOutputType<op, dType>::type;

public:
  Vector<Input<Vector<dType, SPAN, 8>>> data;
  Vector<Output<Vector<OutType, ONE_PTR, 8>>, ONE_PTR> out;
  Vector<Input<dType>, ONE_PTR> B;
  bool compute() {
    unsigned limI = data.size();
    for (unsigned i = 0; i < limI; i++) {
      BroadcastOpDispatch<op, dType, OutType, false, true>::compute(
          data[i].size(), &data[i][0], &out[i][0], B[i]);
    }
    return true;
  }
};

template <BinaryOpType op, typename dType>
class BroadcastScalar2DInPlace : public Vertex {
public:
  Vector<InOut<Vector<dType, SPAN, 8>>> data;
  Vector<Input<dType>, ONE_PTR> B;
  bool compute() {
    unsigned limI = data.size();
    for (unsigned i = 0; i < limI; i++) {
      BroadcastOpDispatch<op, dType, dType, false, true>::compute(
          data[i].size(), &data[i][0], &data[i][0], B[i]);
    }
    return true;
  }
};

INSTANTIATE_SCALAR(BroadcastScalar2D);
INSTANTIATE_SCALAR_RELOP(BroadcastScalar2D);
INSTANTIATE_SCALAR(BroadcastScalar2DInPlace);
INSTANTIATE_SCALAR_RELOP_IN_PLACE(BroadcastScalar2DInPlace);
INSTANTIATE_OP(BroadcastScalar2D, BinaryOpType::INV_STD_DEV_TO_VARIANCE, half,
               float)
INSTANTIATE_OP(BroadcastScalar2DInPlace, BinaryOpType::INV_STD_DEV_TO_VARIANCE,
               half, float)

template <BinaryOpType op, typename inT, typename outT>
class BroadcastScalar1DRelationalOpDualOutput : public MultiVertex {
public:
  Input<Vector<inT, SPAN, 8>> data;
  Output<Vector<outT, ONE_PTR, 8>> out;
  Output<Vector<outT, ONE_PTR, 8>> outInv;
  Input<inT> B;
  IS_EXTERNAL_CODELET(false);
  bool compute(unsigned wid) {
    BroadcastRelationalOpDualOutputDispatchMultiVertex<
        op, inT, outT, true>::compute(data.size(), wid, &data[0], &out[0],
                                      &outInv[0], *B);
    return true;
  }
};

INSTANTIATE_SCALAR_DUAL_OUT_RELOP(BroadcastScalar1DRelationalOpDualOutput);

template <BinaryOpType op, typename dType>
class BroadcastScalar1D : public MultiVertex {
  using OutType = typename BinaryOpOutputType<op, dType>::type;

public:
  Input<Vector<dType, SPAN, 8>> data;
  Output<Vector<OutType, ONE_PTR, 8>> out;
  Input<dType> B;
  IS_EXTERNAL_CODELET((hasAssembly<op, dType>()));
  bool compute(unsigned wid) {
    BroadcastOpDispatchMultiVertex<op, dType, OutType, false, true>::compute(
        data.size(), wid, &data[0], &out[0], *B);
    return true;
  }
};

template <BinaryOpType op, typename dType>
class BroadcastScalar1DInPlace : public MultiVertex {
public:
  InOut<Vector<dType, SPAN, 8>> data;
  Input<dType> B;
  IS_EXTERNAL_CODELET((hasAssembly<op, dType>()));
  bool compute(unsigned wid) {
    BroadcastOpDispatchMultiVertex<op, dType, dType, false, true>::compute(
        data.size(), wid, &data[0], &data[0], *B);
    return true;
  }
};

template class BroadcastScalar1D<BinaryOpType::INV_STD_DEV_TO_VARIANCE, float>;
template class BroadcastScalar1DInPlace<BinaryOpType::INV_STD_DEV_TO_VARIANCE,
                                        float>;

INSTANTIATE_SCALAR(BroadcastScalar1D);
INSTANTIATE_SCALAR_RELOP(BroadcastScalar1D);
INSTANTIATE_SCALAR(BroadcastScalar1DInPlace);
INSTANTIATE_SCALAR_RELOP_IN_PLACE(BroadcastScalar1DInPlace);

// Ensure that internal arithmetic is always done in full precision for
// INV_STD_DEV_TO_VARIANCE
#define DEF_BROADCAST_1D_DATA_VERTEX_FP(vertexName, inOutType, outDef,         \
                                        outName)                               \
  template <>                                                                  \
  class vertexName<BinaryOpType::INV_STD_DEV_TO_VARIANCE, half>                \
      : public MultiVertex {                                                   \
  public:                                                                      \
    inOutType<Vector<half, SPAN, 8>> data;                                     \
    outDef Input<half> B;                                                      \
    IS_EXTERNAL_CODELET(true);                                                 \
    bool compute(unsigned wid) {                                               \
      for (unsigned i = wid; i < data.size(); i += numWorkers()) {             \
        outName[i] = static_cast<half>(                                        \
            BinaryOpFn<BinaryOpType::INV_STD_DEV_TO_VARIANCE, float,           \
                       architecture::active>::fn(static_cast<float>(data[i]),  \
                                                 static_cast<float>(*B)));     \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_1D_DATA_VERTEX_FP(BroadcastScalar1D, Input, OUT_1D_DEF_HALF, out)
DEF_BROADCAST_1D_DATA_VERTEX_FP(BroadcastScalar1DInPlace, InOut, , data)

#define DEF_BROADCAST_VECT_OUTER_BY_COLUMN_VERTEX(vertexName, inOutType,       \
                                                  outDef, outName, isInPlace)  \
  template <BinaryOpType op, typename dType, bool allowMisaligned>             \
  class vertexName : public MultiVertex {                                      \
    static constexpr std::size_t inputAlign =                                  \
        (allowMisaligned && isInPlace) ? alignof(dType) : 8;                   \
                                                                               \
  public:                                                                      \
    inOutType<Vector<dType, ONE_PTR, inputAlign>> data;                        \
    outDef Input<Vector<dType, SPAN>> B;                                       \
    unsigned short columns;                                                    \
    unsigned short rows;                                                       \
    IS_EXTERNAL_CODELET(!allowMisaligned);                                     \
    bool compute(unsigned wid) {                                               \
      std::size_t bIndex = 0;                                                  \
      auto bLen = B.size();                                                    \
      for (unsigned i = 0; i < rows; i++) {                                    \
        BroadcastOpDispatchMultiVertex<                                        \
            op, dType, dType, allowMisaligned,                                 \
            allowMisaligned>::compute(columns, wid, &data[i * columns],        \
                                      &outName[i * columns], B[bIndex]);       \
        ++bIndex;                                                              \
        if (bIndex == bLen) {                                                  \
          bIndex = 0;                                                          \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_VECT_OUTER_BY_COLUMN_VERTEX(BroadcastVectorOuterByColumn1D, Input,
                                          OUT_1D_DEF, out, false)
DEF_BROADCAST_VECT_OUTER_BY_COLUMN_VERTEX(BroadcastVectorOuterByColumn1DInPlace,
                                          InOut, , data, true)

INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByColumn1D);
INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByColumn1DInPlace);

// The template below will normally divide work by assigning one worker per
// row.  However in the case where the data type is half, and the data is
// not guaranteed aligned while processing every row, workers are assigned
// consecutive pairs of rows.  This avoids the case where an odd number of
// halves results in a read-modify-write conflict between one worker processing
// the last element of a row and another processing the first. At least 32 bit
// alignment is guaranteed at the end of 2 rows, providing that the start of the
// first row has at least 32 bit alignment itself.

#define DEF_BROADCAST_VECT_OUTER_BY_ROW_VERTEX(vertexName, inOutType, outDef,  \
                                               outName, isInPlace)             \
  template <BinaryOpType op, typename dType, bool allowMisaligned>             \
  class vertexName : public MultiVertex {                                      \
    static constexpr std::size_t inputAlign =                                  \
        (allowMisaligned && isInPlace) ? 4 : 8;                                \
    static constexpr bool assignWorkersPairsOfRows =                           \
        std::is_same<dType, half>::value && allowMisaligned;                   \
                                                                               \
  public:                                                                      \
    inOutType<Vector<dType, ONE_PTR, inputAlign>> data;                        \
    outDef Input<Vector<dType, SPAN>> B;                                       \
    unsigned short columns;                                                    \
    unsigned short rows;                                                       \
    IS_EXTERNAL_CODELET(!allowMisaligned);                                     \
    bool compute(unsigned wid) {                                               \
      std::size_t bIndex = assignWorkersPairsOfRows ? 2 * wid : wid;           \
      auto bLen = B.size();                                                    \
      unsigned i = bIndex;                                                     \
      unsigned increment = assignWorkersPairsOfRows ? 1 : numWorkers();        \
      while (i < rows) {                                                       \
        while (bIndex >= bLen) {                                               \
          bIndex -= bLen;                                                      \
        }                                                                      \
        BroadcastOpDispatch<op, dType, dType, allowMisaligned,                 \
                            allowMisaligned>::compute(columns,                 \
                                                      &data[i * columns],      \
                                                      &outName[i * columns],   \
                                                      B[bIndex]);              \
        bIndex += increment;                                                   \
        i += increment;                                                        \
        if (assignWorkersPairsOfRows) {                                        \
          if (increment == 1) {                                                \
            increment = (2 * numWorkers()) - 1;                                \
          } else {                                                             \
            increment = 1;                                                     \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_VECT_OUTER_BY_ROW_VERTEX(BroadcastVectorOuterByRow1D, Input,
                                       OUT_1D_DEF, out, false)
DEF_BROADCAST_VECT_OUTER_BY_ROW_VERTEX(BroadcastVectorOuterByRow1DInPlace,
                                       InOut, , data, true)

INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByRow1D);
INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByRow1DInPlace);

// VARIANCE_TO_INV_STD_DEV and INV_STD_DEV_TO_VARIANCE have the option of a
// different output type to input type, and all arithmetic is to be carried out
// in single precision.  Variance will be float, ISD, half
template <BinaryOpType op, typename inType, typename outType>
class BroadcastScalar2Types2DData : public Vertex {
public:
  Vector<Input<Vector<inType, SPAN, 8>>> data;
  Vector<Output<Vector<outType, ONE_PTR, 8>>, ONE_PTR> out;
  Input<inType> B;
  IS_EXTERNAL_CODELET(true);
  bool compute() {
    unsigned limI = data.size();
    for (unsigned i = 0; i < limI; i++) {
      for (unsigned j = 0; j < data[i].size(); j++) {
        out[i][j] = static_cast<outType>(
            BinaryOpFn<op, float, architecture::active>::fn(
                static_cast<float>(data[i][j]), static_cast<float>(*B)));
      }
    }
    return true;
  }
};

template class BroadcastScalar2Types2DData<
    BinaryOpType::INV_STD_DEV_TO_VARIANCE, half, float>;
template class BroadcastScalar2Types2DData<
    BinaryOpType::VARIANCE_TO_INV_STD_DEV, float, half>;

template <BinaryOpType op, typename inType, typename outType>
class BroadcastScalar2Types1D : public MultiVertex {
public:
  Input<Vector<inType, SPAN, 8>> data;
  Output<Vector<outType, ONE_PTR, 8>> out;
  Input<inType> B;
  IS_EXTERNAL_CODELET(true);
  bool compute(unsigned wid) {
    for (unsigned i = wid; i < data.size(); i += numWorkers()) {
      out[i] =
          static_cast<outType>(BinaryOpFn<op, float, architecture::active>::fn(
              static_cast<float>(data[i]), static_cast<float>(*B)));
    }
    return true;
  }
};
template class BroadcastScalar2Types1D<BinaryOpType::INV_STD_DEV_TO_VARIANCE,
                                       half, float>;
template class BroadcastScalar2Types1D<BinaryOpType::VARIANCE_TO_INV_STD_DEV,
                                       float, half>;

} // namespace popops
