// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>

#include "elemwiseBinaryOps.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;
using namespace popops;

using BinaryOpType = popops::expr::BinaryOpType;

// Defines which broadcast codelets have an assembly implementation
template <BinaryOpType op, typename dType> constexpr static bool hasAssembly() {
  return (std::is_same<dType, float>::value ||
          std::is_same<dType, half>::value) &&
         (op == BinaryOpType::ADD || op == BinaryOpType::SUBTRACT ||
          op == BinaryOpType::MULTIPLY);
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

// Note: run in the context of one of the worker (started by the supervisor),
// not the supervisor itself.
template <BinaryOpType op, typename inT, typename outT,
          bool allowUnaligned, // Allow input/output that isn't 64-bit aligned.
          bool allowRemainder>
struct BroadcastOpDispatchSupervisor {
public:
  static void compute(unsigned size, unsigned worker, const inT *in, outT *out,
                      const inT K) {
    // No vectorisation but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = BinaryOpFn<op, inT, architecture::active>::fn(in[j], K);
  }
};

#ifdef __IPU__

// This function is marked as 'noinline' because this, when used together with
// maskForRepeat to generate a loop count, encourages the compiler to create
// the loop using the more efficient 'rpt' instruction.
// See T9902  mad the definitions of maskForRepeat
__attribute__((noinline)) unsigned divideWork(const unsigned size,
                                              const unsigned vectorWidthShifts,
                                              const unsigned worker) {
  // Integer divide by 6.
  const unsigned loopCount = ((size >> vectorWidthShifts) * 0xaaab) >> 18;
  const unsigned remainder = (size >> vectorWidthShifts) - loopCount * 6;
  return loopCount + static_cast<unsigned>(worker < remainder);
}

/// Perform the bulk of a *broadcast scalar* comparison operator
/// [FLOAT, FLOAT] => BOOL, processing 2 'float2' in each cycle (i.e. 4 floats),
/// so that we write 4 boolean results (1 full word) at a time, avoiding calls
/// to __st8/__st16.
/// This is run both by the '2DData' vertices and by the 1D workers started
/// by the '1DSupervisor'
///
///  \tparam     op         Operation to perform. One of the comparison ops
///                        (EQUAL, LESS_THAN, ...)
///
///  \tparam     stride    Stride (in units of 4 floats/4 bools) to advance
///                        in/out at each cycle of the loop.
///
///  \param[in]  loopCount How many groups of 4 floats to process.
///
///  \param[in]  in        Pointer to an array of floats with the input data
///                        (first operand)
///
///  \param[out] out       Pointer to an array of boolean (1 byte each) that
///                        will be populated with the results
///
///  \param[in]  K         The second operand (the one that is broadcasted)
template <BinaryOpType op, unsigned stride>
void broadcastCompareOpBulk(unsigned loopCount, const float2 *in, int *out,
                            const float K) {
  const float2 K2 = {K, K};
  for (unsigned j = 0; j < loopCount; j++) {
    float2 load = ipu::load_postinc(&in, 1);
    long2 calc_lo = BinaryOpFn<op, float2, architecture::active>::fn(load, K2);
    load = ipu::load_postinc(&in, 2 * stride - 1);
    long2 calc_hi = BinaryOpFn<op, float2, architecture::active>::fn(load, K2);

    // Pack the two pair of results as 4 bytes in a 32 bit word and keep
    // only the least significant bit of each bytes
    *out = copy_cast<int>(tochar4(calc_lo, calc_hi)) & 0x01010101;
    out += stride;
  }
}

/// Same as for broadcastCompareOpBulk above but for [HALF, HALF] => BOOL,
/// processing a 'half4' at each cycle in the loop.
template <BinaryOpType op, unsigned stride>
void broadcastCompareOpBulk(unsigned loopCount, const half4 *in, int *out,
                            const half K) {
  const half4 K4 = {K, K, K, K};
  for (unsigned j = 0; j < loopCount; j++) {
    half4 load = ipu::load_postinc(&in, stride);
    short4 calc = BinaryOpFn<op, half4, architecture::active>::fn(load, K4);

    // Pack the four results as 4 bytes in a 32 bit word and keep only the
    // least significant bit of each bytes
    *out = copy_cast<int>(tochar4(calc)) & 0x01010101;
    out += stride;
  }
}

/// Performs the broadcast comparison operator ([FLOAT, FLOAT] => BOOL or
/// [HALF, HALF] => BOOL) for the trailing 0..3 elements of the input data.
/// This is run both by the '2DData' vertices and by the 1D workers started
/// by the '1DSupervisor'
///
///  \tparam     op        Operation to perform. One of the comparison ops
///                        (EQUAL, LESS_THAN, ...)
///
///  \tparam     T         data type )(float or half)
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
void broadcastCompareOpRemainder(unsigned size,
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

/// Processing for FLOAT, HALF operators returning BOOL (2D workers started by
/// Supervisor vertices).
///  \tparam  T    The data type (FLOAT or HALF)
///  \tparam  VT   The type for a 64 bit vector of type T (i.e. float2 or half4)
template <BinaryOpType op, typename T, typename VT, bool allowRemainder>
static void computeBool(unsigned size,
                        const __attribute__((align_value(8))) T *in,
                        __attribute__((align_value(8))) bool *out, const T K) {
  if (size >= 4) {
    const unsigned loopCount = maskForRepeat(size / 4u);
    broadcastCompareOpBulk<op, 1>(loopCount, reinterpret_cast<const VT *>(in),
                                  reinterpret_cast<int *>(out), K);
  }
  if (allowRemainder) {
    broadcastCompareOpRemainder<op, T>(size, in, out, K);
  }
}

// Run by the 2D worker for the comparison operators (EQUAL, LESS_THAN, etc )
// for float data (they return bool results).
// Optimised to perform word writes.
template <BinaryOpType op, bool allowRemainder>
struct BroadcastOpDispatch<op, float, bool, false, allowRemainder> {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) bool *out,
                      const float K) {
    computeBool<op, float, float2, allowRemainder>(size, in, out, K);
  }
};

// Run by the 2D worker for the comparison operators (EQUAL, LESS_THAN, etc )
// for half data (they return bool results).
// Optimised to perform word writes.
template <BinaryOpType op, bool allowRemainder>
struct BroadcastOpDispatch<op, half, bool, false, allowRemainder> {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) bool *out, const half K) {
    computeBool<op, half, half4, allowRemainder>(size, in, out, K);
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
        half2 res = {(*h2Out)[0],
                     BinaryOpFn<op, half, architecture::active>::fn(
                         ipu::load_postinc(&h2In, 1)[1], K)};
        *h2Out++ = res;
        size -= 1;
        in = reinterpret_cast<const half *>(h2In);
        out = reinterpret_cast<half *>(h2Out);
      }
      if (size >= 2 && reinterpret_cast<std::uintptr_t>(in) & 7) {
        half2 K2 = {K, K};
        const half2 *h2In = reinterpret_cast<const half2 *>(in);
        half2 *h2Out = reinterpret_cast<half2 *>(out);
        *h2Out++ = BinaryOpFn<op, half2, architecture::active>::fn(
            ipu::load_postinc(&h2In, 1), K2);
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
      const unsigned loopCount = maskForRepeat((size / 4u) - 1u);
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < loopCount; i++) {
        half4 calc = BinaryOpFn<op, half4, architecture::active>::fn(load, K4);
        load = ipu::load_postinc(&h4In, 1);
        *h4Out++ = calc;
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
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
        *h2Out++ = BinaryOpFn<op, half2, architecture::active>::fn(
            ipu::load_postinc(&h2In, 1), K2);
        size -= 2;
      }

      if (size == 1) {
        half2 res = (half2){
            BinaryOpFn<op, half, architecture::active>::fn((*h2In)[0], K),
            (*h2Out)[1],
        };
        *h2Out = res;
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
        *out++ = BinaryOpFn<op, float, architecture::active>::fn(
            ipu::load_postinc(&in, 1), K);
        size -= 1;
      }
    }
    if (size >= 2) {
      float2 K2 = {K, K};
      const float2 *f2In = reinterpret_cast<const float2 *>(in);
      float2 *f2Out = reinterpret_cast<float2 *>(out);
      float2 load = ipu::load_postinc(&f2In, 1);
      const unsigned loopCount = maskForRepeat((size / 2u) - 1);
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < loopCount; i++) {
        float2 calc =
            BinaryOpFn<op, float2, architecture::active>::fn(load, K2);
        load = ipu::load_postinc(&f2In, 1);
        *f2Out++ = calc;
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
      *f2Out++ = BinaryOpFn<op, float2, architecture::active>::fn(load, K2);
      if (allowRemainder) {
        in = reinterpret_cast<const float *>(f2In);
        out = reinterpret_cast<float *>(f2Out);
      }
    }
    if (allowRemainder) {
      if (size & 1) {
        float load = ipu::load_postinc(&in, 1);
        *out = BinaryOpFn<op, float, architecture::active>::fn(load, K);
      }
    }
  }
};

/// Processing for FLOAT, HALF operators returning BOOL (for workers started by
/// Supervisor vertices).
///
///  \tparam T              The data type (FLOAT or HALF)
///  \tparam VT             Type for a 64 bit vector of type T (i.e.
///                         float2 or half4)
///  \tparam allowRemainder Same as defined in BroadcastOpDispatchSupervisor
template <BinaryOpType op, typename T, typename VT, bool allowRemainder>
static void computeBoolSupervisor(unsigned size, unsigned worker,
                                  const __attribute__((align_value(8))) T *in,
                                  __attribute__((align_value(8))) bool *out,
                                  const T K) {
  const unsigned loopCount = maskForRepeat(divideWork(size, 2, worker));
  // We need to process 4 values at a time; for float this means: 2 x float2
  // while for half means: 1 x half4
  constexpr unsigned multTo4 = std::is_same<T, float>::value ? 2 : 1;
  broadcastCompareOpBulk<op, CTXT_WORKERS>(
      loopCount, reinterpret_cast<const VT *>(in) + multTo4 * worker,
      reinterpret_cast<int *>(out) + worker, K);

  // To process the trailing elements (if any) use the last worker as it is
  // most likely to have less to do than the others.
  if (allowRemainder) {
    if (worker == (CTXT_WORKERS - 1)) {
      broadcastCompareOpRemainder<op, T>(size, in, out, K);
    }
  }
}

// Run by the worker (started by supervisor vertex) for the comparison
// operators (EQUAL, LESS_THAN, etc that return bool results) for float data.
template <BinaryOpType op, bool allowRemainder>
class BroadcastOpDispatchSupervisor<op, float, bool, false, allowRemainder> {
public:
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) bool *out,
                      const float K) {
    computeBoolSupervisor<op, float, float2, allowRemainder>(size, worker, in,
                                                             out, K);
  }
};

// Run by the worker (started by supervisor vertex) for the comparison
// operators (EQUAL, LESS_THAN, etc ) for half data (return bool results).
template <BinaryOpType op, bool allowRemainder>
struct BroadcastOpDispatchSupervisor<op, half, bool, false, allowRemainder> {
public:
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) bool *out, const half K) {
    computeBoolSupervisor<op, half, half4, allowRemainder>(size, worker, in,
                                                           out, K);
  }
};

// Run by the worker (started by supervisor vertex) for the operators that
// work on floats and return floats.
// Optimised to perform vector read/writes.
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
class BroadcastOpDispatchSupervisor<op, float, float, allowUnaligned,
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
          auto val = ipu::load_postinc(&in, 1);
          *out++ = BinaryOpFn<op, float, architecture::active>::fn(val, K);
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
    const unsigned loopCount = maskForRepeat(divideWork(size, 1, worker));
    for (unsigned j = 0; j < loopCount; j++) {
      float2 load = ipu::load_postinc(&f2In, CTXT_WORKERS);
      float2 calc = BinaryOpFn<op, float2, architecture::active>::fn(load, K2);
      *f2Out = calc;
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

// Run by the worker (started by supervisor vertex) for the operators that
// work on half data and return half.
// Optimised to perform vector read/writes.
template <BinaryOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatchSupervisor<op, half, half, allowUnaligned,
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
        if (worker == NUM_WORKERS - 1) {
          const half2 *h2In = reinterpret_cast<const half2 *>(
              reinterpret_cast<std::uintptr_t>(in) & ~std::uintptr_t(0x3));
          half2 *h2Out = reinterpret_cast<half2 *>(
              reinterpret_cast<std::uintptr_t>(out) & ~std::uintptr_t(0x3));
          half2 res = {(*h2Out)[0],
                       BinaryOpFn<op, half, architecture::active>::fn(
                           ipu::load_postinc(&h2In, 1)[1], K)};
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
        if (worker == NUM_WORKERS - 1) {
          half2 K2 = {K, K};
          const half2 *h2In = reinterpret_cast<const half2 *>(in);
          half2 *h2Out = reinterpret_cast<half2 *>(out);
          *h2Out++ = BinaryOpFn<op, half2, architecture::active>::fn(
              ipu::load_postinc(&h2In, 1), K2);
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

    asm volatile("# Thwart loop rotation (start)" ::: "memory");
    const unsigned loopCount = maskForRepeat(divideWork(size, 2, worker));
    for (unsigned j = 0; j < loopCount; j++) {
      half4 load = ipu::load_postinc(&h4In, CTXT_WORKERS);
      half4 calc = BinaryOpFn<op, half4, architecture::active>::fn(load, K4);
      *h4Out = calc;
      h4Out += CTXT_WORKERS;
    }
    asm volatile("# Thwart loop rotation (end)" ::: "memory");
    // Use the same worker to deal with the leading elements as deals with
    // the trailing elements to avoid read-modify-write conflicts in
    // dealing with the odd single element
    if (allowRemainder) {
      if (worker == NUM_WORKERS - 1) {
        const half2 *h2In =
            reinterpret_cast<const half2 *>(&in[size & ~unsigned(3)]);
        half2 *h2Out = reinterpret_cast<half2 *>(&out[size & ~unsigned(3)]);
        if (size & 2) {
          half2 K2 = {K, K};
          *h2Out++ = BinaryOpFn<op, half2, architecture::active>::fn(
              ipu::load_postinc(&h2In, 1), K2);
        }
        if (size & 1) {
          h2Out = reinterpret_cast<half2 *>(&out[size - 1]);
          half2 res = {
              BinaryOpFn<op, half, architecture::active>::fn((*h2In)[0], K),
              (*h2Out)[1],
          };
          *h2Out = res;
        }
      }
    }
  }
};

#endif

namespace popops {

#define INSTANTIATE_SCALAR(name)                                               \
  INSTANTIATE_OP(name, BinaryOpType::ADD, float, half, int, unsigned)          \
  INSTANTIATE_OP(name, BinaryOpType::ATAN2, float, half)                       \
  INSTANTIATE_OP(name, BinaryOpType::BITWISE_AND, int, unsigned)               \
  INSTANTIATE_OP(name, BinaryOpType::BITWISE_OR, int, unsigned)                \
  INSTANTIATE_OP(name, BinaryOpType::BITWISE_XOR, int, unsigned)               \
  INSTANTIATE_OP(name, BinaryOpType::BITWISE_XNOR, int, unsigned)              \
  INSTANTIATE_OP(name, BinaryOpType::DIVIDE, float, half, int, unsigned)       \
  INSTANTIATE_OP(name, BinaryOpType::LOGICAL_AND, bool)                        \
  INSTANTIATE_OP(name, BinaryOpType::LOGICAL_OR, bool)                         \
  INSTANTIATE_OP(name, BinaryOpType::MAXIMUM, float, half, int, unsigned)      \
  INSTANTIATE_OP(name, BinaryOpType::MINIMUM, float, half, int, unsigned)      \
  INSTANTIATE_OP(name, BinaryOpType::MULTIPLY, float, half, int, unsigned)     \
  INSTANTIATE_OP(name, BinaryOpType::POWER, float, half)                       \
  INSTANTIATE_OP(name, BinaryOpType::REMAINDER, float, half, int, unsigned)    \
  INSTANTIATE_OP(name, BinaryOpType::SHIFT_LEFT, int, unsigned)                \
  INSTANTIATE_OP(name, BinaryOpType::SHIFT_RIGHT, int, unsigned)               \
  INSTANTIATE_OP(name, BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int, unsigned)   \
  INSTANTIATE_OP(name, BinaryOpType::SUBTRACT, float, half, int, unsigned)     \
  INSTANTIATE_OP(name, BinaryOpType::VARIANCE_TO_INV_STD_DEV, float, half)

// Relational (comparison) operators are separate because InPlace vertices
// have them only for boolean
#define INSTANTIATE_SCALAR_RELOP(name)                                         \
  INSTANTIATE_OP(name, BinaryOpType::EQUAL, float, half, int, unsigned, bool)  \
  INSTANTIATE_OP(name, BinaryOpType::GREATER_THAN, float, half, int, unsigned, \
                 bool)                                                         \
  INSTANTIATE_OP(name, BinaryOpType::GREATER_THAN_EQUAL, float, half, int,     \
                 unsigned, bool)                                               \
  INSTANTIATE_OP(name, BinaryOpType::LESS_THAN, float, half, int, unsigned,    \
                 bool)                                                         \
  INSTANTIATE_OP(name, BinaryOpType::LESS_THAN_EQUAL, float, half, int,        \
                 unsigned, bool)                                               \
  INSTANTIATE_OP(name, BinaryOpType::NOT_EQUAL, float, half, int, unsigned,    \
                 bool)

#define INSTANTIATE_SCALAR_RELOP_IN_PLACE(name)                                \
  INSTANTIATE_OP(name, BinaryOpType::EQUAL, bool)                              \
  INSTANTIATE_OP(name, BinaryOpType::GREATER_THAN, bool)                       \
  INSTANTIATE_OP(name, BinaryOpType::GREATER_THAN_EQUAL, bool)                 \
  INSTANTIATE_OP(name, BinaryOpType::LESS_THAN, bool)                          \
  INSTANTIATE_OP(name, BinaryOpType::LESS_THAN_EQUAL, bool)                    \
  INSTANTIATE_OP(name, BinaryOpType::NOT_EQUAL, bool)

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

#define DEF_BROADCAST_VECT_OUTER_BY_COLUMN_VERTEX(vertexName, inOutType,       \
                                                  outDef, outName, isInPlace)  \
  template <BinaryOpType op, typename dType, bool allowMisaligned>             \
  class vertexName : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {         \
    static constexpr std::size_t inputAlign =                                  \
        (allowMisaligned && isInPlace) ? alignof(dType) : 8;                   \
                                                                               \
  public:                                                                      \
    inOutType<Vector<dType, ONE_PTR, inputAlign>> data;                        \
    outDef Input<Vector<dType, SPAN>> B;                                       \
    short columns;                                                             \
    short rows;                                                                \
    IS_EXTERNAL_CODELET(true);                                                 \
    bool compute() {                                                           \
      std::size_t bIndex = 0;                                                  \
      auto bLen = B.size();                                                    \
      for (unsigned i = 0; i < rows; i++) {                                    \
        BroadcastOpDispatch<op, dType, dType, allowMisaligned,                 \
                            allowMisaligned>::compute(columns,                 \
                                                      &data[i * columns],      \
                                                      &outName[i * columns],   \
                                                      B[bIndex]);              \
        ++bIndex;                                                              \
        if (bIndex == bLen) {                                                  \
          bIndex = 0;                                                          \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_VECT_OUTER_BY_COLUMN_VERTEX(
    BroadcastVectorOuterByColumnSupervisor, Input, OUT_1D_DEF, out, false)
DEF_BROADCAST_VECT_OUTER_BY_COLUMN_VERTEX(
    BroadcastVectorOuterByColumnInPlaceSupervisor, InOut, , data, true)

INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByColumnSupervisor);
INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByColumnInPlaceSupervisor);

#define DEF_BROADCAST_VECT_OUTER_BY_ROW_VERTEX(vertexName, inOutType, outDef,  \
                                               outName, isInPlace)             \
  template <BinaryOpType op, typename dType, bool allowMisaligned>             \
  class vertexName : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {         \
    static constexpr std::size_t inputAlign =                                  \
        (allowMisaligned && isInPlace) ? alignof(dType) : 8;                   \
                                                                               \
  public:                                                                      \
    inOutType<Vector<dType, ONE_PTR, inputAlign>> data;                        \
    outDef Input<Vector<dType, SPAN>> B;                                       \
    short columns;                                                             \
    short rows;                                                                \
    IS_EXTERNAL_CODELET(true);                                                 \
    bool compute() {                                                           \
      std::size_t bIndex = 0;                                                  \
      auto bLen = B.size();                                                    \
      for (unsigned i = 0; i < rows; i++) {                                    \
        BroadcastOpDispatch<op, dType, dType, allowMisaligned,                 \
                            allowMisaligned>::compute(columns,                 \
                                                      &data[i * columns],      \
                                                      &outName[i * columns],   \
                                                      B[bIndex]);              \
        ++bIndex;                                                              \
        if (bIndex == bLen) {                                                  \
          bIndex = 0;                                                          \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_VECT_OUTER_BY_ROW_VERTEX(BroadcastVectorOuterByRowSupervisor,
                                       Input, OUT_1D_DEF, out, false)
DEF_BROADCAST_VECT_OUTER_BY_ROW_VERTEX(
    BroadcastVectorOuterByRowInPlaceSupervisor, InOut, , data, true)

INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByRowSupervisor);
INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByRowInPlaceSupervisor);

template <BinaryOpType op, typename dType>
class BroadcastScalar1DSupervisor
    : public SupervisorVertexIf<binaryOp1DIsSupervisor<op, dType>() &&
                                ASM_CODELETS_ENABLED> {
  typedef typename BinaryOpOutputType<op, dType>::type OutputType;

public:
  Input<Vector<dType, SPAN, 8>> data;
  Output<Vector<OutputType, ONE_PTR, 8>> out;
  Input<dType> B;
  IS_EXTERNAL_CODELET((binaryOp1DIsSupervisor<op, dType>()));
  bool compute() {
    BroadcastOpDispatch<op, dType, OutputType, false, true>::compute(
        data.size(), &data[0], &out[0], *B);
    return true;
  }
};

template <BinaryOpType op, typename dType>
class BroadcastScalar1DInPlaceSupervisor
    : public SupervisorVertexIf<binaryOp1DInPlaceIsSupervisor<op, dType>() &&
                                ASM_CODELETS_ENABLED> {
  typedef typename BinaryOpOutputType<op, dType>::type OutputType;

public:
  InOut<Vector<OutputType, SPAN, 8>> data;
  Input<dType> B;
  IS_EXTERNAL_CODELET((binaryOp1DInPlaceIsSupervisor<op, dType>()));
  bool compute() {
    BroadcastOpDispatch<op, dType, OutputType, false, true>::compute(
        data.size(), &data[0], &data[0], *B);
    return true;
  }
};

// Ensure that internal arithmetic is always done in full precision for
// INV_STD_DEV_TO_VARIANCE
#define DEF_BROADCAST_1D_VERTEX_FP(vertexName, inOutType, outDef, outName)     \
  template <>                                                                  \
  class vertexName<BinaryOpType::INV_STD_DEV_TO_VARIANCE, half>                \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    inOutType<Vector<half, SPAN, 8>> data;                                     \
    outDef Input<half> B;                                                      \
    IS_EXTERNAL_CODELET(true);                                                 \
    bool compute() {                                                           \
      unsigned limI = data.size();                                             \
      for (unsigned i = 0; i < limI; i++) {                                    \
        outName[i] = static_cast<half>(                                        \
            BinaryOpFn<BinaryOpType::INV_STD_DEV_TO_VARIANCE, float,           \
                       architecture::active>::fn(static_cast<float>(data[i]),  \
                                                 static_cast<float>(*B)));     \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_1D_VERTEX_FP(BroadcastScalar1DSupervisor, Input, OUT_1D_DEF_HALF,
                           out)
DEF_BROADCAST_1D_VERTEX_FP(BroadcastScalar1DInPlaceSupervisor, InOut, , data)

template class BroadcastScalar1DInPlaceSupervisor<
    BinaryOpType::INV_STD_DEV_TO_VARIANCE, float>;
template class BroadcastScalar1DSupervisor<
    BinaryOpType::INV_STD_DEV_TO_VARIANCE, float>;

INSTANTIATE_SCALAR(BroadcastScalar1DSupervisor);
INSTANTIATE_SCALAR_RELOP(BroadcastScalar1DSupervisor);
INSTANTIATE_SCALAR(BroadcastScalar1DInPlaceSupervisor);
INSTANTIATE_SCALAR_RELOP_IN_PLACE(BroadcastScalar1DInPlaceSupervisor);

#ifdef __IPU__

// Create worker vertex code, which will do the actual work, when the
// supervisor vertices compile to an external codelet.  Called via an
// assembly stub.

template <BinaryOpType op, typename dType>
class BroadcastScalar1D : public Vertex {
  using OutType = typename BinaryOpOutputType<op, dType>::type;

public:
  Input<Vector<dType, SPAN, 8>> data;
  Output<Vector<OutType, ONE_PTR, 8>> out;
  Input<dType> B;
  IS_EXTERNAL_CODELET((hasAssembly<op, dType>()));
  bool compute() {
    BroadcastOpDispatchSupervisor<op, dType, OutType, false, true>::compute(
        data.size(), getWsr(), &data[0], &out[0], *B);
    return true;
  }
};

template <BinaryOpType op, typename dType>
class BroadcastScalar1DInPlace : public Vertex {
public:
  InOut<Vector<dType, SPAN, 8>> data;
  Input<dType> B;
  IS_EXTERNAL_CODELET((hasAssembly<op, dType>()));
  bool compute() {
    BroadcastOpDispatchSupervisor<op, dType, dType, false, true>::compute(
        data.size(), getWsr(), &data[0], &data[0], *B);
    return true;
  }
};

template class BroadcastScalar1D<BinaryOpType::INV_STD_DEV_TO_VARIANCE, float>;
template class BroadcastScalar1DInPlace<BinaryOpType::INV_STD_DEV_TO_VARIANCE,
                                        float>;

INSTANTIATE_SCALAR(BroadcastScalar1D);
INSTANTIATE_SCALAR_RELOP(BroadcastScalar1D);
INSTANTIATE_SCALAR(BroadcastScalar1DInPlace);

#define DEF_BROADCAST_VECT_OUTER_BY_COLUMN_WK_VERTEX(                          \
    vertexName, inOutType, outDef, outName, isInPlace)                         \
  template <BinaryOpType op, typename dType, bool allowMisaligned>             \
  class vertexName : public Vertex {                                           \
    static constexpr std::size_t inputAlign =                                  \
        (allowMisaligned && isInPlace) ? alignof(dType) : 8;                   \
                                                                               \
  public:                                                                      \
    inOutType<Vector<dType, ONE_PTR, inputAlign>> data;                        \
    outDef Input<Vector<dType, SPAN>> B;                                       \
    unsigned short columns;                                                    \
    unsigned short rows;                                                       \
    IS_EXTERNAL_CODELET(!allowMisaligned);                                     \
    bool compute() {                                                           \
      std::size_t bIndex = 0;                                                  \
      auto bLen = B.size();                                                    \
      for (unsigned i = 0; i < rows; i++) {                                    \
        BroadcastOpDispatchSupervisor<                                         \
            op, dType, dType, allowMisaligned,                                 \
            allowMisaligned>::compute(columns, getWsr(), &data[i * columns],   \
                                      &outName[i * columns], B[bIndex]);       \
        ++bIndex;                                                              \
        if (bIndex == bLen) {                                                  \
          bIndex = 0;                                                          \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_VECT_OUTER_BY_COLUMN_WK_VERTEX(BroadcastVectorOuterByColumn,
                                             Input, OUT_1D_DEF, out, false)
DEF_BROADCAST_VECT_OUTER_BY_COLUMN_WK_VERTEX(
    BroadcastVectorOuterByColumnInPlace, InOut, , data, true)

INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByColumn);
INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByColumnInPlace);

// The template below will normally divide work by assigning one worker per
// row.  However in the case where the data type is half, and the data is
// not guaranteed aligned while processing every row, workers are assigned
// consecutive pairs of rows.  This avoids the case where an odd number of
// halves results in a read-modify-write conflict between one worker processing
// the last element of a row and another processing the first. At least 32 bit
// alignment is guaranteed at the end of 2 rows, providing that the start of the
// first row has at least 32 bit alignment itself.

#define DEF_BROADCAST_VECT_OUTER_BY_ROW_WK_VERTEX(vertexName, inOutType,       \
                                                  outDef, outName, isInPlace)  \
  template <BinaryOpType op, typename dType, bool allowMisaligned>             \
  class vertexName : public Vertex {                                           \
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
    bool compute() {                                                           \
      std::size_t bIndex = assignWorkersPairsOfRows ? 2 * getWsr() : getWsr(); \
      auto bLen = B.size();                                                    \
      unsigned i = bIndex;                                                     \
      unsigned increment = assignWorkersPairsOfRows ? 1 : CTXT_WORKERS;        \
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
            increment = (2 * CTXT_WORKERS) - 1;                                \
          } else {                                                             \
            increment = 1;                                                     \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_VECT_OUTER_BY_ROW_WK_VERTEX(BroadcastVectorOuterByRow, Input,
                                          OUT_1D_DEF, out, false)
DEF_BROADCAST_VECT_OUTER_BY_ROW_WK_VERTEX(BroadcastVectorOuterByRowInPlace,
                                          InOut, , data, true)

INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByRow);
INSTANTIATE_VECTOR_OUTER(BroadcastVectorOuterByRowInPlace);

#endif // __IPU__

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
class BroadcastScalar2Types1DSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  Input<Vector<inType, SPAN, 8>> data;
  Output<Vector<outType, ONE_PTR, 8>> out;
  Input<inType> B;
  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < data.size(); i++) {
      out[i] =
          static_cast<outType>(BinaryOpFn<op, float, architecture::active>::fn(
              static_cast<float>(data[i]), static_cast<float>(*B)));
    }
    return true;
  }
};
template class BroadcastScalar2Types1DSupervisor<
    BinaryOpType::INV_STD_DEV_TO_VARIANCE, half, float>;
template class BroadcastScalar2Types1DSupervisor<
    BinaryOpType::VARIANCE_TO_INV_STD_DEV, float, half>;
} // namespace popops
