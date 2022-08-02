// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <cstring>

#include "elemwiseBinaryOps.hpp"

#ifdef __IPU__
#include "inlineAssembler.hpp"
#endif

using namespace poplar;

namespace popops {

using BinaryOpType = popops::expr::BinaryOpType;

template <expr::BinaryOpType op, typename inT, typename outT, typename A>
struct BinaryOpDispatch {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) inT *in1,
                      const __attribute__((align_value(8))) inT *in2,
                      __attribute__((align_value(8))) outT *out) {

    for (unsigned j = 0; j != size; ++j) {
      out[j] = BinaryOpFn<op, inT, A>::fn(in1[j], in2[j]);
    }
  }
};

#ifdef __IPU__

/// Performs the bulk of a binary 'op' that has bool as output
/// type:  T op T => BOOL (i.e. the comparison operator EQUAL, LESS_THAN etc
/// plus LOGICAL_AND/OR).
/// This processes 4 elements of type T in each cycle of the loop, and writes
/// the 4 aggregated boolean results (1 full word) at a time, avoiding calls
/// to __st8/__st16.
/// This is run both by the '2D' and 'MultiVertex' vertices.
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
///  \param[in]  in1       Pointer to an array of T with first operand.
///
///  \param[in]  in2       Pointer to an array of T with the second operand.
///
///  \param[out] out       Pointer to an array of boolean (1 byte each) that
///                        will be populated with the results
///
template <BinaryOpType op, typename T, unsigned stride>
struct binaryBoolOpBulk {
  static void compute(unsigned loopCount, const T *in1, const T *in2,
                      int *out) {
    for (unsigned i = 0; i < loopCount; i++) {
      unsigned result4 = 0;
      // Accumulate in 'result4' the 4 bytes for this loop
      for (unsigned j = 0, shifts = 0; j != 4; ++j, shifts += 8) {
        bool res = BinaryOpFn<op, T, architecture::active>::fn(in1[j], in2[j]);
        result4 |= res << shifts;
      }
      in1 += 4 * stride;
      in2 += 4 * stride;
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
  template <unsigned stride> struct binaryBoolOpBulk<op, bool, stride> {       \
    static void compute(unsigned loopCount, const bool *in1, const bool *in2,  \
                        int *out) {                                            \
      const unsigned *b4In1 = reinterpret_cast<const unsigned *>(in1);         \
      const unsigned *b4In2 = reinterpret_cast<const unsigned *>(in2);         \
      for (unsigned i = 0; i < loopCount; i++) {                               \
        unsigned word1 = *b4In1;                                               \
        unsigned word2 = *b4In2;                                               \
        *out = OP_FN;                                                          \
        b4In1 += stride;                                                       \
        b4In2 += stride;                                                       \
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

/// Binary 'op' for bool output type ( T op T => BOOL) for the
/// trailing 0..3 elements of the input data (used in conjunction with
/// 'binaryBoolOpBulk')
/// This is run both by the '2D' and 'MultiVertex' vertices.
///
///  \tparam     op        Operation to perform. One of the comparison ops
///                        (EQUAL, LESS_THAN, ...)
///
///  \tparam     T         data type (float or half)
///
///  \param[in]  size      Total number of data elements
///
///  \param[in]  in1       Pointer to an array of T with first operand.
///
///  \param[in]  in2       Pointer to an array of T with the second operand.
///
///  \param[out] out       Pointer to an array of boolean (1 byte each) that
///                        is/will be populated with the results
///
template <BinaryOpType op, typename T>
void binaryBoolOpRemainder(unsigned size,
                           const __attribute__((align_value(8))) T *in1,
                           const __attribute__((align_value(8))) T *in2,
                           __attribute__((align_value(8))) bool *out) {
  unsigned remainder = size & 3;
  if (remainder) {
    unsigned offs = size - remainder;
    in1 = &in1[offs]; // make it point to the 'remainder'
    in2 = &in2[offs]; // make it point to the 'remainder'
    // Read the word of the output in memory that will contain the 1-3 bytes
    // of the remainder, so that we can write back the byte(s) we will not be
    // updating.
    unsigned *out4 = reinterpret_cast<unsigned *>(&out[offs]);
    unsigned result4 = *out4;
    // Accumulate in 'result4' the 1-3 bytes of the remainder
    unsigned mask = 0xff;
    for (unsigned j = 0, shifts = 0; j != remainder; ++j, shifts += 8) {
      bool res = BinaryOpFn<op, T, architecture::active>::fn(in1[j], in2[j]);
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
///  \param[in]  loopCount How many vectors process.
///
///  \param[in]  in1       Pointer to an array of input data for first operand
///
///  \param[in]  in2       Pointer to an array of input data for second operand
///
///  \param[out] out       Pointer to an array that will be populated with the
///                        results
///
template <BinaryOpType op, typename T, unsigned stride>
static void binaryShort2Bulk(unsigned loopCount,
                             const __attribute__((align_value(8))) T *in1,
                             const __attribute__((align_value(8))) T *in2,
                             __attribute__((align_value(8))) T *out) {
  const short2 *s2In1 = reinterpret_cast<const short2 *>(in1);
  const short2 *s2In2 = reinterpret_cast<const short2 *>(in2);
  short2 *s2Out = reinterpret_cast<short2 *>(out);

  for (unsigned j = 0; j < loopCount; j++) {
    short2 load1 = *s2In1;
    s2In1 += stride;
    short2 load2 = *s2In2;
    s2In2 += stride;
    short2 calc =
        BinaryOpFn<op, short2, architecture::active>::fn(load1, load2);
    *s2Out = calc;
    s2Out += stride;
  }
}

/// Processes the 'trailing' 1-3 elements (if present) for any operation that
/// can be performed on 'short4'/'short2' vectors (starting from arrays of
/// 16 bit short integers)
///
/// \tparam op            Operation to perform
///
/// \tparam vectorWidthShifts  1 or 2 to indicate if we have a vector of 2
///                            or 4 elements, i.e.: log2(vectorWidth)
///
/// \param[in]  size      Total number of elements contained in 'in1', 'in2'
///
/// \param[in]  in1       Pointer to an array of input data for first operand
///
/// \param[in]  in2       Pointer to an array of input data for second operand
///
/// \param[out] out       Pointer to an array that will be populated with the
///                       results
///
template <BinaryOpType op, typename T, unsigned vectorWidthShifts>
static void
binaryShort2Short4Remainder(unsigned size,
                            const __attribute__((align_value(8))) T *in1,
                            const __attribute__((align_value(8))) T *in2,
                            __attribute__((align_value(8))) T *out) {
  constexpr unsigned mask = (1 << vectorWidthShifts) - 1;
  const unsigned rem = size & mask;
  const short2 *s2In1 = reinterpret_cast<const short2 *>(&in1[size - rem]);
  const short2 *s2In2 = reinterpret_cast<const short2 *>(&in2[size - rem]);
  short2 *s2Out = reinterpret_cast<short2 *>(&out[size - rem]);
  if constexpr (mask == 0x3) {
    if (size & 3) {
      *s2Out = BinaryOpFn<op, short2, architecture::active>::fn(*s2In1, *s2In2);
      s2In1++;
      s2In2++;
      s2Out++;
    }
  }
  // This works only for short, unsigned short and is for use with bitwise
  // operators only
  static_assert(std::is_same<T, short>::value ||
                std::is_same<T, unsigned short>::value);
  if (size & 1) {
    short2 res = {
        static_cast<short>(BinaryOpFn<op, T, architecture::active>::fn(
            (*s2In1)[0], (*s2In2)[0])),
        (*s2Out)[1],
    };
    *s2Out = res;
  }
}

template <BinaryOpType op, typename T>
static void
binaryShort2_MultiVertex(unsigned size, unsigned worker,
                         const __attribute__((align_value(4))) T *in1,
                         const __attribute__((align_value(4))) T *in2,
                         __attribute__((align_value(4))) T *out) {
  const rptsize_t loopCount = divideWork(size, 1, worker);

  binaryShort2Bulk<op, T, CTXT_WORKERS>(loopCount, in1 + 2 * worker,
                                        in2 + 2 * worker, out + 2 * worker);

  // To process the trailing elements (if any) use the last worker as it is
  // most likely to have less to do than the others.
  if (worker == (CTXT_WORKERS - 1)) {
    binaryShort2Short4Remainder<op, T, 1>(size, in1, in2, out);
  }
}

template <BinaryOpType op, typename T>
static void binaryShort2_2D(unsigned size,
                            const __attribute__((align_value(4))) T *in1,
                            const __attribute__((align_value(4))) T *in2,
                            __attribute__((align_value(4))) T *out) {
  const rptsize_t loopCount = size / 2u;
  binaryShort2Bulk<op, T, 1>(loopCount, in1, in2, out);
  binaryShort2Short4Remainder<op, T, 1>(size, in1, in2, out);
}

// Run by the 2D worker for  binary ops that have bool as output
// type:  T op T => BOOL (i.e. the comparison operator EQUAL, LESS_THAN etc
// plus LOGICAL_AND/OR)
template <BinaryOpType op, typename T>
struct BinaryOpDispatch<op, T, bool, architecture::ipu> {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) T *in1,
                      const __attribute__((align_value(8))) T *in2,
                      __attribute__((align_value(8))) bool *out) {
    if (size >= 4) {
      const rptsize_t loopCount = size / 4u;
      binaryBoolOpBulk<op, T, 1>::compute(loopCount, in1, in2,
                                          reinterpret_cast<int *>(out));
    }
    binaryBoolOpRemainder<op, T>(size, in1, in2, out);
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, float, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = BinaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in1,
                      const __attribute__((align_value(8))) float *in2,
                      __attribute__((align_value(8))) bool *out) {

    if (size >= 4) {
      const float2 *f2In1 = reinterpret_cast<const float2 *>(in1);
      const float2 *f2In2 = reinterpret_cast<const float2 *>(in2);
      int *iOut = reinterpret_cast<int *>(out);
      const rptsize_t loopCount = size / 4u;
      for (unsigned i = 0; i < loopCount; ++i) {
        long2 calc_lo =
            static_cast<long2>(FuncTy<float2>::fn(*f2In1++, *f2In2++));
        long2 calc_hi =
            static_cast<long2>(FuncTy<float2>::fn(*f2In1++, *f2In2++));

        char4 result = tochar4(calc_lo, calc_hi);
        int ires = copy_cast<int>(result) & 0x01010101;
        *iOut++ = ires;
      }
      in1 = reinterpret_cast<const float *>(f2In1);
      in2 = reinterpret_cast<const float *>(f2In2);
      out = reinterpret_cast<bool *>(iOut);
    }
    // process any remainder, up to 3 of
    size = size & 3;
    for (unsigned j = 0; j != size; ++j) {
      *out++ = FuncTy<float>::fn(*in1++, *in2++);
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, half, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = BinaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in1,
                      const __attribute__((align_value(8))) half *in2,
                      __attribute__((align_value(8))) bool *out) {

    if (size >= 4) {
      const half4 *h4In1 = reinterpret_cast<const half4 *>(in1);
      const half4 *h4In2 = reinterpret_cast<const half4 *>(in2);
      int *iOut = reinterpret_cast<int *>(out);

      const rptsize_t loopCount = size / 4u;
      for (unsigned i = 0; i < loopCount; ++i) {
        short4 calc =
            static_cast<short4>(FuncTy<half4>::fn(*h4In1++, *h4In2++));
        char4 result = tochar4(calc);
        int ires = copy_cast<int>(result) & 0x01010101;
        *iOut++ = ires;
      }
      in1 = reinterpret_cast<const half *>(h4In1);
      in2 = reinterpret_cast<const half *>(h4In2);
      out = reinterpret_cast<bool *>(iOut);
    }
    // process any remainder, up to 3 of
    size = size & 3;
    for (unsigned j = 0; j != size; ++j) {
      *out++ = FuncTy<half>::fn(*in1++, *in2++);
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, half, half, architecture::ipu> {
  static_assert(
      std::is_same<half, typename BinaryOpOutputType<op, half>::type>::value,
      "");
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in1,
                      const __attribute__((align_value(8))) half *in2,
                      __attribute__((align_value(8)))
                      typename BinaryOpOutputType<op, half>::type *out) {
    using arch = architecture::ipu;

    if (size >= 4) {
      const half4 *h4In1 = reinterpret_cast<const half4 *>(in1);
      const half4 *h4In2 = reinterpret_cast<const half4 *>(in2);
      half4 *h4Out = reinterpret_cast<half4 *>(out);

      // LLVM currently chooses to rotate the loop in a way that is not optimal
      // for our hardware. The inline asm blocks this. The loop is pipelined
      // sufficiently to overlap load with calculation. This was used a it seems
      // a reasonable compromise over zero overlap and unrolling far enough to
      // overlap the store with calculation.

      half4 load1 = *h4In1++;
      half4 load2 = *h4In2++;
      const rptsize_t loopCount = (size / 4u) - 1u;
      for (unsigned i = 0; i < loopCount; ++i) {
        half4 calc = BinaryOpFn<op, half4, arch>::fn(load1, load2);
        load1 = *h4In1++;
        load2 = *h4In2++;
        *h4Out++ = calc;
      }
      *h4Out++ = BinaryOpFn<op, half4, arch>::fn(load1, load2);

      in1 = reinterpret_cast<const half *>(h4In1);
      in2 = reinterpret_cast<const half *>(h4In2);
      half *tmp = reinterpret_cast<half *>(h4Out);
      size -= (tmp - out);
      out = tmp;
    }

    const half2 *h2In1 = reinterpret_cast<const half2 *>(in1);
    const half2 *h2In2 = reinterpret_cast<const half2 *>(in2);
    half2 *h2Out = reinterpret_cast<half2 *>(out);

    if (size >= 2) {
      *h2Out++ = BinaryOpFn<op, half2, arch>::fn(*h2In1++, *h2In2++);
      size -= 2;
    }

    if (size == 1) {
      write16Aligned32(BinaryOpFn<op, half, arch>::fn((*h2In1)[0], (*h2In2)[0]),
                       h2Out);
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, float, float, architecture::ipu> {
  static_assert(
      std::is_same<float, typename BinaryOpOutputType<op, float>::type>::value,
      "");

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in1,
                      const __attribute__((align_value(8))) float *in2,
                      __attribute__((align_value(8)))
                      typename BinaryOpOutputType<op, float>::type *out) {
    using arch = architecture::ipu;

    if (size >= 2) {
      const float2 *f2In1 = reinterpret_cast<const float2 *>(in1);
      const float2 *f2In2 = reinterpret_cast<const float2 *>(in2);
      float2 *f2Out = reinterpret_cast<float2 *>(out);

      // Partially unroll the loop
      // Can this be done automatically? See T59317
      float2 load1 = *f2In1++;
      float2 load2 = *f2In2++;
      const rptsize_t loopCount = (size / 2u) - 1u;
      for (unsigned i = 0; i < loopCount; ++i) {
        auto calc = BinaryOpFn<op, float2, arch>::fn(load1, load2);
        load1 = ipu::load_postinc(&f2In1, 1);
        load2 = ipu::load_postinc(&f2In2, 1);
        *f2Out++ = calc;
      }
      *f2Out++ = BinaryOpFn<op, float2, arch>::fn(load1, load2);

      in1 = reinterpret_cast<const float *>(f2In1);
      in2 = reinterpret_cast<const float *>(f2In2);
      float *tmp = reinterpret_cast<float *>(f2Out);
      size -= (tmp - out);
      out = tmp;
    }

    if (size == 1) {
      *out = BinaryOpFn<op, float, arch>::fn(*in1, *in2);
    }
  }
};

// This works only (and is instantiated) for BITWISE operators
template <BinaryOpType op>
struct BinaryOpDispatch<op, short, short, architecture::ipu> {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) short *in1,
                      const __attribute__((align_value(8))) short *in2,
                      __attribute__((align_value(8))) short *out) {
    binaryShort2_2D<op, short>(size, in1, in2, out);
  }
};

// Works only (and is instantiated) for BITWISE operators for UNSIGNED SHORT
template <BinaryOpType op>
struct BinaryOpDispatch<op, unsigned short, unsigned short, architecture::ipu> {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) unsigned short *in1,
                      const __attribute__((align_value(8))) unsigned short *in2,
                      __attribute__((align_value(8))) unsigned short *out) {
    binaryShort2_2D<op, unsigned short>(size, in1, in2, out);
  }
};

#endif

// Condition for being an external op for several vertices
template <BinaryOpType op, typename T> constexpr static bool isExternal() {
  const bool isExternalArithmeticOp =
      (std::is_same<T, float>::value || std::is_same<T, half>::value) &&
      (op == expr::BinaryOpType::ADD || op == expr::BinaryOpType::SUBTRACT ||
       op == expr::BinaryOpType::MULTIPLY);
  const bool isExternalBitwiseOp = (std::is_same<T, short>::value ||
                                    std::is_same<T, unsigned short>::value) &&
                                   (op == expr::BinaryOpType::BITWISE_AND ||
                                    op == expr::BinaryOpType::BITWISE_OR);
  return isExternalArithmeticOp || isExternalBitwiseOp;
}

template <expr::BinaryOpType op, typename T> class BinaryOp2D : public Vertex {
  typedef typename BinaryOpOutputType<op, T>::type outputType;

public:
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in1;
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in2;
  Vector<Output<Vector<outputType, SPAN, 8>>> out;
  IS_EXTERNAL_CODELET((isExternal<op, outputType>()));

  void compute() {
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;
    const unsigned limI = out.size();
    for (unsigned i = 0; i != limI; ++i) {
      BinaryOpDispatch<op, T, outputType, arch>::compute(
          out[i].size(), &in1[i][0], &in2[i][0], &out[i][0]);
    }
  }
};

template <expr::BinaryOpType op, typename T>
class BinaryOp2DInPlace : public Vertex {
  typedef typename BinaryOpOutputType<op, T>::type outputType;
  static_assert(std::is_same<T, outputType>::value,
                "In, Out types must match for in place operations");

public:
  Vector<InOut<Vector<outputType, SPAN, 8>>> in1Out;
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in2;
  IS_EXTERNAL_CODELET((isExternal<op, outputType>()));

  void compute() {
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;
    const unsigned limI = in1Out.size();
    for (unsigned i = 0; i != limI; ++i) {
      BinaryOpDispatch<op, T, outputType, arch>::compute(
          in1Out[i].size(), &in1Out[i][0], &in2[i][0], &in1Out[i][0]);
    }
  }
};

//******************************************************************************
// Dispatch for use with Binary Operation MultiVertex vertices
//******************************************************************************
template <expr::BinaryOpType op, typename inT, typename outT, typename A>
struct BinaryOpDispatchMultiVertex {
public:
  static void compute(unsigned size, unsigned worker, const inT *in1,
                      const inT *in2, outT *out) {
    // No vectorisation for int, unsigned int, but still split over workers
    // However cannot use this when writing bool
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = BinaryOpFn<op, inT, architecture::generic>::fn(in1[j], in2[j]);
  }
};

#ifdef __IPU__

/// Processing for operators that return a bool (comparison: EQUAL, LESS_THAN,
/// etc plus LOGICAL_AND/OR), for MultiVertex vertices.
template <BinaryOpType op, typename T, typename A>
class BinaryOpDispatchMultiVertex<op, T, bool, A> {
public:
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) T *in1,
                      const __attribute__((align_value(8))) T *in2,
                      __attribute__((align_value(8))) bool *out) {
    const rptsize_t loopCount = divideWork(size, 2, worker);
    binaryBoolOpBulk<op, T, CTXT_WORKERS>::compute(
        loopCount, in1 + 4 * worker, in2 + 4 * worker,
        reinterpret_cast<int *>(out) + worker);

    // To process the trailing elements (if any) use the last worker as it is
    // most likely to have less to do than the others.
    if (worker == (CTXT_WORKERS - 1)) {
      binaryBoolOpRemainder<op, T>(size, in1, in2, out);
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatchMultiVertex<op, half, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = BinaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) half *in1,
                      const __attribute__((align_value(8))) half *in2,
                      __attribute__((align_value(8))) bool *out) {

    const half4 *h4In1 = reinterpret_cast<const half4 *>(in1) + worker;
    const half4 *h4In2 = reinterpret_cast<const half4 *>(in2) + worker;
    int *iOut = reinterpret_cast<int *>(out) + worker;

    const rptsize_t loopCount = divideWork(size, 2, worker);
    for (unsigned j = 0; j < loopCount; j++) {
      short4 calc = static_cast<short4>(FuncTy<half4>::fn(*h4In1, *h4In2));
      h4In1 += CTXT_WORKERS;
      h4In2 += CTXT_WORKERS;
      char4 result = tochar4(calc);
      int ires = copy_cast<int>(result) & 0x01010101;
      *iOut = ires;
      iOut += CTXT_WORKERS;
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    // As we are writing bools it's dangerous to share this between workers
    unsigned remainder = size & 3;
    if (worker == (CTXT_WORKERS - 1) && remainder) {
      in1 = &in1[size - remainder];
      in2 = &in2[size - remainder];
      out = &out[size - remainder];
      for (unsigned j = 0; j != remainder; ++j) {
        *out++ = FuncTy<half>::fn(*in1++, *in2++);
      }
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatchMultiVertex<op, float, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = BinaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) float *in1,
                      const __attribute__((align_value(8))) float *in2,
                      __attribute__((align_value(8))) bool *out) {

    const float2 *f2In1 = reinterpret_cast<const float2 *>(in1) + 2 * worker;
    const float2 *f2In2 = reinterpret_cast<const float2 *>(in2) + 2 * worker;
    int *iOut = reinterpret_cast<int *>(out) + worker;

    const rptsize_t loopCount = divideWork(size, 2, worker);
    for (unsigned j = 0; j < loopCount; j++) {
      long2 calc_lo =
          static_cast<long2>(FuncTy<float2>::fn(*f2In1++, *f2In2++));

      long2 calc_hi = static_cast<long2>(FuncTy<float2>::fn(*f2In1, *f2In2));
      f2In1 += 2 * CTXT_WORKERS - 1;
      f2In2 += 2 * CTXT_WORKERS - 1;

      char4 result = tochar4(calc_lo, calc_hi);
      int ires = copy_cast<int>(result) & 0x01010101;
      *iOut = ires;
      iOut += CTXT_WORKERS;
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    // As we are writing bools it's dangerous to share this between workers
    unsigned remainder = size & 3;
    if (worker == (CTXT_WORKERS - 1) && remainder) {
      in1 = &in1[size - remainder];
      in2 = &in2[size - remainder];
      out = &out[size - remainder];
      for (unsigned j = 0; j != remainder; ++j) {
        *out++ = FuncTy<float>::fn(*in1++, *in2++);
      }
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatchMultiVertex<op, half, half, architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker, const half *in1,
                      const half *in2,
                      typename BinaryOpOutputType<op, half>::type *out) {

    const half4 *h4In1 = reinterpret_cast<const half4 *>(in1) + worker;
    const half4 *h4In2 = reinterpret_cast<const half4 *>(in2) + worker;
    half4 *h4Out = reinterpret_cast<half4 *>(out) + worker;

    const rptsize_t loopCount = divideWork(size, 2, worker);
    for (unsigned i = 0; i < loopCount; i++) {
      *h4Out = BinaryOpFn<op, half4, architecture::ipu>::fn(*h4In1, *h4In2);
      h4In1 += CTXT_WORKERS;
      h4In2 += CTXT_WORKERS;
      h4Out += CTXT_WORKERS;
    }
    if (size & 3) {
      const half2 *h2In1 = reinterpret_cast<const half2 *>(h4In1);
      const half2 *h2In2 = reinterpret_cast<const half2 *>(h4In2);
      half2 *h2Out = reinterpret_cast<half2 *>(h4Out);
      if (size & 2) {
        if (h4Out == (half4 *)&out[size & (~3)]) {
          *h2Out++ =
              BinaryOpFn<op, half2, architecture::ipu>::fn(*h2In1++, *h2In2++);
        }
      }
      assert(size != 0);
      if (h2Out == (half2 *)&out[size - 1]) {
        write16Aligned32(BinaryOpFn<op, half, architecture::ipu>::fn(
                             (*h2In1)[0], (*h2In2)[0]),
                         h2Out);
      }
    }
  }
};

template <expr::BinaryOpType op>
class BinaryOpDispatchMultiVertex<op, float, float, architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker, const float *in1,
                      const float *in2,
                      typename BinaryOpOutputType<op, float>::type *out) {

    const float2 *f2In1 = reinterpret_cast<const float2 *>(in1) + worker;
    const float2 *f2In2 = reinterpret_cast<const float2 *>(in2) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;

    const rptsize_t loopCount = divideWork(size, 1, worker);
    if (loopCount) {
      // Partially unroll the loop, carefully avoiding overreads (which will
      // be strided).
      // Can this be done automatically? See T59317
      auto load1 = *f2In1;
      auto load2 = *f2In2;
      f2In1 += CTXT_WORKERS;
      f2In2 += CTXT_WORKERS;

      for (unsigned j = 0; j < rptsize_t(loopCount - 1); j++) {
        auto calc = BinaryOpFn<op, float2, architecture::ipu>::fn(load1, load2);
        load1 = *f2In1;
        load2 = *f2In2;
        f2In1 += CTXT_WORKERS;
        f2In2 += CTXT_WORKERS;
        *f2Out = calc;
        f2Out += CTXT_WORKERS;
      }
      *f2Out = BinaryOpFn<op, float2, architecture::ipu>::fn(load1, load2);
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if (worker == (CTXT_WORKERS - 1) && (size & 1)) {
      out[size - 1] = BinaryOpFn<op, float, architecture::ipu>::fn(
          in1[size - 1], in2[size - 1]);
    }
  }
};

// This works only (and is instantiated) for BITWISE operators
template <BinaryOpType op>
struct BinaryOpDispatchMultiVertex<op, short, short, architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker, const short *in1,
                      const short *in2, short *out) {
    binaryShort2_MultiVertex<op, short>(size, worker, in1, in2, out);
  }
};

// Works only (and is instantiated) for BITWISE operators for UNSIGNED SHORT
template <BinaryOpType op>
struct BinaryOpDispatchMultiVertex<op, unsigned short, unsigned short,
                                   architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker, const unsigned short *in1,
                      const unsigned short *in2, unsigned short *out) {
    binaryShort2_MultiVertex<op, unsigned short>(size, worker, in1, in2, out);
  }
};

#endif

template <expr::BinaryOpType op, typename T>
class BinaryOp1D : public MultiVertex {
  typedef typename BinaryOpOutputType<op, T>::type outputType;

public:
  Input<Vector<T, ONE_PTR, 8>> in1;
  Input<Vector<T, ONE_PTR, 8>> in2;
  Output<Vector<outputType, SPAN, 8>> out;

  IS_EXTERNAL_CODELET((isExternal<op, T>()));
  void compute(unsigned wid) {
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;

    popops::BinaryOpDispatchMultiVertex<op, T, outputType, arch>::compute(
        out.size(), wid, &in1[0], &in2[0], &out[0]);
  }
};

template <expr::BinaryOpType op, typename T>
class BinaryOp1DInPlace : public MultiVertex {
  typedef typename BinaryOpOutputType<op, T>::type outputType;
  static_assert(std::is_same<T, outputType>::value,
                "In, Out types must match for in place operations");

public:
  InOut<Vector<outputType, SPAN, 8>> in1Out;
  Input<Vector<T, ONE_PTR, 8>> in2;

  IS_EXTERNAL_CODELET((isExternal<op, T>()));
  void compute(unsigned wid) {
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;

    popops::BinaryOpDispatchMultiVertex<op, T, outputType, arch>::compute(
        in1Out.size(), wid, &in1Out[0], &in2[0], &in1Out[0]);
  }
};

INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::ADD, float, half, int, unsigned,
               unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_AND, int, unsigned,
               short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_OR, int, unsigned, short,
               unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_XOR, int, unsigned,
               short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_XNOR, int, unsigned,
               short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::DIVIDE, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::EQUAL, float, half, bool, int,
               unsigned, short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::GREATER_THAN_EQUAL, float, half,
               int, unsigned, bool, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::GREATER_THAN, float, half, int,
               unsigned, bool, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LESS_THAN_EQUAL, float, half,
               int, unsigned, bool, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LESS_THAN, float, half, int,
               unsigned, bool, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MAXIMUM, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MINIMUM, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MULTIPLY, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::NOT_EQUAL, float, half, int,
               unsigned, bool, short, unsigned short, unsigned long long,
               long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::POWER, float, half, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::REMAINDER, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_LEFT, int, unsigned,
               unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_RIGHT, int, unsigned,
               unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int,
               long long)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SUBTRACT, float, half, int,
               unsigned long long, long long, unsigned)

INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::ADD, float, half, int, unsigned,
               unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_AND, int, unsigned,
               short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_OR, int, unsigned, short,
               unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_XOR, int, unsigned,
               short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_XNOR, int, unsigned,
               short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::DIVIDE, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::EQUAL, float, half, int,
               unsigned, bool, short, unsigned short, unsigned long long,
               long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::GREATER_THAN_EQUAL, float, half,
               int, unsigned, bool, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::GREATER_THAN, float, half, int,
               unsigned, bool, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::LESS_THAN_EQUAL, float, half,
               int, unsigned, bool, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::LESS_THAN, float, half, int,
               unsigned, bool, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MAXIMUM, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MINIMUM, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MULTIPLY, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::NOT_EQUAL, float, half, int,
               unsigned, bool, short, unsigned short, unsigned long long,
               long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::POWER, float, half, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::REMAINDER, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SHIFT_LEFT, int, unsigned,
               unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SHIFT_RIGHT, int, unsigned,
               unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int,
               long long)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SUBTRACT, float, half, int,
               unsigned, unsigned long long, long long)

INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::ADD, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_AND, int,
               unsigned, short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_OR, int, unsigned,
               short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_XOR, int,
               unsigned, short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_XNOR, int,
               unsigned, short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::DIVIDE, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::GREATER_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::GREATER_THAN, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LESS_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LESS_THAN, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MAXIMUM, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MINIMUM, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MULTIPLY, float, half,
               int, unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::NOT_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::POWER, float, half, int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::REMAINDER, float, half,
               int, unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_LEFT, int, unsigned,
               unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_RIGHT, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
               int, long long)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SUBTRACT, float, half,
               int, unsigned, unsigned long long, long long)

INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::ADD, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_AND, int,
               unsigned, short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_OR, int, unsigned,
               short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_XOR, int,
               unsigned, short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_XNOR, int,
               unsigned, short, unsigned short, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::DIVIDE, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MAXIMUM, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MINIMUM, float, half, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MULTIPLY, float, half,
               int, unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::POWER, float, half, int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::REMAINDER, float, half,
               int, unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SHIFT_LEFT, int, unsigned,
               unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SHIFT_RIGHT, int,
               unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
               int, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SUBTRACT, float, half,
               int, unsigned, unsigned long long, long long)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::EQUAL, bool)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::GREATER_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::GREATER_THAN, bool)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::LESS_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::NOT_EQUAL, bool)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::LESS_THAN, bool)
} // namespace popops
