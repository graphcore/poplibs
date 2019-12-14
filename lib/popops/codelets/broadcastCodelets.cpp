// Copyright (c) Graphcore Ltd, All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include "popops/ExprOp.hpp"
#include "util.hpp"

#ifdef __IPU__
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>

static __attribute__((always_inline)) unsigned getWsr(void) {
  return __builtin_ipu_get(CSR_W_WSR__INDEX) & CSR_W_WSR__CTXTID_M1__MASK;
}
// Use attributes to ensure that the maskForRepeat is inline just prior to the
// repeat instruction and therefore picked up by the compiler pass which
// optimises for repeat instructions.
// See T9902.
static __attribute__((always_inline)) unsigned maskForRepeat(unsigned input) {
  return input & CSR_W_REPEAT_COUNT__VALUE__MASK;
}
__attribute__((noinline)) unsigned divideWork(const unsigned size,
                                              const unsigned vectorWidthShifts,
                                              const unsigned worker) {
  // Integer divide by 6.
  const unsigned loopCount = ((size >> vectorWidthShifts) * 0xaaab) >> 18;
  const unsigned remainder = (size >> vectorWidthShifts) - loopCount * 6;
  return loopCount + static_cast<unsigned>(worker < remainder);
}
#endif

using namespace poplar;
using namespace popops;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;

template <expr::BroadcastOpType op, typename T> struct BroadcastOpFn {};

#define DEFINE_BROADCAST_OP_FN(op, body)                                       \
  template <typename T> struct BroadcastOpFn<op, T> {                          \
    static T fn(T x, T K) { body }                                             \
  };

DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::ADD, return x + K;)
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::SUBTRACT, return x - K;)
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::MULTIPLY, return x * K;)
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE,
                       return (1 / (x * x)) - K;)
#ifdef __IPU__
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV,
                       return ipu::rsqrt(x + K);)
#else
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV,
                       return 1 / (std::sqrt(x + K));)
#endif
//******************************************************************************
// Dispatch functions for the broadcast codelets
//******************************************************************************

template <typename T, expr::BroadcastOpType op,
          bool allowUnaligned, // Allow input/output that isn't 64-bit aligned
          bool allowRemainder>
struct BroadcastOpDispatch {
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(T) : 8;
  static void compute(unsigned size,
                      const __attribute__((align_value(minAlign))) T *in,
                      __attribute__((align_value(minAlign))) T *out,
                      const T K) {
    for (unsigned j = 0; j != size; j++) {
      out[j] = BroadcastOpFn<op, T>::fn(in[j], K);
    }
  }
};

template <typename T, expr::BroadcastOpType op,
          bool allowUnaligned, // Allow input/output that isn't 64-bit aligned.
          bool allowRemainder>
struct BroadcastOpDispatchSupervisor {
public:
  static void compute(unsigned size, unsigned worker, const T *in, T *out,
                      const T K) {
    // No vectorisation but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = BroadcastOpFn<op, T>::fn(in[j], K);
  }
};

#ifdef __IPU__

template <expr::BroadcastOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatch<half, op, allowUnaligned, allowRemainder> {
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
        half2 res = {(*h2Out)[0], BroadcastOpFn<op, half>::fn(
                                      ipu::load_postinc(&h2In, 1)[1], K)};
        ipu::store_postinc(&h2Out, res, 1);
        size -= 1;
        in = reinterpret_cast<const half *>(h2In);
        out = reinterpret_cast<half *>(h2Out);
      }
      if (size >= 2 && reinterpret_cast<std::uintptr_t>(in) & 7) {
        half2 K2 = {K, K};
        const half2 *h2In = reinterpret_cast<const half2 *>(in);
        half2 *h2Out = reinterpret_cast<half2 *>(out);
        ipu::store_postinc(
            &h2Out,
            BroadcastOpFn<op, half2>::fn(ipu::load_postinc(&h2In, 1), K2), 1);
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
        half4 calc = BroadcastOpFn<op, half4>::fn(load, K4);
        load = ipu::load_postinc(&h4In, 1);
        ipu::store_postinc(&h4Out, calc, 1);
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
      ipu::store_postinc(&h4Out, BroadcastOpFn<op, half4>::fn(load, K4), 1);
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
        ipu::store_postinc(
            &h2Out,
            BroadcastOpFn<op, half2>::fn(ipu::load_postinc(&h2In, 1), K2), 1);
        size -= 2;
      }

      if (size == 1) {
        half2 res = (half2){
            BroadcastOpFn<op, half>::fn((*h2In)[0], K),
            (*h2Out)[1],
        };
        *h2Out = res;
      }
    }
  }
};

template <expr::BroadcastOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatch<float, op, allowUnaligned, allowRemainder> {
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
        ipu::store_postinc(
            &out, BroadcastOpFn<op, float>::fn(ipu::load_postinc(&in, 1), K),
            1);
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
        float2 calc = BroadcastOpFn<op, float2>::fn(load, K2);
        load = ipu::load_postinc(&f2In, 1);
        ipu::store_postinc(&f2Out, calc, 1);
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
      ipu::store_postinc(&f2Out, BroadcastOpFn<op, float2>::fn(load, K2), 1);
      if (allowRemainder) {
        in = reinterpret_cast<const float *>(f2In);
        out = reinterpret_cast<float *>(f2Out);
      }
    }
    if (allowRemainder) {
      if (size & 1) {
        float load = ipu::load_postinc(&in, 1);
        *out = BroadcastOpFn<op, float>::fn(load, K);
      }
    }
  }
};

template <expr::BroadcastOpType op, bool allowUnaligned, bool allowRemainder>
class BroadcastOpDispatchSupervisor<float, op, allowUnaligned, allowRemainder> {
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
          ipu::store_postinc(&out, BroadcastOpFn<op, float>::fn(val, K), 1);
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
      float2 calc = BroadcastOpFn<op, float2>::fn(load, K2);
      ipu::store_postinc(&f2Out, calc, CTXT_WORKERS);
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if (allowRemainder) {
      if (worker == (CTXT_WORKERS - 1) && (size & 1)) {
        out[size - 1] = BroadcastOpFn<op, float>::fn(in[size - 1], K);
      }
    }
  }
};

template <expr::BroadcastOpType op, bool allowUnaligned, bool allowRemainder>
struct BroadcastOpDispatchSupervisor<half, op, allowUnaligned, allowRemainder> {
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
          half2 res = {(*h2Out)[0], BroadcastOpFn<op, half>::fn(
                                        ipu::load_postinc(&h2In, 1)[1], K)};
          ipu::store_postinc(&h2Out, res, 1);
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
          ipu::store_postinc(
              &h2Out,
              BroadcastOpFn<op, half2>::fn(ipu::load_postinc(&h2In, 1), K2), 1);
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
      half4 calc = BroadcastOpFn<op, half4>::fn(load, K4);
      ipu::store_postinc(&h4Out, calc, CTXT_WORKERS);
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
          ipu::store_postinc(
              &h2Out,
              BroadcastOpFn<op, half2>::fn(ipu::load_postinc(&h2In, 1), K2), 1);
        }
        if (size & 1) {
          h2Out = reinterpret_cast<half2 *>(&out[size - 1]);
          half2 res = {
              BroadcastOpFn<op, half>::fn((*h2In)[0], K),
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

#define INSTANTIATE_TYPES(name, opType)                                        \
  template class name<opType, float>;                                          \
  template class name<opType, half>

#define INSTANTIATE_SCALAR_BASIC(name)                                         \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::ADD);                         \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::SUBTRACT);                    \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::MULTIPLY)

#define INSTANTIATE_SCALAR(name)                                               \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV);     \
  INSTANTIATE_SCALAR_BASIC(name)

#define INSTANTIATE_VECTOR_OUTER_TYPES(name, opType)                           \
  template class name<opType, float, true>;                                    \
  template class name<opType, half, true>;                                     \
  template class name<opType, float, false>;                                   \
  template class name<opType, half, false>

#define INSTANTIATE_VECTOR_OUTER(name)                                         \
  INSTANTIATE_VECTOR_OUTER_TYPES(name, expr::BroadcastOpType::ADD);            \
  INSTANTIATE_VECTOR_OUTER_TYPES(name, expr::BroadcastOpType::SUBTRACT);       \
  INSTANTIATE_VECTOR_OUTER_TYPES(name, expr::BroadcastOpType::MULTIPLY)

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

#define DEF_BROADCAST_2D_DATA_VERTEX(vertexName, inOutType, outDef, outName)   \
  template <expr::BroadcastOpType op, typename dType>                          \
  class vertexName : public Vertex {                                           \
  public:                                                                      \
    constexpr static bool isExternal() {                                       \
      return (std::is_same<dType, float>::value ||                             \
              std::is_same<dType, half>::value) &&                             \
             (op == expr::BroadcastOpType::ADD ||                              \
              op == expr::BroadcastOpType::SUBTRACT ||                         \
              op == expr::BroadcastOpType::MULTIPLY);                          \
    }                                                                          \
    Vector<inOutType<Vector<dType, SPAN, 8>>> data;                            \
    outDef Input<dType> B;                                                     \
    IS_EXTERNAL_CODELET(isExternal());                                         \
    bool compute() {                                                           \
      unsigned limI = data.size();                                             \
      for (unsigned i = 0; i < limI; i++) {                                    \
        BroadcastOpDispatch<dType, op, false, true>::compute(                  \
            data[i].size(), &data[i][0], &outName[i][0], *B);                  \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_2D_DATA_VERTEX(BroadcastScalar2DData, Input, OUT_2D_DEF, out)
DEF_BROADCAST_2D_DATA_VERTEX(BroadcastScalar2DDataInPlace, InOut, , data)

// Ensure that internal arithmetic is always done in full precision for
// INV_STD_DEV_TO_VARIANCE
#define DEF_BROADCAST_2D_DATA_VERTEX_FP(vertexName, inOutType, outDef,         \
                                        outName)                               \
  template <>                                                                  \
  class vertexName<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, half>       \
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
              BroadcastOpFn<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE,    \
                            float>::fn(static_cast<float>(data[i][j]),         \
                                       static_cast<float>(*B)));               \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_2D_DATA_VERTEX_FP(BroadcastScalar2DData, Input, OUT_2D_DEF_HALF,
                                out)
DEF_BROADCAST_2D_DATA_VERTEX_FP(BroadcastScalar2DDataInPlace, InOut, , data)

template class BroadcastScalar2DDataInPlace<
    expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, float>;
template class BroadcastScalar2DData<
    expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, float>;

INSTANTIATE_SCALAR(BroadcastScalar2DData);
INSTANTIATE_SCALAR(BroadcastScalar2DDataInPlace);

#define DEF_BROADCAST_2D_VERTEX(vertexName, inOutType, outDef, outName)        \
  template <expr::BroadcastOpType op, typename dType>                          \
  class vertexName : public Vertex {                                           \
  public:                                                                      \
    Vector<inOutType<Vector<dType, SPAN, 8>>> data;                            \
    outDef Vector<Input<dType>, ONE_PTR> B;                                    \
    bool compute() {                                                           \
      unsigned limI = data.size();                                             \
      for (unsigned i = 0; i < limI; i++) {                                    \
        BroadcastOpDispatch<dType, op, false, true>::compute(                  \
            data[i].size(), &data[i][0], &outName[i][0], B[i]);                \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_2D_VERTEX(BroadcastScalar2D, Input, OUT_2D_DEF, out)
DEF_BROADCAST_2D_VERTEX(BroadcastScalar2DInPlace, InOut, , data)

INSTANTIATE_SCALAR_BASIC(BroadcastScalar2D);
INSTANTIATE_SCALAR_BASIC(BroadcastScalar2DInPlace);

#define DEF_BROADCAST_VECT_OUTER_BY_COLUMN_VERTEX(vertexName, inOutType,       \
                                                  outDef, outName, isInPlace)  \
  template <expr::BroadcastOpType op, typename dType, bool allowMisaligned>    \
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
        BroadcastOpDispatch<dType, op, allowMisaligned,                        \
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
  template <expr::BroadcastOpType op, typename dType, bool allowMisaligned>    \
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
        BroadcastOpDispatch<dType, op, allowMisaligned,                        \
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

#define DEF_BROADCAST_1D_VERTEX(vertexName, inOutType, outDef, outName)        \
  template <expr::BroadcastOpType op, typename dType>                          \
  class vertexName : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {         \
  public:                                                                      \
    inOutType<Vector<dType, SPAN, 8>> data;                                    \
    outDef Input<dType> B;                                                     \
    IS_EXTERNAL_CODELET(true);                                                 \
    bool compute() {                                                           \
      BroadcastOpDispatch<dType, op, false, true>::compute(                    \
          data.size(), &data[0], &outName[0], *B);                             \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_1D_VERTEX(BroadcastScalar1DSupervisor, Input, OUT_1D_DEF, out)
DEF_BROADCAST_1D_VERTEX(BroadcastScalar1DInPlaceSupervisor, InOut, , data)

// Ensure that internal arithmetic is always done in full precision for
// INV_STD_DEV_TO_VARIANCE
#define DEF_BROADCAST_1D_VERTEX_FP(vertexName, inOutType, outDef, outName)     \
  template <>                                                                  \
  class vertexName<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, half>       \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    inOutType<Vector<half, SPAN, 8>> data;                                     \
    outDef Input<half> B;                                                      \
    IS_EXTERNAL_CODELET(true);                                                 \
    bool compute() {                                                           \
      unsigned limI = data.size();                                             \
      for (unsigned i = 0; i < limI; i++) {                                    \
        outName[i] = static_cast<half>(                                        \
            BroadcastOpFn<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE,      \
                          float>::fn(static_cast<float>(data[i]),              \
                                     static_cast<float>(*B)));                 \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_1D_VERTEX_FP(BroadcastScalar1DSupervisor, Input, OUT_1D_DEF_HALF,
                           out)
DEF_BROADCAST_1D_VERTEX_FP(BroadcastScalar1DInPlaceSupervisor, InOut, , data)

template class BroadcastScalar1DInPlaceSupervisor<
    expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, float>;
template class BroadcastScalar1DSupervisor<
    expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, float>;

INSTANTIATE_SCALAR(BroadcastScalar1DSupervisor);
INSTANTIATE_SCALAR(BroadcastScalar1DInPlaceSupervisor);

#ifdef __IPU__

// Create worker vertex code, which will do the actual work, when the
// supervisor vertices compile to an external codelet.  Called via an
// assembly stub.

#define DEF_BROADCAST_1D_WK_VERTEX(vertexName, inOutType, outDef, outName)     \
  template <expr::BroadcastOpType op, typename dType>                          \
  class vertexName : public Vertex {                                           \
  public:                                                                      \
    constexpr static bool isExternal() {                                       \
      return (std::is_same<dType, float>::value ||                             \
              std::is_same<dType, half>::value) &&                             \
             (op == expr::BroadcastOpType::ADD ||                              \
              op == expr::BroadcastOpType::SUBTRACT ||                         \
              op == expr::BroadcastOpType::MULTIPLY);                          \
    }                                                                          \
    inOutType<Vector<dType, SPAN, 8>> data;                                    \
    outDef Input<dType> B;                                                     \
    IS_EXTERNAL_CODELET(isExternal());                                         \
    bool compute() {                                                           \
      BroadcastOpDispatchSupervisor<dType, op, false, true>::compute(          \
          data.size(), getWsr(), &data[0], &outName[0], *B);                   \
      return true;                                                             \
    }                                                                          \
  };

DEF_BROADCAST_1D_WK_VERTEX(BroadcastScalar1D, Input, OUT_1D_DEF, out)
DEF_BROADCAST_1D_WK_VERTEX(BroadcastScalar1DInPlace, InOut, , data)

template class BroadcastScalar1D<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE,
                                 float>;
template class BroadcastScalar1DInPlace<
    expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, float>;

INSTANTIATE_SCALAR(BroadcastScalar1D);
INSTANTIATE_SCALAR(BroadcastScalar1DInPlace);

#define DEF_BROADCAST_VECT_OUTER_BY_COLUMN_WK_VERTEX(                          \
    vertexName, inOutType, outDef, outName, isInPlace)                         \
  template <expr::BroadcastOpType op, typename dType, bool allowMisaligned>    \
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
            dType, op, allowMisaligned,                                        \
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
  template <expr::BroadcastOpType op, typename dType, bool allowMisaligned>    \
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
        BroadcastOpDispatch<dType, op, allowMisaligned,                        \
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
template <expr::BroadcastOpType op, typename inType, typename outType>
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
        out[i][j] = static_cast<outType>(BroadcastOpFn<op, float>::fn(
            static_cast<float>(data[i][j]), static_cast<float>(*B)));
      }
    }
    return true;
  }
};

template class BroadcastScalar2Types2DData<
    expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, half, float>;
template class BroadcastScalar2Types2DData<
    expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV, float, half>;

template <expr::BroadcastOpType op, typename inType, typename outType>
class BroadcastScalar2Types1DSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  Input<Vector<inType, SPAN, 8>> data;
  Output<Vector<outType, ONE_PTR, 8>> out;
  Input<inType> B;
  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < data.size(); i++) {
      out[i] = static_cast<outType>(BroadcastOpFn<op, float>::fn(
          static_cast<float>(data[i]), static_cast<float>(*B)));
    }
    return true;
  }
};
template class BroadcastScalar2Types1DSupervisor<
    expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, half, float>;
template class BroadcastScalar2Types1DSupervisor<
    expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV, float, half>;
} // namespace popops
