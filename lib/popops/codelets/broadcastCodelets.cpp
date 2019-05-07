#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

#include <cassert>
#include <cmath>


#include "util.hpp"
#include "popops/ExprOp.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

#ifdef __IPU__
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>

inline unsigned getWsr(void) {
  return __builtin_ipu_get(CSR_W_WSR__INDEX) & CSR_W_WSR__CTXTID_M1__MASK;
}

#endif

using namespace poplar;
using namespace popops;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;

template <expr::BroadcastOpType op, typename T>
struct BroadcastOpFn {};


#define DEFINE_BROADCAST_OP_FN(op, body)                                       \
  template<typename T> struct BroadcastOpFn<op, T> {                           \
  static T fn(T x, T K) { body } };

DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::ADD, return x + K;)
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::SUBTRACT, return x - K;)
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::MULTIPLY, return x * K;)
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE,
                                return (1/(x * x)) - K;)
#ifdef __IPU__
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV,
                                            return ipu::rsqrt(x + K);)
#else
DEFINE_BROADCAST_OP_FN(expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV,
                                            return 1/(std::sqrt(x + K));)
#endif
//******************************************************************************
// Dispatch functions for the broadcast codelets
//******************************************************************************

template <typename T,
          expr::BroadcastOpType op,
          bool allowUnaligned> // Allow input/output that isn't 64-bit aligned
struct BroadcastOpDispatch {
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(T) : 8;
  static void compute(unsigned size,
                const __attribute__((align_value(minAlign))) T *in,
                __attribute__((align_value(minAlign))) T *out,
                const T K) {
    for(unsigned j = 0; j != size; j++) {
        out[j] = BroadcastOpFn<op, T>::fn(in[j], K);
      }
  }
};

template <typename T,
          expr::BroadcastOpType op,
          bool allowUnaligned> // Allow input/output that isn't 64-bit aligned.
struct BroadcastOpDispatchSupervisor {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      const T *in,
                      T *out,
                      const T K) {
    // No vectorisation but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = BroadcastOpFn<op, T>::fn(in[j], K);
  }
};


#ifdef __IPU__

template <expr::BroadcastOpType op, bool allowUnaligned>
struct BroadcastOpDispatch<half, op, allowUnaligned> {
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
          reinterpret_cast<std::uintptr_t>(in) & ~std::uintptr_t(3)
        );
        half2 *h2Out = reinterpret_cast<half2 *>(
          reinterpret_cast<std::uintptr_t>(out) & ~std::uintptr_t(3)
        );
        half2 res = {
          (*h2Out)[0],
          BroadcastOpFn<op, half>::fn(
            ipu::load_postinc(&h2In, 1)[1], K
          )
        };
        ipu::store_postinc(&h2Out, res, 1);
        size -= 1;
        in = reinterpret_cast<const half *>(h2In);
        out = reinterpret_cast<half *>(h2Out);
      }
      if (size >= 2 && reinterpret_cast<std::uintptr_t>(in) & 7) {
        half2 K2 = {K,K};
        const half2 *h2In = reinterpret_cast<const half2 *>(in);
        half2 *h2Out = reinterpret_cast<half2 *>(out);
        ipu::store_postinc(&h2Out,
          BroadcastOpFn<op, half2>::fn(ipu::load_postinc(&h2In, 1), K2), 1);
        size -= 2;
        in = reinterpret_cast<const half *>(h2In);
        out = reinterpret_cast<half *>(h2Out);
      }
    }

    if (size >= 4) {
      half4 K4 = {K,K,K,K};
      const half4 *h4In = reinterpret_cast<const half4 *>(in);
      half4 *h4Out = reinterpret_cast<half4 *>(out);

      half4 load = ipu::load_postinc(&h4In, 1);
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < (size / 4u) - 1u; i++) {
        half4 calc = BroadcastOpFn<op, half4>::fn(load, K4);
        load = ipu::load_postinc(&h4In, 1);
        ipu::store_postinc(&h4Out, calc, 1);
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
      ipu::store_postinc(&h4Out, BroadcastOpFn<op, half4>::fn(load, K4), 1);

      in = reinterpret_cast<const half *>(h4In);
      half *tmp = reinterpret_cast<half *>(h4Out);
      size -= (tmp - out);
      out = tmp;
    }

    const half2 *h2In = reinterpret_cast<const half2 *>(in);
    half2 *h2Out = reinterpret_cast<half2 *>(out);

    if (size >= 2) {
      half2 K2 = {K,K};
      ipu::store_postinc(&h2Out,
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
};

template <expr::BroadcastOpType op, bool allowUnaligned>
struct BroadcastOpDispatch<float, op, allowUnaligned> {
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
        ipu::store_postinc(&out,
            BroadcastOpFn<op, float>::fn(ipu::load_postinc(&in, 1), K), 1);
        size -= 1;
      }
    }
    if (size >= 2) {
      float2 K2 = {K,K};
      const float2 *f2In = reinterpret_cast<const float2 *>(in);
      float2 *f2Out = reinterpret_cast<float2 *>(out);
      float2 load = ipu::load_postinc(&f2In, 1);
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < (size / 2u) - 1u; i++) {
        float2 calc = BroadcastOpFn<op, float2>::fn(load, K2);
        load = ipu::load_postinc(&f2In, 1);
        ipu::store_postinc(&f2Out, calc, 1);
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
      ipu::store_postinc(&f2Out, BroadcastOpFn<op, float2>::fn(load, K2), 1);
      in = reinterpret_cast<const float *>(f2In);
      out = reinterpret_cast<float *>(f2Out);
    }
    if (size & 1) {
      float load = ipu::load_postinc(&in, 1);
      *out = BroadcastOpFn<op, float>::fn(load, K);
    }
  }
};

template <expr::BroadcastOpType op, bool allowUnaligned>
class BroadcastOpDispatchSupervisor<float, op, allowUnaligned> {
public:
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(float) : 8;
  // Assumes in and out both point to the same memory (or at least have
  // same alignment at function start) and that there is at least one element
  // to process.
  static void compute(unsigned size,
                      unsigned worker,
                      const __attribute__((align_value(minAlign))) float *in,
                      __attribute__((align_value(minAlign))) float *out,
                      const float K) {

    if (allowUnaligned) {
      // TODO: Using the same worker as for trailing elements to handle
      // leading elements, is this really wise for performance?
      if (reinterpret_cast<std::uintptr_t>(in) & 0x7) {
        if (worker == 0) {
          auto val = ipu::load_postinc(&in, 1);
          ipu::store_postinc(&out, BroadcastOpFn<op, float>::fn(val, K), 1);
        } else {
          ++in;
          ++out; }
        size -= 1;
      }
    }
    const float2 *f2In = reinterpret_cast<const float2 *>(in) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;
    float2 K2 = {K,K};

    for (unsigned j = worker; j < (size>>1) ; j += CTXT_WORKERS) {
      float2 load = ipu::load_postinc(&f2In, CTXT_WORKERS);
      float2 calc = BroadcastOpFn<op, float2>::fn(load, K2);
      ipu::store_postinc(&f2Out, calc, CTXT_WORKERS);
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if (worker == (CTXT_WORKERS - 1) && (size & 1)) {
      out[size-1] = BroadcastOpFn<op,float>::fn(in[size-1], K);
    }
  }
};

template <expr::BroadcastOpType op, bool allowUnaligned>
struct BroadcastOpDispatchSupervisor<half, op, allowUnaligned> {
public:
  static constexpr std::size_t minAlign = allowUnaligned ? alignof(half) : 8;
  // Assumes in and out both point to the same memory (or at least have
  // same alignment at function start) and that there is at least one element
  // to process.
  static void compute(unsigned size,
                      unsigned worker,
                      const __attribute__((align_value(minAlign))) half *in,
                      __attribute__((align_value(minAlign))) half *out,
                      const half K) {

    if (allowUnaligned) {
      if (reinterpret_cast<std::uintptr_t>(in) & 0x3) {
        if (worker == 0) {
          const half2 *h2In = reinterpret_cast<const half2 *>(
            reinterpret_cast<std::uintptr_t>(in) & ~std::uintptr_t(0x3)
          );
          half2 *h2Out = reinterpret_cast<half2 *>(
            reinterpret_cast<std::uintptr_t>(out) & ~std::uintptr_t(0x3)
          );
          half2 res = {
            (*h2Out)[0],
            BroadcastOpFn<op, half>::fn(
              ipu::load_postinc(&h2In, 1)[1], K
            )
          };
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
        if (worker == 0) {
          half2 K2 = {K,K};
          const half2 *h2In = reinterpret_cast<const half2 *>(in);
          half2 *h2Out = reinterpret_cast<half2 *>(out);
          ipu::store_postinc(&h2Out,
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
    half4 K4 = {K,K,K,K};

    asm volatile ("# Thwart loop rotation (start)" ::: "memory");
    for (unsigned i = worker; i < size>>2; i += CTXT_WORKERS) {
      half4 load = ipu::load_postinc(&h4In, CTXT_WORKERS);
      half4 calc = BroadcastOpFn<op, half4>::fn(load, K4);
      ipu::store_postinc(&h4Out, calc, CTXT_WORKERS);
    }
    asm volatile ("# Thwart loop rotation (end)" ::: "memory");
    // Handle the remaining elements with the worker with the correct
    // pointer.
    const half2 *h2In = reinterpret_cast<const half2*>(h4In);
    half2 *h2Out = reinterpret_cast<half2 *>(h4Out);
    if (size & 2 &&
        h2Out == reinterpret_cast<half2 *>(&out[size & ~unsigned(3)])) {
      half2 K2 = {K,K};
      ipu::store_postinc(&h2Out,
        BroadcastOpFn<op, half2>::fn(ipu::load_postinc(&h2In, 1), K2), 1);
    }
    if (size & 1 &&
        h2Out == reinterpret_cast<half2 *>(&out[size-1])) {
      half2 res = {
        BroadcastOpFn<op, half>::fn((*h2In)[0], K), (*h2Out)[1],
      };
      *h2Out = res;
    }
  }
};

#endif

namespace popops {

#define INSTANTIATE_TYPES(name, opType) \
  template class name<opType, float>; \
  template class name<opType, half>

#define INSTANTIATE(name) \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::ADD); \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::SUBTRACT); \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::MULTIPLY)

#define INSTANTIATE_SCALAR(name) \
  INSTANTIATE(name); \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE); \
  INSTANTIATE_TYPES(name, expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV);

// The not-in-place broadcast vertices have an 'out' member for the output of
// the operation.
// The in-place broadcast vertices instead use the 'data' member for input and
// output.
// We will add the 'out' member, or not, passing this macro, or an empty
// argument, to each DEF_BROADCAST_xxx_VERTEX() macros.
#define OUT_1D_DEF Output<Vector<dType, ONE_PTR, 8>> out;
#define OUT_2D_DEF Vector<Output<Vector<dType, ONE_PTR, 8>>, ONE_PTR> out;


#define DEF_BROADCAST_2D_DATA_VERTEX(vertexName, inOutType, outDef, outName)\
template <expr::BroadcastOpType op, typename dType>\
class vertexName : public Vertex {\
public:\
  Vector<inOutType<Vector<dType, SPAN, 8>>> data;\
  outDef\
  Input<dType> B;\
  bool compute() {\
    unsigned limI = data.size();\
    for (unsigned i = 0; i < limI; i++) {\
      BroadcastOpDispatch<dType, op, false>::compute(\
        data[i].size(),\
        &data[i][0],\
        &outName[i][0],\
        *B);\
    }\
    return true;\
  }\
};\
INSTANTIATE_SCALAR(vertexName);

DEF_BROADCAST_2D_DATA_VERTEX(BroadcastScalar2DData, Input, OUT_2D_DEF, out)
DEF_BROADCAST_2D_DATA_VERTEX(BroadcastScalar2DDataInPlace, InOut, , data)


#define DEF_BROADCAST_2D_VERTEX(vertexName, inOutType, outDef, outName)\
template <expr::BroadcastOpType op, typename dType>\
class vertexName : public Vertex {\
public:\
  Vector<inOutType<Vector<dType, SPAN, 8>>> data;\
  outDef\
  Vector<Input<dType>, ONE_PTR> B;\
  bool compute() {\
    unsigned limI = data.size();\
    for (unsigned i = 0; i < limI; i++) {\
      BroadcastOpDispatch<dType, op, false>::compute(\
        data[i].size(),\
        &data[i][0],\
        &outName[i][0],\
        B[i]);\
    }\
    return true;\
  }\
};\
INSTANTIATE_SCALAR(vertexName);

DEF_BROADCAST_2D_VERTEX(BroadcastScalar2D, Input, OUT_2D_DEF, out)
DEF_BROADCAST_2D_VERTEX(BroadcastScalar2DInPlace, InOut, , data)


#define DEF_BROADCAST_VECT_OUTER_VERTEX(vertexName, inOutType, outDef,\
                                        outName, isInPlace)\
template <expr::BroadcastOpType op, typename dType>\
class vertexName : public SupervisorVertex {\
  static constexpr std::size_t inputAlign = isInPlace ? alignof(dType) : 8;\
public:\
  inOutType<Vector<dType, ONE_PTR, inputAlign>> data;\
  outDef\
  Input<Vector<dType, SPAN>> B;\
  short columns;\
  short rows;\
  IS_EXTERNAL_CODELET(true);\
  bool compute() {\
    std::size_t bIndex = 0;\
    auto bLen = B.size();\
    for (unsigned i = 0; i < rows; i++) {\
      BroadcastOpDispatch<dType, op, true>::compute(\
        columns,\
        &data[i * columns],\
        &outName[i * columns],\
        B[bIndex]);\
      ++bIndex;\
      if (bIndex == bLen) {\
        bIndex = 0;\
      }\
    }\
    return true;\
  }\
};\
INSTANTIATE(vertexName);

DEF_BROADCAST_VECT_OUTER_VERTEX(BroadcastVectorOuterSupervisor, Input,
                                OUT_1D_DEF, out, false)
DEF_BROADCAST_VECT_OUTER_VERTEX(BroadcastVectorOuterInPlaceSupervisor,
                                InOut, , data, true)


#define DEF_BROADCAST_1D_VERTEX(vertexName, inOutType, outDef, outName)\
template <expr::BroadcastOpType op, typename dType>\
class vertexName : public SupervisorVertex {\
public:\
  inOutType<Vector<dType, SPAN, 8>> data;\
  outDef\
  Input<dType> B;\
  IS_EXTERNAL_CODELET(true);\
  bool compute() {\
    BroadcastOpDispatch<dType, op, false>::compute(\
      data.size(),\
      &data[0],\
      &outName[0],\
      *B);\
    return true;\
  }\
};\
INSTANTIATE_SCALAR(vertexName);

DEF_BROADCAST_1D_VERTEX(BroadcastScalar1DSupervisor, Input, OUT_1D_DEF, out)
DEF_BROADCAST_1D_VERTEX(BroadcastScalar1DInPlaceSupervisor, InOut, , data)


#ifdef __IPU__

// Create worker vertex code, which will do the actual work, when the
// supervisor vertices compile to an external codelet.  Called via an
// assembly stub.

#define DEF_BROADCAST_1D_WK_VERTEX(vertexName, inOutType, outDef, outName)\
template <expr::BroadcastOpType op, typename dType>\
class vertexName : public Vertex {\
public:\
  inOutType<Vector<dType, SPAN, 8>> data;\
  outDef\
  Input<dType> B;\
  bool compute() {\
    BroadcastOpDispatchSupervisor<dType, op, false>::compute(\
      data.size(),\
      getWsr(),\
      &data[0],\
      &outName[0],\
      *B);\
    return true;\
  }\
};\
INSTANTIATE_SCALAR(vertexName);

DEF_BROADCAST_1D_WK_VERTEX(BroadcastScalar1D, Input, OUT_1D_DEF, out)
DEF_BROADCAST_1D_WK_VERTEX(BroadcastScalar1DInPlace, InOut, , data)


#define DEF_BROADCAST_VECT_OUTER_WK_VERTEX(vertexName, inOutType, outDef,\
                                           outName, isInPlace)\
template <expr::BroadcastOpType op, typename dType>\
class vertexName : public Vertex {\
  static constexpr std::size_t inputAlign = isInPlace ? alignof(dType) : 8;\
public:\
  inOutType<Vector<dType, ONE_PTR, inputAlign>> data;\
  outDef\
  Input<Vector<dType, SPAN>> B;\
  unsigned short columns;\
  unsigned short rows;\
  bool compute() {\
    std::size_t bIndex = 0;\
    auto bLen = B.size();\
    for (unsigned i = 0; i < rows; i++) {\
      BroadcastOpDispatchSupervisor<dType, op, true>::compute(\
        columns,\
        getWsr(),\
        &data[i * columns],\
        &outName[i * columns],\
        B[bIndex]);\
      ++bIndex;\
      if (bIndex == bLen) {\
        bIndex = 0;\
      }\
    }\
    return true;\
  }\
};\
INSTANTIATE(vertexName);

DEF_BROADCAST_VECT_OUTER_WK_VERTEX(BroadcastVectorOuter, Input, OUT_1D_DEF,
                                   out, false)
DEF_BROADCAST_VECT_OUTER_WK_VERTEX(BroadcastVectorOuterInPlace, InOut, ,
                                   data, true)

#endif // __IPU__

} // namespace popops
