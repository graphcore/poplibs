#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

#include <cassert>
#include <cmath>

#define __IPU_ARCH_VERSION__ 0
#include "util.hpp"
#include "popops/ExprOp.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <tileimplconsts.h>

#ifdef __IPU__
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#include <tilearch.h>

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

template <expr::BroadcastOpType op, typename T>
struct BroadcastOpDispatch {
  static void compute(unsigned size,
                __attribute__((align_value(8))) T *in,
                __attribute__((align_value(8))) T *out,
                const T K) {
    for(unsigned j = 0; j != size; j++) {
        out[j] = BroadcastOpFn<op, T>::fn(in[j], K);
      }
  }
};

template <expr::BroadcastOpType op, typename T>
struct BroadcastOpDispatchSupervisor {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      T *in,
                      T *out,
                      const T K) {
    // No vectorisation but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = BroadcastOpFn<op, T>::fn(in[j], K);
  }
};


#ifdef __IPU__

template <expr::BroadcastOpType op>
struct BroadcastOpDispatch<op, half> {

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) half *out,
                      const half K) {
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
      ipu::store_postinc(
          &h2Out, BroadcastOpFn<op, half2>::fn(ipu::load_postinc(&h2In, 1), K2),
          1);
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

template <expr::BroadcastOpType op>
struct BroadcastOpDispatch<op, float> {

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) float *out,
                      const float K) {
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
    if(size & 1) {
      float load = ipu::load_postinc(&in, 1);
      *out = BroadcastOpFn<op, float>::fn(load, K);

    }

  }
};

template <expr::BroadcastOpType op>
class BroadcastOpDispatchSupervisor<op, float> {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) float *out,
                      const float K) {

    const float2 *f2In = reinterpret_cast<const float2 *>(in) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;
    float2 K2 = {K,K};

    for(unsigned j = worker; j < (size>>1) ; j += CTXT_WORKERS) {
      float2 load = ipu::load_postinc(&f2In, CTXT_WORKERS);
      float2 calc = BroadcastOpFn<op, float2>::fn(load, K2);
      ipu::store_postinc(&f2Out, calc, CTXT_WORKERS);
     }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if(worker == (CTXT_WORKERS - 1)  && (size & 1)) {
      out[size-1] = BroadcastOpFn<op,float>::fn(in[size-1], K);
    }
  }
};

template <expr::BroadcastOpType op>
struct BroadcastOpDispatchSupervisor<op, half> {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) half *out,
                      const half K) {

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
    if(size & 3) {
      half2 K2 = {K,K};

      const half2 *h2In = reinterpret_cast<const half2*>(h4In);
      half2 *h2Out = reinterpret_cast<half2 *>(h4Out);
      if(size & 2) {
        if(h4Out == (half4*)&out[size & (~3)]) {
          ipu::store_postinc(&h2Out,
                              BroadcastOpFn<op, half2>::fn(
                              ipu::load_postinc(&h2In, 1), K2),
                              1);
        }
      }
      assert(size != 0);
      if(h2Out == (half2*)&out[size-1]) {
        half2 res = (half2)
        {
          BroadcastOpFn<op, half>::fn((*h2In)[0], K), (*h2Out)[1],
        };
        *h2Out = res;
      }
    }
  }
};

#endif

//******************************************************************************
// Codelets
//******************************************************************************

namespace popops {

template <expr::BroadcastOpType op, typename inOutType>
class BroadcastOp2DInPlace : public Vertex {
public:
  Vector<InOut<Vector<inOutType, SPAN, 8>>> data;
  Input<inOutType> B;

  bool compute() {
    unsigned limI = data.size();
    for (unsigned i = 0; i < limI; i++) {
      BroadcastOpDispatch<op, inOutType>::compute(
                      data[i].size(), &data[i][0], &data[i][0], *B);
    }
    return true;
  }
};

template class BroadcastOp2DInPlace<expr::BroadcastOpType::ADD, float>;
template class BroadcastOp2DInPlace<expr::BroadcastOpType::ADD, half>;

template class BroadcastOp2DInPlace<expr::BroadcastOpType::SUBTRACT, float>;
template class BroadcastOp2DInPlace<expr::BroadcastOpType::SUBTRACT, half>;

template class BroadcastOp2DInPlace<expr::BroadcastOpType::MULTIPLY, float>;
template class BroadcastOp2DInPlace<expr::BroadcastOpType::MULTIPLY, half>;

template class
BroadcastOp2DInPlace<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, float>;
template class
BroadcastOp2DInPlace<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, half>;

template class
BroadcastOp2DInPlace<expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV, float>;
template class
BroadcastOp2DInPlace<expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV, half>;

template <expr::BroadcastOpType op, typename inOutType>
class BroadcastOp1DInPlaceSupervisor : public SupervisorVertex {
public:
  InOut<Vector<inOutType, SPAN, 8>> data;
  Input<inOutType> B;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    BroadcastOpDispatch<op, inOutType>::compute(
                      data.size(), &data[0], &data[0], *B);
    return true;
  }
};
template class BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::ADD,
                                                                      float>;
template class BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::ADD, half>;

template class BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::SUBTRACT,
                                                                        float>;
template class BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::SUBTRACT,
                                                                        half>;

template class BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::MULTIPLY,
                                                                        float>;
template class BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::MULTIPLY,
                                                                        half>;

template class
BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE,
                                                                        float>;
template class
BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE,
                                                                        half>;

template class
BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV,
                                                                        float>;
template class
BroadcastOp1DInPlaceSupervisor<expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV,
                                                                        half>;

#ifdef __IPU__

// Create worker vertex code, which will do the actual work, when the
// supervisor vertices copile to an external codelet.  Called via an
// assembly stub.

template <expr::BroadcastOpType op, typename inOutType>
class BroadcastOp1DInPlace : public Vertex {
public:
  InOut<Vector<inOutType, SPAN, 8>> data;
  Input<inOutType> B;

  bool compute() {
    BroadcastOpDispatchSupervisor<op, inOutType>::compute(data.size(), getWsr(),
                                                  &data[0], &data[0], *B);
    return true;
  }
};
template class BroadcastOp1DInPlace<expr::BroadcastOpType::ADD, float>;
template class BroadcastOp1DInPlace<expr::BroadcastOpType::ADD, half>;

template class BroadcastOp1DInPlace<expr::BroadcastOpType::SUBTRACT, float>;
template class BroadcastOp1DInPlace<expr::BroadcastOpType::SUBTRACT, half>;

template class BroadcastOp1DInPlace<expr::BroadcastOpType::MULTIPLY, float>;
template class BroadcastOp1DInPlace<expr::BroadcastOpType::MULTIPLY, half>;

template class
  BroadcastOp1DInPlace<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, float>;
template class
  BroadcastOp1DInPlace<expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE, half>;

template class
  BroadcastOp1DInPlace<expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV, float>;
template class
  BroadcastOp1DInPlace<expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV, half>;

#endif

} // namespace popops
