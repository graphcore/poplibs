// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "Dot.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

namespace poplin {

#ifdef __IPU__
namespace {
__attribute__((always_inline)) float reduce_add_v4(const float4 &v4) {
  const float2 v2 = *(float2 *)&v4 + *(((float2 *)&v4) + 1);
  return v2[0] + v2[1];
}

__attribute__((always_inline)) half reduce_add_v4(const half4 &v4) {
  const half2 v2 = *(half2 *)&v4 + *(((half2 *)&v4) + 1);
  return v2[0] + v2[1];
}
} // namespace
#endif

template <class FloatType, bool lower> class TriangularSolveSIMD4 : Vertex {
public:
  Input<Vector<FloatType, poplar::VectorLayout::ONE_PTR>> a;
  InOut<Vector<FloatType, poplar::VectorLayout::ONE_PTR>> b;
  Output<Vector<FloatType, poplar::VectorLayout::ONE_PTR>> x;
  Input<unsigned> an;
  Input<unsigned> xOffset;

#ifdef __IPU__

  __attribute__((always_inline)) unsigned getWid(void) {
    return __builtin_ipu_get(CSR_W_WSR__INDEX) & CSR_W_WSR__CTXTID_M1__MASK;
  }

  __attribute__((always_inline)) int a_off(int y, int x) {
    return (xOffset + y) * an + xOffset + x;
  }

  void compute() {
    const unsigned wid = getWid();
    const unsigned yOffset = xOffset;
    constexpr bool isFloat32 = std::is_same<FloatType, float>::value;
    using simd2_t = typename std::conditional<isFloat32, float2, half2>::type;
    using simd4_t = typename std::conditional<isFloat32, float4, half4>::type;

    // Compute first four values of x on all workers to avoid
    // the need for synchronization
    // Matrix is unit-diagonal, so the first x is equal to the first b
    x[xOffset] = b[xOffset];

    // Solve for the next three x values
    if (lower) {
      // Solve for the second x
      x[xOffset + 1] = b[xOffset + 1] - a[a_off(1, 0)] * x[xOffset];

      // Solve for the third x using SIMD2
      const simd2_t *xSlice_01 = (simd2_t *)&x[xOffset];

      const simd2_t bUpdate_y2_v2 = *(simd2_t *)&a[a_off(2, 0)] * *xSlice_01;
      const FloatType bUpdate_y2 = bUpdate_y2_v2[0] + bUpdate_y2_v2[1];
      x[xOffset + 2] = b[xOffset + 2] - bUpdate_y2;

      // Solve for the fourth x using SIMD2
      const simd2_t bUpdate_y3_v2 = *(simd2_t *)&a[a_off(3, 0)] * *xSlice_01;
      const FloatType bUpdate_y3 =
          bUpdate_y3_v2[0] + bUpdate_y3_v2[1] + a[a_off(3, 2)] * x[xOffset + 2];
      x[xOffset + 3] = b[xOffset + 3] - bUpdate_y3;

    } else {
      // Solve for the second x from the end
      x[xOffset - 1] = b[xOffset - 1] - a[a_off(-1, 0)] * x[xOffset];

      // Solve for the third x from the end using SIMD 2
      const simd2_t *xSlice_01 = (simd2_t *)&x[xOffset - 1];

      const simd2_t bUpdate_y2_v2 = *(simd2_t *)&a[a_off(-2, -1)] * *xSlice_01;
      const FloatType bUpdate_y2 = bUpdate_y2_v2[0] + bUpdate_y2_v2[1];
      x[xOffset - 2] = b[xOffset - 2] - bUpdate_y2;

      // Solve for the fourth x from the end using SIMD 2
      const simd2_t bUpdate_y3_v2 = *(simd2_t *)&a[a_off(-3, -1)] * *xSlice_01;
      const FloatType bUpdate_y3 = bUpdate_y3_v2[0] + bUpdate_y3_v2[1] +
                                   a[a_off(-3, -2)] * x[xOffset - 2];
      x[xOffset - 3] = b[xOffset - 3] - bUpdate_y3;
    }

    // Nothing left to compute
    if (an - xOffset == 4)
      return;

    // Update matrix b for the rest of rows using SIMD 4 and multiple workers
    constexpr unsigned simdSize = 4;
    // If the datatype is half, each worker must update two values of b
    // vector written on the single 32b word
    constexpr unsigned numElemsPerIter = isFloat32 ? 1 : 2;
    // If a is lower triangular, y is increased in the loop, otherwise decreased
    constexpr int moveDir = lower ? 1 : -1;
    constexpr int yStep = moveDir * numElemsPerIter * CTXT_WORKERS;
    const int aStep = yStep * an / simdSize;
    const unsigned yWorkerOffset = moveDir * (4 + wid * numElemsPerIter);
    const int yStart = yOffset + yWorkerOffset;

    const simd4_t xSlice_0123 = *(simd4_t *)&x[lower ? xOffset : xOffset - 3];
    const int aBaseOffset = a_off(yWorkerOffset, lower ? 0 : -3);
    const simd4_t *a_v4 = (simd4_t *)&a[aBaseOffset];
    for (int y = yStart; lower ? (y < an) : (y >= 0);
         y += yStep, a_v4 += aStep) {
      const simd4_t bUpdate_v4 = *a_v4 * xSlice_0123;
      const FloatType bUpdate = reduce_add_v4(bUpdate_v4);
      if constexpr (isFloat32) {
        b[y] -= bUpdate;
      } else {
        // Calculate the second b element
        const simd4_t *a2_v4 = a_v4 + moveDir * an / simdSize;
        const simd4_t bUpdate2_v4 = *a2_v4 * xSlice_0123;
        const FloatType bUpdate2 = reduce_add_v4(bUpdate2_v4);
        // Subtract two half values at once
        if (lower) {
          const simd2_t bUpdate2Elems = {bUpdate, bUpdate2};
          *(simd2_t *)&b[y] -= bUpdate2Elems;
        } else {
          const simd2_t bUpdate2Elems = {bUpdate2, bUpdate};
          *(simd2_t *)&b[y - 1] -= bUpdate2Elems;
        }
      }
    }
  }
#else
  void compute() {
    // The vertex can be run only on IPU
    assert(false);
  }
#endif
};

template class TriangularSolveSIMD4<float, true>;
template class TriangularSolveSIMD4<float, false>;
template class TriangularSolveSIMD4<half, true>;
template class TriangularSolveSIMD4<half, false>;

template <class FloatType, bool lower>
class TriangularSolveMultiWorker : SupervisorVertex {
public:
  TriangularSolveMultiWorker();

  const unsigned an;
  Input<Vector<FloatType, poplar::VectorLayout::SPAN, 8>> a;
  InOut<Vector<FloatType, poplar::VectorLayout::SPAN, 8>> b;
  Output<Vector<FloatType, poplar::VectorLayout::SPAN, 8>> x;

  struct SolveWorkerBase {
    const FloatType *a;
    const FloatType *b;
    const FloatType *x;
    const unsigned *an;
    unsigned *xOffset;
  };

#ifdef __IPU__

  static __attribute__((always_inline)) void syncWorkers(void) {
    asm volatile(".supervisor\nsync %[sync_zone]\n" ::[sync_zone] "i"(
        TEXCH_SYNCZONE_LOCAL));
  }

  template <typename T>
  static __attribute__((always_inline)) void
  runAll(const unsigned *workerAddress, const T *state) {
    asm volatile(".supervisor\nrunall %[workerAddress], %[state], 0\n"
                 :
                 : [workerAddress] "r"(workerAddress), [state] "r"(state)
                 :);
  }

#define SET_ADDR(RESULT, NAME_STR)                                             \
  asm volatile(".supervisor\nsetzi %[workerAddress], " NAME_STR "\n"           \
               : [workerAddress] "=r"(RESULT)                                  \
               :                                                               \
               :);

  void __attribute__((target("supervisor"))) compute() {
    assert(an != 0);
    assert(a.size() == an * an);
    assert(b.size() == an);
    assert(x.size() == an);

    constexpr unsigned simdSize = 4;
    assert(an % simdSize == 0);
    constexpr bool isFloat32 = std::is_same<FloatType, float>::value;

    SolveWorkerBase solveBase{&a[0], &b[0], &x[0], &an, 0};
    unsigned *solvePtr;
    if (lower) {
      if (isFloat32)
        SET_ADDR(solvePtr,
                 "__runCodelet_poplin__TriangularSolveSIMD4___float_true")
      else
        SET_ADDR(solvePtr,
                 "__runCodelet_poplin__TriangularSolveSIMD4___half_true")

      for (unsigned x = 0; x < an; x += simdSize) {
        solveBase.xOffset = &x;
        runAll(solvePtr, &solveBase);
        syncWorkers();
      }
    } else {
      if (isFloat32)
        SET_ADDR(solvePtr,
                 "__runCodelet_poplin__TriangularSolveSIMD4___float_false")
      else
        SET_ADDR(solvePtr,
                 "__runCodelet_poplin__TriangularSolveSIMD4___half_false")

      for (int x = an - 1; x >= static_cast<int>(simdSize - 1); x -= simdSize) {
        unsigned x_unsigned = x;
        solveBase.xOffset = &x_unsigned;
        runAll(solvePtr, &solveBase);
        syncWorkers();
      }
    }
  }
#else
  void compute() {
    if (lower) {
      for (unsigned y_off = 0; y_off < an; ++y_off) {
        FloatType curr_b = b[y_off];
        for (unsigned x_off = 0; x_off < y_off; ++x_off) {
          curr_b -= a[y_off * an + x_off] * x[x_off];
        }
        x[y_off] = curr_b / a[y_off * an + y_off];
      }
    } else {
      for (int y_off = an - 1; y_off >= 0; --y_off) {
        FloatType curr_b = b[y_off];
        for (int x_off = an - 1; x_off > y_off; --x_off) {
          curr_b -= a[y_off * an + x_off] * x[x_off];
        }
        x[y_off] = curr_b / a[y_off * an + y_off];
      }
    }
  }
#endif
};

template class TriangularSolveMultiWorker<float, true>;
template class TriangularSolveMultiWorker<float, false>;
template class TriangularSolveMultiWorker<half, true>;
template class TriangularSolveMultiWorker<half, false>;

template <class FloatType, bool lower>
class [[poplar::constraint("elem(*a) != elem(*x)")]] TriangularSolve : Vertex {
public:
  TriangularSolve();

  const unsigned an;
  Input<Vector<FloatType, poplar::VectorLayout::SPAN, 8>> a;
  Input<Vector<FloatType>> b;
  Output<Vector<FloatType, poplar::VectorLayout::SPAN, 8>> x;

  void compute() {
    assert(an != 0);
    assert(a.size() == an * an);
    assert(b.size() == an);
    assert(x.size() == an);

    if (lower) {
      x[0] = b[0];
    } else {
      const auto last = an - 1;
      x[last] = b[last];
    }

    std::size_t iBase = lower ? an : a.size() - an - an;
    std::size_t bBase = lower ? 1 : (an - 2);
    const auto *dotSrc1 = &a[iBase];
    if (lower) {
      for (std::size_t i = 1; i < an; ++i) {

        const auto *dotSrc2 = &x[0];
        FloatType dot = Dot<FloatType, true>::compute(dotSrc1, dotSrc2, i);
        dotSrc1 += an - i;
        x[bBase] = b[bBase] - dot;
        ++bBase;
      }
    } else {
      dotSrc1 += an - 1;
      for (std::size_t i = 1; i < an; ++i) {
        auto jBase = an - i;

        const auto *dotSrc2 = &x[jBase];
        FloatType dot = Dot<FloatType, true>::compute(dotSrc1, dotSrc2, i);
        dotSrc1 -= an + i + 1;
        x[bBase] = b[bBase] - dot;
        --bBase;
      }
    }
  }
};

template class TriangularSolve<float, false>;
template class TriangularSolve<float, true>;
template class TriangularSolve<half, false>;
template class TriangularSolve<half, true>;

} // end namespace poplin
