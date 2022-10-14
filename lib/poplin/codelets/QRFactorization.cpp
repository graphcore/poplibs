// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "Dot.hpp"
#include <cmath>
#include <poplar/Vertex.hpp>

using namespace poplar;

namespace poplin {

namespace experimental {

#ifdef __IPU__

#define SET_ADDR(RESULT, NAME_STR)                                             \
  asm volatile(" setzi  %[workerAddress], " NAME_STR "\n"                      \
               : [workerAddress] "=r"(RESULT)                                  \
               :                                                               \
               :);

static __attribute__((always_inline)) unsigned getWid(void) {
  unsigned result;
  return __builtin_ipu_get(CSR_W_WSR__INDEX) & CSR_W_WSR__CTXTID_M1__MASK;
}

#endif

static __attribute__((always_inline)) unsigned
divideWork(const unsigned size, const unsigned vectorWidthShifts,
           const unsigned worker) {
  // Multiply by 0xaaab and shift by 18 is just a multiplication by 1/6 in
  // fixed point (Q14.18)
  return (((size >> vectorWidthShifts) + 5 - worker) * 0xaaab) >> 18;
}

class PartialSquareElements : public MultiVertex {
public:
  InOut<Vector<float>> rowToProcess;
  Input<unsigned> offset;
  Input<Vector<unsigned int>> padding;

  bool compute(unsigned wid) {
    const unsigned nelems = rowToProcess.size();
    const unsigned padd = padding[0];
    unsigned start = *offset;

    const unsigned toZero =
        std::max(std::min((int)(padd - start), (int)nelems), 0);
    // Zero padding.
    for (unsigned i = wid; i < toZero; i += numWorkers()) {
      rowToProcess[i] = 0;
    }
    // Sqare the elements.
    for (unsigned i = toZero + wid; i < nelems; i += numWorkers()) {
      rowToProcess[i] *= rowToProcess[i];
    }

    return true;
  }
};

class Householder : public MultiVertex {
public:
  InOut<Vector<float>> v;
  Input<float> dotProduct;
  Input<unsigned> offset;
  Input<Vector<unsigned int>> padding;
  Input<float> diagonalValue;

  bool compute(unsigned wid) {
    const float dotProductValue = *dotProduct;
    const float diag = *diagonalValue;
    const unsigned vSize = v.size();
    const unsigned workers = numWorkers();
    const unsigned padd = padding[0];
    const unsigned start = *offset;
    const unsigned end = start + vSize;

    // Set the value for the first element in the vector v (according to the
    // formula).
    if (wid == 0) {
      if (start <= padd && padd < end) {
        v[padd - start] =
            -(-(diag / abs(diag)) * sqrtf(dotProductValue) - diag);
      }
    }

    const unsigned toZero =
        std::min(std::max((int)(padd - start), 0), (int)vSize);
    // Zero padding.
    for (unsigned i = wid; i < toZero; i += workers)
      v[i] = 0.f;

    // Update the ip from the formula to update the remaining values.
    const float ip =
        -sqrtf(2 * dotProductValue + 2 * sqrtf(dotProductValue) * abs(diag));
    for (unsigned i = wid + toZero; i < vSize; i += workers)
      v[i] /= ip;
    return true;
  }
};

#ifdef __IPU__
class ScaledSub : public Vertex {
public:
  Input<Vector<float, VectorLayout::ONE_PTR>> v;
  InOut<Vector<float, VectorLayout::ONE_PTR>> AQRows;
  Output<Vector<float, VectorLayout::ONE_PTR>> dotProducts;
  Input<Vector<unsigned, VectorLayout::ONE_PTR>> nelemsPerWorker;
  Input<Vector<unsigned, VectorLayout::ONE_PTR>> offsets;

  bool compute() {
    constexpr bool useCppVersion = false;
    const unsigned wid = getWid();
    const unsigned nelems = nelemsPerWorker[wid];

    // Accumulate values.
    float dotProduct = 0;
    constexpr float multiplier = 2.f;
    for (unsigned j = 0; j < CTXT_WORKERS; j++)
      dotProduct += dotProducts[j];
    dotProduct *= multiplier;

    const unsigned offset = offsets[wid];
    if (!useCppVersion) {
      const float *inputR = &AQRows[offset];
      const float *inputV = &v[offset];
      const bool aligned = poplin::aligned<8, float>(inputR, inputV);
      // Scaled subtract loop (A[row][:] -= v * dotProcut)
      // Assembly version takes 4 cycles per iteration, C++ 6 cycles
      asm(
          R"(
                   brnz %[aligned], 4f
                   ld32step $a1, $mzero, %[vPtr]+=, 1
                   {
                    ld32 $a0, %[inputPtr], $mzero, 0
                    f32mul $a1, %[dotProduct], $a1
                   }
                   {
                     sub %[nelems], %[nelems], 1
                     f32sub $a0, $a0, $a1
                   }
                   st32step $a0, $mzero, %[inputPtr]+=, 1
                 4:
                   and $m1, %[nelems], 1
                   shr %[nelems], %[nelems], 1
                   bri 2f
                  1:
                   st64step $a0:1, $mzero, %[inputPtr]+=, 1
                  2:
                   ld64step $a2:3, $mzero, %[vPtr]+=, 1
                   {
                     ld64 $a0:1, %[inputPtr], $mzero, 0
                     f32v2mul $a2:3, %[dotProduct]:B, $a2:3
                   }
                   {
                     brnzdec %[nelems], 1b
                     f32v2sub $a0:1,  $a0:1, $a2:3
                   }
                   brz $m1, 3f
                   st32 $a0, $mzero, %[inputPtr], 0
                  3:
                  )"
          : [inputPtr] "+r"(inputR), [vPtr] "+r"(inputV)
          : [dotProduct] "r"(dotProduct), [aligned] "r"(aligned),
            [nelems] "r"(nelems)
          : "$a0:1", "$a2:3", "$m1", "memory");
    } else {
      for (unsigned i = 0; i < nelems; i++) {
        AQRows[offset + i] -= v[offset + i] * dotProduct;
      }
    }

    return true;
  }
};

class DotProduct : public Vertex {
public:
  Input<Vector<float, VectorLayout::ONE_PTR>> v;
  InOut<Vector<float, VectorLayout::ONE_PTR>> AQRows;
  Output<Vector<float, VectorLayout::ONE_PTR>> dotProducts;
  Input<Vector<unsigned, VectorLayout::ONE_PTR>> nelemsPerWorker;
  Input<Vector<unsigned, VectorLayout::ONE_PTR>> offsets;

  bool compute() {
    const unsigned wid = getWid();
    const unsigned offset = offsets[wid];
    const float *inputA = &AQRows[offset];
    const float *inputB = &v[offset];
    // Dot product from the row and v.
    dotProducts[wid] = poplin::Dot<float, false>::compute(inputA, inputB,
                                                          nelemsPerWorker[wid]);

    return true;
  }
};
#endif

class Update : public SupervisorVertex {
public:
  Input<Vector<float, VectorLayout::SPAN, 8>> v;
  Vector<InOut<Vector<float, VectorLayout::SPAN, 8>>> AQRows;
  Input<Vector<unsigned int>> padding;

#ifdef __IPU__
  struct WorkerBase {
    const float *v;
    const float *AQRows;
    const float *dotProducts;
    const unsigned *nelemsPerWorker;
    const unsigned *offsets;
  };

  static __attribute__((always_inline)) void syncWorkers(void) {
    asm volatile(
        " sync   %[sync_zone]\n" ::[sync_zone] "i"(TEXCH_SYNCZONE_LOCAL));
  }

  static __attribute__((always_inline)) void
  runAll(const unsigned *workerAddress, const WorkerBase *state) {
    asm volatile(" runall %[workerAddress], %[state], 0\n"
                 :
                 : [workerAddress] "r"(workerAddress), [state] "r"(state)
                 :);
  }

  bool __attribute__((target("supervisor"))) compute() {
    const unsigned rows = AQRows.size();
    const unsigned vSize = v.size();
    const unsigned shift = padding[0] & 0xFFFFFFFE;
    const unsigned nelems = (vSize - shift);
    const unsigned remainder = nelems % CTXT_WORKERS;

    float dotProducts[CTXT_WORKERS];
    unsigned nelemsPerWorker[CTXT_WORKERS];
    unsigned offsets[CTXT_WORKERS];

    unsigned offset = 0;
    for (unsigned w = 0; w < CTXT_WORKERS - 1; w++) {
      const unsigned work = divideWork(nelems, 1, w) * 2;
      nelemsPerWorker[w] = work;
      offsets[w] = offset;
      offset += work;
    }
    nelemsPerWorker[CTXT_WORKERS - 1] = nelems - offset;
    offsets[CTXT_WORKERS - 1] = offset;

    unsigned *dotProductPtr, *scaledSubPtr;
    SET_ADDR(dotProductPtr, "__runCodelet_poplin__experimental__DotProduct")
    SET_ADDR(scaledSubPtr, "__runCodelet_poplin__experimental__ScaledSub")

    WorkerBase wBase{&v[shift], nullptr, &dotProducts[0], &nelemsPerWorker[0],
                     &offsets[0]};

    // A loop in which each worker computes the dot Product part of a given row
    // and v, and then each worker accumulates these values and updates the part
    // of the row with them.
    for (unsigned r = 0; r < rows; r++) {
      wBase.AQRows = &AQRows[r][shift];
      runAll(dotProductPtr, &wBase);
      syncWorkers();
      runAll(scaledSubPtr, &wBase);
      syncWorkers();
    }

    return true;
  }
#else
  bool compute() {
    const unsigned rows = AQRows.size();
    const unsigned vSize = v.size();
    const unsigned shift = padding[0];
    constexpr double multiplier = 2.0;

    for (unsigned r = 0; r < rows; r++) {
      float dotProduct = 0.0;
      for (unsigned i = shift; i < vSize; i++)
        dotProduct += AQRows[r][i] * v[i];
      dotProduct *= multiplier;
      for (unsigned c = shift; c < vSize; c++)
        AQRows[r][c] -= dotProduct * v[c];
    }
    return true;
  }
#endif
};

class RowCopy : public MultiVertex {
public:
  Input<unsigned> offset;
  Output<Vector<float>> copiedRow;
  Output<Vector<float>> diagonalValueVector;
  Vector<Input<Vector<float>>> A;
  Input<Vector<unsigned int>> padding;

  bool compute(unsigned wid) {
    const unsigned rows = A.size();
    const unsigned padd = padding[0];

    for (unsigned i = wid; i < rows; i += numWorkers()) {
      copiedRow[i] = A[i][padd];
      diagonalValueVector[i] = 0.f;
    }

    // If the value of diagonal A is in the memory of this tile, set it in
    // diagonalValueVector, the rest of the vector is set to zero above.
    if (wid == 0) {
      const unsigned offsetValue = *offset;
      if (padd >= offsetValue && padd < offsetValue + rows) {
        diagonalValueVector[0] = A[padd - offsetValue][padd];
      }
    }

    return true;
  }
};
} // namespace experimental

} // end namespace poplin
