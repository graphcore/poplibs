// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <math.h>

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "popops/ExprOp.hpp"
#include "popops/elementwiseCodelets.hpp"

#ifdef VECTOR_AVAIL_SHORT_SPAN
static constexpr auto SPAN_TYPE = poplar::VectorLayout::SHORT_SPAN;
#else
static constexpr auto SPAN_TYPE = poplar::VectorLayout::SPAN;
#endif
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

using namespace poplar;

namespace popops {

template <typename FPType>
class [[poplar::constraint("elem(**A) != elem(**B)")]] HadamardProd
    : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> A;
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> B;

  bool compute() {
    const unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      const unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] *= refIn[j];
      }
    }
    return true;
  }
};

template class HadamardProd<float>;
template class HadamardProd<half>;

template <typename InType> class Fill : public Vertex {
public:
  InType in;
  Output<Vector<InType>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (auto &x : out) {
      x = in;
    }
    return true;
  }
};

template class Fill<float>;
template class Fill<half>;
template class Fill<int>;
template class Fill<unsigned>;

template <typename FPType> constexpr bool fill2dHasAssembly() {
  return std::is_same<FPType, half>() || std::is_same<FPType, float>();
}

template <typename FPType> class Fill2d : public Vertex {
public:
  FPType in;
  Vector<Output<Vector<FPType>>> out;

  IS_EXTERNAL_CODELET(fill2dHasAssembly<FPType>);

  bool compute() {
    for (auto &row : out) {
      for (auto &x : row) {
        x = in;
      }
    }
    return true;
  }
};

template class Fill2d<float>;
template class Fill2d<half>;
template class Fill2d<unsigned int>;
template class Fill2d<int>;

// A couple of macros to instantiate more compactly the templates of the various
// Cast vertices, for all possible combinations of input and output types
// (float, half, signed/unsinged ints and bool)
#define INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, srcType)                  \
  template class CastVertexName<srcType, float>;                               \
  template class CastVertexName<srcType, half>;                                \
  template class CastVertexName<srcType, int>;                                 \
  template class CastVertexName<srcType, unsigned>;                            \
  template class CastVertexName<srcType, unsigned short>;                      \
  template class CastVertexName<srcType, bool>;
#define INSTANTIATE_CAST(CastVertexName)                                       \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, float)                          \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, half)                           \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, int)                            \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, unsigned)                       \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, unsigned short)                 \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, bool)

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Cast : public Vertex {
public:
  Cast();

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  static const bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : alignof(DstType);
  static const unsigned inAlign = ext ? 8 : alignof(SrcType);

  static const poplar::VectorLayout inLayout =
      inAlign == 8 ? PTR_ALIGN64 : ONE_PTR;
  static const poplar::VectorLayout outLayout =
      outAlign == 4 ? PTR_ALIGN32 : (outAlign == 8 ? PTR_ALIGN64 : ONE_PTR);
  Input<Vector<SrcType, inLayout, inAlign>> src;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  const unsigned numElems;

  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    for (unsigned i = 0; i < numElems; ++i) {
      dst[i] = static_cast<DstType>(src[i]);
    }
    return true;
  }
};

INSTANTIATE_CAST(Cast)

#ifdef __IPU__
// The vertices defined by this template will be called by the supervisor
// vertex only
template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] CastWorker
    : public Vertex {
public:
  CastWorker();

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  static const bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : alignof(DstType);
  static const unsigned inAlign = ext ? 8 : alignof(SrcType);

  static const poplar::VectorLayout inLayout =
      inAlign == 8 ? PTR_ALIGN64 : ONE_PTR;
  static const poplar::VectorLayout outLayout =
      outAlign == 4 ? PTR_ALIGN32 : (outAlign == 8 ? PTR_ALIGN64 : ONE_PTR);
  Input<Vector<SrcType, inLayout, inAlign>> src;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  const unsigned partitionParams;

  bool compute() {
    unsigned wId = getWsr();
    // Read the comment for partitionParams in CastSupervisor below to see
    // how work is partitioned to this worker.
    const unsigned deltaLast = partitionParams & 0x7;
    const unsigned workerLast = (partitionParams >> 3) & 0x7;
    const unsigned workerCount = (partitionParams >> 6) & 0x7;
    unsigned workerElems = partitionParams >> 9;
    unsigned offs = wId * workerElems;
    if (wId >= workerCount) {
      workerElems -= 4;
      offs -= (wId - workerCount) * 4;
    }
    if (wId == workerLast) {
      workerElems -= deltaLast;
    }
    for (unsigned i = 0; i < workerElems; ++i) {
      dst[offs + i] = static_cast<DstType>(src[offs + i]);
    }
    return true;
  }
};

// Note that we don't define here the following:
//    1. FLOAT<->HALF conversion (defined in assembly)
//    2. Identity conversions (XXX->XXX) and INT<->UNSIGNED as these will be
//       replaced with Copy() in popops::cast()
template class CastWorker<float, int>;
template class CastWorker<float, unsigned>;
template class CastWorker<float, unsigned short>;
template class CastWorker<float, bool>;

template class CastWorker<half, int>;
template class CastWorker<half, unsigned>;
template class CastWorker<half, unsigned short>;
template class CastWorker<half, bool>;

template class CastWorker<int, float>;
template class CastWorker<int, half>;
template class CastWorker<int, bool>;
template class CastWorker<int, unsigned short>;

template class CastWorker<unsigned int, float>;
template class CastWorker<unsigned int, half>;
template class CastWorker<unsigned int, bool>;
template class CastWorker<unsigned int, unsigned short>;

template class CastWorker<unsigned short, float>;
template class CastWorker<unsigned short, half>;
template class CastWorker<unsigned short, int>;
template class CastWorker<unsigned short, unsigned int>;
template class CastWorker<unsigned short, bool>;

template class CastWorker<bool, float>;
template class CastWorker<bool, half>;
template class CastWorker<bool, int>;
template class CastWorker<bool, unsigned int>;
template class CastWorker<bool, unsigned short>;

#endif

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] CastSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  CastSupervisor();

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  static const bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : alignof(DstType);
  static const unsigned inAlign = ext ? 8 : alignof(SrcType);

  static const poplar::VectorLayout inLayout =
      inAlign == 8 ? PTR_ALIGN64 : ONE_PTR;
  static const poplar::VectorLayout outLayout =
      outAlign == 4 ? PTR_ALIGN32 : (outAlign == 8 ? PTR_ALIGN64 : ONE_PTR);

  Input<Vector<SrcType, inLayout, inAlign>> src;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  // 'partitionParams' contains 4 bit fields defining how the work is
  // partitioned among workers:
  //
  //                           23 bits                    3     3     3
  //  +------------------------------------------------+-----+-----+-----+
  //  |                       Welems                   | Wcnt| Wlst| Dlst|
  //  +------------------------------------------------+-----+-----+-----+
  // MSB                                                               LSB
  //
  // The first 'Wcnt' (Worker Count) workers will process 'Welems' (Worker
  // Elements) elements each ('Welems' always a multiple of 4).
  // The other (6-'Wcnt') workers will process 'Welems-4' elems (could be none).
  // Need to correct the above for the last worker, (index 'Wlst =Worker Last),
  // that will process 'Dlst' (Delta Last, 0..3) fewer elements than specified
  // by 'Welems' or 'Welems-4'.
  // For instance:
  //
  // Total elements : 15  =>   WCnt=4, Welems=4, Wlst=3, Dlst=1
  //
  //  WkId 0     WkId 1     WkId 2     WkId 3     WkId 4     WkId 5
  // 4 elems    4 elems    4 elems    3 elems    0 elems    0 elems
  //   +---------------+------------------+  \
  //                   |                      \
  //                Wcnt=4                  Last one does 'Dlst' fewer elems.
  //
  //
  // Total elements : 30  =>   WCnt=2, Welems=8, Wlst=5, Dlst=2
  //
  //  WkId 0     WkId 1     WkId 2     WkId 3     WkId 4     WkId 5
  // 8 elems    8 elems    4 elems    4 elems    4 elems    2 elems
  // +--------+-------+                                        |
  //          |                                                |
  //        Wcnt=2                       Last one does 'Dlst' fewer elems.
  //
  // This ensures that all workers start on a 4-element boundary.
  const unsigned partitionParams;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const unsigned deltaLast = partitionParams & 0x7;
    const unsigned workerLast = (partitionParams >> 3) & 0x7;
    const unsigned workerCount = (partitionParams >> 6) & 0x7;
    const unsigned workerElems = partitionParams >> 9;
    const unsigned numElems = workerCount * workerElems +
                              (CTXT_WORKERS - workerCount) * (workerElems - 4) -
                              deltaLast;
    for (unsigned i = 0; i < numElems; ++i) {
      dst[i] = static_cast<DstType>(src[i]);
    }
    return true;
  }
};

INSTANTIATE_CAST(CastSupervisor)

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(**src) != elem(**dst)")]] Cast2d
    : public Vertex {
public:
  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  static const bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : alignof(DstType);
  static const unsigned inAlign = ext ? 8 : alignof(SrcType);

  Vector<Input<Vector<SrcType, ONE_PTR, inAlign>>, ONE_PTR> src;
  Vector<Output<Vector<DstType, SPAN, outAlign>>> dst;

  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    const unsigned limI = dst.size();
    for (unsigned i = 0; i != limI; ++i) {
      const unsigned limJ = dst[i].size();
      auto const &refSrc = src[i];
      auto &refDst = dst[i];
      for (unsigned j = 0; j != limJ; ++j) {
        refDst[j] = static_cast<DstType>(refSrc[j]);
      }
    }
    return true;
  }
};

INSTANTIATE_CAST(Cast2d)

template <typename InType> class Clamp : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2; // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3; // upper bound
  Vector<Output<Vector<InType>>> out;

  static const bool ext =
      std::is_same<InType, float>::value || std::is_same<InType, half>::value;
  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {

      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in1[i][j];
        if (out[i][j] < in2[i][j]) {
          out[i][j] = in2[i][j];
        }
        if (out[i][j] > in3[i][j]) {
          out[i][j] = in3[i][j];
        }
      }
    }
    return true;
  }
};

template class Clamp<float>;
template class Clamp<half>;
template class Clamp<int>;

template <typename InType> class Select : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Vector<Input<Vector<bool, ONE_PTR>>, ONE_PTR> in3;
  Vector<Output<Vector<InType, SPAN, 4>>> out;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in3[i][j] ? in1[i][j] : in2[i][j];
      }
    }
    return true;
  }
};

template class Select<float>;
template class Select<half>;
template class Select<int>;
template class Select<bool>;

template <typename InType> class BroadcastClamp : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Input<InType> in2;
  Input<InType> in3;
  Vector<Output<Vector<InType>>> out;

  static const bool ext =
      std::is_same<InType, float>::value || std::is_same<InType, half>::value;
  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned j = 0; j < out[i].size(); ++j) {
        out[i][j] = in1[i][j];
        if (out[i][j] < *in2) {
          out[i][j] = *in2;
        }
        if (out[i][j] > *in3) {
          out[i][j] = *in3;
        }
      }
    }
    return true;
  }
};

template class BroadcastClamp<float>;
template class BroadcastClamp<half>;
template class BroadcastClamp<int>;

// 'Select' ternary operator where the selector (boolean third operand) is a
// tensor, while the 1st and 2nd operands are scalars (that are broadcasted
// into the output)
template <typename InType> class BroadcastSelect : public Vertex {
public:
  Input<InType> in1;
  Input<InType> in2;
  Vector<Input<Vector<bool, ONE_PTR>>, ONE_PTR> in3;
  Vector<Output<Vector<InType, SPAN>>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in3[i][j] ? in1 : in2;
      }
    }
    return true;
  }
};

template class BroadcastSelect<float>;
template class BroadcastSelect<half>;
template class BroadcastSelect<int>;
template class BroadcastSelect<bool>;

// 'Select' ternary operator where the selector (boolean third operand) is a
// scalar and needs broadcasting, while the 1st and 2nd operands are tensors
// Just copy 'in1', or 'in2', into 'out', based on the scalar 'in3'.
template <typename InType> class BroadcastSelectorSelect : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Input<bool> in3;
  Vector<Output<Vector<InType, SPAN>>> out;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    const auto in = in3 ? in1 : in2;
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned j = 0; j < out[i].size(); ++j) {
        out[i][j] = in[i][j];
      }
    }
    return true;
  }
};

template class BroadcastSelectorSelect<float>;
template class BroadcastSelectorSelect<half>;
template class BroadcastSelectorSelect<int>;
template class BroadcastSelectorSelect<bool>;

template <typename InType> class ClampInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2; // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3; // upper bound

  bool compute() {
    for (unsigned i = 0; i != in1Out.size(); ++i) {
      for (unsigned j = 0; j != in1Out[i].size(); ++j) {
        if (in1Out[i][j] < in2[i][j]) {
          in1Out[i][j] = in2[i][j];
        }
        if (in1Out[i][j] > in3[i][j]) {
          in1Out[i][j] = in3[i][j];
        }
      }
    }
    return true;
  }
};

template class ClampInPlace<float>;
template class ClampInPlace<half>;
template class ClampInPlace<int>;

template <typename InType> class BroadcastClampInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Input<InType> in2;
  Input<InType> in3;

  bool compute() {
    for (unsigned i = 0; i < in1Out.size(); ++i) {
      for (unsigned j = 0; j < in1Out[i].size(); ++j) {
        if (in1Out[i][j] < *in2) {
          in1Out[i][j] = *in2;
        }
        if (in1Out[i][j] > *in3) {
          in1Out[i][j] = *in3;
        }
      }
    }
    return true;
  }
};

template class BroadcastClampInPlace<float>;
template class BroadcastClampInPlace<half>;
template class BroadcastClampInPlace<int>;

template <typename InType> class SelectInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Vector<Input<Vector<bool, ONE_PTR>>, ONE_PTR> in3;

  bool compute() {
    for (unsigned i = 0; i != in1Out.size(); ++i) {
      for (unsigned j = 0; j != in1Out[i].size(); ++j) {
        in1Out[i][j] = in3[i][j] ? in1Out[i][j] : in2[i][j];
      }
    }
    return true;
  }
};

template class SelectInPlace<float>;
template class SelectInPlace<half>;
template class SelectInPlace<int>;
template class SelectInPlace<bool>;

template <typename InType>
class BroadcastSelectorSelectInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Input<bool> in3;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    if (in3 == false) {
      for (unsigned i = 0; i != in1Out.size(); ++i) {
        for (unsigned j = 0; j != in1Out[i].size(); ++j) {
          in1Out[i][j] = in2[i][j];
        }
      }
    }
    return true;
  }
};

template class BroadcastSelectorSelectInPlace<float>;
template class BroadcastSelectorSelectInPlace<half>;
template class BroadcastSelectorSelectInPlace<int>;
template class BroadcastSelectorSelectInPlace<bool>;

template <typename InType, bool isAbsolute> class Histogram2D : public Vertex {
public:
  Vector<Input<Vector<InType, SPAN_TYPE>>, SPAN_TYPE> data;
  // There will be `limits` +1 histogram entries
  Input<Vector<InType, PTR_ALIGN32, 4>> limits;
  Output<Vector<float, PTR_ALIGN32, 4>> histogram;
  unsigned short histogramCount;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    auto condAbs = [](auto d) {
      return isAbsolute ? static_cast<InType>(std::fabs(static_cast<float>(d)))
                        : d;
    };

    // Structured like the assembler
    float previous = 0.0;
    for (unsigned i = 0; i < histogramCount - 1; i++) {
      float lessThanCount = 0;
      histogram[histogramCount - 1] = 0;
      for (unsigned j = 0; j < data.size(); j++) {
        histogram[histogramCount - 1] += data[j].size();
        for (unsigned k = 0; k < data[j].size(); k++) {
          lessThanCount += condAbs(data[j][k]) < limits[i];
        }
      }
      const auto thisCount = lessThanCount;
      histogram[i] = lessThanCount - previous;
      previous = thisCount;
    }
    // Adjust the last one
    histogram[histogramCount - 1] -= previous;
    return true;
  }
};

template class Histogram2D<float, true>;
template class Histogram2D<half, true>;
template class Histogram2D<float, false>;
template class Histogram2D<half, false>;

template <typename InType, bool isAbsolute, bool splitByLimits>
class HistogramSupervisor : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  // SPAN required to support usefully large data size and no alignment
  // constraint
  Input<Vector<InType, SPAN>> data;
  // There will be `limits` +1 histogram entries
  Input<Vector<InType, PTR_ALIGN32, 4>> limits;
  // When splitByLimits==false, this array must be histogramCount * NUM_WORKERS
  // in size
  Output<Vector<float, PTR_ALIGN32, 4>> histogram;
  unsigned short histogramCount;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    auto condAbs = [](auto d) {
      return isAbsolute ? static_cast<InType>(std::fabs(static_cast<float>(d)))
                        : d;
    };

    // Structured like the assembler (when split by limits)
    for (unsigned wkrId = 0; wkrId < NUM_WORKERS; wkrId++) {
      for (unsigned i = wkrId; i < histogramCount; i += NUM_WORKERS) {
        float lessThanCount = 0;
        for (unsigned j = 0; j < data.size(); j++) {
          lessThanCount += condAbs(data[j]) < limits[i];
        }
        histogram[i] = lessThanCount;
      }
    }
    // They are all < the max limit of the upper "open bound"
    histogram[histogramCount - 1] = data.size();
    // Post process
    for (unsigned i = histogramCount - 1; i > 0; i--) {
      float count = histogram[i];
      float adjusted = count - histogram[i - 1];
      histogram[i] = adjusted;
    }
    return true;
  }
};

// Note: Other types of histogram vertices may be more efficient in certain
// circumstances.
// For example with a large number of levels a scalar
// comparison of data with a level and a binary search to pick the next level to
// compare to will reduce the number or comparisons required.  This can be
// slow (as it cannot be vectorised and will need decision branches in its inner
// loop) but as the number of levels to compare is increased it will eventually
// be more efficient.
// A second case would be where levels are regular (and constant) each input
// could be used to calculate the histogram entry which it will contribute to
// directly.
//
// The methods used in the exisiting vertices are more general purpose and
// cover many reasonable cases.

template class HistogramSupervisor<float, true, true>;
template class HistogramSupervisor<half, true, true>;
template class HistogramSupervisor<float, false, true>;
template class HistogramSupervisor<half, false, true>;
template class HistogramSupervisor<float, true, false>;
template class HistogramSupervisor<half, true, false>;
template class HistogramSupervisor<float, false, false>;
template class HistogramSupervisor<half, false, false>;
} // namespace popops
