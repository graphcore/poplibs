// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <math.h>
#include <tuple>

#include "elementwiseCodelets.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "popops/ExprOp.hpp"

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

#if __IPU_ARCH_VERSION__ == 21
struct quarter {
  unsigned char data;
  quarter(unsigned x) { data = x; };
};
#endif

#ifdef __IPU__
#include "inlineAssembler.hpp"
#include "inlineAssemblerCast.hpp"
#endif

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

#if __IPU_ARCH_VERSION__ == 21
template class Fill<quarter>;
#endif

template class Fill<float>;
template class Fill<half>;
template class Fill<int>;
template class Fill<unsigned>;
template class Fill<bool>;
template class Fill<char>;
template class Fill<unsigned char>;
template class Fill<signed char>;
template class Fill<unsigned long long>;
template class Fill<long long>;

template <typename FPType> class Fill2d : public Vertex {
public:
  FPType in;
  Vector<Output<Vector<FPType>>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (auto &row : out) {
      for (auto &x : row) {
        x = in;
      }
    }
    return true;
  }
};

#if __IPU_ARCH_VERSION__ == 21
template class Fill2d<quarter>;
#endif

template class Fill2d<float>;
template class Fill2d<half>;
template class Fill2d<unsigned int>;
template class Fill2d<int>;
template class Fill2d<bool>;
template class Fill2d<char>;
template class Fill2d<unsigned char>;
template class Fill2d<signed char>;
template class Fill2d<unsigned long long>;
template class Fill2d<long long>;

// A couple of macros to instantiate more compactly the templates of the various
// Cast vertices, for all possible combinations of input and output types
// (float, half, signed/unsigned ints and bool)
#define INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, srcType)                  \
  template class CastVertexName<srcType, char>;                                \
  template class CastVertexName<srcType, signed char>;                         \
  template class CastVertexName<srcType, unsigned char>;                       \
  template class CastVertexName<srcType, float>;                               \
  template class CastVertexName<srcType, half>;                                \
  template class CastVertexName<srcType, int>;                                 \
  template class CastVertexName<srcType, short>;                               \
  template class CastVertexName<srcType, unsigned>;                            \
  template class CastVertexName<srcType, unsigned short>;                      \
  template class CastVertexName<srcType, bool>;
#define INSTANTIATE_CAST(CastVertexName)                                       \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, char)                           \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, signed char)                    \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, unsigned char)                  \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, float)                          \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, half)                           \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, int)                            \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, short)                          \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, unsigned)                       \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, unsigned short)                 \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, bool)

#define INSTANTIATE_CAST_LONGLONG(CastVertexName)                              \
  template class CastVertexName<short, unsigned long long>;                    \
  template class CastVertexName<short, long long>;                             \
  template class CastVertexName<unsigned short, unsigned long long>;           \
  template class CastVertexName<unsigned short, long long>;                    \
  template class CastVertexName<bool, unsigned long long>;                     \
  template class CastVertexName<bool, long long>;                              \
  template class CastVertexName<unsigned, long long>;                          \
  template class CastVertexName<unsigned, unsigned long long>;                 \
  template class CastVertexName<int, long long>;                               \
  template class CastVertexName<int, unsigned long long>;                      \
  template class CastVertexName<char, long long>;                              \
  template class CastVertexName<char, unsigned long long>;                     \
  template class CastVertexName<signed char, long long>;                       \
  template class CastVertexName<signed char, unsigned long long>;              \
  template class CastVertexName<unsigned char, long long>;                     \
  template class CastVertexName<unsigned char, unsigned long long>;

// Returns some compile time parameters for Cast vertices, based on SrcType
// and DstType, as a tuple where :
//   element 0 is a boolean     : true if the vertex is implemented in assembly
//   element 1 is an unsigned   : Alignment required for input data (bytes)
//   element 2 is an unsigned   : Alignment required for output (bytes)
//   element 3 is a VectorLayout: Layout for input data pointer
//   element 4 is a VectorLayout: Layout for output pointer
template <typename SrcType, typename DstType>
constexpr std::tuple<bool, unsigned, unsigned, VectorLayout, VectorLayout>
getCastParams() {
  bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;
  bool charHalf = (std::is_same<SrcType, unsigned char>::value ||
                   std::is_same<SrcType, signed char>::value ||
                   std::is_same<SrcType, char>::value) &&
                  std::is_same<DstType, half>::value;
  bool charFloat = (std::is_same<SrcType, unsigned char>::value ||
                    std::is_same<SrcType, signed char>::value ||
                    std::is_same<SrcType, char>::value) &&
                   std::is_same<DstType, float>::value;
  bool halfChar = std::is_same<SrcType, half>::value &&
                  (std::is_same<DstType, unsigned char>::value ||
                   std::is_same<DstType, signed char>::value ||
                   std::is_same<DstType, char>::value);
  bool floatChar = std::is_same<SrcType, float>::value &&
                   (std::is_same<DstType, unsigned char>::value ||
                    std::is_same<DstType, signed char>::value ||
                    std::is_same<DstType, char>::value);

  bool inlineAsm =
      floatHalf || halfFloat || floatChar || halfChar || charFloat || charHalf;

  unsigned inAlign = alignof(SrcType);
  unsigned outAlign = alignof(DstType);
  if (halfFloat) {
    inAlign = outAlign = 8;
  } else if (floatHalf) {
    inAlign = 8;
    outAlign = 4;
  } else if (charHalf || charFloat) {
    inAlign = 8;
    outAlign = 8;
  } else if (halfChar || floatChar) {
    inAlign = 8;
    outAlign = 4;
  }

  VectorLayout inLayout = ONE_PTR;
  VectorLayout outLayout = ONE_PTR;
  if (inlineAsm) {
    inLayout = PTR_ALIGN64;
    outLayout = PTR_ALIGN32;
  }
  if (halfFloat) {
    outLayout = PTR_ALIGN64;
  }

  return {inlineAsm, inAlign, outAlign, inLayout, outLayout};
}

template <typename SrcType, typename DstType, bool inlineAsm>
struct CastDispatch {
public:
  static void compute(unsigned numElems, const SrcType *src, DstType *dst) {
    constexpr unsigned elemsPerLoop = 4;
    for (unsigned i = 0; i < numElems / elemsPerLoop; ++i) {
      *dst++ = static_cast<DstType>(*src++);
      *dst++ = static_cast<DstType>(*src++);
      *dst++ = static_cast<DstType>(*src++);
      *dst++ = static_cast<DstType>(*src++);
    }
    for (unsigned i = 0; i < (numElems & 3); i++) {
      dst[i] = static_cast<DstType>(src[i]);
    }
  }
};

template <typename SrcType, typename DstType, bool inlineAsm>
struct CastDispatchFp8 {
public:
  static void compute(unsigned numElems, const SrcType *src, DstType *dst,
                      const unsigned char *metaData) {}
};

template <typename SrcType, typename DstType, bool inlineAsm>
struct CastDispatchMultiVertexFp8 {
public:
  static void compute(unsigned numElems, unsigned wid, const SrcType *src,
                      DstType *dst, const unsigned char *metaData) {}
};

template <typename SrcType, typename DstType, bool inlineAsm>
struct CastDispatchMultiVertex {
public:
  static void compute(unsigned numElems, unsigned wid, const SrcType *src,
                      DstType *dst) {
    constexpr unsigned elemsPerLoop = 4;
    const SrcType *loopSrc = &src[wid * elemsPerLoop];
    DstType *loopDst = &dst[wid * elemsPerLoop];

    for (unsigned i = 0; i < divideWork(numElems, 2, wid); ++i) {
      *loopDst++ = static_cast<DstType>(*loopSrc++);
      *loopDst++ = static_cast<DstType>(*loopSrc++);
      *loopDst++ = static_cast<DstType>(*loopSrc++);
      *loopDst++ = static_cast<DstType>(*loopSrc++);
      loopDst += elemsPerLoop * CTXT_WORKERS - elemsPerLoop;
      loopSrc += elemsPerLoop * CTXT_WORKERS - elemsPerLoop;
    }
    if (wid == CTXT_WORKERS - 1 && numElems & 3) {
      const unsigned offset = numElems & ~3;
      for (unsigned i = 0; i < (numElems & 3); i++) {
        dst[offset + i] = static_cast<DstType>(src[offset + i]);
      }
    }
  }
};

#ifdef __IPU__

#if __IPU_ARCH_VERSION__ == 21

template <typename SrcType, typename DstType>
struct CastDispatchFp8<SrcType, DstType, true> {
public:
  static constexpr auto fp8ToFp8 = std::is_same<SrcType, quarter>::value &&
                                   std::is_same<DstType, quarter>::value;
  static void compute(unsigned numElems, const SrcType *src, DstType *dst,
                      const unsigned char *metaData) {
    float2 metaData0, metaData1;
    if constexpr (!fp8ToFp8) {
      // Setup the Fp8 config once for the codelet
      if constexpr (std::is_same<SrcType, quarter>::value) {
        setFp8Config(metaData);
      } else {
        setFp8ConfigNegScale(metaData);
      }
    } else {
      // We need to keep changing these so extract the bitfield
      metaData0 = extractMetaData(&metaData[0]);
      metaData1 = extractMetaDataNegScale(&metaData[1]);
    }
    inLineAssemblerCastFp8<const SrcType *, DstType *, true, 1>::loopBody(
        numElems / 8, src, dst, metaData0, metaData1);
    src += numElems & (~7);
    dst += numElems & (~7);
    for (unsigned i = 0; i < (numElems & 7); i++) {
      *dst++ = inLineAssemblerCastFp8<const SrcType *, DstType *, true,
                                      1>::singleCast(src, metaData0, metaData1);
      src++;
    }
  }
};

template <typename SrcType, typename DstType>
struct CastDispatchMultiVertexFp8<SrcType, DstType, true> {
public:
  static constexpr auto fp8ToFp8 = std::is_same<SrcType, quarter>::value &&
                                   std::is_same<DstType, quarter>::value;
  static void compute(unsigned numElems, unsigned wid, const SrcType *src,
                      DstType *dst, const unsigned char *metaData) {

    constexpr unsigned elemsPerLoop = 8;
    float2 metaData0, metaData1;
    if constexpr (!fp8ToFp8) {
      // Setup the Fp8 config once for the codelet
      if constexpr (std::is_same<SrcType, quarter>::value) {
        setFp8Config(metaData);
      } else {
        setFp8ConfigNegScale(metaData);
      }
    } else {
      // We need to keep changing these so extract the bitfield
      metaData0 = extractMetaData(&metaData[0]);
      metaData1 = extractMetaDataNegScale(&metaData[1]);
    }
    inLineAssemblerCastFp8<const SrcType *, DstType *, true,
                           CTXT_WORKERS>::loopBody(divideWork(numElems, 3, wid),
                                                   &src[wid * elemsPerLoop],
                                                   &dst[wid * elemsPerLoop],
                                                   metaData0, metaData1);
    if (wid == CTXT_WORKERS - 1) {
      src += numElems & (~7);
      dst += numElems & (~7);
      for (unsigned i = 0; i < (numElems & 7); i++) {
        *dst++ =
            inLineAssemblerCastFp8<const SrcType *, DstType *, true,
                                   CTXT_WORKERS>::singleCast(src, metaData0,
                                                             metaData1);
        src++;
      }
    }
  }
};

#endif

template <> struct CastDispatch<float, half, true> {
public:
  static void compute(unsigned numElems, const float *src, half *dst) {
    constexpr unsigned elemsPerLoop = 4;

    if (reinterpret_cast<unsigned>(dst) & 4 && numElems >= 2) {
      auto dst2 = reinterpret_cast<half2 *>(&dst[0]);
      *dst2 = {static_cast<half>(src[0]), static_cast<half>(src[1])};
      numElems -= 2;
      src += 2;
      dst += 2;
    }
    inLineAssemblerCast<const float *, half *, false, 1>::loopBody(
        numElems / elemsPerLoop, src, dst);
    if (numElems & 2) {
      auto idx = numElems & (~3);
      auto dst2 = reinterpret_cast<half2 *>(&dst[idx]);
      *dst2 = {static_cast<half>(src[idx]), static_cast<half>(src[idx + 1])};
    }
    if (numElems & 1) {
      auto idx = numElems & (~1);
      write16Aligned32(static_cast<half>(src[idx]),
                       reinterpret_cast<half2 *>(&dst[idx]));
    }
  }
};

template <> struct CastDispatch<half, float, true> {
public:
  static void compute(unsigned numElems, const half *src, float *dst) {
    constexpr unsigned elemsPerLoop = 4;

    inLineAssemblerCast<const half *, float *, false, 1>::loopBody(
        numElems / elemsPerLoop, src, dst);
    if (numElems & 2) {
      auto idx = numElems & (~3);
      auto dst2 = reinterpret_cast<float2 *>(&dst[idx]);
      *dst2 = {static_cast<float>(src[idx]), static_cast<float>(src[idx + 1])};
    }
    if (numElems & 1) {
      auto idx = numElems & (~1);
      dst[idx] = static_cast<float>(src[idx]);
    }
  }
};

template <typename SrcType, typename DstType>
struct CastDispatch<SrcType, DstType, true> {
public:
  static constexpr bool toFpType =
      std::is_same<DstType, float>::value || std::is_same<DstType, half>::value;

  static void compute(unsigned numElems, const SrcType *src, DstType *dst) {
    constexpr unsigned elemsPerLoop = 4;
    float2 limits;
    if constexpr (std::is_same<DstType, unsigned char>::value) {
      limits = {0.0f, 255.0f};
    } else {
      limits = {-128.0f, 127.0f};
    }

    inLineAssemblerCast<const SrcType *, DstType *, toFpType, 1>::loopBody(
        numElems / elemsPerLoop, src, dst, limits);
    dst += numElems & ~3;
    src += numElems & ~3;
    for (unsigned i = 0; i < (numElems & 3); i++) {
      *dst++ = inLineAssemblerCast<const SrcType *, DstType *, toFpType,
                                   1>::singleCast(*src++, limits);
    }
  }
};

template <> struct CastDispatchMultiVertex<float, half, true> {
public:
  static void compute(unsigned numElems, unsigned wid, const float *src,
                      half *dst) {
    constexpr unsigned elemsPerLoop = 4;

    if (reinterpret_cast<unsigned>(dst) & 4 && numElems >= 2) {
      if (wid == CTXT_WORKERS - 1) {
        auto dst2 = reinterpret_cast<half2 *>(&dst[0]);
        *dst2 = {static_cast<half>(src[0]), static_cast<half>(src[1])};
      }
      numElems -= 2;
      src += 2;
      dst += 2;
    }

    inLineAssemblerCast<const float *, half *, false, CTXT_WORKERS>::loopBody(
        divideWork(numElems, 2, wid), &src[elemsPerLoop * wid],
        &dst[elemsPerLoop * wid]);

    if (numElems & 3) {
      if (wid == CTXT_WORKERS - 2 && numElems & 2) {
        auto idx = numElems & (~3);
        auto dst2 = reinterpret_cast<half2 *>(&dst[idx]);
        *dst2 = {static_cast<half>(src[idx]), static_cast<half>(src[idx + 1])};
      }
      if (wid == CTXT_WORKERS - 3 && numElems & 1) {
        auto idx = numElems & (~1);
        write16Aligned32(static_cast<half>(src[idx]),
                         reinterpret_cast<half2 *>(&dst[idx]));
      }
    }
  }
};

template <> struct CastDispatchMultiVertex<half, float, true> {
public:
  static void compute(unsigned numElems, unsigned wid, const half *src,
                      float *dst) {
    constexpr unsigned elemsPerLoop = 4;
    inLineAssemblerCast<const half *, float *, false, CTXT_WORKERS>::loopBody(
        divideWork(numElems, 2, wid), &src[elemsPerLoop * wid],
        &dst[elemsPerLoop * wid]);

    if (numElems & 3) {
      if (wid == CTXT_WORKERS - 1 && numElems & 2) {
        auto idx = numElems & (~3);
        auto dst2 = reinterpret_cast<float2 *>(&dst[idx]);
        *dst2 = {static_cast<float>(src[idx]),
                 static_cast<float>(src[idx + 1])};
      }
      if (wid == CTXT_WORKERS - 2 && numElems & 1) {
        auto idx = numElems & (~1);
        dst[idx] = static_cast<float>(src[idx]);
      }
    }
  }
};

template <typename SrcType, typename DstType>
struct CastDispatchMultiVertex<SrcType, DstType, true> {
public:
  static constexpr bool toFpType =
      std::is_same<DstType, float>::value || std::is_same<DstType, half>::value;

  static void compute(unsigned numElems, unsigned wid, const SrcType *src,
                      DstType *dst) {
    constexpr unsigned elemsPerLoop = 4;
    float2 limits;
    if constexpr (std::is_same<DstType, unsigned char>::value) {
      limits = {0.0f, 255.0f};
    } else {
      limits = {-128.0f, 127.0f};
    }

    inLineAssemblerCast<const SrcType *, DstType *, toFpType,
                        CTXT_WORKERS>::loopBody(divideWork(numElems, 2, wid),
                                                &src[elemsPerLoop * wid],
                                                &dst[elemsPerLoop * wid],
                                                limits);
    if (wid == CTXT_WORKERS - 1) {
      dst += numElems & ~3;
      src += numElems & ~3;
      for (unsigned i = 0; i < (numElems & 3); i++) {
        *dst++ = inLineAssemblerCast<const SrcType *, DstType *, toFpType,
                                     1>::singleCast(*src++, limits);
      }
    }
  }
};

#endif

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Cast1DSingleWorker
    : public Vertex {
public:
  Cast1DSingleWorker();

  // Structured binding would be nicer, but it doesn't work here
  constexpr static auto t = getCastParams<SrcType, DstType>();
  constexpr static bool inlineAsm = std::get<0>(t);
  constexpr static unsigned inAlign = std::get<1>(t);
  constexpr static unsigned outAlign = std::get<2>(t);
  constexpr static VectorLayout inLayout = std::get<3>(t);
  constexpr static VectorLayout outLayout = std::get<4>(t);

  Input<Vector<SrcType, inLayout, inAlign>> src;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  const unsigned numElems;

  bool compute() {
    CastDispatch<SrcType, DstType, inlineAsm>::compute(numElems, &src[0],
                                                       &dst[0]);
    return true;
  }
};

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Cast1DSingleWorkerFp8
    : public Vertex {
public:
  Cast1DSingleWorkerFp8();

  // Structured binding would be nicer, but it doesn't work here
  constexpr static auto t = getCastParams<SrcType, DstType>();
  constexpr static bool inlineAsm = true;
  constexpr static unsigned inAlign = 8;
  constexpr static unsigned outAlign = 8;
  constexpr static VectorLayout inLayout = std::get<3>(t);
  constexpr static VectorLayout outLayout = std::get<4>(t);

  Input<Vector<SrcType, inLayout, inAlign>> src;
  Input<Vector<unsigned char, ONE_PTR>> metaData;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  const unsigned numElems;

  bool compute() {
    CastDispatchFp8<SrcType, DstType, inlineAsm>::compute(
        numElems, &src[0], &dst[0], &metaData[0]);
    return true;
  }
};

INSTANTIATE_CAST(Cast1DSingleWorker)
INSTANTIATE_CAST_LONGLONG(Cast1DSingleWorker)

#if __IPU_ARCH_VERSION__ == 21
template class Cast1DSingleWorkerFp8<half, quarter>;
template class Cast1DSingleWorkerFp8<quarter, half>;
template class Cast1DSingleWorkerFp8<char, quarter>;
template class Cast1DSingleWorkerFp8<quarter, char>;
template class Cast1DSingleWorkerFp8<unsigned char, quarter>;
template class Cast1DSingleWorkerFp8<quarter, unsigned char>;
template class Cast1DSingleWorkerFp8<quarter, quarter>;
#endif

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Cast1D
    : public MultiVertex {
public:
  Cast1D();

  constexpr static auto t = getCastParams<SrcType, DstType>();
  constexpr static bool inlineAsm = std::get<0>(t);
  constexpr static unsigned inAlign = std::get<1>(t);
  constexpr static unsigned outAlign = std::get<2>(t);
  constexpr static VectorLayout inLayout = std::get<3>(t);
  constexpr static VectorLayout outLayout = std::get<4>(t);

  Input<Vector<SrcType, inLayout, inAlign>> src;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  const unsigned numElems;

  bool compute(unsigned wid) {

    CastDispatchMultiVertex<SrcType, DstType, inlineAsm>::compute(
        numElems, wid, &src[0], &dst[0]);
    return true;
  }
};

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Cast1DFp8
    : public MultiVertex {
public:
  Cast1DFp8();

  constexpr static auto t = getCastParams<SrcType, DstType>();
  constexpr static bool inlineAsm = true;
  constexpr static unsigned inAlign = 8;
  constexpr static unsigned outAlign = 8;
  constexpr static VectorLayout inLayout = std::get<3>(t);
  constexpr static VectorLayout outLayout = std::get<4>(t);

  Input<Vector<SrcType, inLayout, inAlign>> src;
  Input<Vector<unsigned char, ONE_PTR>> metaData;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  const unsigned numElems;

  bool compute(unsigned wid) {

    CastDispatchMultiVertexFp8<SrcType, DstType, inlineAsm>::compute(
        numElems, wid, &src[0], &dst[0], &metaData[0]);
    return true;
  }
};

INSTANTIATE_CAST(Cast1D)
INSTANTIATE_CAST_LONGLONG(Cast1D)

#if __IPU_ARCH_VERSION__ == 21
template class Cast1DFp8<half, quarter>;
template class Cast1DFp8<quarter, half>;
template class Cast1DFp8<char, quarter>;
template class Cast1DFp8<quarter, char>;
template class Cast1DFp8<unsigned char, quarter>;
template class Cast1DFp8<quarter, unsigned char>;
template class Cast1DFp8<quarter, quarter>;
#endif

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(**src) != elem(**dst)")]] Cast2D
    : public Vertex {
public:
  constexpr static auto t = getCastParams<SrcType, DstType>();
  constexpr static bool inlineAsm = std::get<0>(t);
  constexpr static unsigned inAlign = std::get<1>(t);
  constexpr static unsigned outAlign = std::get<2>(t);

  Vector<Input<Vector<SrcType, ONE_PTR, inAlign>>, ONE_PTR> src;
  Vector<Output<Vector<DstType, SPAN, outAlign>>> dst;

  bool compute() {
    const unsigned limI = dst.size();
    for (unsigned i = 0; i != limI; ++i) {
      CastDispatch<SrcType, DstType, inlineAsm>::compute(
          dst[i].size(), &src[i][0], &dst[i][0]);
    }
    return true;
  }
};

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(**src) != elem(**dst)")]] Cast2DFp8
    : public Vertex {
public:
  constexpr static auto t = getCastParams<SrcType, DstType>();
  constexpr static bool inlineAsm = true;
  constexpr static unsigned inAlign = 8;
  constexpr static unsigned outAlign = 8;

  Vector<Input<Vector<SrcType, ONE_PTR, inAlign>>, ONE_PTR> src;
  Input<Vector<unsigned char, ONE_PTR>> metaData;
  Vector<Output<Vector<DstType, SPAN, outAlign>>> dst;

  bool compute() {
    const unsigned limI = dst.size();
    for (unsigned i = 0; i != limI; ++i) {
      CastDispatchFp8<SrcType, DstType, inlineAsm>::compute(
          dst[i].size(), &src[i][0], &dst[i][0], &metaData[0]);
    }
    return true;
  }
};

INSTANTIATE_CAST(Cast2D)
INSTANTIATE_CAST_LONGLONG(Cast2D)

#if __IPU_ARCH_VERSION__ == 21
template class Cast2DFp8<half, quarter>;
template class Cast2DFp8<quarter, half>;
template class Cast2DFp8<quarter, char>;
template class Cast2DFp8<char, quarter>;
template class Cast2DFp8<quarter, unsigned char>;
template class Cast2DFp8<unsigned char, quarter>;
template class Cast2DFp8<quarter, quarter>;
#endif

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
template class Select<unsigned>;
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
template class BroadcastSelect<unsigned>;
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
template class BroadcastSelectorSelect<unsigned>;
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
template class SelectInPlace<unsigned>;
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
template class BroadcastSelectorSelectInPlace<unsigned>;
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
class Histogram1D : public MultiVertex {
public:
  // SPAN required to support usefully large data size and no alignment
  // constraint
  Input<Vector<InType, SPAN>> data;
  // There will be `limits` +1 histogram entries
  Input<Vector<InType, PTR_ALIGN32, 4>> limits;
  // When splitByLimits==false, this array must be histogramCount * CTXT_WORKERS
  // in size
  Output<Vector<float, PTR_ALIGN32, 4>> histogram;
  unsigned short histogramCount;

  IS_EXTERNAL_CODELET(true);
  bool compute(unsigned wid) {
    if (wid == 0) {
      auto condAbs = [](auto d) {
        return isAbsolute
                   ? static_cast<InType>(std::fabs(static_cast<float>(d)))
                   : d;
      };

      // Structured like the assembler (when split by limits)
      for (unsigned wkrId = 0; wkrId < CTXT_WORKERS; wkrId++) {
        for (unsigned i = wkrId; i < histogramCount; i += CTXT_WORKERS) {
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

template class Histogram1D<float, true, true>;
template class Histogram1D<half, true, true>;
template class Histogram1D<float, false, true>;
template class Histogram1D<half, false, true>;
template class Histogram1D<float, true, false>;
template class Histogram1D<half, true, false>;
template class Histogram1D<float, false, false>;
template class Histogram1D<half, false, false>;

template <typename T>
class ForLoopCounter : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
  static const bool needsAlignWorkers = false;

public:
  InOut<T> count;
  Input<T> limit;
  Output<unsigned> comparisonResult;

  T increment;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    *count += increment;
    *comparisonResult = static_cast<unsigned>(count != limit);
    return true;
  }
};
template class ForLoopCounter<int>;
template class ForLoopCounter<unsigned>;
template class ForLoopCounter<short>;
template class ForLoopCounter<unsigned short>;

// This vertex is used for testing, to ensure that a vector is aligned to
// 8 bytes
template <unsigned ALIGN, typename T> class NopAlignVertex : public Vertex {
public:
  InOut<Vector<T, SPAN, ALIGN>> t;
  bool compute() { return true; }
};
template class NopAlignVertex<8, half>;
template class NopAlignVertex<8, float>;
template class NopAlignVertex<8, int>;
template class NopAlignVertex<8, unsigned int>;
template class NopAlignVertex<8, short>;
template class NopAlignVertex<8, unsigned short>;
template class NopAlignVertex<8, bool>;
template class NopAlignVertex<8, char>;
template class NopAlignVertex<8, unsigned char>;
template class NopAlignVertex<8, signed char>;
template class NopAlignVertex<8, unsigned long long>;
template class NopAlignVertex<8, long long>;
} // namespace popops
