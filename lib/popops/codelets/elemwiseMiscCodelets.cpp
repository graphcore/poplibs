// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

#define NANOO_BIT 20
#define NANOO_MASK (1 << NANOO_BIT)

using namespace poplar;

namespace popops {

#ifdef __IPU__
#include "inlineAssembler.hpp"
#include "inlineAssemblerCast.hpp"
#include <ipu_builtins.h>
#endif

template <typename FPType>
class [[poplar::constraint("elem(**A) != elem(**B)")]] HadamardProd
    : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> A;
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> B;

  void compute() {
    const unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      const unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] *= refIn[j];
      }
    }
  }
};

template class HadamardProd<float>;
template class HadamardProd<half>;

template <typename InType> class Fill : public Vertex {
public:
  InType in;
  Output<Vector<InType>> out;
  IS_EXTERNAL_CODELET(true);

  void compute() {
    for (auto &x : out) {
      x = in;
    }
  }
};

#ifdef __IPU__

static __attribute__((always_inline)) void
fillMisaligned(quarter *out, unsigned size, unsigned charValue4) {

  constexpr unsigned fullWordMask = 0xffffffff;
  constexpr unsigned sizeOfQuarterInBits = 8;
  constexpr unsigned sizeOfFullWordInBits = 32;
  constexpr unsigned quartersPerWord =
      sizeOfFullWordInBits / sizeOfQuarterInBits;
  constexpr unsigned misalignMask = quartersPerWord - 1;

  auto address = reinterpret_cast<unsigned>(out);
  // Misaligned
  auto ptr4 = reinterpret_cast<unsigned *>(address & (~misalignMask));
  const auto misalignment = (address & misalignMask);
  if (misalignment) {
    auto preserved = *ptr4;
    const unsigned misalignedBytes = quartersPerWord - misalignment;

    // clang-format off
    // Form a mask to preserve the part of the word that is not part of the
    // data.  Need to account for size resulting in only a "middle portion" of
    // the word being written.
    // Expected masks, the least significant byte is
    // that found in memory first. 0xff denotes a byte to be preserved, 0x00 to
    // be filled
    // misalignedBytes=3 size>=3  0x000000ff
    // misalignedBytes=2 size>=2  0x0000ffff
    // misalignedBytes=1 size>=1  0x00ffffff
    // misalignedBytes=3 size==1  0xffff00ff
    // misalignedBytes=3 size==2  0xff0000ff
    // misalignedBytes=2 size==1  0xff00ffff
    // clang-format on

    unsigned mask = fullWordMask >> (sizeOfQuarterInBits * misalignedBytes);
    if (size < misalignedBytes) {
      mask |= fullWordMask << ((misalignment + size) * sizeOfQuarterInBits);
    }
    *ptr4++ = (preserved & mask) | (charValue4 & (~mask));
    if (size <= misalignedBytes) {
      return;
    }
    size -= misalignedBytes;
  }

  // Vectors of 4 to avoid subword writes
  for (unsigned i = 0; i < size / quartersPerWord; i++) {
    *ptr4++ = charValue4;
  }
  // Remainder, shifting forms a similar function to the mask when dealing
  // with misalignment
  const auto shifts = sizeOfQuarterInBits * (size & misalignMask);
  if (shifts) {
    auto preserved = (*ptr4) >> shifts;
    *ptr4 =
        (preserved << shifts) | (charValue4 >> (sizeOfFullWordInBits - shifts));
  }
}

template <> class Fill<quarter> : public Vertex {
public:
  // We store a float value as all representable values of type quarter can
  // be represented by float, but not by half.
  float in;
  Output<Vector<quarter>> out;
  IS_EXTERNAL_CODELET(false);

  void compute() {
    // TODO T62077 - when casting, using the intermediate type of half can
    // result in inaccurate results depending on the value of scale.  A direct
    // conversion float to quarter could be created if necessary to improve
    // this.
    auto value =
        toQuarter(static_cast<half>(in), unpackMetadata(out.getMetadata()));
    auto charValue = *(reinterpret_cast<unsigned char *>(&value));
    uchar4 charValue4 = {charValue, charValue, charValue, charValue};
    fillMisaligned(&out[0], out.size(), unsigned(charValue4));
  }
};

#else

template <> class Fill<quarter> : public Vertex {
public:
  // We store a float value as all representable values of type quarter can
  // be represented by float, but not by half.
  float in;
  Output<Vector<quarter>> out;
  IS_EXTERNAL_CODELET(false);

  void compute() {
    // TODO T62077 - when casting, using the intermediate type of half can
    // result in inaccurate results depending on the value of scale.  A direct
    // conversion float to quarter could be created if necessary to improve
    // this.
    const auto value =
        toQuarter(static_cast<half>(in), unpackMetadata(out.getMetadata()));
    for (auto &x : out) {
      x = value;
    }
  }
};

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

  void compute() {
    for (auto &row : out) {
      for (auto &x : row) {
        x = in;
      }
    }
  }
};
#ifdef __IPU__
template <> class Fill2d<quarter> : public Vertex {
public:
  half in;
  Vector<Output<Vector<quarter>>> out;

  IS_EXTERNAL_CODELET(false);

  void compute() {
    auto value = toQuarter(in, unpackMetadata(out.getMetadata()));
    auto charValue = *(reinterpret_cast<unsigned char *>(&value));
    uchar4 charValue4 = {charValue, charValue, charValue, charValue};

    for (auto &row : out) {
      fillMisaligned(&row[0], row.size(), unsigned(charValue4));
    }
  }
};
#else
template <> class Fill2d<quarter> : public Vertex {
public:
  half in;
  Vector<Output<Vector<quarter>>> out;

  IS_EXTERNAL_CODELET(false);

  void compute() {
    const auto value = toQuarter(in, unpackMetadata(out.getMetadata()));
    for (auto &row : out) {
      for (auto &x : row) {
        x = value;
      }
    }
  }
};
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

#define INSTANTIATE_CAST_QUARTER(CastVertexName)                               \
  template class CastVertexName<float, quarter>;                               \
  template class CastVertexName<quarter, float>;                               \
  template class CastVertexName<half, quarter>;                                \
  template class CastVertexName<quarter, half>;                                \
  template class CastVertexName<char, quarter>;                                \
  template class CastVertexName<quarter, char>;                                \
  template class CastVertexName<signed char, quarter>;                         \
  template class CastVertexName<quarter, signed char>;                         \
  template class CastVertexName<unsigned char, quarter>;                       \
  template class CastVertexName<quarter, unsigned char>;                       \
  template class CastVertexName<quarter, quarter>;

// Returns some compile time parameters for Cast vertices, based on SrcType
// and DstType, as a tuple where :
//   element 0 is a boolean     : true if the vertex is implemented in assembly
//   element 1 is an unsigned   : Alignment required for input data (bytes)
//   element 2 is an unsigned   : Alignment required for output (bytes)
//   element 3 is a VectorLayout: Layout for input data pointer
//   element 4 is a VectorLayout: Layout for output pointer
template <typename SrcType, typename DstType>
constexpr std::tuple<bool, unsigned, unsigned> getCastParams() {
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
  bool srcOrDstQuarter = std::is_same<SrcType, quarter>::value ||
                         std::is_same<DstType, quarter>::value;

  bool inlineAsm = floatHalf || halfFloat || floatChar || halfChar ||
                   charFloat || charHalf || srcOrDstQuarter;

  unsigned inAlign = alignof(SrcType);
  unsigned outAlign = alignof(DstType);
  if (halfFloat || srcOrDstQuarter) {
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

  return {inlineAsm, inAlign, outAlign};
}

template <typename SrcType, typename DstType, bool inlineAsm,
          bool metadataRequired>
struct CastDispatch {
public:
  static void compute(unsigned numElems, const SrcType *src, DstType *dst) {}
};

template <typename SrcType, typename DstType, bool inlineAsm,
          bool metadataRequired>
struct CastDispatchMultiVertex {
public:
  static void compute(unsigned numElems, unsigned wid, const SrcType *src,
                      DstType *dst) {}
};

template <typename SrcType, typename DstType, bool inlineAsm>
struct CastDispatch<SrcType, DstType, inlineAsm, false> {
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
      *dst++ = static_cast<DstType>(*src++);
    }
  }
};

template <typename SrcType, typename DstType, bool inlineAsm>
struct CastDispatchMultiVertex<SrcType, DstType, inlineAsm, false> {
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

#if !defined(__IPU__) || (__IPU_ARCH_VERSION__ <= 2)
// For non IPU and any ARCH_VERSION compile using cast functions
// for quarter types

template <typename SrcType, typename DstType>
DstType cast(SrcType src, quarter_metadata metadataSrc,
             quarter_metadata metadataDst, bool nanoo) {
  if constexpr (std::is_same<SrcType, quarter>::value &&
                std::is_same<DstType, quarter>::value) {
    return toQuarter(toFloat(src, metadataSrc), metadataDst, nanoo);
  } else if constexpr (std::is_same<SrcType, quarter>::value) {
    return static_cast<DstType>(toFloat(src, metadataSrc));
  } else if constexpr (std::is_same<DstType, quarter>::value) {
    return toQuarter(static_cast<float>(src), metadataDst, nanoo);
  }
}

// Get the state of the nanoo bit in the FP_CTL register.
// IpuModel targets don't have this so we have to pick a behaviour for the cast
// which is nanoo = true;
// Return in an unsigned as managing 8 bit types can be slow
static __attribute__((always_inline)) unsigned getNanoo() {
#ifdef __IPU__
  return __builtin_ipu_uget(CSR_W_FP_CTL__INDEX & CSR_UPPER_MASK) & NANOO_MASK;
#else
  return 1u;
#endif
}

template <typename SrcType, typename DstType>
struct CastDispatch<SrcType, DstType, true, true> {
public:
  static void compute(unsigned numElems, const SrcType *src, DstType *dst,
                      const MetadataType *metadataSrc,
                      const MetadataType *metadataDst) {
    auto nanoo = getNanoo();
    quarter_metadata metadata0, metadata1;
    if constexpr (std::is_same<SrcType, quarter>::value) {
      metadata0 = unpackMetadata(metadataSrc);
    }
    if constexpr (std::is_same<DstType, quarter>::value) {
      metadata1 = unpackMetadata(metadataDst);
    }
    constexpr unsigned elemsPerLoop = 4;
    for (unsigned i = 0; i < numElems / elemsPerLoop; ++i) {
      *dst++ = cast<SrcType, DstType>(*src++, metadata0, metadata1, nanoo);
      *dst++ = cast<SrcType, DstType>(*src++, metadata0, metadata1, nanoo);
      *dst++ = cast<SrcType, DstType>(*src++, metadata0, metadata1, nanoo);
      *dst++ = cast<SrcType, DstType>(*src++, metadata0, metadata1, nanoo);
    }
    for (unsigned i = 0; i < (numElems & 3); i++) {
      *dst++ = cast<SrcType, DstType>(*src++, metadata0, metadata1, nanoo);
    }
  }
};

template <typename SrcType, typename DstType>
struct CastDispatchMultiVertex<SrcType, DstType, true, true> {
public:
  static void compute(unsigned numElems, unsigned wid, const SrcType *src,
                      DstType *dst, const MetadataType *metadataSrc,
                      const MetadataType *metadataDst) {
    constexpr unsigned elemsPerLoop = 4;
    const SrcType *loopSrc = &src[wid * elemsPerLoop];
    DstType *loopDst = &dst[wid * elemsPerLoop];
    auto nanoo = getNanoo();
    quarter_metadata metadata0, metadata1;
    if constexpr (std::is_same<SrcType, quarter>::value) {
      metadata0 = unpackMetadata(metadataSrc);
    }
    if constexpr (std::is_same<DstType, quarter>::value) {
      metadata1 = unpackMetadata(metadataDst);
    }
    for (unsigned i = 0; i < divideWork(numElems, 2, wid); ++i) {
      *loopDst++ =
          cast<SrcType, DstType>(*loopSrc++, metadata0, metadata1, nanoo);
      *loopDst++ =
          cast<SrcType, DstType>(*loopSrc++, metadata0, metadata1, nanoo);
      *loopDst++ =
          cast<SrcType, DstType>(*loopSrc++, metadata0, metadata1, nanoo);
      *loopDst++ =
          cast<SrcType, DstType>(*loopSrc++, metadata0, metadata1, nanoo);
      loopDst += elemsPerLoop * CTXT_WORKERS - elemsPerLoop;
      loopSrc += elemsPerLoop * CTXT_WORKERS - elemsPerLoop;
    }
    if (wid == CTXT_WORKERS - 1 && numElems & 3) {
      const unsigned offset = numElems & ~3;
      for (unsigned i = 0; i < (numElems & 3); i++) {
        dst[offset + i] = cast<SrcType, DstType>(src[offset + i], metadata0,
                                                 metadata1, nanoo);
      }
    }
  }
};

#endif

#ifdef __IPU__

#if __IPU_ARCH_VERSION__ > 2
// For IPU and ARCH VERSION > 2 compile inline assembler implementations
// for quarter types
template <typename SrcType, typename DstType>
struct CastDispatch<SrcType, DstType, true, true> {
  using SrcTypeInternal =
      std::conditional_t<std::is_same<SrcType, signed char>::value, char,
                         SrcType>;
  using DstTypeInternal =
      std::conditional_t<std::is_same<DstType, signed char>::value, char,
                         DstType>;

public:
  static constexpr auto fp8ToFp8 = std::is_same<SrcType, quarter>::value &&
                                   std::is_same<DstType, quarter>::value;
  static constexpr auto floatToFp8 =
      std::is_same<SrcType, float>() && std::is_same<DstType, quarter>();
  static void compute(unsigned numElems, const SrcType *src, DstType *dst,
                      const MetadataType *metadataSrc,
                      const MetadataType *metadataDst) {
    constexpr unsigned elemsPerLoopDivisorShift =
        (__IPU_ARCH_VERSION__ >= 21 && floatToFp8) ? 2 : 3;
    constexpr unsigned elemsPerLoop = 1 << elemsPerLoopDivisorShift;
    constexpr unsigned elemsPerLoopM1 = elemsPerLoop - 1;

    float2 metadata0, metadata1;
    if constexpr (!fp8ToFp8) {
      // Setup the Fp8 config once for the codelet
      if constexpr (std::is_same<SrcType, quarter>::value) {
        if constexpr (std::is_same<DstType, float>::value) {
          // Clear scale and retain the format bit (bit 7)
          setFp8Config(*metadataSrc & 0x80);
          metadata0[1] = __builtin_ipu_exp2(getScaleFloat(*metadataSrc));
        } else {
          setFp8Config(*metadataSrc);
        }
      } else {
        if constexpr (std::is_same<SrcType, float>::value) {
          // Clear scale and retain the format bit (bit 7)
          setFp8Config(*metadataDst & 0x80);
          metadata1[1] = __builtin_ipu_exp2(-getScaleFloat(*metadataDst));
        } else {
          setFp8ConfigNegScale(*metadataDst);
        }
      }
    } else {
      // We need to keep changing these so extract the bitfield
      metadata0 = extractMetadata(metadataSrc);
      metadata1 = extractMetadataNegScale(metadataDst);
    }
    auto srcInternal = reinterpret_cast<const SrcTypeInternal *>(src);
    auto dstInternal = reinterpret_cast<DstTypeInternal *>(dst);
    inLineAssemblerCast<const SrcTypeInternal *, DstTypeInternal *, true,
                        1>::loopBody(numElems / elemsPerLoop, srcInternal,
                                     dstInternal, metadata0, metadata1);
    srcInternal += numElems & (~elemsPerLoopM1);
    dstInternal += numElems & (~elemsPerLoopM1);
    for (unsigned i = 0; i < (numElems & elemsPerLoopM1); i++) {
      *dstInternal++ =
          inLineAssemblerCast<const SrcTypeInternal *, DstTypeInternal *, true,
                              1>::singleCast(srcInternal, metadata0, metadata1);
      srcInternal++;
    }
  }
};

template <typename SrcType, typename DstType>
struct CastDispatchMultiVertex<SrcType, DstType, true, true> {
  using SrcTypeInternal =
      std::conditional_t<std::is_same<SrcType, signed char>::value, char,
                         SrcType>;
  using DstTypeInternal =
      std::conditional_t<std::is_same<DstType, signed char>::value, char,
                         DstType>;

public:
  static constexpr auto fp8ToFp8 = std::is_same<SrcType, quarter>::value &&
                                   std::is_same<DstType, quarter>::value;
  static constexpr auto floatToFp8 =
      std::is_same<SrcType, float>() && std::is_same<DstType, quarter>();
  static void compute(unsigned numElems, unsigned wid, const SrcType *src,
                      DstType *dst, const MetadataType *metadataSrc,
                      const MetadataType *metadataDst) {

    constexpr unsigned elemsPerLoopDivisorShift =
        (__IPU_ARCH_VERSION__ >= 21 && floatToFp8) ? 2 : 3;
    constexpr unsigned elemsPerLoop = 1 << elemsPerLoopDivisorShift;
    constexpr unsigned elemsPerLoopM1 = elemsPerLoop - 1;

    float2 metadata0, metadata1;
    if constexpr (!fp8ToFp8) {
      // Setup the Fp8 config once for the codelet
      if constexpr (std::is_same<SrcType, quarter>::value) {
        if constexpr (std::is_same<DstType, float>::value) {
          // Clear scale and retain the format bit (bit 7)
          setFp8Config(*metadataSrc & 0x80);
          metadata0[1] = __builtin_ipu_exp2(getScaleFloat(*metadataSrc));
        } else {
          setFp8Config(*metadataSrc);
        }
      } else {
        if constexpr (std::is_same<SrcType, float>::value) {
          // Clear scale and retain the format bit (bit 7)
          setFp8Config(*metadataDst & 0x80);
          metadata1[1] = __builtin_ipu_exp2(-getScaleFloat(*metadataDst));
        } else {
          setFp8ConfigNegScale(*metadataDst);
        }
      }
    } else {
      // We need to keep changing these so extract the bitfield
      metadata0 = extractMetadata(metadataSrc);
      metadata1 = extractMetadataNegScale(metadataDst);
    }
    auto srcInternal = reinterpret_cast<const SrcTypeInternal *>(src);
    auto dstInternal = reinterpret_cast<DstTypeInternal *>(dst);
    inLineAssemblerCast<
        const SrcTypeInternal *, DstTypeInternal *, true,
        CTXT_WORKERS>::loopBody(divideWork(numElems, elemsPerLoopDivisorShift,
                                           wid),
                                &srcInternal[wid * elemsPerLoop],
                                &dstInternal[wid * elemsPerLoop], metadata0,
                                metadata1);
    if (wid == CTXT_WORKERS - 1) {
      srcInternal += numElems & (~elemsPerLoopM1);
      dstInternal += numElems & (~elemsPerLoopM1);
      for (unsigned i = 0; i < (numElems & elemsPerLoopM1); i++) {
        *dstInternal++ =
            inLineAssemblerCast<const SrcTypeInternal *, DstTypeInternal *,
                                true, CTXT_WORKERS>::singleCast(srcInternal,
                                                                metadata0,
                                                                metadata1);
        srcInternal++;
      }
    }
  }
};

#else

template <> struct CastDispatch<quarter, half, true, true> {
public:
  static void compute(unsigned numElems, const quarter *src, half *dst,
                      const MetadataType *metadataSrc,
                      const MetadataType *metadataDst) {
    float2 unpackedMetadata;

    auto srcInternal = reinterpret_cast<const quarter *>(src);
    auto dstInternal = reinterpret_cast<half *>(dst);
    extractMetadata(metadataSrc, &unpackedMetadata);

    constexpr unsigned vectorShift = 2;
    constexpr unsigned vectorSize = 4;
    constexpr unsigned remainderMask = 3;
    unsigned vectors = numElems >> vectorShift;
    unsigned remainder = numElems & remainderMask;
    vectors -= (remainder == 0);

    half4 lastResults;
    if (unpackedMetadata[0] == 0.f) {
      lastResults =
          inLineAssemblerCast<const quarter *, half *, true, 1>::loopCast152(
              vectors, srcInternal, dstInternal, &unpackedMetadata);
    } else {
      lastResults =
          inLineAssemblerCast<const quarter *, half *, true, 1>::loopCast143(
              vectors, srcInternal, dstInternal, &unpackedMetadata);
    }

    const auto remainderOffset = numElems - remainder;
    switch (remainder) {
    case 0: // store the last 4 halves due to the lack of remainder
      *((half4 *)(dstInternal + numElems - vectorSize)) = lastResults;
      break;
    case 3:
      *(dstInternal + remainderOffset + 2) = lastResults[2];
    case 2: {
      half2 lastResultsV2 = {lastResults[0], lastResults[1]};
      *((half2 *)(dstInternal + remainderOffset)) = lastResultsV2;
      break;
    }
    default:
      *(dstInternal + remainderOffset) = lastResults[0];
      break;
    }
  }
};

template <> struct CastDispatchMultiVertex<quarter, half, true, true> {

  static void compute(unsigned numElems, unsigned wid, const quarter *src,
                      half *dst, const MetadataType *metadataSrc,
                      const MetadataType *metadataDst) {
    float2 unpackedMetadata;

    constexpr unsigned vectorSize = 4;
    constexpr unsigned vectorShift = 2;
    constexpr unsigned remainderMask = 3;
    const unsigned remainder = numElems & remainderMask;
    numElems += remainder ? vectorSize - remainder : 0;
    const int vectorsPerWorker = divideWork(numElems, vectorShift, wid) - 1;
    const unsigned workerOffset = wid * vectorSize;

    constexpr unsigned stride = vectorSize * CTXT_WORKERS;
    const auto lastVectorOffset = workerOffset + stride * vectorsPerWorker;

    auto srcInternal = reinterpret_cast<const quarter *>(src);
    auto dstInternal = reinterpret_cast<half *>(dst);
    extractMetadata(metadataSrc, &unpackedMetadata);

    if (vectorsPerWorker < 0)
      return;

    half4 lastResults;
    if (unpackedMetadata[0] == 0.f) {
      lastResults =
          inLineAssemblerCast<const quarter *, half *, true,
                              CTXT_WORKERS>::loopCast152(vectorsPerWorker,
                                                         srcInternal +
                                                             workerOffset,
                                                         dstInternal +
                                                             workerOffset,
                                                         &unpackedMetadata);
    } else {
      lastResults =
          inLineAssemblerCast<const quarter *, half *, true,
                              CTXT_WORKERS>::loopCast143(vectorsPerWorker,
                                                         srcInternal +
                                                             workerOffset,
                                                         dstInternal +
                                                             workerOffset,
                                                         &unpackedMetadata);
    }

    const auto remainderVectorOffset = numElems - 4;
    const auto tail = remainder && lastVectorOffset == remainderVectorOffset
                          ? remainder
                          : vectorSize;
    switch (tail) {
    case 4:
      *((half4 *)(dstInternal + lastVectorOffset)) = lastResults;
      break;
    case 3:
      *(dstInternal + lastVectorOffset + 2) = lastResults[2];
    case 2: {
      half2 lastResultsV2 = {lastResults[0], lastResults[1]};
      *((half2 *)(dstInternal + lastVectorOffset)) = lastResultsV2;
      break;
    }
    default:
      *(dstInternal + lastVectorOffset) = lastResults[0];
      break;
    }
  }
};

// TODO: Enable after T71257 fix
#if 0
template <> struct CastDispatch<half, quarter, true, true> {
public:
  static void compute(unsigned numElems, const half *src, quarter *dst,
                      const MetadataType *metadataSrc,
                      const MetadataType *metadataDst) {
    float2 unpackedMetadata;

    auto srcInternal = reinterpret_cast<const half *>(src);
    auto dstInternal = reinterpret_cast<quarter *>(dst);
    extractMetadata(metadataDst, &unpackedMetadata);

    constexpr unsigned vectorShift = 2;
    constexpr unsigned vectorSize = 4;
    constexpr unsigned remainderMask = 3;
    unsigned vectors = numElems >> vectorShift;
    unsigned remainder = numElems & remainderMask;
    vectors -= (remainder == 0);

    unsigned lastResults;
    if (unpackedMetadata[0] == 0.f) {
      lastResults =
          inLineAssemblerCast<const half *, quarter *, true, 1>::loopCast152(
              vectors, srcInternal, dstInternal, &unpackedMetadata);
    } else {
      lastResults =
          inLineAssemblerCast<const half *, quarter *, true, 1>::loopCast143(
              vectors, srcInternal, dstInternal, &unpackedMetadata);
    }

    const auto remainderOffset = numElems - remainder;
    switch (remainder) {
    case 0: // store the last 4 halves due to the lack of remainder
      *((unsigned *)(dstInternal + numElems - vectorSize)) = lastResults;
      break;
    case 3: {
      quarter val = *((quarter *)(&lastResults) + 2);
      *(dstInternal + remainderOffset + 2) = val;
    }
    case 2: {
      unsigned short val = *((short *)(&lastResults));
      *((unsigned short *)(dstInternal + remainderOffset)) = val;
      break;
    }
    default: {
      quarter val = *((quarter *)(&lastResults));
      *(dstInternal + remainderOffset) = val;
      break;
    }
    }
  }
};

template <> struct CastDispatchMultiVertex<half, quarter, true, true> {

  static void compute(unsigned numElems, unsigned wid, const half *src,
                      quarter *dst, const MetadataType *metadataSrc,
                      const MetadataType *metadataDst) {
    float2 unpackedMetadata;

    constexpr unsigned vectorSize = 4;
    constexpr unsigned vectorShift = 2;
    constexpr unsigned remainderMask = 3;
    const unsigned remainder = numElems & remainderMask;
    numElems += remainder ? vectorSize - remainder : 0;
    const int vectorsPerWorker = divideWork(numElems, vectorShift, wid) - 1;
    const unsigned workerOffset = wid * vectorSize;

    constexpr unsigned stride = vectorSize * CTXT_WORKERS;
    const auto lastVectorOffset = workerOffset + stride * vectorsPerWorker;

    auto srcInternal = reinterpret_cast<const half *>(src);
    auto dstInternal = reinterpret_cast<quarter *>(dst);
    extractMetadata(metadataDst, &unpackedMetadata);

    if (vectorsPerWorker < 0)
      return;

    unsigned lastResults;
    if (unpackedMetadata[0] == 0.f) {
      lastResults =
          inLineAssemblerCast<const half *, quarter *, true,
                              CTXT_WORKERS>::loopCast152(vectorsPerWorker,
                                                         srcInternal +
                                                             workerOffset,
                                                         dstInternal +
                                                             workerOffset,
                                                         &unpackedMetadata);
    } else {
      lastResults =
          inLineAssemblerCast<const half *, quarter *, true,
                              CTXT_WORKERS>::loopCast143(vectorsPerWorker,
                                                         srcInternal +
                                                             workerOffset,
                                                         dstInternal +
                                                             workerOffset,
                                                         &unpackedMetadata);
    }

    const auto remainderVectorOffset = numElems - 4;
    const auto tail = remainder && lastVectorOffset == remainderVectorOffset
                          ? remainder
                          : vectorSize;

    switch (tail) {
    case 4:
      *((unsigned *)(dstInternal + lastVectorOffset)) = lastResults;
      break;
    case 3: {
      quarter val = *((quarter *)(&lastResults) + 2);
      *(dstInternal + lastVectorOffset + 2) = val;
    }
    case 2: {
      unsigned short val = *((short *)(&lastResults));
      *((unsigned short *)(dstInternal + lastVectorOffset)) = val;
      break;
    }
    default: {
      quarter val = *((quarter *)(&lastResults));
      *(dstInternal + lastVectorOffset) = val;
      break;
    }
    }
  }
};
#endif
#endif // __IPU_ARCH_VERSION__ > 2

template <> struct CastDispatch<float, half, true, false> {
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

template <> struct CastDispatch<half, float, true, false> {
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
struct CastDispatch<SrcType, DstType, true, false> {
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

template <> struct CastDispatchMultiVertex<float, half, true, false> {
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

template <> struct CastDispatchMultiVertex<half, float, true, false> {
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
struct CastDispatchMultiVertex<SrcType, DstType, true, false> {
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

#endif // __IPU__

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

  Input<Vector<SrcType, ONE_PTR, inAlign>> src;
  Output<Vector<DstType, ONE_PTR, outAlign>> dst;
  const unsigned numElems;

  void compute() {
    if constexpr (std::is_same<SrcType, quarter>::value ||
                  std::is_same<DstType, quarter>::value) {
      const MetadataType *metadataSrc, *metadataDst;
      if constexpr (std::is_same<SrcType, quarter>::value) {
        metadataSrc = src.getMetadata();
      }
      if constexpr (std::is_same<DstType, quarter>::value) {
        metadataDst = dst.getMetadata();
      }
      CastDispatch<SrcType, DstType, inlineAsm, true>::compute(
          numElems, &src[0], &dst[0], metadataSrc, metadataDst);
    } else {
      CastDispatch<SrcType, DstType, inlineAsm, false>::compute(
          numElems, &src[0], &dst[0]);
    }
  }
};

INSTANTIATE_CAST(Cast1DSingleWorker)
INSTANTIATE_CAST_LONGLONG(Cast1DSingleWorker)
INSTANTIATE_CAST_QUARTER(Cast1DSingleWorker)

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Cast1D
    : public MultiVertex {
public:
  Cast1D();

  constexpr static auto t = getCastParams<SrcType, DstType>();
  constexpr static bool inlineAsm = std::get<0>(t);
  constexpr static unsigned inAlign = std::get<1>(t);
  constexpr static unsigned outAlign = std::get<2>(t);

  Input<Vector<SrcType, ONE_PTR, inAlign>> src;
  Output<Vector<DstType, ONE_PTR, outAlign>> dst;
  const unsigned numElems;

  void compute(unsigned wid) {
    if constexpr (std::is_same<SrcType, quarter>::value ||
                  std::is_same<DstType, quarter>::value) {
      const MetadataType *metadataSrc, *metadataDst;
      if constexpr (std::is_same<SrcType, quarter>::value) {
        metadataSrc = src.getMetadata();
      }
      if constexpr (std::is_same<DstType, quarter>::value) {
        metadataDst = dst.getMetadata();
      }
      CastDispatchMultiVertex<SrcType, DstType, inlineAsm, true>::compute(
          numElems, wid, &src[0], &dst[0], metadataSrc, metadataDst);

    } else {
      CastDispatchMultiVertex<SrcType, DstType, inlineAsm, false>::compute(
          numElems, wid, &src[0], &dst[0]);
    }
  }
};

INSTANTIATE_CAST(Cast1D)
INSTANTIATE_CAST_LONGLONG(Cast1D)
INSTANTIATE_CAST_QUARTER(Cast1D)

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

  void compute() {
    const unsigned limI = dst.size();
    const MetadataType *metadataSrc, *metadataDst;
    if constexpr (std::is_same<SrcType, quarter>::value ||
                  std::is_same<DstType, quarter>::value) {
      if constexpr (std::is_same<SrcType, quarter>::value) {
        metadataSrc = src.getMetadata();
      }
      if constexpr (std::is_same<DstType, quarter>::value) {
        metadataDst = dst.getMetadata();
      }
      const unsigned limI = dst.size();
      for (unsigned i = 0; i != limI; ++i) {
        CastDispatch<SrcType, DstType, inlineAsm, true>::compute(
            dst[i].size(), &src[i][0], &dst[i][0], metadataSrc, metadataDst);
      }
    } else {
      for (unsigned i = 0; i != limI; ++i) {
        CastDispatch<SrcType, DstType, inlineAsm, false>::compute(
            dst[i].size(), &src[i][0], &dst[i][0]);
      }
    }
  }
};

INSTANTIATE_CAST(Cast2D)
INSTANTIATE_CAST_LONGLONG(Cast2D)
INSTANTIATE_CAST_QUARTER(Cast2D)

template <typename InType> class Clamp : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2; // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3; // upper bound
  Vector<Output<Vector<InType>>> out;

  static const bool ext =
      std::is_same<InType, float>::value || std::is_same<InType, half>::value;
  IS_EXTERNAL_CODELET(ext);

  void compute() {
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
  void compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in3[i][j] ? in1[i][j] : in2[i][j];
      }
    }
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

  void compute() {
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

  void compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in3[i][j] ? in1 : in2;
      }
    }
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
  void compute() {
    const auto in = in3 ? in1 : in2;
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned j = 0; j < out[i].size(); ++j) {
        out[i][j] = in[i][j];
      }
    }
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

  void compute() {
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

  void compute() {
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

  void compute() {
    for (unsigned i = 0; i != in1Out.size(); ++i) {
      for (unsigned j = 0; j != in1Out[i].size(); ++j) {
        in1Out[i][j] = in3[i][j] ? in1Out[i][j] : in2[i][j];
      }
    }
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
  void compute() {
    if (in3 == false) {
      for (unsigned i = 0; i != in1Out.size(); ++i) {
        for (unsigned j = 0; j != in1Out[i].size(); ++j) {
          in1Out[i][j] = in2[i][j];
        }
      }
    }
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
  void compute() {
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
  void compute(unsigned wid) {
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
  void compute() {
    *count += increment;
    *comparisonResult = static_cast<unsigned>(count != limit);
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
  void compute() {}
};
template class NopAlignVertex<8, quarter>;
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
