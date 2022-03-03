// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/QuarterFloat.hpp>

#ifndef _conv_cast_support_hpp_
#define _conv_cast_support_hpp_

static quarter_metadata unpackMetadata(const MetadataType *in) {
  quarter_metadata out;
  constexpr auto formatBit = 7;
  constexpr auto formatBitMask = (1 << formatBit);
  constexpr auto scaleSignBit = 5;
  constexpr auto scaleSignBitMask = (1 << scaleSignBit);
  constexpr auto scaleMask = (1 << (scaleSignBit + 1)) - 1;
  constexpr auto scaleSignExtendMask = 0xff ^ scaleMask;

  out.fmt =
      *in & formatBitMask ? quarter_metadata::f143 : quarter_metadata::f152;
  out.scale = static_cast<char>(*in & scaleMask);
  if (out.scale & scaleSignBitMask) {
    out.scale |= scaleSignExtendMask;
  }
  return out;
}

template <typename FPType, typename AccumType>
static AccumType promoteType(FPType in, quarter_metadata metadata) {
#ifndef __IPU__
  if constexpr (std::is_same<FPType, quarter>::value) {
    AccumType result;
    return poplar::toHalf(in, metadata);
  } else {
    return AccumType(in);
  }
#else
  if constexpr (std::is_same<FPType, quarter>::value) {
    AccumType result;
    // Currently unused
    return result;
  } else {
    return AccumType(in);
  }
#endif
}

#endif // _conv_cast_support_hpp_
