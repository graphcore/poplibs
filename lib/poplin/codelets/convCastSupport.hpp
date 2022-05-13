// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/QuarterFloat.hpp>

#ifndef _conv_cast_support_hpp_
#define _conv_cast_support_hpp_

template <typename FPType, typename AccumType>
static AccumType promoteType(FPType in, quarter_metadata metadata) {
#ifndef __IPU__
  if constexpr (std::is_same_v<FPType, quarter>) {
    AccumType result;
    return poplar::toHalf(in, metadata);
  } else {
    return AccumType(in);
  }
#else
  if constexpr (std::is_same_v<FPType, quarter>) {
#if __IPU_ARCH_VERSION__ >= 21
    return AccumType(poplar::toHalf(in, metadata));
#else
    // Currently unused
    AccumType result;
    return result;
#endif
  } else {
    return AccumType(in);
  }
#endif
}

#endif // _conv_cast_support_hpp_
