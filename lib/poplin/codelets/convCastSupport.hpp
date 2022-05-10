// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/QuarterFloat.hpp>

#ifndef _conv_cast_support_hpp_
#define _conv_cast_support_hpp_

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
