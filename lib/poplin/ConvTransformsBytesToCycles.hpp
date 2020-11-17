// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poplin_internal_ConvTransformsBytesToCycles_hpp
#define poplin_internal_ConvTransformsBytesToCycles_hpp

#include "ConvPlan.hpp"
#include <unordered_map>

namespace poplin {

typedef int64_t (*convExpr)(uint64_t);

template <typename T> struct ConvTransformsHasher {
  std::size_t operator()(const T &it) const;
};

using ConvTransformsMap =
    std::unordered_map<ConvTransform, convExpr,
                       ConvTransformsHasher<ConvTransform>>;

extern const ConvTransformsMap conversionTable;

} // namespace poplin

#endif // poplin_internal_ConvTransformsBytesToCycles_hpp
