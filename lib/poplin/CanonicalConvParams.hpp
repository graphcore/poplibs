// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef poplin_CanonicalConvParams_hpp
#define poplin_CanonicalConvParams_hpp

#include "poplin/ConvParams.hpp"
#include <boost/optional.hpp>

namespace poplin {

class CanonicalConvParams {
  boost::optional<ConvParams> params;

public:
  CanonicalConvParams(const ConvParams &params_)
      : params{params_.canonicalize()} {
    assert(params == params_.canonicalize() &&
           "canonicalizeParams is not idempotent");
  }

  const ConvParams *operator->() const { return &params.get(); }

  friend bool operator==(const CanonicalConvParams &a,
                         const CanonicalConvParams &b) {
    return a.params == b.params;
  }

  friend bool operator<(const CanonicalConvParams &a,
                        const CanonicalConvParams &b) {
    return a.params < b.params;
  }

  const ConvParams &getParams() const { return params.get(); }

  ConvParams &&releaseParams() { return std::move(params.get()); }
};

} // namespace poplin

#endif // poplin_CanonicalConvParams_hpp
