// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popsparse_SparseTensor_hpp
#define popsparse_SparseTensor_hpp

#include <poplar/Tensor.hpp>

namespace popsparse {
namespace dynamic {

/// Representation of a sparse tensor.
class SparseTensor {
  /// Tensor containing positional sparsity information.
  poplar::Tensor metaInfo;
  /// Tensor contains non zero values.
  poplar::Tensor nzValues;

public:
  SparseTensor() = default;

  SparseTensor(const SparseTensor &t) = default;

  SparseTensor(const poplar::Tensor &metaInfo, const poplar::Tensor &nzValues)
      : metaInfo(metaInfo), nzValues(nzValues) {}

  poplar::Tensor getMetaInfoTensor() const { return metaInfo; }

  poplar::Tensor getNzValuesTensor() const { return nzValues; }
};

} // end namespace dynamic
} // end namespace popsparse

#endif // popsparse_SparseTensor_hpp
