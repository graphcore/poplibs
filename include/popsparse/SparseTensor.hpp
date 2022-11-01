// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Basic representation of a sparse tensor.
 */
#ifndef popsparse_SparseTensor_hpp
#define popsparse_SparseTensor_hpp

#include <poplar/Tensor.hpp>

#include <poputil/TensorMetaData.hpp>

namespace popsparse {
namespace dynamic {

/// Representation of a sparse tensor.
class SparseTensor {
  /// Tensor containing positional sparsity information.
  poplar::Tensor metaInfo;
  /// Tensor contains non zero values.
  poplar::Tensor nzValues;

  /// Meta-data for this tensor object.
  poputil::TensorMetaData opMetaData;

public:
  SparseTensor() = default;

  SparseTensor(const SparseTensor &t) = default;

  SparseTensor(const poplar::Tensor &metaInfo, const poplar::Tensor &nzValues,
               const poputil::TensorMetaData &opMetaData = {})
      : metaInfo(metaInfo), nzValues(nzValues), opMetaData(opMetaData) {}

  const poplar::Tensor &getMetaInfoTensor() const { return metaInfo; }

  const poplar::Tensor &getNzValuesTensor() const { return nzValues; }

  const poputil::TensorMetaData &getOpMetaData() const { return opMetaData; }
};

} // end namespace dynamic

namespace static_ {

/// Representation of a sparse tensor.
class SparseTensor {
  /// Tensor contains non zero values.
  poplar::Tensor nzValues;

  /// Meta-data for this tensor object.
  poputil::TensorMetaData opMetaData;

public:
  SparseTensor() = default;

  SparseTensor(const SparseTensor &t) = default;

  SparseTensor(const poplar::Tensor &nzValues,
               const poputil::TensorMetaData &opMetaData = {})
      : nzValues(nzValues), opMetaData(opMetaData) {}

  const poplar::Tensor &getNzValuesTensor() const { return nzValues; }
  const poputil::TensorMetaData &getOpMetaData() const { return opMetaData; }
};

} // end namespace static_

} // end namespace popsparse

#endif // popsparse_SparseTensor_hpp
