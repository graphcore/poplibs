// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_MatMulTensorMetaData_hpp
#define popsparse_MatMulTensorMetaData_hpp

#include "FullyConnectedTensorMetaData.hpp"
#include "TensorMetaDataBase.hpp"
#include "popsparse/MatMulParams.hpp"

namespace popsparse {
namespace dynamic {

class MatMulTensorMetaData : public poputil::TensorMetaDataBase {
public:
  FullyConnectedTensorMetaData fc;
  MatMulParams mmParams;
  MatMulOptions mmOptions;
  MatMulTensorMetaData(FullyConnectedTensorMetaData fc, MatMulParams mmParams,
                       MatMulOptions mmOptions)
      : fc(std::move(fc)), mmParams(std::move(mmParams)),
        mmOptions(std::move(mmOptions)) {}
  virtual ~MatMulTensorMetaData() {}
  virtual std::unique_ptr<poputil::TensorMetaDataBase>
  clone() const override final {
    return std::make_unique<MatMulTensorMetaData>(fc, mmParams, mmOptions);
  }
};

} // end namespace dynamic
} // end namespace popsparse

#endif // popsparse_FullyConnectedTensorMetaData_hpp
