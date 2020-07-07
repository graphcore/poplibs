// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_FullyConnectedTensorMetaData_hpp
#define popsparse_FullyConnectedTensorMetaData_hpp

#include "PlanningCacheImpl.hpp"

#include "TensorMetaDataBase.hpp"

namespace popsparse {
namespace dynamic {

/// TensorMetaData for sparse tensors created for the fully connected layer
class FullyConnectedTensorMetaData : public poputil::TensorMetaDataBase {
public:
  // In order to identify the fully connected layer this tensor was
  // created for we just store the planning key which necessarily
  // uniquely identifies this operation.
  PlanningCacheImpl::Key planningKey;
  FullyConnectedTensorMetaData(PlanningCacheImpl::Key planningKey)
      : planningKey(std::move(planningKey)) {}
  FullyConnectedTensorMetaData(const FullyConnectedParams &params,
                               const fullyconnected::Options &options)
      : planningKey(params, options) {}
  virtual ~FullyConnectedTensorMetaData() {}
  virtual std::unique_ptr<TensorMetaDataBase> clone() const override final {
    return std::make_unique<FullyConnectedTensorMetaData>(planningKey);
  }
};

} // end namespace dynamic
} // end namespace popsparse

#endif // popsparse_FullyConnectedTensorMetaData_hpp
