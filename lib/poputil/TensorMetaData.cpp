// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "poputil/TensorMetaData.hpp"
#include "TensorMetaDataBase.hpp"

namespace poputil {

TensorMetaData::TensorMetaData() = default;
TensorMetaData::TensorMetaData(const TensorMetaData &other) {
  if (other.data) {
    data = other.data->clone();
  }
}
TensorMetaData::TensorMetaData(TensorMetaData &&other) = default;
TensorMetaData::TensorMetaData(std::unique_ptr<TensorMetaDataBase> data)
    : data(std::move(data)) {}
TensorMetaData::~TensorMetaData() = default;

TensorMetaData &TensorMetaData::operator=(const TensorMetaData &other) {
  if (other.data) {
    data = other.data->clone();
  }
  return *this;
}

} // end namespace poputil
