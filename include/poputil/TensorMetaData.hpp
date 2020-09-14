// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file TensorMetaData.hpp
 *
 * Class to allow extra data to be associated with a tensor.
 *
 */

#ifndef poputil_TensorMetaData_hpp
#define poputil_TensorMetaData_hpp

#include <memory>

namespace poputil {

class TensorMetaDataBase;

/** Class used to represent some unspecified form of meta-data for a tensor.
 */
class TensorMetaData {
  std::unique_ptr<TensorMetaDataBase> data;

public:
  TensorMetaData();
  TensorMetaData(const TensorMetaData &other);
  TensorMetaData(TensorMetaData &&other);
  TensorMetaData &operator=(const TensorMetaData &other);
  TensorMetaData &operator=(TensorMetaData &&other);

  // Implementation details
  TensorMetaData(std::unique_ptr<TensorMetaDataBase> data);
  ~TensorMetaData();
  const TensorMetaDataBase *getData() const { return data.get(); }
};

} // end namespace poputil

#endif // poputil_TensorMetaData_hpp
