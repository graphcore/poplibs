// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poputil_TensorMetaDataBase_hpp
#define poputil_TensorMetaDataBase_hpp

#include "poputil/exceptions.hpp"

namespace poputil {

/** All meta-data given with a tensor derives from this class
 *  and implements its methods.
 */
class TensorMetaDataBase {
public:
  virtual std::unique_ptr<TensorMetaDataBase> clone() const = 0;
  virtual ~TensorMetaDataBase() {}
};

} // end namespace poputil

#endif // poputil_TensorMetaDataBase_hpp
