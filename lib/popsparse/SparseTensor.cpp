// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/SparseTensor.hpp"
#include "poputil/DebugInfo.hpp"

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const popsparse::dynamic::SparseTensor &t) {
  poplar::ProfileValue::Map v;

  v.insert({"metaInfo", toProfileValue(t.getMetaInfoTensor())});
  v.insert({"nzValues", toProfileValue(t.getNzValuesTensor())});
  v.insert({"opMetaData", toProfileValue(t.getOpMetaData())});

  return v;
}

template <>
poplar::ProfileValue toProfileValue(const popsparse::static_::SparseTensor &t) {
  poplar::ProfileValue::Map v;

  v.insert({"nzValues", toProfileValue(t.getNzValuesTensor())});
  v.insert({"opMetaData", toProfileValue(t.getOpMetaData())});

  return v;
}

} // namespace poputil
