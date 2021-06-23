// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "RandomUtils.hpp"

using namespace poplar;

namespace poprand {

class SetSeed : public MultiVertex {
public:
  SetSeed();

  Input<Vector<unsigned, ONE_PTR, 8>> seed;
  const uint32_t seedModifierUser;
  const uint32_t seedModifierHw;

  IS_EXTERNAL_CODELET(true);

  bool compute(unsigned wid) { return true; }
};

} // namespace poprand
