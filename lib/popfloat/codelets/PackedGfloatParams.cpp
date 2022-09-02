// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "asm/GfloatConst.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

using namespace poplar;

namespace popfloat {
namespace experimental {

class PackedGfloatParams : public Vertex {
public:
  Vector<Output<int>, ONE_PTR> gfStruct;
  unsigned manBits;
  unsigned expBits;
  int expBias;
  unsigned enDenorm;
  unsigned enInf;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  void compute() {
    uint32_t param = 0;
    param += enDenorm << POPFLOAT_GF_STRUCT_ENDENORM_BIT_OFFSET;
    param += enInf << POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET;

    uint32_t gfPacked;
    char packed[4];
    packed[POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET] = manBits;
    packed[POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET] = expBits;
    packed[POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET] = expBias;
    packed[POPFLOAT_GF_STRUCT_PARAMS_OFFSET] = param;

    std::memcpy(&gfPacked, &packed, sizeof(gfPacked));
    gfStruct[0] = gfPacked;
  }
};

} // end namespace experimental
} // end namespace popfloat
