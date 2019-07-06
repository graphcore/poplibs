#include <popfloat/GfloatExprUtil.hpp>
#include "poputil/exceptions.hpp"

using namespace popfloat::gfexpr;

namespace popfloat {
namespace gfexpr {

std::string gfloatCastOpTypeToString(GfloatCastOpType op) {
  switch(op) {
    case GfloatCastOpType::CAST_TO_QUANTISED_GF32:
      return "QUANTISE_GF32";
    case GfloatCastOpType::CAST_TO_QUANTISED_GF16:
      return "QUANTISE_GF16";
    case GfloatCastOpType::CAST_FLOAT_TO_SHORT:
      return "PACK_FP32_AS_GF16";
    case GfloatCastOpType::CAST_HALF_TO_CHAR:
      return "PACK_FP16_AS_GF8";
    case GfloatCastOpType::CAST_SHORT_TO_FLOAT:
      return "UNPACK_GF16_TO_FLOAT";
    case GfloatCastOpType::CAST_CHAR_TO_HALF:
      return "UNPACK_GF8_TO_HALF";
  }
  throw poputil::poplibs_error(
      "popfloat::gfexpr::gfloatCastOpTypeToString: Op not supported");
}

std::string gfloatRoundTypeToString(GfloatRoundType rmode) {
  switch(rmode) {
    case GfloatRoundType::RZ:
      return "RZ";
    case GfloatRoundType::RN:
      return "RN";
    case GfloatRoundType::RA:
      return "RA";
    case GfloatRoundType::RU:
      return "RU";
    case GfloatRoundType::RD:
      return "RD";
    case GfloatRoundType::SR:
      return "SR";
    case GfloatRoundType::SX:
      return "SX";
    case GfloatRoundType::INV:
      return "INV";
  }
  throw poputil::poplibs_error(
      "popfloat::gfexpr::gfloatRoundTypeToString: mode not supported");
}

std::string gfloatFormatTypeToString(GfloatFormatType fmt) {
  switch(fmt) {
    case GfloatFormatType::QUANTISED_FP32:
      return "QUANTISED_FP32";
    case GfloatFormatType::QUANTISED_FP16:
      return "QUANTISED_FP16";
    case GfloatFormatType::MIN_NORM_ALIGN_GF8:
      return "MIN_NORM_ALIGN_GF8";
    case GfloatFormatType::ONE_FIVE_TWO_GF8:
      return "ONE_FIVE_TWO_GF8";
    case GfloatFormatType::MAX_NORM_ALIGN_GF8:
      return "MAX_NORM_ALIGN_GF8";
    case GfloatFormatType::BFLOAT16:
      return "BFLOAT16";
    case GfloatFormatType::NO_DENORM_GF16:
      return "NO_DENORM_GF16";
    case GfloatFormatType::ENABLE_DENORM_GF16:
      return "ENABLE_DENORM_GF16";
  }
  throw poputil::poplibs_error(
      "popfloat::gfexpr::gfloatFormatTypeToString: format not supported");
}

std::string GfloatSRDensityTypeToString(GfloatSRDensityType dist) {
  switch(dist) {
    case GfloatSRDensityType::UNIFORM:
      return "UNIFORM";
    case GfloatSRDensityType::NORMAL:
      return "NORMAL";
    case GfloatSRDensityType::TRUNCATED_NORMAL:
      return "TRUNCATED_NORMAL";
    case GfloatSRDensityType::BERNOULLI:
      return "BERNOULLI";
    case GfloatSRDensityType::LOGISTIC:
      return "LOGISTIC";
    case GfloatSRDensityType::LAPLACE:
      return "LAPLACE";
    case GfloatSRDensityType::LOGIT_NORMAL:
      return "LOGIT_NORMAL";
    case GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL:
      return "TRUNCATED_LOGIT_NORMAL";
  }
  throw poputil::poplibs_error(
     "popfloat::gfexpr::GfloatSRDensityType: Distribution not supported");
}

} // end namespace gfexpr
} // end namespace popfloat
