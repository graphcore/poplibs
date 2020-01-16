// Copyright (c) Graphcore Ltd, All rights reserved.
#include "poputil/exceptions.hpp"
#include <popfloat/experimental/GfloatExprUtil.hpp>

namespace popfloat {
namespace experimental {

std::string roundTypeToString(RoundType rmode) {
  switch (rmode) {
  case RoundType::RZ:
    return "RZ";
  case RoundType::RN:
    return "RN";
  case RoundType::RA:
    return "RA";
  case RoundType::RU:
    return "RU";
  case RoundType::RD:
    return "RD";
  case RoundType::SR:
    return "SR";
  case RoundType::SX:
    return "SX";
  case RoundType::INV:
    return "INV";
  }
  throw poputil::poplibs_error(
      "popfloat::RoundTypeToString: mode not supported");
}

std::string formatTypeToString(FormatType fmt) {
  switch (fmt) {
  case FormatType::IEEE_FP16:
    return "IEEE_FP16";
  case FormatType::QUANTISED_FP32:
    return "QUANTISED_FP32";
  case FormatType::QUANTISED_FP16:
    return "QUANTISED_FP16";
  case FormatType::MIN_NORM_ALIGN_GF8:
    return "MIN_NORM_ALIGN_GF8";
  case FormatType::ONE_FIVE_TWO_GF8:
    return "ONE_FIVE_TWO_GF8";
  case FormatType::MAX_NORM_ALIGN_GF8:
    return "MAX_NORM_ALIGN_GF8";
  case FormatType::BFLOAT16:
    return "BFLOAT16";
  case FormatType::NO_DENORM_GF16:
    return "NO_DENORM_GF16";
  case FormatType::ENABLE_DENORM_GF16:
    return "ENABLE_DENORM_GF16";
  case FormatType::INVALID_FORMAT:
    return "INVALID_FORMAT";
  }
  throw poputil::poplibs_error(
      "popfloat::formatTypeToString: format not supported");
}

std::string srDensityTypeToString(SRDensityType dist) {
  switch (dist) {
  case SRDensityType::UNIFORM:
    return "UNIFORM";
  case SRDensityType::NORMAL:
    return "NORMAL";
  case SRDensityType::TRUNCATED_NORMAL:
    return "TRUNCATED_NORMAL";
  case SRDensityType::BERNOULLI:
    return "BERNOULLI";
  case SRDensityType::LOGISTIC:
    return "LOGISTIC";
  case SRDensityType::TRUNCATED_LOGISTIC:
    return "TRUNCATED_LOGISTIC";
  case SRDensityType::LAPLACE:
    return "LAPLACE";
  case SRDensityType::TRUNCATED_LAPLACE:
    return "TRUNCATED_LAPLACE";
  case SRDensityType::LOGIT_NORMAL:
    return "LOGIT_NORMAL";
  case SRDensityType::TRUNCATED_LOGIT_NORMAL:
    return "TRUNCATED_LOGIT_NORMAL";
  case SRDensityType::INVALID:
    return "INVALID";
  }
  throw poputil::poplibs_error(
      "popfloat::SRDensityType: Distribution not supported");
}

poplar::Type specTypeToPoplarType(SpecType specType) {
  if (specType == SpecType::FP32) {
    return poplar::FLOAT;
  } else if (specType == SpecType::FP16) {
    return poplar::HALF;
  } else if (specType == SpecType::INT32) {
    return poplar::INT;
  } else if (specType == SpecType::INT16) {
    return poplar::SHORT;
  } else if (specType == SpecType::INT8) {
    return poplar::CHAR;
  }
  throw poputil::poplibs_error("popfloat::SpecType: format not supported");
}

} // end namespace experimental
} // end namespace popfloat
