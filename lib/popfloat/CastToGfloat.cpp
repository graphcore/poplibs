// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popfloat/experimental/CastToGfloat.hpp"
#include "codelets/asm/GfloatConst.hpp"
#include "popfloat/experimental/CastToHalf.hpp"
#include "popops/ElementWiseUtil.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <popfloat/experimental/GfloatExpr.hpp>
#include <popfloat/experimental/GfloatExprUtil.hpp>
#include <popops/Cast.hpp>

#include <cassert>
#include <cmath>
#include <unordered_set>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;

namespace poputil {
template <>
poplar::ProfileValue
toProfileValue(const popfloat::experimental::GfloatCast::CastConfig &t) {
  poplar::ProfileValue::Map v;
  v.insert({"calculationType", toProfileValue(t.getCalculationType())});
  v.insert({"storageType", toProfileValue(t.getStorageType())});
  v.insert({"srNoiseMin", toProfileValue(t.getSRNoiseMin())});
  v.insert({"srNoiseMax", toProfileValue(t.getSRNoiseMax())});
  return v;
}
} // namespace poputil

namespace popfloat {
namespace experimental {

static std::tuple<unsigned, unsigned>
getInterleavedWorkSplit(const unsigned nElements, const unsigned grainSize,
                        const unsigned numWorkers) {
  const auto elementsPerWorker = (nElements / grainSize) / numWorkers;
  auto remainder = nElements - (grainSize * numWorkers * elementsPerWorker);
  const auto lastWorker = remainder / grainSize;
  remainder = remainder % grainSize;
  return std::make_tuple(elementsPerWorker, (lastWorker + (remainder << 8)));
}
static std::vector<Interval>
flatten(const std::vector<std::vector<Interval>> &intervals2D) {
  std::vector<Interval> flattenedIntervals;
  for (const auto &intervals1D : intervals2D) {
    flattenedIntervals.insert(flattenedIntervals.end(), std::begin(intervals1D),
                              std::end(intervals1D));
  }
  return flattenedIntervals;
}

void GfloatCast::GfloatFormatOptions::parseGfloatFormatOptions(
    const poplar::OptionFlags &options) {
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;

  const OptionSpec gfFormatSpec{
      {"numMantissaBits", OptionHandler::createWithInteger(numMantissaBits)},
      {"numExponentBits", OptionHandler::createWithInteger(numExponentBits)},
      {"numExponentBias", OptionHandler::createWithInteger(numExponentBias)},
      {"enableDenorms", OptionHandler::createWithBool(enableDenorms)},
      {"enableInfsAndNans", OptionHandler::createWithBool(enableInfsAndNans)}};
  for (const auto &entry : options) {
    gfFormatSpec.parse(entry.first, entry.second);
  }
}

void GfloatCast::GfloatCastOptions::parseGfloatCastOptions(
    const poplar::OptionFlags &options) {
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;

  const OptionSpec gfCastSpec{
      {"roundMode",
       OptionHandler::createWithEnum(
           roundMode, {{"INV", popfloat::experimental::RoundType::INV},
                       {"RZ", popfloat::experimental::RoundType::RZ},
                       {"RA", popfloat::experimental::RoundType::RA},
                       {"RN", popfloat::experimental::RoundType::RN},
                       {"RU", popfloat::experimental::RoundType::RU},
                       {"RD", popfloat::experimental::RoundType::RD},
                       {"SR", popfloat::experimental::RoundType::SR},
                       {"SX", popfloat::experimental::RoundType::SX}})},
      {"numSRBits", OptionHandler::createWithInteger(numSRBits)},
      {"srNoiseDensity",
       OptionHandler::createWithEnum(
           srNoiseDensity,
           {{"INVALID", popfloat::experimental::SRDensityType::INVALID},
            {"UNIFORM", popfloat::experimental::SRDensityType::UNIFORM},
            {"NORMAL", popfloat::experimental::SRDensityType::NORMAL},
            {"TRUNCATED_NORMAL",
             popfloat::experimental::SRDensityType::TRUNCATED_NORMAL},
            {"BERNOULLI", popfloat::experimental::SRDensityType::BERNOULLI},
            {"LOGISTIC", popfloat::experimental::SRDensityType::LOGISTIC},
            {"TRUNCATED_LOGISTIC",
             popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC},
            {"LAPLACE", popfloat::experimental::SRDensityType::LAPLACE},
            {"TRUNCATED_LAPLACE",
             popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE},
            {"LOGIT_NORMAL",
             popfloat::experimental::SRDensityType::LOGIT_NORMAL},
            {"TRUNCATED_LOGIT_NORMAL",
             popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL}})},
      {"srNoiseOffset", OptionHandler::createWithDouble(srNoiseOffset)},
      {"srNoiseScale", OptionHandler::createWithDouble(srNoiseScale)},
      {"srNoiseMax", OptionHandler::createWithDouble(srNoiseMax)},
      {"srNoiseMin", OptionHandler::createWithDouble(srNoiseMin)},
      {"bernoulliProb", OptionHandler::createWithDouble(bernoulliProb)},
      {"enableNanooMode", OptionHandler::createWithBool(enableNanooMode)}};

  for (const auto &entry : options) {
    gfCastSpec.parse(entry.first, entry.second);
  }
}

GfloatCast::FormatConfig::FormatConfig(unsigned numMantissaBits,
                                       unsigned numExponentBits,
                                       int exponentBias, bool enableDenorms,
                                       bool enableInfsAndNans,
                                       Type calculationType)
    : calculationType(calculationType), numMantissaBits(numMantissaBits),
      numExponentBits(numExponentBits), exponentBias(exponentBias),
      enableDenorms(enableDenorms),
      enableInfsAndNans(enableInfsAndNans && (numExponentBits > 0)) {
  unsigned gfNumBits = 1 + numMantissaBits + numExponentBits;

  formatType = FormatType::INVALID_FORMAT;

  storageType = poplar::FLOAT;

  blockFloat = false;
  if ((numExponentBits == 0) || ((numExponentBits == 1) && enableInfsAndNans)) {
    blockFloat = true;
  }

  // Infer the generic float attributes from the input parameters. The cast
  // function will use the FP16 codelet if the gfloat mantissa and exponent can
  // fit in FP16's mantissa and exponent, respectively, except if the gfloat
  // format is FP16 (for example when trying different rounding modes) the code
  // will select the cast to FP32.
  if ((numMantissaBits <= POPFLOAT_NUM_FP16_MANTISSA_BITS) &&
      (numExponentBits <= POPFLOAT_NUM_FP16_EXPONENT_BITS) &&
      ((numMantissaBits != POPFLOAT_NUM_FP16_MANTISSA_BITS) ||
       (numExponentBits != POPFLOAT_NUM_FP16_EXPONENT_BITS))) {
    nativeType = poplar::HALF;
    storageType = poplar::HALF;

    // Default format is "quantised FP16"
    formatType = FormatType::QUANTISED_FP16;
    packedFloatBits = 16;
    if (gfNumBits <= 8) {
      packedFloatBits = 8;

      // Set the gfloat pack Op parameters:
      storageType = poplar::SIGNED_CHAR;

      if (numExponentBits == POPFLOAT_NUM_FP16_EXPONENT_BITS) {
        // If the FP8 has 8 exponents, check if Infs/Nans are enabled or not.
        // If Infs/Nans are enabled, FP8 values are the top 8-bits of FP16.
        // If Infs/Nans are not enabled, the format's largest exponent will be
        // represented with FP16's largest exponent (15).
        if (enableInfsAndNans) {
          formatType = FormatType::ONE_FIVE_TWO_GF8;
        } else {
          formatType = FormatType::MAX_NORM_ALIGN_GF8;
        }
      } else {
        // For all other FP8 formats, gfloat's smallest norm exponent will be
        // aligned with the smallest norm (exponent=-14)
        formatType = FormatType::MIN_NORM_ALIGN_GF8;
      }
    }
  } else if ((numMantissaBits <= POPFLOAT_NUM_FP32_MANTISSA_BITS) &&
             (numExponentBits <= POPFLOAT_NUM_FP32_EXPONENT_BITS)) {
    nativeType = poplar::FLOAT;

    formatType = FormatType::QUANTISED_FP32;

    packedFloatBits = 32;

    // If the gfloat size is 16-bits or less (can be stored as 16-bit), set
    // the pack and unpack attributes of the float
    if ((numMantissaBits == POPFLOAT_NUM_FP16_MANTISSA_BITS) &&
        (numExponentBits == POPFLOAT_NUM_FP16_EXPONENT_BITS) &&
        enableInfsAndNans) {
      formatType = FormatType::IEEE_FP16;
      nativeType = poplar::HALF;
      storageType = poplar::HALF;
      packedFloatBits = 16;
    } else if (gfNumBits <= 8) {
      packedFloatBits = 8;
      formatType = FormatType::ENABLE_DENORM_GF16;
      if (numExponentBits <= POPFLOAT_NUM_FP16_EXPONENT_BITS) {
        nativeType = poplar::HALF;
      }
      storageType = poplar::SIGNED_CHAR;
    } else if (gfNumBits <= 16) {
      packedFloatBits = 16;

      // Set the gfloat pack Op parameters:
      storageType = poplar::SHORT;

      // If the exponent size is 8-bits, the format is set to BFLOAT. If the
      // exponent size is smaller, check the denorm flag
      if (numExponentBits == POPFLOAT_NUM_FP32_EXPONENT_BITS) {
        formatType = FormatType::BFLOAT16;
      } else if (enableDenorms) {
        formatType = FormatType::ENABLE_DENORM_GF16;
      } else {
        formatType = FormatType::NO_DENORM_GF16;
      }
    }
  }

  if (formatType == FormatType::INVALID_FORMAT) {
    throw poplibs_error("GfloatCast::FormatConfig: Invalid Gfloat format");
  }

  // Pack the gfloat parameters as INT
  uint32_t param = 0;
  param += enableDenorms << POPFLOAT_GF_STRUCT_ENDENORM_BIT_OFFSET;
  param += enableInfsAndNans << POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET;

  char packed[4];
  packed[POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET] = numMantissaBits;
  packed[POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET] = numExponentBits;
  packed[POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET] = exponentBias;
  packed[POPFLOAT_GF_STRUCT_PARAMS_OFFSET] = param;

  std::memcpy(&packedFloatParameters, &packed, sizeof(packedFloatParameters));
}

GfloatCast::FormatConfig::FormatConfig(GfloatFormatOptions formatOptions,
                                       poplar::Type calculationType)
    : FormatConfig(formatOptions.numMantissaBits, formatOptions.numExponentBits,
                   formatOptions.numExponentBias, formatOptions.enableDenorms,
                   formatOptions.enableInfsAndNans, calculationType) {}

GfloatCast::FormatConfig::FormatConfig(unsigned numMantissaBits,
                                       unsigned numExponentBits,
                                       int exponentBias, bool enableDenorms,
                                       bool enableInfsAndNans,
                                       SpecType specCalculationType)
    : numMantissaBits(numMantissaBits), numExponentBits(numExponentBits),
      exponentBias(exponentBias), enableDenorms(enableDenorms),
      enableInfsAndNans(enableInfsAndNans && (numExponentBits > 0)) {
  unsigned gfNumBits = 1 + numMantissaBits + numExponentBits;

  formatType = FormatType::INVALID_FORMAT;

  storageType = poplar::FLOAT;

  blockFloat = false;
  if ((numExponentBits == 0) || ((numExponentBits == 1) && enableInfsAndNans)) {
    blockFloat = true;
  }

  // Infer the generic float attributes from the input parameters. The cast
  // function will use the FP16 codelet if the gfloat mantissa and exponent can
  // fit in FP16's mantissa and exponent, respectively, except if the gfloat
  // format is FP16 (for example when trying different rounding modes) the code
  // will select the cast to FP32.
  if ((specCalculationType != SpecType::FP32) &&
      (numMantissaBits <= POPFLOAT_NUM_FP16_MANTISSA_BITS) &&
      (numExponentBits <= POPFLOAT_NUM_FP16_EXPONENT_BITS) &&
      ((numMantissaBits != POPFLOAT_NUM_FP16_MANTISSA_BITS) ||
       (numExponentBits != POPFLOAT_NUM_FP16_EXPONENT_BITS))) {
    calculationType = poplar::HALF;
    nativeType = poplar::HALF;
    storageType = poplar::HALF;

    // Default format is "quantised FP16"
    formatType = FormatType::QUANTISED_FP16;
    packedFloatBits = 16;
    if (gfNumBits <= 8) {
      packedFloatBits = 8;

      // Set the gfloat pack Op parameters:
      storageType = poplar::SIGNED_CHAR;

      if (numExponentBits == POPFLOAT_NUM_FP16_EXPONENT_BITS) {
        // If the FP8 has 8 exponents, check if Infs/Nans are enabled or not.
        // If Infs/Nans are enabled, FP8 values are the top 8-bits of FP16.
        // If Infs/Nans are not enabled, the format's largest exponent will be
        // represented with FP16's largest exponent (15).
        if (enableInfsAndNans) {
          formatType = FormatType::ONE_FIVE_TWO_GF8;
        } else {
          formatType = FormatType::MAX_NORM_ALIGN_GF8;
        }
      } else {
        // For all other FP8 formats, gfloat's smallest norm exponent will be
        // aligned with the smallest norm (exponent=-14)
        formatType = FormatType::MIN_NORM_ALIGN_GF8;
      }
    }
  } else if (specCalculationType != SpecType::FP16) {
    calculationType = poplar::FLOAT;
    nativeType = poplar::FLOAT;

    formatType = FormatType::QUANTISED_FP32;

    packedFloatBits = 32;

    // If the gfloat size is 16-bits or less (can be stored as 16-bit), set
    // the pack and unpack attributes of the float
    if ((numMantissaBits == POPFLOAT_NUM_FP16_MANTISSA_BITS) &&
        (numExponentBits == POPFLOAT_NUM_FP16_EXPONENT_BITS) &&
        enableInfsAndNans) {
      formatType = FormatType::IEEE_FP16;
      nativeType = poplar::HALF;
      storageType = poplar::HALF;
      packedFloatBits = 16;
    } else if (gfNumBits <= 8) {
      packedFloatBits = 8;
      storageType = poplar::SIGNED_CHAR;

      formatType = FormatType::ENABLE_DENORM_GF16;
      if (numExponentBits <= POPFLOAT_NUM_FP16_EXPONENT_BITS) {
        nativeType = poplar::HALF;
      }
    } else if (gfNumBits <= 16) {
      packedFloatBits = 16;

      // Set the gfloat pack Op parameters:
      storageType = poplar::SHORT;

      // If the exponent size is 8-bits, the format is set to BFLOAT. If the
      // exponent size is smaller, check the denorm flag
      if (numExponentBits == POPFLOAT_NUM_FP32_EXPONENT_BITS) {
        formatType = FormatType::BFLOAT16;
      } else if (enableDenorms) {
        formatType = FormatType::ENABLE_DENORM_GF16;
      } else {
        formatType = FormatType::NO_DENORM_GF16;
      }
    }
  }

  if (formatType == FormatType::INVALID_FORMAT) {
    throw poplibs_error("GfloatCast::FormatConfig: Invalid Gfloat format");
  }

  // Pack the gfloat parameters as INT
  uint32_t param = 0;
  param += enableDenorms << POPFLOAT_GF_STRUCT_ENDENORM_BIT_OFFSET;
  param += enableInfsAndNans << POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET;

  char packed[4];
  packed[POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET] = numMantissaBits;
  packed[POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET] = numExponentBits;
  packed[POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET] = exponentBias;
  packed[POPFLOAT_GF_STRUCT_PARAMS_OFFSET] = param;

  std::memcpy(&packedFloatParameters, &packed, sizeof(packedFloatParameters));
}

GfloatCast::FormatConfig::FormatConfig(const FormatConfig &formatConfig) {
  calculationType = formatConfig.getCalculationType();
  numMantissaBits = formatConfig.getNumMantissaBits();
  numExponentBits = formatConfig.getNumExponentBits();
  exponentBias = formatConfig.getExponentBias();
  enableDenorms = formatConfig.isDenormEnabled();
  enableInfsAndNans = formatConfig.infAndNansEnabled();
  nativeType = formatConfig.getNativeType();
  storageType = formatConfig.getStorageType();
  formatType = formatConfig.getFormatType();
  packedFloatParameters = formatConfig.getPackedFloatParameters();
  blockFloat = formatConfig.isBlockFloat();
  packedFloatBits = formatConfig.getPackedFloatBits();
}

static unsigned gfloatParamSize(Type calculationType) {
  if (calculationType == HALF) {
    return POPFLOAT_CAST_TO_GF16_TOTAL_PARAM_SIZE;
  } else if (calculationType == FLOAT) {
    return POPFLOAT_CAST_TO_GF32_TOTAL_PARAM_SIZE;
  } else {
    throw poputil::poplibs_error(
        "gfloatParamSize: calculationType not supported");
  }
}

GfloatCast::RoundConfig::RoundConfig(const GfloatCast::RoundConfig &roundCfg) {
  roundModeType = roundCfg.getRoundMode();
  srNoiseDensity = roundCfg.getSRNoiseDensity();
  bernoulliProb = roundCfg.getBernoulliProbability();
  srNoiseOffset = roundCfg.getSRNoiseOffset();
  srNoiseScale = roundCfg.getSRNoiseScale();
  srNoiseMax = roundCfg.getSRNoiseMax();
  srNoiseMin = roundCfg.getSRNoiseMin();
  numSRBits = roundCfg.getNumSRBits();
  noiseParams = roundCfg.getNoiseParams();
  densityParam = roundCfg.getDensityParam();
  srBitMask = roundCfg.getSRBitMask();
  roundingParams = roundCfg.getRoundingParams();
}

GfloatCast::RoundConfig::RoundConfig(GfloatCastOptions castOptions,
                                     poplar::Type calculationType)
    : RoundConfig(RoundType::INV, castOptions.numSRBits, calculationType,
                  SRDensityType::INVALID, castOptions.srNoiseOffset,
                  castOptions.srNoiseScale, castOptions.srNoiseMax,
                  castOptions.srNoiseMin, castOptions.bernoulliProb) {}

GfloatCast::RoundConfig::RoundConfig(RoundType roundMode, unsigned numSRBits,
                                     Type calculationType,
                                     SRDensityType srNoiseDensity,
                                     float srNoiseOffset, float srNoiseScale,
                                     float srNoiseMax, float srNoiseMin,
                                     float bernoulliProb)
    : roundModeType(roundMode), numSRBits(numSRBits),
      srNoiseDensity(srNoiseDensity), bernoulliProb(bernoulliProb),
      srNoiseOffset(srNoiseOffset), srNoiseScale(srNoiseScale),
      srNoiseMax(srNoiseMax), srNoiseMin(srNoiseMin) {

  float minVal_ = srNoiseMin;
  float maxVal_ = srNoiseMax;

  assert(srNoiseMin <= srNoiseMax);

  float scaleOut_ = srNoiseScale;
  float offsetOut_ = 0.5;

  float scaleIn_ = 1.0;
  float offsetIn_ = 0.0;

  unsigned srMask = 0;
  if (calculationType == FLOAT) {
    unsigned usedBits = (numSRBits < POPFLOAT_NUM_FP32_MANTISSA_BITS)
                            ? numSRBits
                            : POPFLOAT_NUM_FP32_MANTISSA_BITS;
    srMask = (1 << (POPFLOAT_NUM_FP32_MANTISSA_BITS - usedBits)) - 1;
    srMask = ~srMask;
  } else if (calculationType == HALF) {
    unsigned usedBits = (numSRBits < POPFLOAT_NUM_FP16_MANTISSA_BITS)
                            ? numSRBits
                            : POPFLOAT_NUM_FP16_MANTISSA_BITS;
    srMask = (1 << (POPFLOAT_NUM_FP16_MANTISSA_BITS - usedBits)) - 1;
    srMask = (~srMask) & 0xFFFF;
    srMask = srMask | (srMask << 16);
  }
  srBitMask = {srMask, srMask};

  roundingParams.resize(POPFLOAT_ROUND_PARAMS_TOTAL_SIZE);
  roundingParams[POPFLOAT_CAST_PARAMS_SR_MASK_OFFSET] = srMask;
  roundingParams[POPFLOAT_CAST_PARAMS_SR_MASK_OFFSET + 1] = srMask;

  if ((roundModeType == RoundType::SX) &&
      (srNoiseDensity != SRDensityType::INVALID)) {
    roundingParams.resize(calculationType == FLOAT
                              ? POPFLOAT_SR_ROUND_PARAMS_FP32_TOTAL_SIZE
                              : POPFLOAT_SR_ROUND_PARAMS_FP16_TOTAL_SIZE);

    if (srNoiseDensity == SRDensityType::UNIFORM) {
      assert((srNoiseMin >= 0.0) && (srNoiseMax <= 1.0));

      scaleOut_ = (maxVal_ - minVal_);
      offsetOut_ = (scaleOut_ / 2.0 + minVal_);
    } else if (srNoiseDensity == SRDensityType::NORMAL) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      offsetOut_ += srNoiseOffset;

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;
    } else if (srNoiseDensity == SRDensityType::TRUNCATED_NORMAL) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      offsetOut_ += srNoiseOffset;
      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;

      const double alpha = std::min(std::abs(minVal_), std::abs(maxVal_));
      const float logProb = -4.0;
      densityParam =
          std::ceil(logProb / std::log10(std::erfc(alpha / std::sqrt(2.0))));
      densityParam = (densityParam > 0) ? densityParam : (densityParam - 1);
    } else if (srNoiseDensity == SRDensityType::BERNOULLI) {
      scaleOut_ = 1.0;
      offsetOut_ = 0.0;
      densityParam = (unsigned)((1.0 - bernoulliProb) * 65536.0);
    } else if (srNoiseDensity == SRDensityType::LAPLACE) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      offsetOut_ += srNoiseOffset;

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;
    } else if (srNoiseDensity == SRDensityType::TRUNCATED_LAPLACE) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;

      float minCdf_ = ((minVal_ < 0.0) ? -1.0 : 1.0) *
                      (1.0 - std::exp(-std::abs(minVal_))) / 2.0;
      float maxCdf_ = ((maxVal_ < 0.0) ? -1.0 : 1.0) *
                      (1.0 - std::exp(-std::abs(maxVal_))) / 2.0;

      scaleIn_ = (maxCdf_ - minCdf_);
      offsetIn_ = (scaleIn_ / 2.0 + std::min<float>(minCdf_, maxCdf_));

      offsetOut_ += srNoiseOffset;

      this->srNoiseDensity = SRDensityType::LAPLACE;
    } else if (srNoiseDensity == SRDensityType::LOGISTIC) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      offsetIn_ = 0.5;

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;
      offsetOut_ += srNoiseOffset;
    } else if (srNoiseDensity == SRDensityType::TRUNCATED_LOGISTIC) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale / 2.0;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale / 2.0;

      minVal_ = 0.5 * (1.0 + std::tanh(minVal_));
      maxVal_ = 0.5 * (1.0 + std::tanh(maxVal_));

      scaleIn_ = (maxVal_ - minVal_);
      offsetIn_ = (scaleIn_ / 2.0 + std::min<float>(maxVal_, minVal_));

      scaleOut_ = srNoiseScale;
      offsetOut_ += srNoiseOffset;

      minVal_ = (srNoiseMin - srNoiseOffset) / srNoiseScale;
      maxVal_ = (srNoiseMax - srNoiseOffset) / srNoiseScale;

      this->srNoiseDensity = SRDensityType::LOGISTIC;
    } else if (srNoiseDensity == SRDensityType::LOGIT_NORMAL) {
      assert((srNoiseMin >= 0.0) && (srNoiseMax <= 1.0));

      offsetOut_ = 0.0;
    } else if (srNoiseDensity == SRDensityType::TRUNCATED_LOGIT_NORMAL) {
      assert((srNoiseMin >= 0.0) && (srNoiseMax <= 1.0));

      offsetOut_ = 0.0;

      minVal_ = std::log((minVal_ + 1e-10) / (1.0 - (minVal_ + 1e-10)));
      maxVal_ = std::log((maxVal_ - 1e-10) / (1.0 - (maxVal_ - 1e-10)));

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;

      const double alpha = std::min(std::abs(minVal_), std::abs(maxVal_));
      const float logProb = -4.0;
      densityParam =
          std::ceil(logProb / std::log10(std::erf(alpha / std::sqrt(2.0))));
    }

    if (calculationType == FLOAT) {
      float corrScaleOut[2], corrClamp[2], corrScaleIn[2];
      corrScaleIn[0] = offsetIn_;
      corrScaleIn[1] = scaleIn_;

      corrScaleOut[0] = offsetOut_;
      corrScaleOut[1] = scaleOut_;

      corrClamp[0] = minVal_;
      corrClamp[1] = maxVal_;

      unsigned corrScaleInBits[2], corrScaleOutBits[2], corrClampBits[2];
      std::memcpy(corrScaleInBits, corrScaleIn, 2 * sizeof(unsigned));
      std::memcpy(corrScaleOutBits, corrScaleOut, 2 * sizeof(unsigned));
      std::memcpy(corrClampBits, corrClamp, 2 * sizeof(unsigned));

      noiseParams = {corrScaleInBits[0],  corrScaleInBits[1],
                     corrScaleOutBits[0], corrScaleOutBits[1],
                     corrClampBits[0],    corrClampBits[1]};

      roundingParams[POPFLOAT_CAST_PARAMS_FP32_SCALE_IN_OFFSET] =
          corrScaleInBits[0];
      roundingParams[POPFLOAT_CAST_PARAMS_FP32_SCALE_IN_OFFSET + 1] =
          corrScaleInBits[1];
      roundingParams[POPFLOAT_CAST_PARAMS_FP32_SCALE_OUT_OFFSET] =
          corrScaleOutBits[0];
      roundingParams[POPFLOAT_CAST_PARAMS_FP32_SCALE_OUT_OFFSET + 1] =
          corrScaleOutBits[1];
      roundingParams[POPFLOAT_CAST_PARAMS_FP32_CLAMP_OUT_OFFSET] =
          corrClampBits[0];
      roundingParams[POPFLOAT_CAST_PARAMS_FP32_CLAMP_OUT_OFFSET + 1] =
          corrClampBits[1];
      roundingParams[POPFLOAT_CAST_PARAMS_FP32_DENSITY_PARAM_OFFSET] =
          densityParam;
    } else {
      short corrScaleIn[2], corrScaleOut[2], corrClamp[2];
      corrScaleIn[0] = singleToHalf(offsetIn_);
      corrScaleIn[1] = singleToHalf(scaleIn_);

      corrScaleOut[0] = singleToHalf(offsetOut_);
      corrScaleOut[1] = singleToHalf(scaleOut_);

      corrClamp[0] = singleToHalf(minVal_);
      corrClamp[1] = singleToHalf(maxVal_);

      unsigned corrScaleInBits, corrScaleOutBits, corrClampBits;
      std::memcpy(&corrScaleInBits, corrScaleIn, sizeof(corrScaleInBits));
      std::memcpy(&corrScaleOutBits, corrScaleOut, sizeof(corrScaleOutBits));
      std::memcpy(&corrClampBits, corrClamp, sizeof(corrClampBits));

      noiseParams = {corrScaleInBits, corrScaleOutBits, corrClampBits};

      roundingParams[POPFLOAT_CAST_PARAMS_FP16_SCALE_IN_OFFSET] =
          corrScaleInBits;
      roundingParams[POPFLOAT_CAST_PARAMS_FP16_SCALE_OUT_OFFSET] =
          corrScaleOutBits;
      roundingParams[POPFLOAT_CAST_PARAMS_FP16_CLAMP_OUT_OFFSET] =
          corrClampBits;
      roundingParams[POPFLOAT_CAST_PARAMS_FP16_DENSITY_PARAM_OFFSET] =
          densityParam;
    }
  }
}

GfloatCast::CastConfig::CastConfig(FormatType floatFormatType,
                                   Type calculationType, Type storageType,
                                   GfloatCast::RoundConfig roundCfg,
                                   bool enableNanooMode)
    : calculationType(calculationType), storageType(storageType),
      roundConfig(roundCfg), enableNanooMode(enableNanooMode),
      floatFormatType(floatFormatType) {
  castParams = roundCfg.getRoundingParams();
  castParams.push_back(enableNanooMode);

  switch (floatFormatType) {
  case FormatType::IEEE_FP16:
  case FormatType::QUANTISED_FP16:
  case FormatType::QUANTISED_FP32:
    storeAsNative = true;
    break;
  case FormatType::ONE_FIVE_TWO_GF8:
  case FormatType::MIN_NORM_ALIGN_GF8:
  case FormatType::MAX_NORM_ALIGN_GF8:
    storeAsNative = (storageType == HALF) || (storageType == FLOAT);
    break;
  case FormatType::BFLOAT16:
  case FormatType::NO_DENORM_GF16:
  case FormatType::ENABLE_DENORM_GF16:
    storeAsNative = (storageType == HALF) || (storageType == FLOAT);
    break;
  case FormatType::INVALID_FORMAT:
    throw poputil::poplibs_error(
        "popfloat::GfloatCastConfig: Expecting a valid gfloat format type.");
    break;
  }
}

GfloatCast::CastConfig GfloatCast::CastConfig::createCastGFToNative(
    FormatType floatFormatType, Type calculationType, Type storageType) {
  return GfloatCast::CastConfig(floatFormatType, calculationType, storageType,
                                GfloatCast::RoundConfig(), false);
}

GfloatCast::CastConfig GfloatCast::CastConfig::createCastNativeToGF(
    FormatType floatFormatType, Type calculationType, Type storageType,
    RoundConfig roundConfig, bool enableNanooMode) {
  return GfloatCast::CastConfig(floatFormatType, calculationType, storageType,
                                roundConfig, enableNanooMode);
}

static std::string gfloatCastVertexName(const GfloatCast::CastConfig &gfCastCfg,
                                        Type inType, Type outType,
                                        bool inPlace = false) {
  if (gfCastCfg.getCalculationType() == HALF) {
    if (gfCastCfg.getSRNoiseDensity() != SRDensityType::INVALID) {
      return inPlace ? templateVertex("popfloat::experimental::"
                                      "CastToGfloat16SrInPlaceSupervisor",
                                      inType, gfCastCfg.isNanooModeEnabled(),
                                      gfCastCfg.getSRNoiseDensity())
                     : templateVertex(
                           "popfloat::experimental::CastToGfloat16SrSupervisor",
                           inType, outType, gfCastCfg.isNanooModeEnabled(),
                           gfCastCfg.getSRNoiseDensity());
    } else {
      return inPlace ? templateVertex("popfloat::experimental::"
                                      "CastToGfloat16InPlaceSupervisor",
                                      inType, gfCastCfg.isNanooModeEnabled(),
                                      gfCastCfg.getRoundMode())
                     : templateVertex(
                           "popfloat::experimental::CastToGfloat16Supervisor",
                           inType, outType, gfCastCfg.isNanooModeEnabled(),
                           gfCastCfg.getRoundMode());
    }
  } else if (gfCastCfg.getCalculationType() == FLOAT) {
    if (gfCastCfg.getSRNoiseDensity() != SRDensityType::INVALID) {
      return inPlace ? templateVertex("popfloat::experimental::"
                                      "CastToGfloat32SrInPlaceSupervisor",
                                      gfCastCfg.isNanooModeEnabled(),
                                      gfCastCfg.getSRNoiseDensity())
                     : templateVertex(
                           "popfloat::experimental::CastToGfloat32SrSupervisor",
                           inType, outType, gfCastCfg.isNanooModeEnabled(),
                           gfCastCfg.getSRNoiseDensity());
    } else {
      return inPlace ? templateVertex("popfloat::experimental::"
                                      "CastToGfloat32InPlaceSupervisor",
                                      gfCastCfg.isNanooModeEnabled(),
                                      gfCastCfg.getRoundMode())
                     : templateVertex(
                           "popfloat::experimental::CastToGfloat32Supervisor",
                           inType, outType, gfCastCfg.isNanooModeEnabled(),
                           gfCastCfg.getRoundMode());
    }
  } else {
    throw poputil::poplibs_error(
        "popfloat::gfloatCastVertexName: Cast calculation type not supported");
  }
}

static std::string gfloatPackVertexName(Type calculationType, Type storageType,
                                        FormatType formatType) {
  if (calculationType == FLOAT) {
    if (storageType == SHORT) {
      return templateVertex("popfloat::experimental::CastFloatToGf16Supervisor",
                            formatType);
    } else if (storageType == SIGNED_CHAR) {
      return "popfloat::experimental::CastFloatToGf8Supervisor";
    }
  } else if (calculationType == HALF) {
    return templateVertex("popfloat::experimental::CastHalfToGf8Supervisor",
                          formatType);
  }
  throw poputil::poplibs_error(
      "popfloat::gfloatPackVertexName: Cast calculation type not supported");
}

const std::string gfloatToNativeVertexName(Type calculationType, Type inType,
                                           FormatType formatType) {
  if (calculationType == FLOAT) {
    if (inType == SHORT) {
      return templateVertex("popfloat::experimental::CastGf16ToFloatSupervisor",
                            formatType);
    } else if (inType == SIGNED_CHAR) {
      return "popfloat::experimental::CastGf8ToFloatSupervisor";
    }
  } else if (calculationType == HALF) {
    return templateVertex("popfloat::experimental::CastGf8ToHalfSupervisor",
                          formatType);
  }
  throw poputil::poplibs_error(
      "popfloat::gfloatToNativeVertexName: Calculation type not supported");
}

GfloatCast::GfloatCast(const FormatConfig &formatCfg,
                       const RoundConfig &roundCfg, const bool enableNanooMode,
                       const SpecType &GFType, const SpecType &NativeType)
    : formatCfg(formatCfg), gfParams(new Tensor()), castOpParamSet(false) {
  Type nativeToGFStorageType = formatCfg.getStorageType();
  if (GFType != SpecType::AUTO) {
    nativeToGFStorageType = specTypeToPoplarType(GFType);
  }

  nativeToGFCastCfg = CastConfig::createCastNativeToGF(
      formatCfg.getFormatType(), formatCfg.getCalculationType(),
      nativeToGFStorageType, roundCfg, enableNanooMode);

  Type gfToNativeStorageType = formatCfg.getNativeType();
  if (NativeType != SpecType::AUTO) {
    gfToNativeStorageType = specTypeToPoplarType(NativeType);
  }

  gfToNativeCastCfg = CastConfig::createCastGFToNative(
      formatCfg.getFormatType(), formatCfg.getCalculationType(),
      gfToNativeStorageType);

  gfParams = nullptr;
}

GfloatCast::GfloatCast(const GfloatFormatOptions &formatOtions,
                       const GfloatCastOptions &castOptions,
                       Type calculationType, const SpecType &GFType,
                       const SpecType &NativeType)
    : GfloatCast(FormatConfig(formatOtions, calculationType),
                 RoundConfig(castOptions, calculationType),
                 castOptions.enableNanooMode, GFType, NativeType) {}

GfloatCast::GfloatCast(const GfloatCast &gfloatCast) {
  nativeToGFCastCfg = gfloatCast.getNativeToGFConfig();
  gfToNativeCastCfg = gfloatCast.getGFToNativeConfig();
  formatCfg = gfloatCast.getFormatConfig();
  castOpParamSet = false;
  if (gfloatCast.isCastOpParamSet()) {
    gfParams = std::make_unique<Tensor>(gfloatCast.getCastOpParams());
    castOpParamSet = true;
  }
}

Tensor GfloatCast::createCastOpParamsTensor(Graph &graph, const ComputeSet &cs,
                                            Type calculationType,
                                            Tensor gfStruct,
                                            const DebugContext &dc) {

  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(cs, calculationType, gfStruct));

  Tensor param;

  std::string paramName = "gfloat" +
                          std::to_string((calculationType == HALF) ? 16 : 32) +
                          "_params";

  unsigned paramsSize = gfloatParamSize(calculationType);
  std::vector<std::size_t> paramShape = {paramsSize};
  param = graph.addVariable(INT, paramShape, {di, paramName});

  di.addOutput(param);

  const std::string vertexName =
      (calculationType == HALF) ? "popfloat::experimental::CastToGfloat16Param"
                                : "popfloat::experimental::CastToGfloat32Param";

  auto v = graph.addVertex(cs, vertexName,
                           {{"gfStruct", gfStruct}, {"param", param}});

  graph.setTileMapping(v, 0);
  graph.setTileMapping(param, 0);

  return param;
}

Tensor GfloatCast::createCastOpParamsTensor(Graph &graph, const ComputeSet &cs,
                                            Type calculationType,
                                            const unsigned gfPacked,
                                            const DebugContext &dc) {

  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(cs, calculationType, gfPacked));

  auto gfStruct = graph.addConstant(INT, {1}, &gfPacked, {di, "aramsTensor"});
  graph.setTileMapping(gfStruct, 0);

  auto output =
      createCastOpParamsTensor(graph, cs, calculationType, gfStruct, {di});
  di.addOutput(output);
  return output;
}

void GfloatCast::createCastOpParamsTensor(Graph &graph, const ComputeSet &cs) {
  gfParams = std::make_unique<Tensor>(
      createCastOpParamsTensor(graph, cs, formatCfg.getCalculationType(),
                               formatCfg.getPackedFloatParameters()));
  castOpParamSet = true;
}

void GfloatCast::createCastOpParamsTensor(
    Graph &graph, Sequence &prog, const poplar::DebugContext &debugContext) {

  const poputil::PoplibsOpDebugInfo di(debugContext);

  auto cs = graph.addComputeSet({di, "gfloatParams"});
  createCastOpParamsTensor(graph, cs);

  prog.add(Execute(cs, {di}));
}

Tensor
GfloatCast::createCastOpParamsTensor(Graph &graph, Sequence &prog,
                                     Type calculationType, Tensor gfStruct,
                                     const poplar::DebugContext &debugContext) {

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(calculationType, gfStruct));

  std::string csName = "gfloat" +
                       std::to_string((calculationType == HALF) ? 16 : 32) +
                       "/params";
  auto cs = graph.addComputeSet({di, csName});

  auto output =
      createCastOpParamsTensor(graph, cs, calculationType, gfStruct, {di});

  di.addOutput(output);

  prog.add(Execute(cs, {di}));

  return output;
}

Tensor GfloatCast::createCastOpParamsTensor(
    Graph &graph, Sequence &prog, Type calculationType, const unsigned gfPacked,
    const poplar::DebugContext &debugContext) {

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(calculationType, gfPacked));

  std::string csName = "gfloat" +
                       std::to_string((calculationType == HALF) ? 16 : 32) +
                       "/params";
  auto cs = graph.addComputeSet({di, csName});

  auto output =
      createCastOpParamsTensor(graph, cs, calculationType, gfPacked, {di});

  di.addOutput(output);

  prog.add(Execute(cs, {di}));

  return output;
}

Tensor GfloatCast::castNativeToGfloat(Graph &graph, Tensor input,
                                      const Tensor &param, const ComputeSet &cs,
                                      const GfloatCast::CastConfig &gfCastCfg,
                                      const DebugContext &debugContext) {

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(input, param, cs, gfCastCfg));

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  Type outType = gfCastCfg.getStoreAsNative() ? gfCastCfg.getStorageType()
                                              : gfCastCfg.getCalculationType();

  auto output = createOutputForElementWiseOp(graph, {input}, outType,
                                             {di, "quantiseGfloatOut"});

  auto inFlat = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat});

  const auto mapping = graph.getTileMapping(outFlat);

  const auto vertexTemplate =
      gfloatCastVertexName(gfCastCfg, input.elementType(), outType);

  unsigned grainSize = (gfCastCfg.getCalculationType() == FLOAT) ? 2 : 4;

  const auto numWorkers = graph.getTarget().getNumWorkerContexts();

  auto castParam = gfCastCfg.getCastParams();
  for (auto tile = 0U; tile != numTiles; ++tile) {
    if (mapping[tile].empty())
      continue;

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, mapping[tile]);

    const auto intervals = flatten(tileContiguousRegions);

    auto inputSlice = concat(inFlat.slices(intervals));
    auto outputSlice = concat(outFlat.slices(intervals));

    unsigned elementsPerWorker, lastWorkerParams;
    std::tie(elementsPerWorker, lastWorkerParams) = getInterleavedWorkSplit(
        inputSlice.numElements(), grainSize, numWorkers);

    auto v = graph.addVertex(
        cs, vertexTemplate,
        {{"param", param}, {"in", inputSlice}, {"out", outputSlice}});
    graph.setInitialValue(v["castParam"], gfCastCfg.getCastParams());
    graph.setInitialValue(v["elementsPerWorker"], elementsPerWorker);
    graph.setInitialValue(v["lastWorkerParams"], lastWorkerParams);
    graph.setTileMapping(v, tile);
  }

  di.addOutput(output);

  return output;
}

static Tensor castGfloatAsInteger(Graph &graph, Tensor input,
                                  const Tensor &param, const ComputeSet &cs,
                                  const GfloatCast::CastConfig &gfCastCfg,
                                  const poplar::DebugNameAndId &dnai) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  auto output = createOutputForElementWiseOp(
      graph, {input}, gfCastCfg.getStorageType(), {dnai, "packGfloatOut"});

  auto inFlat = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat});

  const auto mapping = graph.getTileMapping(outFlat);

  const std::string vertexTemplate = gfloatPackVertexName(
      gfCastCfg.getCalculationType(), gfCastCfg.getStorageType(),
      gfCastCfg.getFormatType());

  unsigned grainSize = (gfCastCfg.getStorageType() == SIGNED_CHAR) ? 4 : 2;
  const auto numWorkers = graph.getTarget().getNumWorkerContexts();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    if (mapping[tile].empty())
      continue;

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, mapping[tile]);
    const auto intervals = flatten(tileContiguousRegions);

    auto inputSlice = concat(inFlat.slices(intervals));
    auto outputSlice = concat(outFlat.slices(intervals));

    unsigned elementsPerWorker, lastWorkerParams;
    std::tie(elementsPerWorker, lastWorkerParams) = getInterleavedWorkSplit(
        inputSlice.numElements(), grainSize, numWorkers);

    auto v = graph.addVertex(
        cs, vertexTemplate,
        {{"param", param}, {"in", inputSlice}, {"out", outputSlice}});
    graph.setInitialValue(v["elementsPerWorker"], elementsPerWorker);
    graph.setInitialValue(v["lastWorkerParams"], lastWorkerParams);
    graph.setTileMapping(v, tile);
  }

  return output;
}

static Tensor castGfloatAsInteger(Graph &graph, Tensor input,
                                  const Tensor &param, Sequence &prog,
                                  const GfloatCast::CastConfig &gfCastCfg,
                                  const poplar::DebugContext &debugContext) {

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(input, param, gfCastCfg));

  const auto cs = graph.addComputeSet(
      {di, "gfloatAsInt/" + formatTypeToString(gfCastCfg.getFormatType())});

  auto output = castGfloatAsInteger(graph, input, param, cs, gfCastCfg, {di});

  prog.add(Execute(cs, {di}));
  di.addOutput(output);
  return output;
}

Tensor
GfloatCast::castNativeToGfloat(Graph &graph, Tensor input, const Tensor &param,
                               Sequence &prog, const CastConfig &gfCastCfg,
                               const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(input, param, gfCastCfg));

  const auto cs = graph.addComputeSet(
      {di,
       "castNativeToGfloat/" + formatTypeToString(gfCastCfg.getFormatType())});

  auto output = castNativeToGfloat(graph, input, param, cs, gfCastCfg, {di});

  prog.add(Execute(cs, {di}));

  if (!gfCastCfg.getStoreAsNative()) {
    output = castGfloatAsInteger(graph, output, param, prog, gfCastCfg, {di});
  }

  di.addOutput(output);
  return output;
}

Tensor
GfloatCast::castNativeToGfloat(Graph &graph, Tensor input, Sequence &prog,
                               const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(input));

  auto output = castNativeToGfloat(graph, input, *gfParams, prog,
                                   nativeToGFCastCfg, {di});

  di.addOutput(output);
  return output;
}

void GfloatCast::castNativeToGfloatInPlace(Graph &graph, Tensor input,
                                           const Tensor &param,
                                           const ComputeSet &cs,
                                           const CastConfig &gfCastCfg,
                                           const poplar::DebugContext &dc) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  auto inFlat = input.flatten();
  const auto mapping = graph.getTileMapping(inFlat);

  const auto vertexTemplate = gfloatCastVertexName(
      gfCastCfg, input.elementType(), input.elementType(), true);

  unsigned grainSize = (gfCastCfg.getCalculationType() == FLOAT) ? 2 : 4;
  const auto numWorkers = graph.getTarget().getNumWorkerContexts();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    if (mapping[tile].empty())
      continue;

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(inFlat, mapping[tile]);
    const auto intervals = flatten(tileContiguousRegions);

    auto inputSlice = concat(inFlat.slices(intervals));

    unsigned elementsPerWorker, lastWorkerParams;
    std::tie(elementsPerWorker, lastWorkerParams) = getInterleavedWorkSplit(
        inputSlice.numElements(), grainSize, numWorkers);

    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"param", param}, {"inOut", inputSlice}});
    graph.setInitialValue(v["castParam"], gfCastCfg.getCastParams());
    graph.setInitialValue(v["elementsPerWorker"], elementsPerWorker);
    graph.setInitialValue(v["lastWorkerParams"], lastWorkerParams);
    graph.setTileMapping(v, tile);
  }
}

void GfloatCast::castNativeToGfloatInPlace(Graph &graph, Tensor input,
                                           const ComputeSet &cs) {
  castNativeToGfloatInPlace(graph, input, *gfParams, cs, nativeToGFCastCfg);
}

void GfloatCast::castNativeToGfloatInPlace(
    Graph &graph, Tensor input, const Tensor &param, Sequence &prog,
    const CastConfig &gfCastCfg, const poplar::DebugContext &debugContext) {

  const poputil::PoplibsOpDebugInfo di(debugContext,
                                       DI_ARGS(input, param, gfCastCfg));

  const auto cs = graph.addComputeSet({di, "castNativeToGfloatInPlace/"});

  castNativeToGfloatInPlace(graph, input, param, cs, gfCastCfg, {di});

  prog.add(Execute(cs, {di}));
}

void GfloatCast::castNativeToGfloatInPlace(
    Graph &graph, Tensor input, Sequence &prog,
    const poplar::DebugContext &debugContext) {

  const poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(input));

  castNativeToGfloatInPlace(graph, input, *gfParams, prog, nativeToGFCastCfg,
                            {di});
}

Tensor GfloatCast::castGfloatToNative(Graph &graph, Tensor input,
                                      const Tensor &param, const ComputeSet &cs,
                                      const CastConfig &gfCastCfg,
                                      const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(input, param, cs, gfCastCfg));

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  auto output = createOutputForElementWiseOp(graph, {input},
                                             gfCastCfg.getCalculationType(),
                                             {di, "castGfloatToNativeOut"});

  auto inFlat = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat});

  const auto mapping = graph.getTileMapping(outFlat);

  const std::string vertexTemplate =
      gfloatToNativeVertexName(gfCastCfg.getCalculationType(),
                               input.elementType(), gfCastCfg.getFormatType());
  unsigned grainSize = (gfCastCfg.getCalculationType() == FLOAT) ? 2 : 4;
  const auto numWorkers = graph.getTarget().getNumWorkerContexts();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    if (mapping[tile].empty())
      continue;

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, mapping[tile]);

    const auto intervals = flatten(tileContiguousRegions);

    auto inputSlice = concat(inFlat.slices(intervals));
    auto outputSlice = concat(outFlat.slices(intervals));

    unsigned elementsPerWorker, lastWorkerParams;
    std::tie(elementsPerWorker, lastWorkerParams) = getInterleavedWorkSplit(
        inputSlice.numElements(), grainSize, numWorkers);

    auto v = graph.addVertex(
        cs, vertexTemplate,
        {{"param", param}, {"in", inputSlice}, {"out", outputSlice}});
    graph.setInitialValue(v["elementsPerWorker"], elementsPerWorker);
    graph.setInitialValue(v["lastWorkerParams"], lastWorkerParams);
    graph.setTileMapping(v, tile);
  }

  di.addOutput(output);
  return output;
}

Tensor GfloatCast::castGfloatToNative(Graph &graph, Tensor input,
                                      const ComputeSet &cs) {
  return castGfloatToNative(graph, input, *gfParams, cs, gfToNativeCastCfg);
}

Tensor
GfloatCast::castGfloatToNative(Graph &graph, Tensor input, const Tensor &param,
                               Sequence &prog, const CastConfig &gfCastCfg,
                               const poplar::DebugContext &debugContext) {

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(input, param, gfCastCfg));

  const auto cs = graph.addComputeSet(
      {di,
       "castGfloatToNative/" + formatTypeToString(gfCastCfg.getFormatType())});

  auto output = castGfloatToNative(graph, input, param, cs, gfCastCfg, {di});

  prog.add(Execute(cs, {di}));

  if (gfCastCfg.getCalculationType() != gfCastCfg.getStorageType()) {
    return popops::cast(graph, output, gfCastCfg.getStorageType(), prog, {di});
  }
  di.addOutput(output);
  return output;
}

Tensor
GfloatCast::castGfloatToNative(Graph &graph, Tensor input, Sequence &prog,
                               const poplar::DebugContext &debugContext) {

  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(input));

  auto output = castGfloatToNative(graph, input, *gfParams, prog,
                                   gfToNativeCastCfg, {di});
  di.addOutput(output);
  return output;
}
} // end namespace experimental
} // end namespace popfloat
