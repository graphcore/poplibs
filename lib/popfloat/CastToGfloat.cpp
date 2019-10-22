#include "experimental/popfloat/CastToGfloat.hpp"
#include "codelets/GfloatConst.hpp"
#include "experimental/popfloat/CastToHalf.hpp"
#include "popops/ElementWiseUtil.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <experimental/popfloat/GfloatExpr.hpp>
#include <experimental/popfloat/GfloatExprUtil.hpp>
#include <popops/Cast.hpp>

#include <cassert>
#include <cmath>
#include <unordered_set>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;

namespace experimental {
namespace popfloat {

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
      storageType = poplar::CHAR;

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
      storageType = poplar::CHAR;
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
      storageType = poplar::CHAR;

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
      storageType = poplar::CHAR;

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
}

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

  float scale_ = srNoiseScale;
  float bias_ = 0.5;

  if ((roundModeType == RoundType::SX) &&
      (srNoiseDensity != SRDensityType::INVALID)) {
    if (srNoiseDensity == SRDensityType::UNIFORM) {
      assert((srNoiseMin >= 0.0) && (srNoiseMax <= 1.0));

      scale_ = (maxVal_ - minVal_);
      bias_ = (scale_ / 2.0 + minVal_);
    } else if (srNoiseDensity == SRDensityType::NORMAL) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      bias_ += srNoiseOffset;

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;
    } else if (srNoiseDensity == SRDensityType::TRUNCATED_NORMAL) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      bias_ += srNoiseOffset;
      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;

      const double alpha = std::min(std::abs(minVal_), std::abs(maxVal_));
      const float logProb = -4.0;
      densityParam =
          std::ceil(logProb / std::log10(std::erfc(alpha / std::sqrt(2.0))));
      densityParam = (densityParam > 0) ? densityParam : (densityParam - 1);
    } else if (srNoiseDensity == SRDensityType::BERNOULLI) {
      scale_ = 1.0;
      bias_ = 0.0;
      densityParam = (unsigned)((1.0 - bernoulliProb) * 65536.0);
    } else if (srNoiseDensity == SRDensityType::LAPLACE) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      bias_ += srNoiseOffset;

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;
    } else if (srNoiseDensity == SRDensityType::TRUNCATED_LAPLACE) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      minVal_ = (minVal_ + 0.5 - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ + 0.5 - srNoiseOffset) / srNoiseScale;

      minVal_ =
          ((minVal_ < 0.0) ? -1.0 : 1.0) * (1.0 - std::exp(-std::abs(minVal_)));
      maxVal_ =
          ((maxVal_ < 0.0) ? -1.0 : 1.0) * (1.0 - std::exp(-std::abs(maxVal_)));

      minVal_ /= 2.0;
      maxVal_ /= 2.0;

      scale_ = (maxVal_ - minVal_);
      bias_ = (scale_ / 2.0 + std::min<float>(maxVal_, minVal_));

      // Truncated Laplace uses the clamp vector to store offset and scaling
      minVal_ = srNoiseOffset;
      maxVal_ = srNoiseScale;
    } else if (srNoiseDensity == SRDensityType::LOGISTIC) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;

      bias_ += srNoiseOffset;
    } else if (srNoiseDensity == SRDensityType::TRUNCATED_LOGISTIC) {
      assert((srNoiseMin >= -0.5) && (srNoiseMax <= 0.5));

      minVal_ = (minVal_ + 0.5 - srNoiseOffset) / srNoiseScale / 2.0;
      maxVal_ = (maxVal_ + 0.5 - srNoiseOffset) / srNoiseScale / 2.0;

      minVal_ = 0.5 * (1.0 + std::tanh(minVal_));
      maxVal_ = 0.5 * (1.0 + std::tanh(maxVal_));

      scale_ = (maxVal_ - minVal_);
      bias_ = (scale_ / 2.0 + std::min<float>(maxVal_, minVal_));

      // Truncated Logistic uses the clamp vector to store offset and scaling
      minVal_ = srNoiseOffset;
      maxVal_ = srNoiseScale;
    } else if (srNoiseDensity == SRDensityType::LOGIT_NORMAL) {
      assert((srNoiseMin >= 0.0) && (srNoiseMax <= 1.0));

      bias_ = srNoiseOffset;
    } else if (srNoiseDensity == SRDensityType::TRUNCATED_LOGIT_NORMAL) {
      assert((srNoiseMin >= 0.0) && (srNoiseMax <= 1.0));

      bias_ = srNoiseOffset;
      minVal_ = std::log(minVal_ / (1.0 - minVal_));
      maxVal_ = std::log(maxVal_ / (1.0 - maxVal_));

      minVal_ = (minVal_ - srNoiseOffset) / srNoiseScale;
      maxVal_ = (maxVal_ - srNoiseOffset) / srNoiseScale;

      const double alpha = std::max(std::abs(minVal_), std::abs(maxVal_));
      const float logProb = -4.0;
      densityParam =
          std::ceil(logProb / std::log10(1 + std::erf(alpha / std::sqrt(2.0))));
    }

    if (calculationType == FLOAT) {
      float corrScale[2], corrClamp[2];
      corrScale[0] = bias_;
      corrScale[1] = scale_;

      corrClamp[0] = minVal_;
      corrClamp[1] = maxVal_;

      unsigned corrScaleBits[2], corrClampBits[2];
      std::memcpy(corrScaleBits, corrScale, 2 * sizeof(unsigned));
      std::memcpy(corrClampBits, corrClamp, 2 * sizeof(unsigned));

      noiseParams = {corrScaleBits[0], corrScaleBits[1], corrClampBits[0],
                     corrClampBits[1]};
    } else {
      short corrScale[2], corrClamp[2];
      corrScale[0] = singleToHalf(bias_);
      corrScale[1] = singleToHalf(scale_);

      corrClamp[0] = singleToHalf(minVal_);
      corrClamp[1] = singleToHalf(maxVal_);

      unsigned corrScaleBits, corrClampBits;
      std::memcpy(&corrScaleBits, corrScale, sizeof(corrScaleBits));
      std::memcpy(&corrClampBits, corrClamp, sizeof(corrClampBits));

      noiseParams = {corrScaleBits, corrClampBits};
    }
  }

  if (calculationType == FLOAT) {
    unsigned srMask = 0;
    unsigned usedBits = (numSRBits < POPFLOAT_NUM_FP32_MANTISSA_BITS)
                            ? numSRBits
                            : POPFLOAT_NUM_FP32_MANTISSA_BITS;
    srMask = (1 << (POPFLOAT_NUM_FP32_MANTISSA_BITS - usedBits)) - 1;
    srMask = ~srMask;

    srBitMask = {srMask, srMask};
  } else if (calculationType == HALF) {
    unsigned srMask = 0;
    unsigned usedBits = (numSRBits < POPFLOAT_NUM_FP16_MANTISSA_BITS)
                            ? numSRBits
                            : POPFLOAT_NUM_FP16_MANTISSA_BITS;
    srMask = (1 << (POPFLOAT_NUM_FP16_MANTISSA_BITS - usedBits)) - 1;
    srMask = (~srMask) & 0xFFFF;
    srMask = srMask | (srMask << 16);

    srBitMask = {srMask, srMask};
  }
}

GfloatCast::CastConfig::CastConfig(FormatType floatFormatType,
                                   Type calculationType, Type storageType,
                                   GfloatCast::RoundConfig roundCfg,
                                   bool enableNanooMode)
    : calculationType(calculationType), storageType(storageType),
      roundConfig(roundCfg), enableNanooMode(enableNanooMode),
      floatFormatType(floatFormatType) {
  switch (floatFormatType) {
  case FormatType::IEEE_FP16:
  case FormatType::QUANTISED_FP16:
  case FormatType::QUANTISED_FP32:
    storeAsNative = true;
    break;
  case FormatType::ONE_FIVE_TWO_GF8:
  case FormatType::MIN_NORM_ALIGN_GF8:
  case FormatType::MAX_NORM_ALIGN_GF8:
    storeAsNative = (storageType != CHAR) && (storageType != SHORT);
    break;
  case FormatType::BFLOAT16:
  case FormatType::NO_DENORM_GF16:
  case FormatType::ENABLE_DENORM_GF16:
    storeAsNative = (storageType != CHAR) && (storageType != SHORT);
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
      return inPlace
                 ? templateVertex(
                       "experimental::popfloat::CastToGfloat16SrInPlace",
                       inType, gfCastCfg.isNanooModeEnabled(),
                       gfCastCfg.getSRNoiseDensity())
                 : templateVertex("experimental::popfloat::CastToGfloat16Sr",
                                  inType, outType,
                                  gfCastCfg.isNanooModeEnabled(),
                                  gfCastCfg.getSRNoiseDensity());
    } else {
      return inPlace
                 ? templateVertex(
                       "experimental::popfloat::CastToGfloat16InPlace", inType,
                       gfCastCfg.isNanooModeEnabled(), gfCastCfg.getRoundMode())
                 : templateVertex("experimental::popfloat::CastToGfloat16",
                                  inType, outType,
                                  gfCastCfg.isNanooModeEnabled(),
                                  gfCastCfg.getRoundMode());
    }
  } else if (gfCastCfg.getCalculationType() == FLOAT) {
    if (gfCastCfg.getSRNoiseDensity() != SRDensityType::INVALID) {
      return inPlace
                 ? templateVertex(
                       "experimental::popfloat::CastToGfloat32SrInPlace",
                       gfCastCfg.isNanooModeEnabled(),
                       gfCastCfg.getSRNoiseDensity())
                 : templateVertex("experimental::popfloat::CastToGfloat32Sr",
                                  inType, outType,
                                  gfCastCfg.isNanooModeEnabled(),
                                  gfCastCfg.getSRNoiseDensity());
    } else {
      return inPlace
                 ? templateVertex(
                       "experimental::popfloat::CastToGfloat32InPlace",
                       gfCastCfg.isNanooModeEnabled(), gfCastCfg.getRoundMode())
                 : templateVertex("experimental::popfloat::CastToGfloat32",
                                  inType, outType,
                                  gfCastCfg.isNanooModeEnabled(),
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
      return templateVertex("experimental::popfloat::CastFloatToGf16",
                            formatType);
    } else if (storageType == CHAR) {
      return "experimental::popfloat::CastFloatToGf8";
    }
  } else if (calculationType == HALF) {
    return templateVertex("experimental::popfloat::CastHalfToGf8", formatType);
  }
  throw poputil::poplibs_error(
      "popfloat::gfloatPackVertexName: Cast calculation type not supported");
}

const std::string gfloatToNativeVertexName(Type calculationType, Type inType,
                                           FormatType formatType) {
  if (calculationType == FLOAT) {
    if (inType == SHORT) {
      return templateVertex("experimental::popfloat::CastGf16ToFloat",
                            formatType);
    } else if (inType == CHAR) {
      return "experimental::popfloat::CastGf8ToFloat";
    }
  } else if (calculationType == HALF) {
    return templateVertex("experimental::popfloat::CastGf8ToHalf", formatType);
  }
  throw poputil::poplibs_error(
      "popfloat::gfloatToNativeVertexName: Calculation type not supported");
}

GfloatCast::GfloatCast(const FormatConfig &formatCfg,
                       const RoundConfig &roundCfg, const bool enableNanooMode,
                       const SpecType &GFType, const SpecType &NativeType)
    : formatCfg(formatCfg), gfParams(new Tensor()) {
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
}

Tensor GfloatCast::createCastOpParamsTensor(Graph &graph, const ComputeSet &cs,
                                            Type calculationType,
                                            Tensor gfStruct) {
  Tensor param;

  std::string paramName = "gfloat" +
                          std::to_string((calculationType == HALF) ? 16 : 32) +
                          "_params";

  unsigned paramsSize = gfloatParamSize(calculationType);
  std::vector<std::size_t> paramShape = {paramsSize};
  param = graph.addVariable(INT, paramShape, paramName);

  const std::string vertexName =
      (calculationType == HALF) ? "experimental::popfloat::CastToGfloat16Param"
                                : "experimental::popfloat::CastToGfloat32Param";

  auto v = graph.addVertex(cs, vertexName,
                           {{"gfStruct", gfStruct}, {"param", param}});

  graph.setTileMapping(v, 0);
  graph.setTileMapping(param, 0);

  return param;
}

Tensor GfloatCast::createCastOpParamsTensor(Graph &graph, const ComputeSet &cs,
                                            Type calculationType,
                                            const unsigned gfPacked) {
  auto gfStruct = graph.addConstant(INT, {1}, &gfPacked, "aramsTensor");
  graph.setTileMapping(gfStruct, 0);

  return createCastOpParamsTensor(graph, cs, calculationType, gfStruct);
}

void GfloatCast::createCastOpParamsTensor(Graph &graph, const ComputeSet &cs) {
  *gfParams =
      createCastOpParamsTensor(graph, cs, formatCfg.getCalculationType(),
                               formatCfg.getPackedFloatParameters());
}

void GfloatCast::createCastOpParamsTensor(Graph &graph, Sequence &prog,
                                          const std::string &debugPrefix) {
  auto cs = graph.addComputeSet(debugPrefix + "/gfloatParams");
  createCastOpParamsTensor(graph, cs);

  prog.add(Execute(cs));
}

Tensor GfloatCast::createCastOpParamsTensor(Graph &graph, Sequence &prog,
                                            Type calculationType,
                                            Tensor gfStruct,
                                            const std::string &debugPrefix) {
  std::string csName = "/gfloat" +
                       std::to_string((calculationType == HALF) ? 16 : 32) +
                       "/params";
  auto cs = graph.addComputeSet(debugPrefix + csName);

  auto param = createCastOpParamsTensor(graph, cs, calculationType, gfStruct);

  prog.add(Execute(cs));

  return param;
}

Tensor GfloatCast::createCastOpParamsTensor(Graph &graph, Sequence &prog,
                                            Type calculationType,
                                            const unsigned gfPacked,
                                            const std::string &debugPrefix) {
  std::string csName = "/gfloat" +
                       std::to_string((calculationType == HALF) ? 16 : 32) +
                       "/params";
  auto cs = graph.addComputeSet(debugPrefix + csName);

  auto param = createCastOpParamsTensor(graph, cs, calculationType, gfPacked);

  prog.add(Execute(cs));

  return param;
}

Tensor GfloatCast::castNativeToGfloat(Graph &graph, Tensor input,
                                      const Tensor &param, const ComputeSet &cs,
                                      const GfloatCast::CastConfig &gfCastCfg) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  Type outType = gfCastCfg.getStoreAsNative() ? gfCastCfg.getStorageType()
                                              : gfCastCfg.getCalculationType();

  auto output = createOutputForElementWiseOp(graph, {input}, outType,
                                             "quantiseGfloatOut");

  auto inFlat = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat});

  const auto mapping = graph.getTileMapping(outFlat);
  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(input.elementType()),
                         target.getAtomicStoreGranularity());

  const auto vertexTemplate =
      gfloatCastVertexName(gfCastCfg, input.elementType(), outType);

  auto vSrMask = gfCastCfg.getSRBitMask();
  auto noiseParams = gfCastCfg.getNoiseParams();
  auto densityParam = gfCastCfg.getDensityParam();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, mapping[tile]);

    auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileContiguousRegions, grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, vertexTemplate,
                               {{"param", param},
                                {"in", inFlat.slices(regions)},
                                {"out", outFlat.slices(regions)}});
      if (gfCastCfg.getSRNoiseDensity() != SRDensityType::INVALID) {
        graph.setInitialValue(v["corrParams"], noiseParams);
        graph.setInitialValue(v["distParam"], densityParam);
      }
      graph.setInitialValue(v["srMask"], vSrMask);
      graph.setTileMapping(v, tile);
    }
  }

  return output;
}

static Tensor castGfloatAsInteger(Graph &graph, Tensor input,
                                  const Tensor &param, const ComputeSet &cs,
                                  const GfloatCast::CastConfig &gfCastCfg) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  auto output = createOutputForElementWiseOp(
      graph, {input}, gfCastCfg.getStorageType(), "packGfloatOut");

  auto inFlat = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat});

  const auto mapping = graph.getTileMapping(outFlat);
  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(input.elementType()),
                         target.getAtomicStoreGranularity());

  const std::string vertexTemplate = gfloatPackVertexName(
      gfCastCfg.getCalculationType(), gfCastCfg.getStorageType(),
      gfCastCfg.getFormatType());

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, mapping[tile]);

    auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileContiguousRegions, grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, vertexTemplate,
                               {{"param", param},
                                {"in", inFlat.slices(regions)},
                                {"out", outFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }

  return output;
}

static Tensor castGfloatAsInteger(Graph &graph, Tensor input,
                                  const Tensor &param, Sequence &prog,
                                  const GfloatCast::CastConfig &gfCastCfg,
                                  const std::string &debugPrefix) {
  const auto cs =
      graph.addComputeSet(debugPrefix + "/gfloatAsInt/" +
                          formatTypeToString(gfCastCfg.getFormatType()));

  auto output = castGfloatAsInteger(graph, input, param, cs, gfCastCfg);

  prog.add(Execute(cs));
  return output;
}

Tensor GfloatCast::castNativeToGfloat(Graph &graph, Tensor input,
                                      const Tensor &param, Sequence &prog,
                                      const CastConfig &gfCastCfg,
                                      const std::string &debugPrefix) {
  const auto cs =
      graph.addComputeSet(debugPrefix + "/castNativeToGfloat/" +
                          formatTypeToString(gfCastCfg.getFormatType()));

  auto output = castNativeToGfloat(graph, input, param, cs, gfCastCfg);

  prog.add(Execute(cs));

  if (gfCastCfg.getStoreAsNative()) {
    return output;
  } else {
    return castGfloatAsInteger(graph, output, param, prog, gfCastCfg,
                               debugPrefix);
  }
}

Tensor GfloatCast::castNativeToGfloat(Graph &graph, Tensor input,
                                      Sequence &prog,
                                      const std::string &debugPrefix) {
  return castNativeToGfloat(graph, input, *gfParams, prog, nativeToGFCastCfg,
                            debugPrefix);
}

void GfloatCast::castNativeToGfloatInPlace(Graph &graph, Tensor input,
                                           const Tensor &param,
                                           const ComputeSet &cs,
                                           const CastConfig &gfCastCfg) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  auto inFlat = input.flatten();
  const auto mapping = graph.getTileMapping(inFlat);
  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(input.elementType()),
                         target.getAtomicStoreGranularity());

  const auto vertexTemplate = gfloatCastVertexName(
      gfCastCfg, input.elementType(), input.elementType(), true);

  auto vSrMask = gfCastCfg.getSRBitMask();
  auto noiseParams = gfCastCfg.getNoiseParams();
  auto densityParam = gfCastCfg.getDensityParam();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(inFlat, mapping[tile]);

    auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileContiguousRegions, grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(
          cs, vertexTemplate,
          {{"param", param}, {"inOut", inFlat.slices(regions)}});
      if (gfCastCfg.getSRNoiseDensity() != SRDensityType::INVALID) {
        graph.setInitialValue(v["corrParams"], noiseParams);
        graph.setInitialValue(v["distParam"], densityParam);
      }
      graph.setInitialValue(v["srMask"], vSrMask);
      graph.setTileMapping(v, tile);
    }
  }
}

void GfloatCast::castNativeToGfloatInPlace(Graph &graph, Tensor input,
                                           const ComputeSet &cs) {
  castNativeToGfloatInPlace(graph, input, *gfParams, cs, nativeToGFCastCfg);
}

void GfloatCast::castNativeToGfloatInPlace(Graph &graph, Tensor input,
                                           const Tensor &param, Sequence &prog,
                                           const CastConfig &gfCastCfg,
                                           const std::string &debugPrefix) {
  const auto cs =
      graph.addComputeSet(debugPrefix + "/castNativeToGfloatInPlace/");

  castNativeToGfloatInPlace(graph, input, param, cs, gfCastCfg);

  prog.add(Execute(cs));
}

void GfloatCast::castNativeToGfloatInPlace(Graph &graph, Tensor input,
                                           Sequence &prog,
                                           const std::string &debugPrefix) {
  castNativeToGfloatInPlace(graph, input, *gfParams, prog, nativeToGFCastCfg,
                            debugPrefix);
}

Tensor GfloatCast::castGfloatToNative(Graph &graph, Tensor input,
                                      const Tensor &param, const ComputeSet &cs,
                                      const CastConfig &gfCastCfg) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  auto output = createOutputForElementWiseOp(
      graph, {input}, gfCastCfg.getCalculationType(), "castGfloatToNativeOut");

  auto inFlat = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat});

  const auto mapping = graph.getTileMapping(outFlat);
  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(output.elementType()),
                         target.getAtomicStoreGranularity());

  const std::string vertexTemplate =
      gfloatToNativeVertexName(gfCastCfg.getCalculationType(),
                               input.elementType(), gfCastCfg.getFormatType());

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, mapping[tile]);

    auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileContiguousRegions, grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, vertexTemplate,
                               {{"param", param},
                                {"in", inFlat.slices(regions)},
                                {"out", outFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }

  return output;
}

Tensor GfloatCast::castGfloatToNative(Graph &graph, Tensor input,
                                      const ComputeSet &cs) {
  return castGfloatToNative(graph, input, *gfParams, cs, gfToNativeCastCfg);
}

Tensor GfloatCast::castGfloatToNative(Graph &graph, Tensor input,
                                      const Tensor &param, Sequence &prog,
                                      const CastConfig &gfCastCfg,
                                      const std::string &debugPrefix) {
  const auto cs =
      graph.addComputeSet(debugPrefix + "/castGfloatToNative/" +
                          formatTypeToString(gfCastCfg.getFormatType()));

  auto output = castGfloatToNative(graph, input, param, cs, gfCastCfg);

  prog.add(Execute(cs));

  if (gfCastCfg.getCalculationType() != gfCastCfg.getStorageType()) {
    return popops::cast(graph, output, gfCastCfg.getStorageType(), prog,
                        debugPrefix);
  }
  return output;
}

Tensor GfloatCast::castGfloatToNative(Graph &graph, Tensor input,
                                      Sequence &prog,
                                      const std::string &debugPrefix) {
  return castGfloatToNative(graph, input, *gfParams, prog, gfToNativeCastCfg,
                            debugPrefix);
}
} // end namespace popfloat
} // end namespace experimental
