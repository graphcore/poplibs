#include "popfloat/CastToGfloat.hpp"
#include "popfloat/CastToHalf.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <popfloat/GfloatExpr.hpp>
#include <popfloat/GfloatExprUtil.hpp>
#include "codelets/GfloatConst.hpp"

#include <unordered_set>
#include <cassert>
#include <cmath>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popfloat::gfexpr;

namespace popfloat {

GfloatFormatConfig::GfloatFormatConfig(int numMantissaBits,
                                       int numExponentBits,
                                       int exponentBias,
                                       bool enableDenorms,
                                       bool enableInfsAndNans) :
  numMantissaBits(numMantissaBits),
  numExponentBits(numExponentBits),
  exponentBias(exponentBias),
  enableDenorms(enableDenorms),
  enableInfsAndNans(enableInfsAndNans && (numExponentBits > 0)) {
  unsigned gfNumBits = 1 + numMantissaBits + numExponentBits;

  floatFormatType = GfloatFormatType::INVALID_FORMAT;

  packOpType = GfloatCastOpType::INVALID_OP;
  packedOutputType = poplar::FLOAT;

  unpackOpType = GfloatCastOpType::INVALID_OP;

  blockFloat = false;
  if ((numExponentBits == 0) ||
      ((numExponentBits == 1) && enableInfsAndNans)) {
    blockFloat = true;
  }

  // The quantisation of FP16 floats scales the inputs to allow the use of
  // FP16's denorm range.
  quantisedOutputScale = 1.0;

  // Infer the generic float attributes from the input parameters. The cast
  // function will use the FP16 codelet if the gfloat mantissa and exponent can
  // fit in FP16's mantissa and exponent, respectively, except if the gfloat
  // format is FP16 (for example when trying different rounding modes) the code
  // will select the cast to FP32.
  if ((numMantissaBits <= POPFLOAT_NUM_FP16_MANTISSA_BITS) &&
      (numExponentBits <= POPFLOAT_NUM_FP16_EXPONENT_BITS) &&
      ((numMantissaBits != POPFLOAT_NUM_FP16_MANTISSA_BITS) ||
       (numExponentBits != POPFLOAT_NUM_FP16_EXPONENT_BITS))) {
    quantisedOpType = GfloatCastOpType::CAST_TO_QUANTISED_GF16;
    quantisedOutputType = poplar::HALF;

    // Default format is "quantised FP16"
    floatFormatType = GfloatFormatType::QUANTISED_FP16;

    // If the gfloat size is 8-bits or less (can be stored as 8-bit), set the
    // pack and unpack attributs of the float. The default scaling value is
    // 2^(bias-15). This will align gfloat's smallest norm exponent to FP16's
    // smallest norm (exponent=-14)
    quantisedOutputScale = std::pow(2.0, exponentBias - 15.0);
    packedFloatBits = 16;
    if (gfNumBits <= 8) {
      packedFloatBits = 8;

      // Set the gfloat pack Op parameters:
      packOpType = GfloatCastOpType::CAST_HALF_TO_CHAR;
      packedOutputType = poplar::CHAR;

      // Set the gfloat unpack Op parameters:
      unpackOpType = GfloatCastOpType::CAST_CHAR_TO_HALF;
      unpackedOutputType = poplar::HALF;

      if (numExponentBits == POPFLOAT_NUM_FP16_EXPONENT_BITS) {
        // If the FP8 has 8 exponents, check if Infs/Nans are enabled or not.
        // If Infs/Nans are enabled, FP8 values are the top 8-bits of FP16.
        // If Infs/Nans are not enabled, the format's largest exponent will be
        // represented with FP16's largest exponent (15).
        if (enableInfsAndNans) {
          floatFormatType = GfloatFormatType::ONE_FIVE_TWO_GF8;
        } else {
          floatFormatType = GfloatFormatType::MAX_NORM_ALIGN_GF8;
          quantisedOutputScale = std::pow(2.0, exponentBias - 16.0);
        }
      } else {
        // For all other FP8 formats, gfloat's smallest norm exponent will be
        // aligned with the smallest norm (exponent=-14)
        floatFormatType = GfloatFormatType::MIN_NORM_ALIGN_GF8;
      }
    }
  } else if ((numMantissaBits <= POPFLOAT_NUM_FP32_MANTISSA_BITS) &&
             (numExponentBits <= POPFLOAT_NUM_FP32_EXPONENT_BITS)) {
    quantisedOpType = GfloatCastOpType::CAST_TO_QUANTISED_GF32;
    quantisedOutputType = poplar::FLOAT;

    floatFormatType = GfloatFormatType::QUANTISED_FP32;
    packedFloatBits = 32;

    // If the gfloat size is 16-bits or less (can be stored as 16-bit), set
    // the pack and unpack attributes of the float
    if ((numMantissaBits == POPFLOAT_NUM_FP16_MANTISSA_BITS) &&
        (numExponentBits == POPFLOAT_NUM_FP16_EXPONENT_BITS) &&
        enableInfsAndNans) {
      quantisedOutputType = poplar::HALF;
      packedFloatBits = 16;
    } else if (gfNumBits <= 16) {
      packedFloatBits = 16;

      // Set the gfloat pack Op parameters:
      packOpType = GfloatCastOpType::CAST_FLOAT_TO_SHORT;
      packedOutputType = poplar::SHORT;

      // Set the gfloat unpack Op parameters:
      unpackOpType = GfloatCastOpType::CAST_SHORT_TO_FLOAT;
      unpackedOutputType = poplar::FLOAT;

      // If the exponent size is 8-bits, the format is set to BFLOAT. If the
      // exponent size is smaller, check the denorm flag
      if (numExponentBits == POPFLOAT_NUM_FP32_EXPONENT_BITS) {
        floatFormatType = GfloatFormatType::BFLOAT16;
      } else if (enableDenorms) {
        floatFormatType = GfloatFormatType::ENABLE_DENORM_GF16;
      } else {
        floatFormatType = GfloatFormatType::NO_DENORM_GF16;
      }
    }
  }
  quantisedOutputScaleRecip = 1.0 / quantisedOutputScale;

  if (floatFormatType == GfloatFormatType::INVALID_FORMAT) {
    throw poplibs_error(
       "popfloat::GfloatParamsOp: Float format not supported");
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

GfloatCastOpType
GfloatFormatConfig::getCastOpType(Type inType, bool packOp) {
  if (packOp) {
    return packOpType;
  } else if ((inType == SHORT) || (inType == CHAR)) {
    return unpackOpType;
  } else if ((inType == FLOAT) || (inType == HALF)) {
    return quantisedOpType;
  } else {
    throw poputil::poplibs_error(
       "popfloat::genericFloatParamCastOpType: Op not supported");
  }
}

static unsigned genericFloatParamSize(GfloatCastOpType gfCastOpType) {
  switch (gfCastOpType) {
    case GfloatCastOpType::CAST_TO_QUANTISED_GF16:
      return POPFLOAT_CAST_TO_GF16_TOTAL_PARAM_SIZE;
      break;
    case GfloatCastOpType::CAST_TO_QUANTISED_GF32:
      return POPFLOAT_CAST_TO_GF32_TOTAL_PARAM_SIZE;
      break;
    case GfloatCastOpType::CAST_CHAR_TO_HALF:
      return POPFLOAT_GF8_TO_FP16_TOTAL_PARAM_SIZE;
      break;
    case GfloatCastOpType::CAST_SHORT_TO_FLOAT:
      return POPFLOAT_GF16_TO_FP32_TOTAL_PARAM_SIZE;
      break;
    case GfloatCastOpType::CAST_HALF_TO_CHAR:
      return POPFLOAT_FP16_TO_GF8_TOTAL_PARAM_SIZE;
      break;
    case GfloatCastOpType::CAST_FLOAT_TO_SHORT:
      return POPFLOAT_FP32_TO_GF16_TOTAL_PARAM_SIZE;
      break;
    case GfloatCastOpType::INVALID_OP:
      throw poputil::poplibs_error(
         "popfloat::genericFloatParamSize: Op not supported");
      break;
  }
}

static std::string paramVertexName(GfloatCastOpType gfCastOpType) {
  switch (gfCastOpType) {
    case GfloatCastOpType::CAST_TO_QUANTISED_GF16:
      return "popfloat::CastToGfloat16Param";
      break;
    case GfloatCastOpType::CAST_TO_QUANTISED_GF32:
      return "popfloat::CastToGfloat32Param";
      break;
    case GfloatCastOpType::CAST_CHAR_TO_HALF:
      return "popfloat::CastGf8ToHalfParam";
      break;
    case GfloatCastOpType::CAST_SHORT_TO_FLOAT:
      return "popfloat::CastGf16ToFloatParam";
      break;
    case GfloatCastOpType::CAST_HALF_TO_CHAR:
      return "popfloat::CastHalfToGf8Param";
      break;
    case GfloatCastOpType::CAST_FLOAT_TO_SHORT:
      return "popfloat::CastFloatToGf16Param";
      break;
    case GfloatCastOpType::INVALID_OP:
      throw poputil::poplibs_error(
         "popfloat::paramVertexName: Op not supported");
      break;
  }
}

GfloatCastConfig::
GfloatCastConfig(Type castInputType, Type castOutputType,
                 popfloat::gfexpr::GfloatCastOpType castOpType,
                 GfloatRoundType roundMode, bool enableNanooMode,
                 unsigned numSRBits) :
  castInputType(castInputType), castOutputType(castOutputType),
  castOpType(castOpType), roundModeType(roundMode),
  enableNanooMode(enableNanooMode), numSRBits(numSRBits),
  srNoiseDensity(GfloatSRDensityType::INVALID),
  SRNoiseOffset(0.0), SRNoiseScale(0.0),
  SRNoiseMax(0.0), SRNoiseMin(0.0) {
  if (castOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    floatFormatType = GfloatFormatType::QUANTISED_FP16;
  } else if (castOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF32) {
    floatFormatType = GfloatFormatType::QUANTISED_FP32;
  }
}

GfloatCastConfig::
GfloatCastConfig(Type castInputType, Type castOutputType,
                 popfloat::gfexpr::GfloatCastOpType castOpType,
                 GfloatSRDensityType srNoiseDensity,
                 unsigned numSRBits,
                 bool enableNanooMode, float SRNoiseOffset,
                 float SRNoiseScale, float SRNoiseMax,
                 float SRNoiseMin, float bernoulliProb) :
  castInputType(castInputType),
  castOutputType(castOutputType),
   castOpType(castOpType),
  roundModeType(GfloatRoundType::SR),
  enableNanooMode(enableNanooMode),
  numSRBits(numSRBits),
  srNoiseDensity(srNoiseDensity),
  bernoulliProb(bernoulliProb),
  SRNoiseOffset(SRNoiseOffset),
  SRNoiseScale(SRNoiseScale),
  SRNoiseMax(SRNoiseMax),
  SRNoiseMin(SRNoiseMin) {
  if (castOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    floatFormatType = GfloatFormatType::QUANTISED_FP16;
  } else if (castOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF32) {
    floatFormatType = GfloatFormatType::QUANTISED_FP32;
  }

  float minReq = -0.5;
  float maxReq = 0.5;
  if ((srNoiseDensity == GfloatSRDensityType::LOGISTIC) ||
      (srNoiseDensity == GfloatSRDensityType::LOGIT_NORMAL) ||
      (srNoiseDensity == GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL)) {
    minReq = 0.0;
    maxReq = 1.0;
  }
  float minVal_ = SRNoiseMin;
  float maxVal_ = SRNoiseMax;

  minVal_ = std::max<float>(std::min<float>(minVal_, maxReq), minReq);
  maxVal_ = std::max<float>(std::min<float>(maxVal_, maxReq), minReq);

  float scale_ = SRNoiseScale;
  float bias_  = 0.5;

  if (srNoiseDensity == GfloatSRDensityType::UNIFORM) {
    scale_ = (maxVal_ - minVal_);
    bias_ += (scale_ /  2.0 + minVal_);
  } else if (srNoiseDensity == GfloatSRDensityType::NORMAL) {
    bias_  += SRNoiseOffset;

    minVal_ = (minVal_ - SRNoiseOffset) / SRNoiseScale;
    maxVal_ = (maxVal_ - SRNoiseOffset) / SRNoiseScale;
  } else if (srNoiseDensity == GfloatSRDensityType::TRUNCATED_NORMAL) {
    bias_  += SRNoiseOffset;
    minVal_ = (minVal_ - SRNoiseOffset) / SRNoiseScale;
    maxVal_ = (maxVal_ - SRNoiseOffset) / SRNoiseScale;

    const double alpha = std::min(std::abs(minVal_), std::abs(maxVal_));
    const float logProb = -4.0;
    densityParam =
      std::ceil(logProb / std::log10(std::erfc(alpha / std::sqrt(2.0))));
    densityParam = (densityParam > 0) ? densityParam : (densityParam - 1);
  } else if (srNoiseDensity == GfloatSRDensityType::BERNOULLI) {
    scale_ = 1.0;
    bias_ = 0.0;
    densityParam = (unsigned)(bernoulliProb * 65536.0);
  } else if (srNoiseDensity == GfloatSRDensityType::LAPLACE) {
    bias_ += SRNoiseOffset;

    minVal_ = (minVal_ - SRNoiseOffset) / SRNoiseScale;
    maxVal_ = (maxVal_ - SRNoiseOffset) / SRNoiseScale;
  } else if (srNoiseDensity == GfloatSRDensityType::LOGISTIC) {
    bias_ += SRNoiseOffset;
  } else if (srNoiseDensity == GfloatSRDensityType::LOGIT_NORMAL) {
    bias_ = SRNoiseOffset;
  } else if (srNoiseDensity == GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL) {
    bias_ = SRNoiseOffset;
    minVal_ = std::log(minVal_ / (1.0 - minVal_));
    maxVal_ = std::log(maxVal_ / (1.0 - maxVal_));

    minVal_ = (minVal_ - SRNoiseOffset) / SRNoiseScale;
    maxVal_ = (maxVal_ - SRNoiseOffset) / SRNoiseScale;

    const double alpha = std::max(std::abs(minVal_), std::abs(maxVal_));
    const float logProb = -4.0;
    densityParam =
      std::ceil(logProb / std::log10(1 + std::erf(alpha / std::sqrt(2.0))));
  }

  if (castOpType == GfloatCastOpType::CAST_TO_QUANTISED_GF32) {
    float corrScale[2], corrClamp[2];
    corrScale[0] = bias_;
    corrScale[1] = scale_;

    corrClamp[0] = minVal_;
    corrClamp[1] = maxVal_;

    unsigned corrScaleBits[2], corrClampBits[2];
    std::memcpy(corrScaleBits, corrScale, 2 * sizeof(unsigned));
    std::memcpy(corrClampBits, corrClamp, 2 * sizeof(unsigned));

    noiseParams = { corrScaleBits[0], corrScaleBits[1], corrClampBits[0],
      corrClampBits[1] };
  } else {
    short corrScale[2], corrClamp[2];
    corrScale[0] = singleToHalf(bias_);
    corrScale[1] = singleToHalf(scale_);
    corrClamp[0] = singleToHalf(minVal_);
    corrClamp[1] = singleToHalf(maxVal_);

    unsigned corrScaleBits, corrClampBits;
    std::memcpy(&corrScaleBits, corrScale, sizeof(corrScaleBits));
    std::memcpy(&corrClampBits, corrClamp, sizeof(corrClampBits));

    noiseParams = { corrScaleBits, corrClampBits };
  }
}

GfloatCastConfig::GfloatCastConfig(Type castInputType,
                                   Type castOutputType,
                                   GfloatFormatType gfFrmt) :
   castInputType(castInputType), castOutputType(castOutputType),
   floatFormatType(gfFrmt) {
  if ((castInputType == FLOAT) && (castOutputType == SHORT)) {
    castOpType = GfloatCastOpType::CAST_FLOAT_TO_SHORT;
  } else   if ((castInputType == HALF) && (castOutputType == CHAR)) {
    castOpType = GfloatCastOpType::CAST_HALF_TO_CHAR;
  } else if ((castInputType == SHORT) && (castOutputType == FLOAT)) {
    castOpType = GfloatCastOpType::CAST_SHORT_TO_FLOAT;
  } else   if ((castInputType == CHAR) && (castOutputType == HALF)) {
    castOpType = GfloatCastOpType::CAST_CHAR_TO_HALF;
  } else {
    throw poputil::poplibs_error(
       "setGfloatCastOpType: Op not supported");
  }
}

static std::string genericFloatCastVertexName(const GfloatCastConfig &gfCastCfg,
                                              bool inPlace=false) {
  switch (gfCastCfg.getCastOp()) {
    case GfloatCastOpType::CAST_TO_QUANTISED_GF16:
      if (gfCastCfg.getSrNoiseDensity() != GfloatSRDensityType::INVALID) {
        return inPlace ?
               templateVertex("popfloat::CastToGfloat16SrInPlace",
                              gfCastCfg.isNanooModeEnabled(),
                              gfCastCfg.getSrNoiseDensity()) :
               templateVertex("popfloat::CastToGfloat16Sr",
                              gfCastCfg.getInputType(),
                              gfCastCfg.getOutputType(),
                              gfCastCfg.isNanooModeEnabled(),
                              gfCastCfg.getSrNoiseDensity());
      } else {
        return inPlace ?
               templateVertex("popfloat::CastToGfloat16InPlace",
                              gfCastCfg.isNanooModeEnabled(),
                              gfCastCfg.getRoundMode()) :
               templateVertex("popfloat::CastToGfloat16",
                              gfCastCfg.getInputType(),
                              gfCastCfg.getOutputType(),
                              gfCastCfg.isNanooModeEnabled(),
                              gfCastCfg.getRoundMode());
      }
    case GfloatCastOpType::CAST_TO_QUANTISED_GF32:
      if (gfCastCfg.getSrNoiseDensity() != GfloatSRDensityType::INVALID) {
        return inPlace ?
               templateVertex("popfloat::CastToGfloat32SrInPlace",
                              gfCastCfg.isNanooModeEnabled(),
                              gfCastCfg.getSrNoiseDensity()) :
               templateVertex("popfloat::CastToGfloat32Sr",
                              gfCastCfg.getInputType(),
                              gfCastCfg.getOutputType(),
                              gfCastCfg.isNanooModeEnabled(),
                              gfCastCfg.getSrNoiseDensity());
      } else {
        return inPlace ?
               templateVertex("popfloat::CastToGfloat32InPlace",
                              gfCastCfg.isNanooModeEnabled(),
                              gfCastCfg.getRoundMode()) :
               templateVertex("popfloat::CastToGfloat32",
                              gfCastCfg.getInputType(),
                              gfCastCfg.getOutputType(),
                              gfCastCfg.isNanooModeEnabled(),
                              gfCastCfg.getRoundMode());
      }
    case GfloatCastOpType::CAST_FLOAT_TO_SHORT:
      return
        templateVertex("popfloat::CastFloatToGf16", gfCastCfg.getFormatType());
    case GfloatCastOpType::CAST_HALF_TO_CHAR:
      return
        templateVertex("popfloat::CastHalfToGf8", gfCastCfg.getFormatType());
    case GfloatCastOpType::CAST_SHORT_TO_FLOAT:
      return
        templateVertex("popfloat::CastGf16ToFloat", gfCastCfg.getFormatType());
    case GfloatCastOpType::CAST_CHAR_TO_HALF:
      return
        templateVertex("popfloat::CastGf8ToHalf", gfCastCfg.getFormatType());
    case GfloatCastOpType::INVALID_OP:
      throw poputil::poplibs_error(
         "popfloat::gfQuantiseVertexName: Op not supported");

  }
}

static
Tensor createCastOpParamsTensor(Graph &graph, const ComputeSet &CS,
                                GfloatCastOpType gfCastOpType,
                                Tensor gfStruct) {
  Tensor param;

  unsigned paramsSize = genericFloatParamSize(gfCastOpType);
  std::vector<std::size_t> paramShape = { paramsSize };
  param = graph.addVariable(INT, paramShape,
                            gfloatCastOpTypeToString(gfCastOpType));

  auto v = graph.addVertex(CS,
                           paramVertexName(gfCastOpType),
                           { { "gfStruct", gfStruct },
                             { "param", param } });

  graph.setTileMapping(v, 0);
  graph.setTileMapping(param, 0);

  return param;
}

Tensor createCastOpParamsTensor(Graph &graph, const ComputeSet &CS,
                                GfloatCastOpType gfCastOpType,
                                const unsigned gfPacked) {
  auto gfStruct = graph.addConstant(INT,
                                    { 1 },
                                    &gfPacked,
                                    "createCastOpParamsTensor/gfStruct");
  graph.setTileMapping(gfStruct, 0);

  return createCastOpParamsTensor(graph, CS, gfCastOpType, gfStruct);
}

Tensor createCastOpParamsTensor(Graph &graph, Sequence &prog,
                                GfloatCastOpType gfCastOpType,
                                Tensor gfStruct,
                                const std::string  &debugPrefix) {
  auto CS =
    graph.addComputeSet("Params_" + gfloatCastOpTypeToString(gfCastOpType));

  auto param = createCastOpParamsTensor(graph, CS, gfCastOpType, gfStruct);

  prog.add(Execute(CS));

  return param;
}

poplar::Tensor setPackedGfloatParams(poplar::Graph &graph, Sequence &prog,
                                     const GfloatFormatConfig &gfFormatCfg) {
  auto gfPacked = graph.addVariable(INT, { 1 }, "CompressedGfloatParams");
  auto CS = graph.addComputeSet("PackedGfloatParamsCS");
  auto v = graph.addVertex(CS,
                           "popfloat::PackedGfloatParams",
                           { { "gfStruct", gfPacked } });

  graph.setInitialValue(v["manBits"], gfFormatCfg.getNumMantissaBits());
  graph.setInitialValue(v["expBits"], gfFormatCfg.getNumExponentBits());
  graph.setInitialValue(v["expBias"], gfFormatCfg.getExponentBias());
  graph.setInitialValue(v["enDenorm"], gfFormatCfg.isDenormEnabled());
  graph.setInitialValue(v["enInf"], gfFormatCfg.infAndNansEnabled());

  graph.setTileMapping(v, 0);
  graph.setTileMapping(gfPacked, 0);
  prog.add(Execute(CS));

  return gfPacked;
}


static std::vector<uint32_t> createSRMask(unsigned srBits,
                                          GfloatCastOpType quantiseOp) {
  unsigned srMask = 0;
  if (quantiseOp == GfloatCastOpType::CAST_TO_QUANTISED_GF32) {
    unsigned usedBits = (srBits < POPFLOAT_NUM_FP32_MANTISSA_BITS) ?
                        srBits : POPFLOAT_NUM_FP32_MANTISSA_BITS;
    srMask = (1 << (POPFLOAT_NUM_FP32_MANTISSA_BITS - usedBits)) - 1;
    srMask = ~srMask;
  } else if (quantiseOp == GfloatCastOpType::CAST_TO_QUANTISED_GF16) {
    unsigned usedBits = (srBits < POPFLOAT_NUM_FP16_MANTISSA_BITS) ?
      srBits : POPFLOAT_NUM_FP16_MANTISSA_BITS;
    srMask = (1 << (POPFLOAT_NUM_FP16_MANTISSA_BITS - usedBits)) - 1;
    srMask = (~srMask) & 0xFFFF;
    srMask = srMask | (srMask << 16);
  } else {
    throw poputil::poplibs_error(
       "popfloat::createSRMask: Op not supported");
  }
  std::vector<uint32_t> vSrMask{srMask, srMask};
  return vSrMask;
}

static
Tensor gfloatQuantise(Graph &graph, Tensor input, Tensor param,
                      const ComputeSet &CS,
                      const GfloatCastConfig &gfCastCfg) {
  const auto &target  = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  Tensor output;
  output = graph.clone(gfCastCfg.getOutputType(), input,
                       "quantiseGfloatOut");
  poputil::mapOutputForElementWiseOp(graph, { input }, output);

  auto inFlat  = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, { &inFlat });

  const auto mapping = graph.getTileMapping(outFlat);
  const auto grainSize =
    std::max<unsigned>(target.getVectorWidth(input.elementType()),
                       target.getAtomicStoreGranularity());

  const auto vertexTemplate = genericFloatCastVertexName(gfCastCfg);

  auto vSrMask = createSRMask(gfCastCfg.getNumSRBits(),
                              gfCastCfg.getCastOp());

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(outFlat, mapping[tile]);

    auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(CS,
                               vertexTemplate,
                               { { "param", param },
                                 { "in", inFlat.slices(regions) },
                                 { "out", outFlat.slices(regions) } });
      if (gfCastCfg.getSrNoiseDensity() != GfloatSRDensityType::INVALID) {
        graph.setInitialValue(v["corrParams"], gfCastCfg.getNoiseParams());
        graph.setInitialValue(v["distParam"], gfCastCfg.getDensityParam());
      }
      graph.setInitialValue(v["srMask"], vSrMask);
      graph.setTileMapping(v, tile);
    }
  }

  return output;
}

static Tensor gfloatPack(Graph &graph, Tensor input,
                         Tensor param, const ComputeSet &CS,
                         const GfloatCastConfig &gfCastCfg) {
  const auto &target     = graph.getTarget();
  const auto numTiles    = target.getNumTiles();

  Tensor output;
  output = graph.clone(gfCastCfg.getOutputType(), input, "packGfloatOut");
  poputil::mapOutputForElementWiseOp(graph, { input }, output);

  auto inFlat  = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, { &inFlat });

  const auto mapping = graph.getTileMapping(outFlat);
  const auto grainSize =
    std::max<unsigned>(target.getVectorWidth(input.elementType()),
                       target.getAtomicStoreGranularity());

  const auto vertexTemplate = genericFloatCastVertexName(gfCastCfg);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(outFlat, mapping[tile]);

    auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(CS,
                               vertexTemplate,
                               { { "param", param },
                                 { "in", inFlat.slices(regions) },
                                 { "out", outFlat.slices(regions) } });
      graph.setTileMapping(v, tile);
    }
  }

  return output;
}

static Tensor gfloatUnpack(Graph &graph, Tensor input,
                           Tensor param, const ComputeSet &CS,
                           const GfloatCastConfig &gfCastCfg) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  Tensor output;
  output = graph.clone(gfCastCfg.getOutputType(), input, "unpackGfloatOut");
  poputil::mapOutputForElementWiseOp(graph, { input }, output);

  auto inFlat  = input.flatten();
  auto outFlat = output.flatten();
  graph.reorderToSimplify(&outFlat, { &inFlat });

  const auto mapping = graph.getTileMapping(outFlat);
  const auto grainSize =
    std::max<unsigned>(target.getVectorWidth(input.elementType()),
                       target.getAtomicStoreGranularity());

  const auto vertexTemplate = genericFloatCastVertexName(gfCastCfg);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(outFlat, mapping[tile]);

    auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(CS,
                               vertexTemplate,
                               { { "param", param },
                                 { "in", inFlat.slices(regions) },
                                 { "out", outFlat.slices(regions) } });
      graph.setTileMapping(v, tile);
    }
  }

  return output;
}

poplar::Tensor castToGfloat(Graph &graph, Tensor input, Tensor param,
                            const ComputeSet &CS,
                            const GfloatCastConfig &gfCastCfg,
                            const std::string &debugPrefix) {
  if ((gfCastCfg.getCastOp() == GfloatCastOpType::CAST_TO_QUANTISED_GF16) ||
      (gfCastCfg.getCastOp() == GfloatCastOpType::CAST_TO_QUANTISED_GF32)) {
    return gfloatQuantise(graph, input, param, CS, gfCastCfg);
  } else if ((gfCastCfg.getCastOp() == GfloatCastOpType::CAST_FLOAT_TO_SHORT) ||
             (gfCastCfg.getCastOp() == GfloatCastOpType::CAST_HALF_TO_CHAR)) {
    return gfloatPack(graph, input, param, CS, gfCastCfg);
  } else if ((gfCastCfg.getCastOp() == GfloatCastOpType::CAST_SHORT_TO_FLOAT) ||
             (gfCastCfg.getCastOp() == GfloatCastOpType::CAST_CHAR_TO_HALF)) {
    return gfloatUnpack(graph, input, param, CS, gfCastCfg);
  } else {
    throw poplibs_error(
       "popfloat::GenericFloatCastOp: Cast Op not supported");
  }
}

poplar::Tensor castToGfloat(Graph &graph, Tensor input, Tensor param,
                            Sequence &prog, const GfloatCastConfig &gfCastCfg,
                            const std::string &debugPrefix) {
  const auto CS =
    graph.addComputeSet(debugPrefix + "/castToGfloat/" +
                        gfloatCastOpTypeToString(gfCastCfg.getCastOp()));

  auto output = castToGfloat(graph, input, param, CS, gfCastCfg);

  prog.add(Execute(CS));

  return output;
}

void castToGfloatInPlace(Graph &graph, Tensor input, Tensor param,
                         const ComputeSet &CS,
                         const GfloatCastConfig &gfCastCfg,
                         const std::string &debugPrefix) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  auto inFlat  = input.flatten();
  const auto mapping = graph.getTileMapping(inFlat);
  const auto grainSize =
    std::max<unsigned>(target.getVectorWidth(input.elementType()),
                       target.getAtomicStoreGranularity());

  const auto vertexTemplate = genericFloatCastVertexName(gfCastCfg, true);
  auto vSrMask = createSRMask(gfCastCfg.getNumSRBits(),
                              gfCastCfg.getCastOp());

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(inFlat, mapping[tile]);

    auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(CS,
                               vertexTemplate,
                               { { "param", param },
                                 { "inOut", inFlat.slices(regions) } });
      if (gfCastCfg.getSrNoiseDensity() != GfloatSRDensityType::INVALID) {
        graph.setInitialValue(v["corrParams"], gfCastCfg.getNoiseParams());
        graph.setInitialValue(v["distParam"], gfCastCfg.getDensityParam());
      }
      graph.setInitialValue(v["srMask"], vSrMask);
      graph.setTileMapping(v, tile);
    }
  }
}

void castToGfloatInPlace(Graph &graph, Tensor input, Tensor param,
                         Sequence &prog, const GfloatCastConfig &gfCastCfg,
                         const std::string &debugPrefix) {
  const auto CS = graph.addComputeSet(debugPrefix + "/castToGfloatInPlace/");

  castToGfloatInPlace(graph, input, param, CS, gfCastCfg);

  prog.add(Execute(CS));
}

}
