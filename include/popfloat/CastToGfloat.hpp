// Copyright (c) 2019, Graphcore Ltd, All rights reserved.
#ifndef popfloat_CastToGfloat_hpp
#define popfloat_CastToGfloat_hpp
#include "poputil/exceptions.hpp"
#include <poplar/Program.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Engine.hpp>
#include <popfloat/GfloatExpr.hpp>
#include <popfloat/GfloatExprUtil.hpp>
#include <poplar/Graph.hpp>

#include <functional>

/*
 * Using precision lower than FP32 allows more efficient computation and
 * reduces memory usage, permitting the deployment of larger networks, the use
 * of large batch sizes for memory limited systems and leads to faster data
 * transfers.  In order to study the effect of limited precision data
 * representation and computation on neural network training on the IPU, this
 * library introduces a set of functions that will allow the definition of
 * custom numerical formats on the IPU.
 *
 * The popfloat library allows the emulation of custom floating point formats on
 * the IPU. The user will be able to specify the float format's mantissa size,
 * exponent size, exponent bias, and also enable or disable denormals as well as
 * Inf/Nan signalling. Once the format has been defined, the user can cast an
 * input (float/half) tensor to the new format and can choose from a selection
 * of rounding modes (deterministic and stochastic) for quantisation.
 * Supported deterministic rounding modes are:
 *    - round-to-zero,
 *    - round to nearest with ties to nearest even,
 *    - round to nearest with ties away from zero,
 *    - round towards positive infinity (ceil)
 *    - round towards negative infinity (floor).
 *
 * In the case of stochastic rounding, the density function of the noise samples
 * that are used as well as the number of SR bits can be specified.
 *
 * The library also allows the quantised float format to be saved as 8-bit or
 * 16-bit values, provided that float format size (1+mantissa+exponent) can be
 * represented as an 8- or a 16-bit value.
 * Saving and loading to/from a higher precision float format (half/float) is
 * done with the pack and unpack functions.
*/
namespace popfloat {

struct GfloatFormatConfig {
  /*
   * GfloatFormatConfig: This structure stores the configuration parameters of
   * the generic float format, `gfloat`, and defines attributes of the format
   * used by the cast operations.
   */
  GfloatFormatConfig(int    numMantissaBits,
                     int    numExponentBits,
                     int    exponentBias,
                     bool   enableDenorms,
                     bool   enableInfsAndNans);

  poplar::Type getQuantisedOutputType() const { return quantisedOutputType; };
  poplar::Type getPackedOutputType() const { return packedOutputType; };
  poplar::Type getUnpackedOutputType() const { return unpackedOutputType; };

  popfloat::gfexpr::GfloatCastOpType getCastOpType(poplar::Type inType,
                                                   bool packOp);

  popfloat::gfexpr::GfloatCastOpType getQuantisedOpType() const {
    return quantisedOpType;
  };

  popfloat::gfexpr::GfloatCastOpType getPackOpType() const {
    return packOpType;
  };

  popfloat::gfexpr::GfloatCastOpType getUnpackOpType() const {
    return unpackOpType;
  };

  popfloat::gfexpr::GfloatFormatType getFormatType() const {
    return floatFormatType;
  };

  int  getNumMantissaBits() const { return numMantissaBits; };
  int  getNumExponentBits() const { return numExponentBits; };
  int  getExponentBias() const { return exponentBias; };
  bool isDenormEnabled() const { return enableDenorms; };
  bool infAndNansEnabled() const { return enableInfsAndNans; };
  bool isPackedFloatFormat() const {
    return
      ((floatFormatType != popfloat::gfexpr::GfloatFormatType::QUANTISED_FP16)
       &&
       (floatFormatType != popfloat::gfexpr::GfloatFormatType::QUANTISED_FP32));
  };

  unsigned getPackedFloatBits() const { return packedFloatBits; };
  bool isBlockFloat() const { return blockFloat; };

  unsigned getPackedFloatParameters() const { return packedFloatParameters; };

  bool isQuantisedHalfOutput(const poplar::Type &inType) const {
    return
      (inType == poplar::FLOAT) &&
      (quantisedOpType ==
       popfloat::gfexpr::GfloatCastOpType::CAST_TO_QUANTISED_GF16);
  };

  float getQuantisedOutputScale() const { return quantisedOutputScale; };

  float getQuantisedOutputScaleRecip() const {
    return quantisedOutputScaleRecip;
  };

private:
  /*
   * numMantissaBits: The Gfloat format's mantissa field size, or the number of
   * fraction bits of the significand.
   */
  int numMantissaBits;

  /*
   * numExponentBits: The Gfloat format's exponent field size.
   */
  int numExponentBits;

  /*
   * exponentBias: The Gfloat format's exponent bias.
   */
  int exponentBias;

  /*
   * enableDenorms: to enable the Gfloat format's denormals.
   */
  bool enableDenorms;

  /*
   * enableInfsAndNans: to enable the Gfloat format's Infs/Nans signalling. This
   * is ignored if numExponentBits=0. If enabled, input Infs/Nans are always
   * propagated.
   */
  bool enableInfsAndNans;

  /*
   * quantisedOpType: operation of casting an IEEE FP32 or FP16 to a gfloat
   * while representing the output as an IEEE FP16 or FP32. Supported
   * quantisation operations are:
   *  - Cast an IEEE FP32/FP16 to gfloat and represent the output as IEEE FP16
   *  - Cast an IEEE FP32 to gfloat and represent the output as IEEE FP32
   */
  popfloat::gfexpr::GfloatCastOpType quantisedOpType;

  /*
   * packOpType: operation of casting a gfloat from its IEEE FP16 or FP32
   * representation to its integer representation. Supported gfloat packing
   * operations are:
   *  - Cast a custom FP8 from its IEEE FP16 representation to INT8
   *  - Cast a custom FP16 from its IEEE FP32 representation to INT16
   */
  popfloat::gfexpr::GfloatCastOpType packOpType;

  /*
   * unpackOpType: operation of casting a gfloat format from its integer
   * representation to its IEEE FP16 or FP32 representation. Supported gfloat
   * unpacking cast operations are:
   *  - Cast a custom FP8 from its INT8 representation to IEEE FP16
   *  - Cast a custom FP16 from its INT16 representation to IEEE FP32
   */
  popfloat::gfexpr::GfloatCastOpType unpackOpType;


  /*
   * quantisedOutputType: IEEE float type used to represent the gfloat format
   *  The output type for a quantisation operation is
   *  - poplar::HALF if the gfloat format can be represented as an IEEE FP16.
   *    i.e. when the number of gfloat mantissa bits is less than or equal the
   *    IEEE FP16 mantissa size (10) and the number of gfloat exponent bits is
   *    less than or equal to IEEE FP16 mantissa size (5). If the number of
   *    mantissa bits and exponent bits of the gfloat format and IEEE FP16 are
   *    equal, Infs/Nans must be enabled.
   *  - poplar::FLOAT if the gfloat format cannot be represented as an IEEE
   *    FP16. i.e. when the number of gfloat mantissa bits is greater than the
   *    IEEE FP16 mantissa size (10) or the number of gfloat exponent bits is
   *    greater than the IEEE FP16 mantissa size (5). Or when casting to a
   *    (1/5/10) format with Infs/Nans turned off.
   */
  poplar::Type quantisedOutputType;

  /*
   * packedOutputType: type used to represent a custom FP8 or a custom FP16
   * gfloat format. The output type of a packing operation is
   *   - poplar::CHAR for a custom FP8 format.
   *   - poplar::SHORT for a custom FP16 format.
   */
  poplar::Type packedOutputType;

  /*
   * packedOutputType: the IEEE float type used to represent a custom FP8 or
   * a custom FP16 gfloat format. The output type of a packing operation is
   *   - poplar::HALF for a custom FP8 format.
   *   - poplar::SHORT for a custom FP16 format.
   */
  poplar::Type unpackedOutputType;

  /*
   * floatFormatType :Gfloat format type. The different gfloat format types are:
   *  - QUANTISED_FP32: Any Gfloat format that is stored as IEEE FP32
   *  - QUANTISED_FP16: Any Gfloat format that is stored as IEEE FP16
   *  - MIN_NORM_ALIGN_GF8: Any custom FP8 format with less than 5 exponent bits
   *  - ONE_FIVE_TWO_GF8: A 1/5/2 format with Infs/Nans enabled
   *  - MAX_NORM_ALIGN_GF8: A 1/5/2 format with Infs/Nans disabled
   *  - BFLOAT16: Google's Bfloat format (1/8/7)
   *  - NO_DENORM_GF16: A custom FP16 format with denorms disabled
   *  - ENABLE_DENORM_GF16: A custom FP16 with denorms enabled
   */
  popfloat::gfexpr::GfloatFormatType floatFormatType;

  /*
   * packedFloatParameters: Pack a Gfloat format's parameters as INT32.
   */
  unsigned packedFloatParameters;

  /*
   * blockFloat: Indicate if the format is INT or block-float. Bloack floating-
   * point values are gfloat formats with zero exponent bits or gfloat formats
   * with one exponent bit (numExponentBits=1) and Infs/Nans disabled.
   */
  bool blockFloat;

  /*
   * quantisedOutputScale: Scaling introduced by quantisation op:
   *  - QUANTISED_FP32: the quantisation does not introduce any scaling (=1.0)
   *  - QUANTISED_FP16: the value of the scaling applied to the input is
   *        2^(bias-15). The only exception to this is when the gfloat format's
   *        number of exponent bits is 5 and Infs/Nans are disabled, the scaling
   *        is 2^(bias-16)
   */
  float quantisedOutputScale;

  /*
   * quantisedOutputScaleRecip: reciprocal of the scaling applied by
   *   quantisation op
   */
  float quantisedOutputScaleRecip;

  /*
   * packedFloatBits: the number of bits used to pack the gfloat format
   */
  unsigned packedFloatBits;
};

struct GfloatCastConfig {
  /*
   * GfloatCastConfig: This structure stores the configuration parameters of the
   * gfloat cast operations.
   */

  GfloatCastConfig(poplar::Type inType, poplar::Type outType,
                   popfloat::gfexpr::GfloatCastOpType castOpType,
                   popfloat::gfexpr::GfloatRoundType roundMode,
                   bool enNanoo, unsigned srBits);

  GfloatCastConfig(poplar::Type inType, poplar::Type outType,
                   popfloat::gfexpr::GfloatCastOpType castOpType,
                   popfloat::gfexpr::GfloatSRDensityType srDist,
                   unsigned srBits, bool enNanoo, float mean,
                   float stdDev, float maxVal, float minVal, float prob);

  GfloatCastConfig(poplar::Type inType, poplar::Type outType,
                   popfloat::gfexpr::GfloatFormatType formatType);

  const popfloat::gfexpr::GfloatCastOpType getCastOp() const {
    return castOpType;
  };

  const popfloat::gfexpr::GfloatRoundType getRoundMode() const {
    return roundModeType;
  };

  const bool isNanooModeEnabled() const { return enableNanooMode; };
  const unsigned getNumSRBits() const { return numSRBits; };

  const popfloat::gfexpr::GfloatSRDensityType getSrNoiseDensity() const {
    return srNoiseDensity;
  };
  std::vector<unsigned> getNoiseParams() const { return noiseParams; };
  const unsigned getDensityParam() const { return densityParam; };

  const poplar::Type getInputType() const { return castInputType; };

  const poplar::Type getOutputType() const { return castOutputType; };

  float getBernoulliProbability() const { return bernoulliProb; };

  bool inPlaceOp(poplar::Type inType) const {
    return (inType == castOutputType);
  };

  popfloat::gfexpr::GfloatFormatType getFormatType() const {
    return floatFormatType;
  };

  const float getSRNoiseOffset() const { return SRNoiseOffset; };
  const float getSRNoiseScale() const { return SRNoiseScale; };
  const float getSRNoiseMax() const { return SRNoiseMax; };
  const float getSRNoiseMin() const { return SRNoiseMin; };

private:
  /*
   * castOpType: Cast op type. The supported cast operation types are
   *  - Cast IEEE FP32/FP16 to gfloat and represent the output as IEEE FP16
   *  - Cast IEEE FP32  to gfloat and represent the output as IEEE FP32
   *  - Cast a custom FP8 from its IEEE FP16 representation to INT8
   *  - Cast a custom FP16 from its IEEE FP32 representation to INT16
   *  - Cast a custom FP8 from its INT8 representation to IEEE FP16
   *  - Cast a custom FP16 from its INT16 representation to IEEE FP32
   */
  popfloat::gfexpr::GfloatCastOpType castOpType;

  /*
   * roundModeType: Quantisation rounding mode. Supported rounding modes are:
   *  - RZ: round-to-zero (truncate)
   *  - RA: round-to-nearest with ties rounding away from zero
   *  - RN: round-to-nearest with ties rounding to nearest even value
   *  - RU: round-towards positive infinity (ceil)
   *  - RD: round-towards negative infinity (floor)
   *  - SR: stochastic rounding using as many random bits as the truncated
   *        mantissa for rounding.
   *  - SX: stochastic rounding eXtension to limit the maximum number of random
   *        bit together with the truncated mantissa bits for rounding.
  */
  popfloat::gfexpr::GfloatRoundType roundModeType;

  /*
   * enableNanooMode: If true, the cast codelet will generate a QNaN on overflow
   * and for input values whose magnitudes are greater than the format's maximum
   * value (this is similar to the IPU's NaNOO mode). If false, the codelet will
   * clip on overflow and for input values outside the range.
   * Regardless of whether the Nanoo mode is turned on or off, input Infs/Nans
   * will always propagate.
   * NOTE: This mode should be disabled if the format's Infs/Nans are not
   * enabled or if the number of exponent bits is zero. Otherwise, when a
   * quantised gfloat is packed (INT8 for custom FP8 and INT16 for custom FP16),
   * propagated Infs/Nans will be packed with the all one-exponent. When the
   * packed value are unpacked, the values that used to be Infs/Nans, after
   * quantisation, will become values with the format's maximum exponent. This
   * is equivalent to disabling the propagation of Infs/Nans in quantisation.
   */
  bool enableNanooMode;

  /*
   * numSRBits: The number of random bits (N) used for stochastic rounding.
   * If T mantissa bits the higher precision input are to be truncated, a
   * maximum of N or T random bits are used for stochastic rounding, whichever
   * is smallest. i.e., min(N,T) bits below the Gfloat's mantissa LSB are used
   * for stochastic rounding.
   */
  unsigned numSRBits;

  /*
   * srNoiseDensity: Stochastic rounding noise density. Supported densities are:
   *  - Uniform: the noise samples are uniformly distributed between two user
   *       defined values min and max
   *  - Normal: the noise samples are normally distributed with a user defined
   *       mean and standard deviation (stdDev). The values are clipped to a
   *       defined [min,max] range
   *  - Truncated-Normal: the noise samples have a truncated normal distribution
   *       with a user defined mean and standard deviation. Unlike the normal
   *       distribution, for truncated normal we sample from the normal
   *       distribution until all samples are in the [min,max] range.
   *   - Laplace: the noise samples have a Laplace distribution with a user
   *        defined offset (mu) and scale (b). The values are clipped to a
   *        defined [min,max] range.
   *   - Logistic: the noise samples have a logistic distribution with a user
   *        defined mean and scale (s). The values are clipped to a defined
   *        [min,max] range.
   *   - Logit-Normal: the noise samples have a logit-normal distribution with a
   *        defined mean and scale parameter (standard of the normal values used
   *        whose logit is used). The values are clipped to a [min,max] range.
   *   - Truncated Logit-Normal: the noise samples have a logit-normal
   *        distribution clipped to a [min,max] range. The values whose
   *        logit is used, have a truncated normal distribution.
   *   - Bernoulli: the probability of rounding down is set for all inputs.
   */
  popfloat::gfexpr::GfloatSRDensityType srNoiseDensity;

  /*
   * bernoulliProb: used by the Bernoulli distribution as the stochastic
   * rounding probability of truncating the mantissa.
   */
  float bernoulliProb;

  /*
   * SRNoiseOffset: Stochastic rounding noise samples offset. This is used
   * by the following densities:
   *  - Normal: to set the distribution mean
   *  - Truncated Normal: to set the distribution mean
   *  - Laplace: to set the distribution offset parameter mu
   *  - Logistic: to set the distribution mean
   *  - Logit-normal: to set the mean of the normal distribution used to
   *       generate the samples
   *  - Truncated logit-normal: to set the mean of the normal distribution
   *       used to generate the samples
   */
  float SRNoiseOffset;

  /*
   * SRNoiseScale: Stochastic rounding noise samples scale factor. This is
   * used by the following densities:
   *  - Normal: to set the distribution standard deviation
   *  - Truncated Normal: to set the distribution standard deviation
   *  - Laplace: to set the distribution scale parameter b
   *  - Logistic: to set the distribution scale parameter s
   *  - Logit-normal: to set the standard deviation of the normal distribution
   *       used to generate the samples
   *  - Truncated logit-normal: to set the standard deviation of the normal
   *       distribution used to generate the samples
   */
  float SRNoiseScale;

  /*
   * SRNoiseMax: Stochastic rounding noise samples maximum value. For the
   * following densities SRNoiseMax must satisfy:
   *  - Uniform: must be a value in the range [0,1]
   *  - Normal: must be a value in the range [-0.5,0.5]
   *  - Truncated must Normal: be a value in the range [-0.5,0.5]
   *  - Laplace: must be a value in the range [-0.5,0.5]
   *  - Logistic: must be a value in the range [0,1]
   *  - Logit-normal: must be a value in the range [0,1]
   *  - Truncated logit-normal: must be a value in the range [0,1]
   */
  float SRNoiseMax;

  /*
   * SRNoiseMin: Stochastic rounding noise samples minimum value. For the
   * different densities, SRNoiseMin must satisfy:
   *  - Uniform: must be a value in the range [0,1]
   *  - Normal: must be a value in the range [-0.5,0.5]
   *  - Truncated must Normal: be a value in the range [-0.5,0.5]
   *  - Laplace: must be a value in the range [-0.5,0.5]
   *  - Logistic: must be a value in the range [0,1]
   *  - Logit-normal: must be a value in the range [0,1]
   *  - Truncated logit-normal: must be a value in the range [0,1]
   */
  float SRNoiseMin;

  /*
   * Note on stochastic rounding noise density:
   * For a given higher precision input, x, the cast output is either y1 or y2,
   * such that y1<=x< y2. For a noise sample, n, with a given density, the
   * probability of x rounding down is given by:
   *      p(y1|x,n) = p(x+(y2-y1)*n<y2)=p(n<(y2-x)/(y2-y1))
   * Scaling by (y2-y1) allows the noise samples to align below the mantissa
   * LSB. After adding noise, the bottom T bits of the mantissa are lost and
   * the result is truncated (RZ) or rounded-to-nearest away (RA), depending
   * on the density. The rounding modes for the different distribution are:
   *  - Uniform: truncate (RZ),
   *  - Normal: round to nearest (RA),
   *  - Truncated Normal: round to nearest (RA),
   *  - Laplace: round to nearest (RA),
   *  - Logistic: truncate (RZ),
   *  - Logit-Normal: truncate (RZ),
   *  - Truncated Logit-Normal: truncate (RZ)
   */

  /*
   * castInputType: A cast op input tensor type
   */
  poplar::Type castInputType;

  /*
   * castInputType: A cast op output tensor type
   */
  poplar::Type castOutputType;

  /*
   * noiseParams: the user defined stochastic rounding density parameters
   * (offset, scale, min, and max) will stored in one vector.
   */
  std::vector<unsigned> noiseParams;

  /*
   * densityParam: Other stochastic rounding density parameters:
   *   - For truncated normal and truncated logit-normal this is the maximum
   *     number of times to sample from the Normal distribution per iteration
   *   - For Bernoulli, this is the scaled probability used by the `rmask`
   *     instruction
   */
  unsigned densityParam;

  /*
   * floatFormatType : the Gfloat format type. the different format types are:
   *  QUANTISED_FP32: Any Gfloat format that can only be stored as IEEE FP32
   *  QUANTISED_FP16: Any Gfloat format that is stored as IEEE FP16
   *  MIN_NORM_ALIGN_GF8: Any custom FP8 format with less than 5 exponent bits
   *  ONE_FIVE_TWO_GF8: A 1/5/2 format with Infs/Nans enabled
   *  MAX_NORM_ALIGN_GF8: A 1/5/2 format with Infs/Nans disabled
   *  BFLOAT16: Google's Bfloat format (1/8/7) with denorms not enabled.
   *  NO_DENORM_GF16: A custom FP16 format with denorms not enabled
   *  ENABLE_DENORM_GF16: A custom FP16 with denorms enabled
   */
  popfloat::gfexpr::GfloatFormatType floatFormatType;
};

/** Packed parameters that define a Gfloat format in a tensor suitable for cast
 * ops' ()
 *
 * The shape of the tensor will be {1}
 *
 * \param graph         The tensor will be added to this graph
 * \param prog          Poplar program sequence to append op onto
 * \param gfFormatCfg   Gfloat format config structure
 * \return              A tensor of the packed Gfloat parameters
 */
poplar::Tensor setPackedGfloatParams(poplar::Graph &graph,
                                     poplar::program::Sequence &prog,
                                     const GfloatFormatConfig &gfFormatCfg);

/** Create a cast function's parameters tensor.
 *
 * The shape of the tensor will be depend on the Op
 *
 * \param graph         The tensor will be added to this graph
 * \param CS            Poplar compute set to append op onto
 * \param gfCastOpType  Cast op type
 * \param gfPacked      The format's config parameters packed in structure
 * \return              A tensor of a Gfloat cast op's parameters
 */
poplar::Tensor
createCastOpParamsTensor(poplar::Graph &graph, const poplar::ComputeSet &CS,
                         popfloat::gfexpr::GfloatCastOpType gfCastOpType,
                         const unsigned gfPacked);

/** Create a cast function's parameters tensor.
 *
 * The shape of the tensor will be depend on the Op
 *
 * \param graph         The tensor will be added to this graph
 * \param prog          Poplar program sequence to append op onto
 * \param gfCastOpType  Cast op type
 * \param gfStruct      The format's config parameters packed in structure
 * \return              A tensor of a Gfloat cast op's parameters
 */
poplar::Tensor
createCastOpParamsTensor(poplar::Graph &graph,
                         poplar::program::Sequence &prog,
                         popfloat::gfexpr::GfloatCastOpType gfCastOpType,
                         poplar::Tensor gfStruct,
                         const std::string  &debugPrefix = "");

/** Cast an input tensor to a gfloat format.
 *
 * The shape of the tensor will be the same as the input's
 *
 * \param graph            The tensor will be added to this graph
 * \param input            Input tensor to be quantised
 * \param param            Cast op's parameter tensor
 * \param prog             Poplar program sequence to append op onto
 * \param GfloatCastConfig Structure storing op's arguments
 * \return                 A tensor of quantised elements
 */
poplar::Tensor castToGfloat(poplar::Graph &graph, poplar::Tensor input,
                            poplar::Tensor param,
                            poplar::program::Sequence &prog,
                            const GfloatCastConfig &gfCastCfg,
                            const std::string &debugPrefix = "");

/** Cast an input tensor to a gfloat format.
 *
 * The shape of the tensor will be the same as the input's
 *
 * \param graph            The tensor will be added to this graph
 * \param input            Input tensor to be quantised
 * \param param            Cast op's parameter tensor
 * \param CS               Poplar compute set to append op onto
 * \param GfloatCastConfig Structure storing op's arguments
 * \return                 A tensor of quantised elements
 */
poplar::Tensor castToGfloat(poplar::Graph &graph, poplar::Tensor input,
                            poplar::Tensor param,
                            const poplar::ComputeSet &CS,
                            const GfloatCastConfig &gfCastCfg,
                            const std::string &debugPrefix = "");

/** Cast an input tensor inplace
 *
 * \param graph          The tensor will be added to this graph
 * \param input          Input tensor to be quantised
 * \param param          Quantise op's parameter tensor
 * \param prog           Poplar program sequence to append op onto
 * \param gfQuantiseArgs Structure storing op's arguments
 * \return               A tensor of quantised elements
 */
void castToGfloatInPlace(poplar::Graph &graph, poplar::Tensor input,
                         poplar::Tensor param,
                         poplar::program::Sequence &prog,
                         const GfloatCastConfig &gfCastCfg,
                         const std::string &debugPrefix = "");

/** Cast an input tensor inplace
 *
 * \param graph          The tensor will be added to this graph
 * \param input          Input tensor to be quantised
 * \param param          Quantise op's parameter tensor
 * \param CS             Poplar compute set to append op onto
 * \param gfQuantiseArgs Structure storing op's arguments
 * \return               A tensor of quantised elements
 */
void castToGfloatInPlace(poplar::Graph &graph, poplar::Tensor input,
                         poplar::Tensor param,
                         const poplar::ComputeSet &CS,
                         const GfloatCastConfig &gfCastCfg,
                         const std::string &debugPrefix = "");
}
#endif
