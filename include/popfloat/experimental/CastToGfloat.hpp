// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef popfloat_CastToGfloat_hpp
#define popfloat_CastToGfloat_hpp
#include "poputil/exceptions.hpp"
#include <popfloat/experimental/GfloatExpr.hpp>
#include <popfloat/experimental/GfloatExprUtil.hpp>
#include <poplar/DebugContext.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>

#include <functional>

/*
 * README: THIS CODE IS NOT SUPPORTED AND MAY BE REMOVED WITHOUT NOTICE.
 *
 * Using precision lower than FP32 allows more efficient computation and
 * reduces memory usage, permitting the deployment of larger networks, the
 * use of large batch sizes for memory limited systems and leads to faster
 * data transfers.  In order to study the effect of limited precision data
 * representation and computation on neural network training on the IPU,
 * this library introduces a set of functions that will allow the definition
 * of custom numerical formats.
 *
 * The popfloat library allows the emulation of custom numerical formats on
 * the IPU referred to as Gfloat. The user will be able to specify the float
 * format's mantissa size, exponent size, exponent bias, and also enable or
 * disable denormals as well as Inf/Nan signalling. Once the format has
 * been defined, the user can cast an input (float/half) tensor to the new
 * format and can choose from a selection of rounding modes (deterministic and
 * stochastic) for quantisation.
 * Supported deterministic rounding modes are:
 *    - round-to-zero,
 *    - round to nearest with ties to nearest even,
 *    - round to nearest with ties away from zero,
 *    - round towards positive infinity (ceil)
 *    - round towards negative infinity (floor).
 *
 * In the case of Stochastic Rounding (SR), the density function of the noise
 * samples that are used as well as the number of SR bits can be specified.
 *
 * The library also allows the quantised float format to be saved as 8-bit
 * or 16-bit values, provided that float format size (1+mantissa+exponent)
 * can be represented as an 8- or a 16-bit value.
 * Casting a gfloat format from its packed representation to a native higher
 * precision IEEE float format (IEEE float formats supported on the IPU for
 * calculation) is done with the castGfloatToNative functions.
 *
 */

namespace popfloat {
namespace experimental {

class GfloatCast {
public:
  struct GfloatFormatOptions {
    unsigned numMantissaBits;
    unsigned numExponentBits;
    unsigned numExponentBias;
    bool enableDenorms;
    bool enableInfsAndNans;

    void parseGfloatFormatOptions(const poplar::OptionFlags &options);

    GfloatFormatOptions(const poplar::OptionFlags &options) {
      parseGfloatFormatOptions(options);
    }

    GfloatFormatOptions() {
      numMantissaBits = 10;
      numExponentBits = 5;
      numExponentBias = 15;
      enableDenorms = true;
      enableInfsAndNans = true;
    }
  };

  struct FormatConfig {
    FormatConfig() = default;

    /*
     * GfloatFormatConfig: This structure stores the configuration parameters
     * of the generic float format, `gfloat`, and defines attributes of the
     * format used by the cast operations. Format parameters are:
     *  - numMantissaBits: defines the format's precision
     *  - numExponentBits: defines the number of magnitudes, for radix-2 format
     *  - exponentBias: offset of the stored exponent from the actual value
     *  - enableDenorms: enable gradual underflow
     *  - enableInfsAndNans: allow Inf/Nan signalling
     *  - specCalculationType: The native IPU IEEE float type used to
     *    calculate the gfloat format. Possible values are:
     *      - FP32: can be used for all gfloat formats
     *      - FP16: can only be used for gfloat formats that can be
     *              represented as IEEE FP16. i.e., when numMantissaBits
     *              and numExponentBits less than IEEE FP16's mantissa
     *              and exponent sizes, respectively).
     *      - AUTO: Will let the FormatConfig constructor choose the smallest
     *              native IEEE float to calculate the format.
     */
    FormatConfig(unsigned numMantissaBits, unsigned numExponentBits,
                 int exponentBias, bool enableDenorms, bool enableInfsAndNans,
                 popfloat::experimental::SpecType specCalculationType =
                     popfloat::experimental::SpecType::AUTO);

    FormatConfig(GfloatFormatOptions formatOptions,
                 poplar::Type calculationType);

    FormatConfig(unsigned numMantissaBits, unsigned numExponentBits,
                 int exponentBias, bool enableDenorms, bool enableInfsAndNans,
                 poplar::Type calculationType);

    /* Copy constructor */
    FormatConfig(const FormatConfig &formatConfig);

    poplar::Type getCalculationType() const { return calculationType; }
    poplar::Type getNativeType() const { return nativeType; }
    poplar::Type getStorageType() const { return storageType; }

    popfloat::experimental::FormatType getFormatType() const {
      return formatType;
    }

    unsigned getNumMantissaBits() const { return numMantissaBits; }
    unsigned getNumExponentBits() const { return numExponentBits; }
    int getExponentBias() const { return exponentBias; }
    bool isDenormEnabled() const { return enableDenorms; }
    bool infAndNansEnabled() const { return enableInfsAndNans; }
    bool isPackedFloatFormat() const {
      return (formatType !=
              popfloat::experimental::FormatType::QUANTISED_FP16) &&
             (formatType !=
              popfloat::experimental::FormatType::QUANTISED_FP32) &&
             (formatType != popfloat::experimental::FormatType::IEEE_FP16);
    };

    unsigned getPackedFloatBits() const { return packedFloatBits; };
    bool isBlockFloat() const { return blockFloat; };

    unsigned getPackedFloatParameters() const { return packedFloatParameters; };

    bool operator==(FormatConfig &other) const {
      const auto numMantissaBits_ = other.getNumMantissaBits();
      const auto numExponentBits_ = other.getNumExponentBits();
      const auto exponentBias_ = other.getExponentBias();
      const auto enableDenorms_ = other.isDenormEnabled();
      const auto enableInfsAndNans_ = other.infAndNansEnabled();
      const auto calculationType_ = other.getCalculationType();

      return std::tie(numMantissaBits, numExponentBits, exponentBias,
                      enableDenorms, enableInfsAndNans, calculationType) ==
             std::tie(numMantissaBits_, numExponentBits_, exponentBias_,
                      enableDenorms_, enableInfsAndNans_, calculationType_);
    }

  private:
    /*
     * calculationType: IEEE float type used to calculate the gfloat format.
     * To cast a native IEEE float type to a gfloat format, we can use
     *  - poplar::HALF only if the gfloat format can be represented as an IEEE
     *    FP16. i.e. when the number of gfloat mantissa bits is less than or
     *    equal the IEEE FP16 mantissa size (10) and the number of gfloat
     *    exponent bits is less than or equal to IEEE FP16 mantissa size (5).
     *  - poplar::FLOAT can be used for all gfloat formats.
     */
    poplar::Type calculationType;

    /*
     * numMantissaBits: The Gfloat format's mantissa field size, which
     * determines the number of fraction bits of the significand.
     */
    unsigned numMantissaBits;

    /*
     * numExponentBits: The Gfloat format's exponent field size.
     */
    unsigned numExponentBits;

    /*
     * exponentBias: The Gfloat format's exponent bias.
     */
    int exponentBias;

    /*
     * enableDenorms: to enable the Gfloat format's denormals. If false,
     * gradual underflow is disabled, and denormal values will not
     * represented. This means that the all-zero exponent field will
     * represent zero.
     */
    bool enableDenorms;

    /*
     * enableInfsAndNans: to enable the Gfloat format's Infs/Nans signalling.
     * This is ignored if numExponentBits=0. If enabled, input Infs/Nans are
     * always propagated.
     */
    bool enableInfsAndNans;

    /*
     * nativeType: the format config will choose the smallest IEEE float type
     * to represent the gfloat format. The result of quantisation is
     *  - poplar::HALF if the gfloat format can be represented as an IEEE FP16.
     *  - poplar::FLOAT if the gfloat format cannot be represented as an IEEE
     *    FP16.
     * NOTE:
     *  - If the calculationType is IEEE FP32 and the gfloat format can be
     *    represented as IEEE FP16, nativeType will be set to IEEE FP16. (For
     *    instance, when casting to a 1/3/4 format and using IEEE FP32 as a
     *    calculationType). Otherwise, nativeType and calculationType will be
     *    the same.
     *  - When creating a CastConfig the user can override the native type
     *    to use to represent a gfloat format. (See CastConfig).
     */
    poplar::Type nativeType;

    /*
     * storageType: the format config will choose the samllest type that can be
     * used to store a gfloat format
     *  - poplar::CHAR for a custom FP8 format.
     *  - poplar::SHORT for a custom FP16 format.
     *  - poplar::HALF for formats that can be represented as IEEE FP16.
     *  - poplar::FLOAT for all other format.
     */
    poplar::Type storageType;

    /*
     * formatType: Gfloat format type. The different gfloat format types are:
     *  - IEEE_FP16: This format denotes a cast from IEEE FP32 to IEEE FP16
     *       using rounding schemes not supported by the IPU
     *  - QUANTISED_FP32: Any Gfloat format that is stored as IEEE FP32
     *  - QUANTISED_FP16: Any Gfloat format that is stored as IEEE FP16
     *  - MIN_NORM_ALIGN_GF8: Any custom FP8 format with less than 5 exponent
     *        bits
     *  - ONE_FIVE_TWO_GF8: A 1/5/2 format with Infs/Nans enabled
     *  - MAX_NORM_ALIGN_GF8: A 1/5/2 format with Infs/Nans disabled
     *  - BFLOAT16: Google's Bfloat format (1/8/7)
     *  - NO_DENORM_GF16: A custom FP16 format with denorms disabled
     *  - ENABLE_DENORM_GF16: A custom FP16 with denorms enabled
     */
    popfloat::experimental::FormatType formatType;

    /*
     * packedFloatParameters: This is a packed representation of the gfloat
     * format's parameters using 4 bytes (stored as INT32). The parameter
     * packing is done such that:
     *    - one byte is used to store the number of mantissa bits
     *    - one byte is used to store the number of exponent bits
     *    - one byte is used to store the exponent bias
     *    - one bit is used to store the enableDenorms flag
     *    - one bit is used to store the enableInfsAndNans flag
     */
    unsigned packedFloatParameters;

    /*
     * blockFloat: This indicate if the format is INT or block-float. Block
     * floating-point values are gfloat formats with zero exponent bits or
     * gfloat formats with one exponent bit (numExponentBits=1) and Infs/Nans
     * disabled.
     */
    bool blockFloat;

    /*
     * packedFloatBits: the number of bits used to pack the gfloat format
     */
    unsigned packedFloatBits;
  };

  struct GfloatCastOptions {
    popfloat::experimental::RoundType roundMode;
    popfloat::experimental::SRDensityType srNoiseDensity;
    unsigned numSRBits;
    double srNoiseOffset;
    double srNoiseScale;
    double srNoiseMax;
    double srNoiseMin;
    double bernoulliProb;
    bool enableNanooMode;

    void parseGfloatCastOptions(const poplar::OptionFlags &options);

    GfloatCastOptions(const poplar::OptionFlags &options) {
      parseGfloatCastOptions(options);
    }

    GfloatCastOptions() {
      roundMode = popfloat::experimental::RoundType::INV;
      srNoiseDensity = popfloat::experimental::SRDensityType::INVALID;
      numSRBits = 24;
      srNoiseOffset = 0.0;
      srNoiseScale = 0.0;
      srNoiseMax = 0.0;
      srNoiseMin = 0.0;
      bernoulliProb = 0.0;
      enableNanooMode = true;
    }
  };

  struct RoundConfig {
    /*
     * RoundConfig: This structure stores the configuration parameters for
     * the rounding mode used in a castNativeToGfloat operation:
     *  - roundMode: quantisation rounding mode
     *  - numSRBits: number of random bits used for stochastic rounding,
     *  - srNoiseDensity: Stochasting rounding noise density,
     *  - srNoiseOffset: Stochastic rounding noise offset,
     *  - srNoiseScale: Stochastic rounding noise scaling factor,
     *  - srNoiseMax: Stochastic rounding maximum noise value,
     *  - srNoiseMin: Stochastic rounding minimum noise value,
     *  - bernoulliProb: Probability of rounding down for stochastic
     *      rounding with Bernoulli density
     */

    RoundConfig() = default;

    RoundConfig(popfloat::experimental::RoundType roundMode, unsigned numSRBits,
                poplar::Type calculationType,
                popfloat::experimental::SRDensityType srNoiseDensity =
                    popfloat::experimental::SRDensityType::INVALID,
                float srNoiseOffset = 0.0, float srNoiseScale = 0.0,
                float srNoiseMax = 0.0, float srNoiseMin = 0.0,
                float bernoulliProb = 0.0);

    RoundConfig(const GfloatCast::RoundConfig &roundCfg);

    RoundConfig(GfloatCastOptions castOptions, poplar::Type calculationType);

    popfloat::experimental::RoundType getRoundMode() const {
      return roundModeType;
    }

    unsigned getNumSRBits() const { return numSRBits; }

    popfloat::experimental::SRDensityType getSRNoiseDensity() const {
      return srNoiseDensity;
    }
    std::vector<unsigned> getRoundingParams() const { return roundingParams; }
    std::vector<unsigned> getNoiseParams() const { return noiseParams; }
    unsigned getDensityParam() const { return densityParam; }

    float getBernoulliProbability() const { return bernoulliProb; }

    float getSRNoiseOffset() const { return srNoiseOffset; }
    float getSRNoiseScale() const { return srNoiseScale; }
    float getSRNoiseMax() const { return srNoiseMax; }
    float getSRNoiseMin() const { return srNoiseMin; }

    std::vector<unsigned> getSRBitMask() const { return srBitMask; }

  private:
    /*
     * roundModeType: Quantisation rounding mode. Supported rounding modes are:
     *  - RZ: round-to-zero (truncate)
     *  - RA: round-to-nearest with ties rounding away from zero
     *  - RN: round-to-nearest with ties rounding to nearest even value
     *  - RU: round-towards positive infinity (ceil)
     *  - RD: round-towards negative infinity (floor)
     *  - SR: stochastic rounding using as many random bits as the truncated
     *        mantissa for rounding.
     *  - SX: Stochastic rounding eXtension to limit the maximum number of
     *        random bits and to use different noise distributions for
     *        stochastic rounding.
     */
    popfloat::experimental::RoundType roundModeType;

    /*
     * numSRBits: The number of random bits (N) used for stochastic rounding.
     * If T mantissa bits of the higher precision input are to be truncated, a
     * maximum of N or T random bits are used for stochastic rounding,
     * whichever is smallest. i.e., min(N,T) bits below the Gfloat's mantissa
     * LSB are used for stochastic rounding.
     */
    unsigned numSRBits;

    /*
     * srNoiseDensity: Stochastic rounding noise density.
     * Supported densities are
     *  - Uniform: the noise samples are uniformly distributed between
     *       two user-defined values min and max
     *  - Normal: the noise samples are normally distributed with a user-
     *       defined mean and standard deviation (stdDev). The values are
     *       clipped to a defined [min,max] range.
     *  - Truncated-Normal: the noise samples have a truncated normal
     *       distribution with a user-defined mean and standard deviation.
     *       Unlike the normal distribution, for truncated normal we sample
     *       from the normal distribution until all samples are in the range
     *       [min,max].
     *   - Laplace: the noise samples have a Laplace distribution with a user-
     *        defined offset (mu) and scale (b). The values are clipped to a
     *        defined [min,max] range.
     *   - Logistic: the noise samples have a logistic distribution with a
     *        user defined mean and scale (s). The values are clipped to a
     *        defined [min,max] range.
     *   - Logit-Normal: the noise samples have a logit-normal distribution
     *        with defined mean and scale parameter (standard of the normal
     *        values us whose logit is used). The values are clipped to a
     *        [min,max] range.
     *   - Truncated Logit-Normal: the noise samples have a logit-normal
     *        distribution clipped to a [min,max] range. The values whose
     *        logit is used, have a truncated normal distribution.
     *   - Bernoulli: the probability of rounding down is set for all inputs.
     */
    popfloat::experimental::SRDensityType srNoiseDensity;

    /*
     * bernoulliProb: used by the Bernoulli distribution as the stochastic
     * rounding probability of truncating the mantissa.
     */
    float bernoulliProb;

    /*
     * srNoiseOffset: Stochastic rounding noise samples offset. This is used
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
    float srNoiseOffset;

    /*
     * srNoiseScale: Stochastic rounding noise samples scale factor. This is
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
    float srNoiseScale;

    /*
     * srNoiseMax: Stochastic rounding noise samples maximum value. For the
     * following densities SRNoiseMax must satisfy:
     *  - Uniform: must be a value in the range [0,1]
     *  - Normal: must be a value in the range [-0.5,0.5]
     *  - Truncated must Normal: be a value in the range [-0.5,0.5]
     *  - Laplace: must be a value in the range [-0.5,0.5]
     *  - Logistic: must be a value in the range [0,1]
     *  - Logit-normal: must be a value in the range [0,1]
     *  - Truncated logit-normal: must be a value in the range [0,1]
     */
    float srNoiseMax;

    /*
     * srNoiseMin: Stochastic rounding noise samples minimum value. For the
     * different densities, SRNoiseMin must satisfy:
     *  - Uniform: must be a value in the range [0,1]
     *  - Normal: must be a value in the range [-0.5,0.5]
     *  - Truncated must Normal: be a value in the range [-0.5,0.5]
     *  - Laplace: must be a value in the range [-0.5,0.5]
     *  - Logistic: must be a value in the range [0,1]
     *  - Logit-normal: must be a value in the range [0,1]
     *  - Truncated logit-normal: must be a value in the range [0,1]
     */
    float srNoiseMin;

    /*
     * NOTE: Stochastic rounding noise density:
     * For a given higher precision input, x, the cast output is either
     * y1 or y2 such that y1<=x< y2. For a noise sample, n, with a given
     * density, the probability of x rounding down is given by:
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
     * noiseParams: the user defined stochastic rounding density parameters
     * (offset, scale, min, and max) will stored in one vector.
     */
    std::vector<unsigned> noiseParams;

    /*
     * densityParam: Other density parameters:
     *   - For truncated normal and truncated logit-normal this is the
     *     maximum number of times to sample from the Normal distribution
     *     per iteration
     *   - For Bernoulli, this is the scaled probability used by the `rmask`
     *     instruction
     */
    unsigned densityParam;

    /*
     * srBitMask: Bit mask used for stochastic rounding
     */
    std::vector<unsigned> srBitMask;

    /*
     * roundingParams: a vector of all rounding parameters
     */
    std::vector<unsigned> roundingParams;
  };

  /*
   * CastConfig: This structure stores the configuration parameters
   * of the gfloat cast operations. The different cast operations are:
   *  - Cast to quantised FP32 with the possibility to save the output
   *    as INT16 for custom FP16 formats.
   *  - Cast to quantised FP16 with the possibility to save the output
   *    as INT8 for custom FP8 formats.
   *  - Cast a custom FP16 to IEEE FP32 from the INT16 representation of the
   *    format.
   *  - Cast a custom FP8 to IEEE FP16 from the INT8 representation of the
   *    format.
   */
  struct CastConfig {
    CastConfig() = default;

    /** CastConfig constructor for ops casting an IEEE FP16/FP32 input to
     * a gfloat format (castNativeToGfloat)
     *
     * \param formatType Gfloat format type
     * \param calculationType the native IEEE float used for calculation
     * \param storageType type used to store the gfloat format
     * \param roundCfg Rounding parameters
     * \param enableNanooMode: Enable signalling Nanoo on overflow
     */
    static CastConfig
    createCastNativeToGF(popfloat::experimental::FormatType formatType,
                         poplar::Type calculationType, poplar::Type storageType,
                         RoundConfig roundCfg, bool enableNanooMode);

    /** CastConfig constructor for ops casting a gfloat format stored in
     * the smallest representation (INT8 or INT16 for custom FP8 and FP16,
     * respectively) to a native IEEE FP16/FP32 format (castGfloatToNative)
     *
     * \param formatType Gfloat format type
     * \param calculationType the native IEEE float used for calculation
     * \param nativeType the native IEEE float used to represent the gfloat
     *        format
     * \param storageType type used to store the gfloat format
     */
    static CastConfig
    createCastGFToNative(popfloat::experimental::FormatType formatType,
                         poplar::Type calculationType,
                         poplar::Type storageType);

    popfloat::experimental::RoundType getRoundMode() const {
      return roundConfig.getRoundMode();
    }

    unsigned getNumSRBits() const { return roundConfig.getNumSRBits(); }

    popfloat::experimental::SRDensityType getSRNoiseDensity() const {
      return roundConfig.getSRNoiseDensity();
    }

    std::vector<unsigned> getNoiseParams() const {
      return roundConfig.getNoiseParams();
    }

    unsigned getDensityParam() const { return roundConfig.getDensityParam(); }

    float getBernoulliProbability() const {
      return roundConfig.getBernoulliProbability();
    }

    float getSRNoiseOffset() const { return roundConfig.getSRNoiseOffset(); }

    float getSRNoiseScale() const { return roundConfig.getSRNoiseScale(); }

    float getSRNoiseMax() const { return roundConfig.getSRNoiseMax(); }

    float getSRNoiseMin() const { return roundConfig.getSRNoiseMin(); }

    std::vector<unsigned> getSRBitMask() const {
      return roundConfig.getSRBitMask();
    }

    bool isNanooModeEnabled() const { return enableNanooMode; }

    std::vector<unsigned> getCastParams() const { return castParams; }

    poplar::Type getCalculationType() const { return calculationType; }
    poplar::Type getStorageType() const { return storageType; }

    bool inPlaceOp(poplar::Type inType) const {
      return (inType == storageType);
    }

    popfloat::experimental::FormatType getFormatType() const {
      return floatFormatType;
    }

    bool getStoreAsNative() const { return storeAsNative; }

    std::vector<unsigned> getRoundingParams() const {
      return roundConfig.getRoundingParams();
    }

  private:
    CastConfig(popfloat::experimental::FormatType floatFormatType,
               poplar::Type calculationType, poplar::Type storageType,
               RoundConfig roundCfg, bool enableNanooMode);

    /*
     * calculationType: IEEE float type used to calculate the gfloat format
     *  - poplar::HALF iff the gfloat format can be represented as an
     *    IEEE FP16. i.e. when the number of gfloat mantissa bits is less
     *    than or equal theIEEE FP16 mantissa size (10) and the number of
     *    gfloat exponent bits is less than or equal to IEEE FP16 mantissa
     *    size (5).
     *  - poplar::FLOAT any gfloat format can be represented as an IEEE
     *    FP32.
     *  NOTE: This is copied from the calculationType used for FormatConfig.
     */
    poplar::Type calculationType;

    /*
     * storageType: type used to represent a custom float format
     *  - poplar::FLOAT for quantised FP32 formats.
     *  - poplar::HALF for quantised FP16 formats.
     *  - poplar::CHAR for a custom FP8 formats
     *  - poplar::SHORT for a custom FP16 formats
     *  NOTE: This can be copied from the storageType chosen by FormatConfig,
     *  or can set by the user.
     */
    poplar::Type storageType;

    /*
     * An instance of RoundConfig storing attributes of the rounding method
     * used in this cast operation.
     */
    RoundConfig roundConfig;

    /*
     * enableNanooMode: this is similar to the IPU's NaNOO mode
     *  - If true, the cast will generate QNaNs on overflow or when input
     *    values have magnitudes greater than the format's maximum value.
     *  - If false, the cast will clip on overflow and when input values
     *    are outside the range.
     * NOTE:
     *  - Regardless of whether the Nanoo mode is turned on or off, input
     *    Infs/Nans will always propagate.
     *  - This mode should be disabled if the format's Infs/Nans are not
     *    enabled or if the number of exponent bits is zero. Otherwise,
     *    when a quantised gfloat is packed (INT8 for custom FP8 and INT16
     *    for custom FP16 propagated Infs/Nans will be packed with the all
     *    one-exponent. When the packed value are unpacked, the values that
     *    used to be Infs/Nans, after quantisation, will become values with
     *    the format's maximum exponent. This is equivalent to disabling
     *    the propagation of Infs/Nans in quantisation.
     */
    bool enableNanooMode;

    /*
     * floatFormatType: the Gfloat format type. The different types are:
     *  - IEEE_FP16: When casting from IEEE FP32 to IEEE FP16 using
     *               rounding modes not supported by the IPU
     *  - QUANTISED_FP32: Any Gfloat format that can only be stored as
     *                    IEEE FP32
     *  - QUANTISED_FP16: Any Gfloat format that is stored as IEEE FP16
     *  - MIN_NORM_ALIGN_GF8: Any custom FP8 format with less than 5
     *                        exponent bits
     *  - ONE_FIVE_TWO_GF8: A 1/5/2 format with Infs/Nans enabled
     *  - MAX_NORM_ALIGN_GF8: A 1/5/2 format with Infs/Nans disabled
     *  - BFLOAT16: Google's Bfloat format (1/8/7) with denorms not enabled.
     *  - NO_DENORM_GF16: A custom FP16 format with denorms not enabled
     *  - ENABLE_DENORM_GF16: A custom FP16 with denorms enabled
     */
    popfloat::experimental::FormatType floatFormatType;

    /*
     * storeAsNative: Indicates if a gfloat format is stored as a Native IEEE
     * float if true, or if the gfloat format is packed to the smallest
     * bit representation.
     */
    bool storeAsNative;

    /*
     * castParams: A vector of all parameters used by cast vertex
     */
    std::vector<unsigned> castParams;
  };

  /** GfloatCast class constructor
   *
   * A GfloatCast is a cast engine that that defines the different methods to
   * cast to a user defined custom format.
   *
   * \param formatCfg: A FormatConfig with the format definition information
   * \param roundCfg: A RoundConfig with the rounding configuration
   *        parameters for the castNativeToGfloat function
   * \param enableNanooMode: Enable Nanoo mode for the cast ops
   * \param GFType: castNativeToGfloat output storage type. This can be
   *          - AUTO to use the storage type set by FormatConfig.
   *          - FP32 to use IEEE FP32 as a native type to store the output
   *          of the castNativeToGfloat instead of the FormatConfig's
   *          storageType. This means that if the FormatConfig's nativeType
   *          is IEEE FP16, castNativeToGfloat will cast the output to IEEE
   *          FP32.
   *          - FP16 to use IEEE FP16 as a native type to store this gfloat
   *          format instead of the native type chosen by FormatConfig. This
   *          means that, if the FormatConfig's nativeType is IEEE FP32,
   *          using IEEE FP16 will result of losing precision.
   * \param nativeType: the native floating point Type used to store the
   *        gfloat format when casting a Gfloat to a native type.This can be
   *          - AUTO to use the FormatConfig's nativeType.
   *          - FP32 to use IEEE FP32 as a native type to store the output of
   *          castGfloatToNative instead of FormatConfig's nativeType
   *          - FP16 to use IEEE FP16 as a native type to store the output of
   *          castGfloatToNative instead of FormatConfig's nativeType. This
   *          means that, if the FormatConfig's nativeType is IEEE FP32,the
   *          output of castNativeToGfloat will be cast to FP16 with a risk
   *          losing precision.
   */
  GfloatCast(const FormatConfig &formatCfg, const RoundConfig &roundCfg,
             const bool enableNanooMode,
             const popfloat::experimental::SpecType &GFType =
                 popfloat::experimental::SpecType::AUTO,
             const popfloat::experimental::SpecType &NativeType =
                 popfloat::experimental::SpecType::AUTO);

  GfloatCast(const GfloatFormatOptions &formatOtions,
             const GfloatCastOptions &castOptions, poplar::Type calculationType,
             const popfloat::experimental::SpecType &GFType =
                 popfloat::experimental::SpecType::AUTO,
             const popfloat::experimental::SpecType &NativeType =
                 popfloat::experimental::SpecType::AUTO);

  /** GfloatCast class copy constructor
   *
   * \param GfloatCast: A GfloatCast instance.
   */
  GfloatCast(const GfloatCast &gfloatCast);

  /** Create a cast function's parameters tensor.
   *
   * The shape of the tensor will depend on the calculation type
   *
   * \param graph           The tensor will be added to this graph
   * \param cs              Poplar compute set to append op onto
   * \param calculationType Native type used for Gfloat format calculation
   * \param gfPacked        The format's config parameters packed as INT32
   * \param debugContext    Optional debug information.
   * \return                A tensor of a Gfloat cast op's parameters
   */
  static poplar::Tensor
  createCastOpParamsTensor(poplar::Graph &graph, const poplar::ComputeSet &cs,
                           poplar::Type calculationType,
                           const unsigned gfPacked,
                           const poplar::DebugContext &debugContext = {});

  /** Create a cast function's parameters tensor.
   *
   * The shape of the tensor will depend on the calculation type
   *
   * \param graph             The tensor will be added to this graph
   * \param cs                Poplar compute set to append op onto
   * \param calculationType   Native type used for Gfloat format calculation
   * \param gfPacked          The format's config parameters packed as INT32
   * \param debugContext      Optional debug information.
   * \return                  A tensor of a Gfloat cast op's parameters
   */
  static poplar::Tensor
  createCastOpParamsTensor(poplar::Graph &graph, const poplar::ComputeSet &cs,
                           poplar::Type calculationType,
                           poplar::Tensor gfPacked,
                           const poplar::DebugContext &debugContext = {});

  /** Create a cast function's parameters tensor.
   *
   * The shape of the tensor will depend on the calculation type
   *
   * \param graph           The tensor will be added to this graph
   * \param prog            Poplar program sequence to append op onto
   * \param calculationType Native type used for Gfloat format calculation
   * \param gfStruct        The format's config parameters packed in structure
   * \param debugContext    Optional debug information.
   * \return                A tensor of a Gfloat cast op's parameters
   */
  static poplar::Tensor createCastOpParamsTensor(
      poplar::Graph &graph, poplar::program::Sequence &prog,
      poplar::Type calculationType, const unsigned gfStruct,
      const poplar::DebugContext &debugContext = {});

  /** Create a cast function's parameters tensor.
   *
   * The shape of the tensor will depend on the calculation type
   *
   * \param graph           The tensor will be added to this graph
   * \param prog            Poplar program sequence to append op onto
   * \param calculationType Native type used for Gfloat format calculation
   * \param gfStruct        The format's config parameters packed in structure
   * \param debugContext    Optional debug information.
   * \return                A tensor of a Gfloat cast op's parameters
   */
  static poplar::Tensor createCastOpParamsTensor(
      poplar::Graph &graph, poplar::program::Sequence &prog,
      poplar::Type calculationType, poplar::Tensor gfStruct,
      const poplar::DebugContext &debugContext = {});

  /** Initialise Class's cast function's parameters tensor.
   *
   * The shape of the tensor will depend on the calculation type
   *
   * \param graph         The tensor will be added to this graph
   * \param prog          Poplar program sequence to append op onto
   * \param debugContext  Optional debug information.
   * \return              A tensor of a Gfloat cast op's parameters
   */
  void createCastOpParamsTensor(poplar::Graph &graph,
                                poplar::program::Sequence &prog,
                                const poplar::DebugContext &debugContext = {});

  /** Initialise Class's cast function's parameters tensor.
   *
   * The shape of the tensor will depend on the calculation type
   *
   * \param graph         The tensor will be added to this graph
   * \param cs            Poplar compute set to append op onto
   */
  void createCastOpParamsTensor(poplar::Graph &graph,
                                const poplar::ComputeSet &cs);

  /** Cast an input tensor of a native IPU type to a gfloat format.
   *
   * The shape of the tensor will be the same as the input's
   *
   * \param graph         The tensor will be added to this graph
   * \param input         Input tensor to be quantised
   * \param param         Cast op's parameter tensor
   * \param prog          Poplar program sequence to append op onto
   * \param gfCastConfig  Structure storing op's arguments
   * \param debugContext  Optional debug information.
   * \return              A tensor of quantised elements
   */
  static poplar::Tensor castNativeToGfloat(
      poplar::Graph &graph, poplar::Tensor input, const poplar::Tensor &param,
      poplar::program::Sequence &prog, const CastConfig &gfCastCfg,
      const poplar::DebugContext &debugContext = {});

  /** Cast an input tensor of a native IPU type to a gfloat format.
   *
   * If the gfloat storage type is a native IPU type (IEEE FP32 or FP16),
   * the element type of the tensor will be the storage type, otherwise,
   * the element type will be the calculation type.
   *
   * The shape of the tensor will be the same as the input's
   *
   * \param graph         The tensor will be added to this graph
   * \param input         Input tensor to be quantised
   * \param param         Cast op's parameter tensor
   * \param cs            Poplar compute set to append op onto
   * \param gfCastCfg     Structure storing op's arguments
   * \param debugContext  Optional debug information.
   * \return              A tensor of quantised elements
   */
  static poplar::Tensor
  castNativeToGfloat(poplar::Graph &graph, poplar::Tensor input,
                     const poplar::Tensor &param, const poplar::ComputeSet &cs,
                     const CastConfig &gfCastCfg,
                     const poplar::DebugContext &debugContext = {});

  /** Cast an input tensor of a native IPU type to a gfloat format using
   * instance's cast op params and nativeToGFCastCfg CastConfig.
   *
   * The shape of the tensor will be the same as the input's
   *
   * \param graph         The tensor will be added to this graph
   * \param input         Input tensor to be quantised
   * \param prog          Poplar program sequence to append op onto
   * \param debugContext  Optional debug information.
   * \return              A tensor of quantised elements
   */
  poplar::Tensor
  castNativeToGfloat(poplar::Graph &graph, poplar::Tensor input,
                     poplar::program::Sequence &prog,
                     const poplar::DebugContext &debugContext = {});

  /** Cast an input tensor of a native IPU type to Gfloat inplace
   *
   * \param graph        The tensor will be added to this graph
   * \param input        Input tensor to be quantised
   * \param param        Cast op's parameter tensor
   * \param prog         Poplar program sequence to append op onto
   * \param gfCastCfg    Structure storing op's arguments
   * \param debugContext Optional debug information.
   * \return             A tensor of quantised elements
   */
  static void castNativeToGfloatInPlace(
      poplar::Graph &graph, poplar::Tensor input, const poplar::Tensor &param,
      poplar::program::Sequence &prog, const CastConfig &gfCastCfg,
      const poplar::DebugContext &debugContext = {});

  /** Cast an input tensor of a native IPU type to Gfloat inplace using
   * instance's cast op param tensor and gfToNativeCastCfg CastConfig.
   *
   * \param graph        The tensor will be added to this graph
   * \param input        Input tensor to be quantised
   * \param prog         Poplar program sequence to append op onto
   * \param gfCastCfg    Structure storing op's arguments
   * \param debugContext Optional debug information.
   * \return             A tensor of quantised elements
   */
  void castNativeToGfloatInPlace(poplar::Graph &graph, poplar::Tensor input,
                                 poplar::program::Sequence &prog,
                                 const poplar::DebugContext &debugContext = {});

  /** Cast an input tensor  of a native IPU type inplace
   *
   * \param graph         The tensor will be added to this graph
   * \param input         Input tensor to be quantised
   * \param param         Quantise op's parameter tensor
   * \param cs            Poplar compute set to append op onto
   * \param gfCastArgs    Structure storing op's arguments
   * \param debugContext  Optional debug information.
   * \return              A tensor of quantised elements
   */
  static void castNativeToGfloatInPlace(
      poplar::Graph &graph, poplar::Tensor input, const poplar::Tensor &param,
      const poplar::ComputeSet &cs, const CastConfig &gfCastCfg,
      const poplar::DebugContext &debugContext = {});

  /** Class method to cast an input tensor  of a native IPU type inplace
   * using instance's param tensor and gfToNativeCastCfg CastConfig.
   *
   * \param graph   The tensor will be added to this graph
   * \param input   Input tensor to be quantised
   * \param cs      Poplar compute set to append op onto
   * \return        A tensor of quantised elements
   */
  void castNativeToGfloatInPlace(poplar::Graph &graph, poplar::Tensor input,
                                 const poplar::ComputeSet &cs);

  /** Cast an input tensor of a gfloat type to a tensor of a native IPU type.
   *
   * The shape of the tensor will be the same as the input's
   *
   * \param graph        The tensor will be added to this graph
   * \param input        Input tensor to be quantised
   * \param param        Cast op's parameter tensor
   * \param prog         Poplar program sequence to append op onto
   * \param gfCastCfg    Structure storing op's arguments
   * \param debugContext Optional debug information.
   * \return             A tensor of quantised elements
   */
  static poplar::Tensor castGfloatToNative(
      poplar::Graph &graph, poplar::Tensor input, const poplar::Tensor &param,
      poplar::program::Sequence &prog, const CastConfig &gfCastCfg,
      const poplar::DebugContext &debugContext = {});

  /** Cast an input tensor of a gfloat type to a tensor of a native IPU type
   * using instance's param tensor and gfToNativeCastCfg CastConfig.
   *
   * The shape of the tensor will be the same as the input's
   *
   * \param graph         The tensor will be added to this graph
   * \param input         Input tensor to be quantised
   * \param prog          Poplar program sequence to append op onto
   * \param debugContext  Optional debug information.
   * \return              A tensor of quantised elements
   */
  poplar::Tensor
  castGfloatToNative(poplar::Graph &graph, poplar::Tensor input,
                     poplar::program::Sequence &prog,
                     const poplar::DebugContext &debugContext = {});

  /** Cast an input tensor of a gfloat type to a tensor of a native IPU type.
   *
   * The shape of the tensor will be the same as the input's
   *
   * \param graph         The tensor will be added to this graph
   * \param input         Input tensor to be quantised
   * \param param         Cast op's parameter tensor
   * \param cs            Poplar compute set to append op onto
   * \param gfCastCfg     Structure storing op's arguments
   *                      representation. If false, the input is quantised.
   * \param debugContext  Optional debug information.
   * \return              A tensor of quantised elements
   */
  static poplar::Tensor
  castGfloatToNative(poplar::Graph &graph, poplar::Tensor input,
                     const poplar::Tensor &param, const poplar::ComputeSet &cs,
                     const CastConfig &gfCastCfg,
                     const poplar::DebugContext &debugContext = {});

  /** Cast an input tensor of a gfloat type to a tensor of a native IPU type
   * using instance's param tensor and gfToNativeCastCfg CastConfig.
   *
   * The shape of the tensor will be the same as the input's
   *
   * \param graph         The tensor will be added to this graph
   * \param input         Input tensor to be quantised
   * \param cs            Poplar compute set to append op onto
   *                      representation. If false, the input is quantised.
   * \return              A tensor of quantised elements
   */
  poplar::Tensor castGfloatToNative(poplar::Graph &graph, poplar::Tensor input,
                                    const poplar::ComputeSet &cs);

  /** Get storage Type used to represent the output of the castNativeToGF op
   *
   * \return poplar::Type
   */
  poplar::Type getGFStorageType() const {
    return nativeToGFCastCfg.getStorageType();
  }

  /** Get calculation Type used in the cast functions
   *
   * \return poplar::Type
   */
  poplar::Type getCalculationType() const {
    return nativeToGFCastCfg.getCalculationType();
  }

  /** Get storage Type used to represent the output of the castGFToNative op
   *
   * \return poplar::Type
   */
  poplar::Type getNativeStorageType() const {
    return gfToNativeCastCfg.getStorageType();
  }

  /** Get FormatConfig member
   *
   * \return FormatConfig
   */
  FormatConfig getFormatConfig() const { return formatCfg; }

  /** Get nativeToGFCastCfg
   *
   * \return CastConfig
   */
  CastConfig getNativeToGFConfig() const { return nativeToGFCastCfg; }

  /** Get gfToNativeCastCfg
   *
   * \return CastConfig
   */
  CastConfig getGFToNativeConfig() const { return gfToNativeCastCfg; }

  /** Get nativeToGFCastCfg's storeAsNative member
   *
   * \return bool
   */
  bool getStoreAsNative() const { return nativeToGFCastCfg.getStoreAsNative(); }

  /** Get Cast op params tensor
   *
   * \return Tensor
   */
  poplar::Tensor getCastOpParams() const { return *gfParams; }

  /** Get Cast op params tensor
   *
   * \return Tensor
   */
  bool isCastOpParamSet() const { return castOpParamSet; }

  /** Set Cast op params tensor from constant tensor
   *
   */
  void setGfloatCastParameters(poplar::Tensor *gfParams_) {
    gfParams.reset(gfParams_);
  }

  bool isNanooModeEnabled() const {
    return nativeToGFCastCfg.isNanooModeEnabled();
  }

  std::vector<unsigned> getSRBitMask() const {
    return nativeToGFCastCfg.getSRBitMask();
  }

  popfloat::experimental::RoundType getRoundMode() const {
    return nativeToGFCastCfg.getRoundMode();
  }

  bool inPlaceOp(poplar::Type outType) const {
    return nativeToGFCastCfg.inPlaceOp(outType);
  }

  std::vector<unsigned> getRoundingParams() const {
    return nativeToGFCastCfg.getRoundingParams();
  }

protected:
  CastConfig nativeToGFCastCfg;
  CastConfig gfToNativeCastCfg;
  FormatConfig formatCfg;
  std::unique_ptr<poplar::Tensor> gfParams;
  bool castOpParamSet;
};

} // end namespace experimental
} // end namespace popfloat
#endif
