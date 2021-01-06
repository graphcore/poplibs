// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poprand/RandomGen.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include "poplar/RandomSeed.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/exceptions.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Expr.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/optional.hpp>
#include <cstdint>
#include <limits>

using namespace poputil;
using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
namespace pe = popops::expr;

namespace poprand {

// flatten 2D vector of intervals to a 1D vector
static std::vector<Interval>
flatten(const std::vector<std::vector<Interval>> &intervals2D) {
  std::vector<Interval> flattenedIntervals;
  for (const auto &intervals1D : intervals2D) {
    flattenedIntervals.insert(flattenedIntervals.end(), std::begin(intervals1D),
                              std::end(intervals1D));
  }
  return flattenedIntervals;
}

static void seedTensorChecks(const Tensor *seed) {
  if (seed) {
    if (seed->rank() != 1) {
      // We could allow seed of any shape as long as it has the required number
      // of elements. For now, impose the stricter condition
      throw poputil::poplibs_error("seed tensor must have rank 1");
    }
    if (seed->numElements() != 2) {
      throw poputil::poplibs_error("seed tensor must have 2 elements");
    }
    if (seed->elementType() != poplar::UNSIGNED_INT) {
      throw poputil::poplibs_error("seed tensor must be of type UNSIGNED_INT");
    }
  }
}

// wrapper for nextafterf which doesn't take a step if 'from' is already
// representable as a float
template <typename T, typename T2> static float maybenextafterf(T from, T2 to) {
  float fromF = from;
  if ((from > to && fromF > from) || (from < to && fromF < from))
    fromF = std::nextafterf(fromF, to);
  return fromF;
}

// Basic implementation of maybenextafterf for IEEE FP16 (10 mantissa bits)
// Should generalise for any number of mantissa or exponent bits.
template <typename T, typename T2> static double maybenextafterh(T d, T2 to) {
  // Hard code for IEEE half
  const int mantissaBits = 10;
  const int exponentBits = 5;

  const int exponentBias = std::pow(2, exponentBits - 1) - 1;
  const int exponentMin = 1 - exponentBias;
  const double minSubnormal = std::pow(2, exponentMin - mantissaBits);
  const double minNormal = std::pow(2, exponentMin);
  // Maximum representable value is 1 step lower than maximum exponent
  const double maxNormal =
      std::pow(2, exponentBias + 1) - std::pow(2, exponentBias - mantissaBits);

  // Remember the sign and treat negatives like positives
  int sign = std::signbit(d) ? -1 : 1;
  int toSign = std::signbit(to) ? -1 : 1;
  d = std::abs(d);
  to = std::abs(to);
  bool sameSign = sign == toSign;

  // Handle d in [maxNormal, infinity)
  if (d > maxNormal) {
    if (to < d || sameSign)
      return sign * maxNormal;
    throw poputil::poplibs_error(
        "maybenextafterh: Next representable value in"
        " HALF type for " +
        std::to_string(d) + " towards " + std::to_string(to) +
        " failed. Make sure your range fits inside a HALF.");
  }

  // Handle d in [0, minSubnormal)
  if (d < minSubnormal)
    // Note: Either go up to the min subnormal or down to 0 based on 'to'
    return minSubnormal * (to > d) * sign * sameSign;

  // Determine the range d is enclosed in: 2^x <= d < 2^(x+1)
  double x = std::floor(std::log2(d));

  // Determine the step size from the range 2^x/2^m = 2^(x-m)
  double stepSize = std::pow(2, x - mantissaBits); // x in [2^x, 2^(x+1))

  // Handle d in [minSubnormal, minNormal) - special case for stepSize
  if (d < minNormal)
    stepSize = minSubnormal;

  // Round to the next multiple of stepSize from d ...
  double rem = std::fmod(d - std::pow(2, x), stepSize);
  // Note: If rem is 0 then d is representable in half
  if (rem == 0)
    return sign * d;
  // ... in the direction of 'to'
  return sign * (d + stepSize * (to > d) * sameSign - rem);
}

// Squeezes a double range [a, b] inward based on a poplar Type.
static std::pair<double, double> squeezeRange(Type ptype, double a, double b) {
  if (ptype == INT || ptype == UNSIGNED_INT)
    return std::make_pair(std::ceil(a), std::floor(b));
  if (ptype == FLOAT)
    return std::make_pair(maybenextafterf(a, b), maybenextafterf(b, a));
  if (ptype == HALF)
    return std::make_pair(maybenextafterh(a, b), maybenextafterh(b, a));
  throw poputil::poplibs_error("squeezeRange: unsupported Poplar type");
}

// Convert a range [minVal, maxVal] for uniform number generation into a
// scale and offset used internally by the uniform random number generator
static std::pair<double, double>
uniformScaleAndOffset(double minVal, double maxVal, const Type &dType) {
  if (dType != INT) {
    // round the limits inwards to representable floats
    // avoid STDC FPENV_ACCESS due to incomplete clang support
    float minValF = maybenextafterf(minVal, maxVal);
    float maxValF = maybenextafterf(maxVal, minVal);
    minVal = minValF;
    maxVal = maxValF;

    double scale = double(maxValF) - minValF;
    float scaleF = maybenextafterf(scale, 0.f);
    float offsetF = scaleF / 2.f + minValF;

    // The core generator returns numbers in the range [-0.5:+0.5] not [0:1]
    // and may have a different rounding mode to the host.
    // Ensure that generated values will be within [minValF:maxValF] by
    // reducing the scale so worst-case rounding respects the limits.

    if (dType == FLOAT) {
      // Check whether rounding will happen due to the quantisation to float;
      // note this calculation is in double-precision.
      if (-0.5 * scaleF + offsetF < minVal ||
          +0.5 * scaleF + offsetF > maxVal) {
        // The random generator output is scaled by 0.5, so make two steps
        // which will move both max and min in by equal amounts. This may be
        // more pessimistic than is strictly required.
        scaleF = std::nextafterf(scaleF, 0.f);
        scaleF = std::nextafterf(scaleF, 0.f);
        logging::poprand::debug(
            "uniformScaleAndOffset(float) coerced scale to {}", scaleF);
      }
    } else if (dType == HALF) {
      // For halves we only check that we're not going to include zero when
      // it's pulled within the limits by rounding. Other values may still round
      // and give out-of-interval samples. TODO: T13265 Improve this situation.

      const float halfThreshold = 2.0f * powf(2.0f, -14.f); // 2*min normal
      if (minValF > 0.f && minValF < halfThreshold)
        minValF = halfThreshold;
      if (maxValF < 0.f && maxValF > -halfThreshold)
        maxValF = -halfThreshold;
      if (minValF == halfThreshold || maxValF == -halfThreshold) {
        scaleF = maxValF - minValF;
        // shrink the scale by 1-2^-10 to reduce the product by 2 representable
        // values
        scaleF = scaleF * 0x3ff / 0x400;
        logging::poprand::debug(
            "uniformScaleAndOffset(half) coerced scale to {}", scaleF);
      }
    }
    return std::make_pair(scaleF, offsetF);
  } else {
    if (minVal < std::numeric_limits<int32_t>::min() ||
        maxVal > static_cast<double>(std::numeric_limits<int32_t>::max())) {
      throw poputil::poplibs_error("range for uniform distribution invalid");
    }
    double scale = maxVal - minVal;
    scale += 1.0;
    if (scale ==
        static_cast<double>(std::numeric_limits<uint32_t>::max()) + 1) {
      scale = 0;
    }
    return std::make_pair(scale, minVal);
  }
}

// If master seed tensor is not null then read hw seeds tensor and
// program master seed
// TODO: T12982 To avoid creating vertex state for each worker within the random
// generator codelets we add the getHwSeeds and setSeed program followed by the
// setHwSeeds program. This is not efficient in both cycles and memory but
// is an expedient solution. We can revisit this if memory and performance
// becomes an issue.
static boost::optional<Tensor>
maybeSaveHwSeedsAndSetSeeds(Graph &graph, const Tensor *masterSeed,
                            uint32_t seedModifier, Sequence &prog,
                            const DebugNameAndId &dnai) {
  if (masterSeed) {
    auto hwSeeds = getHwSeeds(graph, prog, dnai.getPathName());
    setSeed(graph, *masterSeed, seedModifier, prog, dnai.getPathName());
    return hwSeeds;
  }
  return boost::none;
}

// Restore Hw seeds
static void maybeRestoreHwSeeds(Graph &graph,
                                const boost::optional<Tensor> &hwSeeds,
                                Sequence &prog, const DebugNameAndId &dnai) {
  if (hwSeeds != boost::none) {
    setHwSeeds(graph, *hwSeeds, prog, dnai.getPathName());
  }
}

Tensor uniform(Graph &graph, const Tensor *masterSeed, uint32_t seedModifier,
               const Tensor &reference, const Type &outType, double minVal,
               double maxVal, Sequence &prog,
               const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(masterSeed, reference, seedModifier, outType, minVal, maxVal));

  if (outType != FLOAT && outType != HALF && outType != INT)
    throw poputil::poplibs_error(
        "uniform only supported for FLOAT/HALF/INT, '" + outType.toString() +
        "' not supported");
  seedTensorChecks(masterSeed);
  const std::string fnPrefix = "uniform";
  auto out = graph.clone(outType, reference, {di, fnPrefix + "/out"});

  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, {di, fnPrefix});
  auto cs = graph.addComputeSet({di, fnPrefix});
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {}, false);
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  double scale, offset;
  std::tie(scale, offset) = uniformScaleAndOffset(minVal, maxVal, outType);

  unsigned int shift = 31;
  if (outType == INT) {
    unsigned tmpScale = (scale < 1.0) ? 1.0 : scale;
    shift = 31 - std::log2(tmpScale);
    int shiftR = (shift < 24) ? (24 - shift) : 0;
    int shiftL = (shift > 24) ? (shift - 24) : 0;

    tmpScale = scale;
    tmpScale += (1 << shiftR) - 1;
    tmpScale >>= shiftR;
    tmpScale <<= shiftL;
    scale = (tmpScale < 255) ? tmpScale : 255;
  }

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);

    const auto vertexTemplate =
        templateVertex("poprand::UniformSupervisor", outType);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", concat(outFlat.slices(intervals))}});
    graph.setInitialValue(v["scale"], scale);
    graph.setInitialValue(v["offset"], offset);
    graph.setInitialValue(v["shift"], shift);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs, {di}));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, {di, fnPrefix});
  di.addOutput(out);
  return out;
}

Tensor logUniform(Graph &graph, const Tensor *masterSeed, uint32_t seedModifier,
                  const Tensor &reference, const Type &outType, double minVal,
                  double maxVal, Sequence &prog, double base,
                  const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(masterSeed, reference, seedModifier,
                                         outType, minVal, maxVal, base));
  const std::string fnPrefix = "logUniform";

  if (minVal < 1)
    throw poputil::poplibs_error("logUniform: minVal must be >= 1");
  if (maxVal <= minVal)
    throw poputil::poplibs_error("logUniform: maxVal must be > minVal");
  if (outType == INT &&
      ((std::trunc(minVal) != minVal) || (std::trunc(maxVal) != maxVal)))
    throw poputil::poplibs_error(
        "logUniform: For INT outType, minVal and"
        " maxVal should represent integers, otherwise there could be"
        " significant bias at the edges of the distribution.");

  // Determine if we're using the default, natural base
  double tolerance = 1e-7;
  float lnBase = std::log(base);
  bool useNaturalLog = std::abs(1.0 - lnBase) < tolerance;
  logging::poprand::debug(
      "logUniform: Using base [{}] - recognised as natural log: {}", base,
      useNaturalLog);

  // Determine range of the underlying uniform distribution in log space
  // Note: narrow the range inward to float representable values
  float minValF = maybenextafterf(minVal, maxVal);
  float maxValF = maybenextafterf(maxVal, minVal);
  float logMinVal = std::log(minValF);
  float logMaxVal = std::log(maxValF);
  if (!useNaturalLog) {
    logMinVal /= lnBase;
    logMaxVal /= lnBase;
  }
  logging::poprand::debug(
      "logUniform: Squeezed uniform range [{}, {}] to [{}, {}] based on"
      " type 'float'",
      minVal, maxVal, minValF, maxValF);

  // Generate uniformly distributed values in the log space
  poplar::Tensor x = uniform(graph, masterSeed, seedModifier, reference, FLOAT,
                             logMinVal, logMaxVal, prog, {di, fnPrefix});

  // Exponentiate back into initial space
  if (!useNaturalLog) {
    // Note: avoid using Pow by rearranging y = b^x into y = e^(x * log(b))
    popops::mapInPlace(graph, pe::Mul(pe::_1, pe::Const(lnBase)), {x}, prog,
                       {di, fnPrefix});
  }
  // Note: exp(log(x)) is not necessarily an identity in fp arithmetic, which
  // means samples in x could go out of bounds at the borders. Additionally,
  // the cast can go OOB too. To fix these, we clamp to a narrower range that
  // is based on the output type.
  auto squeezed = squeezeRange(outType, minValF, maxValF);
  logging::poprand::debug(
      "logUniform: Squeezed range [{}, {}] to [{}, {}] based on type '{}'",
      minValF, maxValF, squeezed.first, squeezed.second, outType);
  // Make sure that the squeezed range still satisfies minVal < maxVal
  // e.g. [28392,28393] maps to [28400,28384] for half
  if (squeezed.second < squeezed.first)
    throw poputil::poplibs_error(
        "logUniform: Range is too small for outType's"
        " representability. Try widening the range for this output type.");
  // Give a warning if squeezing loses more than 1% of the range
  if (squeezed.second - squeezed.first < (maxVal - minVal) * 0.99)
    logging::poprand::warn(
        "logUniform: Reducing size of range by more than one"
        " percent due to output type representability. Make sure your range is"
        " appropriate for your output type.");

  popops::mapInPlace(graph,
                     pe::Clamp(pe::Exp(pe::_1),
                               pe::Const(static_cast<float>(squeezed.first)),
                               pe::Const(static_cast<float>(squeezed.second))),
                     {x}, prog, {di, fnPrefix});
  auto out = popops::cast(graph, x, outType, prog, {di, fnPrefix});
  di.addOutput(out);
  return out;
}

Tensor bernoulli(Graph &graph, const Tensor *masterSeed, uint32_t seedModifier,
                 const Tensor &reference, const Type &outType, double prob,
                 Sequence &prog, const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(masterSeed, reference, seedModifier, outType, prob));
  seedTensorChecks(masterSeed);

  const std::string fnPrefix = "bernoulli";
  auto out = graph.clone(outType, reference, {di, fnPrefix + "/out"});
  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, {di, fnPrefix});

  auto cs = graph.addComputeSet({di, fnPrefix});
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {}, false);
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate =
        templateVertex("poprand::BernoulliSupervisor", outType);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", concat(outFlat.slices(intervals))}});
    // The probability used by f16v4rmask/f32v2rmask is the bottom 17-bits of
    // the 2nd input operand. Hence the scaling by 2^16.
    graph.setInitialValue(v["prob"], (unsigned)(prob * 65536.0));
    graph.setTileMapping(v, tile);
  }

  prog.add(Execute(cs, {di}));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, {di, fnPrefix});
  di.addOutput(out);
  return out;
}

Tensor normal(Graph &graph, const Tensor *masterSeed, uint32_t seedModifier,
              const Tensor &reference, const Type &outType, double mean,
              double stdDev, Sequence &prog,
              const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(masterSeed, reference, seedModifier, outType, mean, stdDev));
  seedTensorChecks(masterSeed);
  const std::string fnPrefix = "normal";
  auto out = graph.clone(outType, reference, {di, fnPrefix + "/out"});
  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, {di, fnPrefix});

  auto cs = graph.addComputeSet({di, fnPrefix});
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {}, false);
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate =
        templateVertex("poprand::NormalSupervisor", outType);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", concat(outFlat.slices(intervals))}});
    graph.setInitialValue(v["mean"], mean);
    graph.setInitialValue(v["stdDev"], stdDev);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs, {di}));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, {di, fnPrefix});
  di.addOutput(out);
  return out;
}

Tensor truncatedNormal(Graph &graph, const Tensor *masterSeed,
                       uint32_t seedModifier, const Tensor &reference,
                       const Type &outType, double mean, double stdDev,
                       double alpha, Sequence &prog,
                       const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(masterSeed, reference, seedModifier,
                                         outType, mean, stdDev, alpha));
  seedTensorChecks(masterSeed);
  const std::string fnPrefix = "truncatedNormal";
  auto out = graph.clone(outType, reference, {di, fnPrefix + "/out"});
  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, {di, fnPrefix});
  auto cs = graph.addComputeSet({di, fnPrefix});
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {}, false);
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  const float logProb = -4.0;
  const unsigned iterations =
      std::ceil(logProb / std::log10(std::erfc(alpha / std::sqrt(2.0))));

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate =
        templateVertex("poprand::TruncatedNormalSupervisor", outType);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", concat(outFlat.slices(intervals))}});
    graph.setInitialValue(v["mean"], mean);
    graph.setInitialValue(v["stdDev"], stdDev);
    graph.setInitialValue(v["alpha"], alpha);
    graph.setInitialValue(v["iterations"], iterations);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs, {di}));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, {di, fnPrefix});
  di.addOutput(out);
  return out;
}

Tensor dropout(Graph &graph, const Tensor *masterSeed,
               const uint32_t seedModifier, const Tensor &in,
               const Tensor &reference, double keepProbability, double scale,
               Sequence &prog, const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(masterSeed, in, reference, seedModifier, keepProbability, scale));
  seedTensorChecks(masterSeed);
  static const unsigned maxProbInHw = 65536;
  const std::string fnPrefix = "dropout";
  if (in.shape() != reference.shape()) {
    throw poputil::poplibs_error("Input and reference shapes must match in "
                                 "dropout");
  }

  if (keepProbability > 1 || keepProbability < 0) {
    throw poputil::poplibs_error("keep probability must be in the range [0,1]");
  }

  // The probability used by f16v4rmask/f32v2rmask
  unsigned probHw = static_cast<unsigned>(keepProbability * maxProbInHw);

  // Maximum probability in hw implies no dropout
  if (probHw == maxProbInHw) {
    return poputil::duplicate(graph, in, prog, {di});
  }

  auto out = graph.clone(in.elementType(), reference, {di, fnPrefix + "/out"});

  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, {di, fnPrefix});

  auto cs = graph.addComputeSet({di, fnPrefix});
  auto outFlat = out.flatten();
  auto inFlat = in.flatten();
  graph.reorderToSimplify(&inFlat, {&outFlat}, false);

  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate =
        templateVertex("poprand::DropoutSupervisor", in.elementType());
    auto inTile = concat(inFlat.slices(intervals));
    const auto &target = graph.getTarget();
    if (inTile.numElements() > target.getRptCountMax() *
                                   target.getNumWorkerContexts() *
                                   (inTile.elementType() == FLOAT ? 2 : 4)) {
      throw poputil::poplibs_error("Elements on tile exceed number that can "
                                   "be processed by codelet");
    }
    auto v = graph.addVertex(
        cs, vertexTemplate,
        {{"in", inTile}, {"out", concat(outFlat.slices(intervals))}});

    graph.setInitialValue(v["prob"], probHw);
    graph.setInitialValue(v["numElems"], inTile.numElements());
    graph.setInitialValue(v["scale"], scale);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs, {di}));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, {di, fnPrefix});
  di.addOutput(out);
  return out;
}

Tensor shapedDropout(Graph &graph, const Tensor *masterSeed,
                     const uint32_t seedModifier, const Tensor &in,
                     const Tensor &reference, double keepProbability,
                     double scale, Sequence &prog,
                     const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(masterSeed, in, reference, seedModifier, keepProbability, scale));
  seedTensorChecks(masterSeed);
  static const unsigned maxProbInHw = 65536;
  const std::string fnPrefix = "shaped_dropout";

  if (keepProbability > 1 || keepProbability < 0) {
    throw poputil::poplibs_error("keep probability must be in the range [0,1]");
  }

  // The probability used by f16v4rmask/f32v2rmask
  unsigned probHw = static_cast<unsigned>(keepProbability * maxProbInHw);

  // Maximum probability in hw implies no dropout
  if (probHw == maxProbInHw) {
    return poputil::duplicate(graph, in, prog, {di, fnPrefix});
  }

  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, {di, fnPrefix});

  auto mask =
      bernoulli(graph, masterSeed, seedModifier, reference, in.elementType(),
                keepProbability, prog, {di, fnPrefix});
  popops::mulInPlace(graph, mask, scale, prog, {di, fnPrefix});
  auto out = popops::mul(graph, in, mask, prog, {di, fnPrefix});

  maybeRestoreHwSeeds(graph, hwSeeds, prog, {di, fnPrefix});
  di.addOutput(out);
  return out;
}

void setSeed(poplar::Graph &graph, const poplar::Tensor &masterSeed,
             uint32_t seedModifier, poplar::program::Sequence &prog,
             const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(masterSeed, seedModifier));
  seedTensorChecks(&masterSeed);
  auto cs = graph.addComputeSet({di, "setMasterSeed"});
  const auto &target = graph.getTarget();
  auto numTiles = target.getNumTiles();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    auto v = graph.addVertex(cs, "poprand::SetSeedSupervisor",
                             {{"seed", masterSeed}});
    graph.setInitialValue(v["seedModifierUser"], seedModifier ^ 0x55555555U);
    // guarantee that even tile id 0 will have at least one bit set
    graph.setInitialValue(v["seedModifierHw"], (tile << 4) ^ 0xAAAAAAA0U);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs, {di}));
}

} // namespace poprand
