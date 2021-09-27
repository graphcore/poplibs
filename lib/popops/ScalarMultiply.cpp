// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "ScalarMultiply.hpp"

#include <cassert>
#include <memory>
#include <sstream>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/OptionParsing.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

namespace popops {

struct ScalarMultiplyOptions {
  double floatToHalfTolerance = 1e-6;
};

ScalarMultiplyOptions parseOptionFlags(const poplar::OptionFlags &options) {
  ScalarMultiplyOptions scalarMultiplyOptions;
  const poplibs::OptionSpec scalarMultiplySpec{
      {"scalarMultiplyFloatToHalfTolerance",
       poplibs::OptionHandler::createWithDouble(
           scalarMultiplyOptions.floatToHalfTolerance)},
  };
  for (const auto &entry : options) {
    scalarMultiplySpec.parse(entry.first, entry.second);
  }
  return scalarMultiplyOptions;
}

static poplar::Tensor
scalarMultiplyImpl(poplar::Graph &graph, const poplar::Tensor &a,
                   const poplar::Tensor &b, poplar::program::Sequence &prog,
                   bool inplace, poputil::PoplibsOpDebugInfo &di,
                   const poplar::OptionFlags &options_) {
  auto options = parseOptionFlags(options_);

  auto aType = a.elementType();
  auto bType = b.elementType();

  auto csName = inplace ? "scalarMultiplyInplace" : "scalarMultiply";
  const auto cs = graph.addComputeSet({di, csName});

  auto c = inplace ? a : graph.clone(a);

  auto aFlat = a.flatten();
  auto cFlat = c.flatten();
  auto b1D = b.reshape({1});

  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(bType);

  graph.reorderToSimplify(&aFlat, {&cFlat}, false);
  const auto &tileMapping = graph.getTileMapping(aFlat);

  for (unsigned tile = 0; tile < tileMapping.size(); tile++) {
    if (tileMapping[tile].empty()) {
      continue;
    }

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, tileMapping[tile]);
    std::string vertexName = "popops::ScalarMultiply";

    if (tileContiguousRegions.size() == 1) {
      vertexName += "1D";
      vertexName += inplace ? "Inplace" : "";
      vertexName = poputil::templateVertex(vertexName, aType, bType);

      const auto aContiguous =
          poplar::concat(aFlat.slices(tileContiguousRegions));
      const auto cContiguous =
          poplar::concat(cFlat.slices(tileContiguousRegions));

      const auto vertex = graph.addVertex(cs, vertexName);
      if (inplace) {
        graph.connect(vertex["in1Out"], aContiguous);
      } else {
        graph.connect(vertex["in1"], aContiguous);
        graph.connect(vertex["out"], cContiguous);
      }
      graph.connect(vertex["in2"], b1D);
      graph.setInitialValue(vertex["tolerance"], options.floatToHalfTolerance);
      graph.setTileMapping(vertex, tile);
    } else {
      vertexName += "2D";
      vertexName += inplace ? "Inplace" : "";
      vertexName = poputil::templateVertex(vertexName, aType, bType);

      const auto regionss = poputil::splitRegionsBetweenWorkers(
          target, tileContiguousRegions, vectorWidth, 2 * vectorWidth);

      for (const auto &regions : regionss) {
        if (regions.empty()) {
          throw poputil::poplibs_error("No regions for " + vertexName +
                                       " worker.");
        }
        const auto vertex = graph.addVertex(cs, vertexName);
        if (inplace) {
          graph.connect(vertex["in1Out"], aFlat.slices(regions));
        } else {
          graph.connect(vertex["in1"], aFlat.slices(regions));
          graph.connect(vertex["out"], cFlat.slices(regions));
        }
        graph.connect(vertex["in2"], b1D);
        graph.setInitialValue(vertex["tolerance"],
                              options.floatToHalfTolerance);
        graph.setTileMapping(vertex, tile);
      }
    }
  }

  prog.add(poplar::program::Execute(cs));

  return c;
}

static std::string invalidOperandsErrMsg(const poplar::Tensor &a,
                                         const poplar::Tensor &b,
                                         bool inplace) {
  using boost::adaptors::transformed;
  using boost::algorithm::join;

  auto toStr = [](const std::vector<size_t> &v) {
    return join(v | transformed([](size_t d) { return std::to_string(d); }),
                ", ");
  };

  auto aShapeStr = toStr(a.shape());
  auto bShapeStr = toStr(b.shape());
  auto funName = inplace ? "scalarMultiplyInplace" : "scalarMultiply";

  std::stringstream ss;
  ss << "Invalid operands of shape and type - ({" << aShapeStr << "}, "
     << a.elementType() << ") and ({" << bShapeStr << "}, " << b.elementType()
     << ") - provided to `popops::" << funName << "()`.";

  return ss.str();
}

void scalarMultiplyInplace(poplar::Graph &graph, const poplar::Tensor &a,
                           const poplar::Tensor &b,
                           poplar::program::Sequence &prog,
                           poputil::PoplibsOpDebugInfo &di,
                           const poplar::OptionFlags &options) {
  if (inputsMatchMixedPrecisionScalarMultiplyPattern(a, b)) {
    scalarMultiplyImpl(graph, a, b, prog, true, di, options);
  } else {
    throw poputil::poplibs_error(invalidOperandsErrMsg(a, b, true));
  }
}

poplar::Tensor scalarMultiply(poplar::Graph &graph, const poplar::Tensor &a,
                              const poplar::Tensor &b,
                              poplar::program::Sequence &prog,
                              poputil::PoplibsOpDebugInfo &di,
                              const poplar::OptionFlags &options) {
  if (inputsMatchMixedPrecisionScalarMultiplyPattern(a, b)) {
    return scalarMultiplyImpl(graph, a, b, prog, false, di, options);
  } else if (inputsMatchMixedPrecisionScalarMultiplyPattern(b, a)) {
    return scalarMultiplyImpl(graph, b, a, prog, false, di, options);
  } else {
    throw poputil::poplibs_error(invalidOperandsErrMsg(a, b, false));
  }
}

bool inputsMatchMixedPrecisionScalarMultiplyPattern(const poplar::Tensor &a,
                                                    const poplar::Tensor &b,
                                                    bool orderInvariant) {
  auto condition = [](const poplar::Tensor &a, const poplar::Tensor &b) {
    return a.elementType() == poplar::HALF &&
           b.elementType() == poplar::FLOAT && b.numElements() == 1;
  };
  return orderInvariant ? condition(a, b) || condition(b, a) : condition(a, b);
}

} // namespace popops
