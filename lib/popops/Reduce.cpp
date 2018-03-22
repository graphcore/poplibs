#include "popops/Reduce.hpp"
#include "popops/Cast.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "popops/Zero.hpp"
#include "popops/ElementWise.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poputil/exceptions.hpp"
#include <algorithm>
#include <numeric>
#include <functional>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

// return vertex name for a given operation
const std::string getVertexStr(Operation operation) {
  switch (operation) {
  case Operation::ADD:
    return "ReduceAdd";
  case Operation::MUL:
    return "ReduceMul";
  case Operation::MIN:
    return "ReduceMin";
  case Operation::MAX:
    return "ReduceMax";
  case Operation::AND:
    return "ReduceAnd";
  case Operation::OR:
    return "ReduceOr";
  }
  POPLIB_UNREACHABLE();
}

// return vertex name for a given Operation
const Type getOutputType(Operation operation, const Type &inType) {
  switch (operation) {
  case Operation::ADD:
  case Operation::MUL:
  case Operation::MIN:
  case Operation::MAX:
    return inType;
  case Operation::AND:
  case Operation::OR:
    return BOOL;
  }
  POPLIB_UNREACHABLE();
}


static unsigned getMaxElementsPerTile(
    const std::vector<
      std::vector<Interval>
    > &reducedMapping) {
  unsigned maxElementsPerTile = 0;
  for (const auto &entry : reducedMapping) {
    unsigned tileElements =
        std::accumulate(entry.begin(), entry.end(), 0U,
                        [](unsigned sum,
                           const Interval &region) {
          return sum + region.end() - region.begin();
        });
    maxElementsPerTile = std::max(maxElementsPerTile, tileElements);
  }
  return maxElementsPerTile;
}

static unsigned estimateReduceAtDstCost(
    Graph &graph,
    Tensor partials,
    const std::vector<
      std::vector<Interval>
    > &reducedMapping) {
  const auto &target = graph.getTarget();
  const auto partialType = partials.elementType();
  const auto partialTypeBytes = target.getTypeSize(partialType);
  const auto partialVectorWidth = target.getVectorWidth(partialType);
  const auto maxElementsPerTile = getMaxElementsPerTile(reducedMapping);
  const auto partialsPerElement = partials.dim(0);
  const auto preComputeExchangeBytes =
      maxElementsPerTile * partialsPerElement * partialTypeBytes;
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();
  const auto syncCycles = target.getMaxIPUSyncDelay();
  unsigned cycles = 0;
  cycles += (preComputeExchangeBytes + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  cycles += partialsPerElement *
            ((maxElementsPerTile + partialVectorWidth - 1) /
             partialVectorWidth);
  cycles += syncCycles;
  return cycles;
}

static unsigned estimateBalancedReduceCost(
    Graph &graph,
    Tensor partials,
    Tensor reduced,
    const std::vector<
      std::vector<Interval>
    > &reducedMapping,
    unsigned grainSize) {
  const auto &target = graph.getTarget();
  const auto partialType = partials.elementType();
  const auto partialTypeBytes = target.getTypeSize(partialType);
  const auto partialVectorWidth = target.getVectorWidth(partialType);
  const auto reducedType = reduced.elementType();
  const auto reducedTypeBytes = target.getTypeSize(reducedType);
  unsigned numReducedElements = reduced.numElements();
  unsigned numReducedGroups = (numReducedElements + grainSize - 1) /
                              grainSize;
  const auto numTiles = graph.getTarget().getNumTiles();
  unsigned maxReducedGroups = (numReducedGroups + numTiles - 1) / numTiles;
  const auto maxElementsPerTile = maxReducedGroups * grainSize;
  const auto partialsPerElement = partials.dim(0);
  const auto preComputeExchangeBytes =
      maxElementsPerTile * partialsPerElement * partialTypeBytes;
  const auto postComputeExchangeBytes =
      getMaxElementsPerTile(reducedMapping) * reducedTypeBytes;
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();
  const auto syncCycles = target.getMaxIPUSyncDelay();
  unsigned cycles = 0;
  cycles += (preComputeExchangeBytes + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  cycles += syncCycles;
  cycles += partialsPerElement *
            ((maxElementsPerTile + partialVectorWidth - 1) /
             partialVectorWidth);
  cycles += (postComputeExchangeBytes + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  cycles += syncCycles;
  return cycles;
}

static std::vector<std::vector<Interval>>
determineReduceVertexMapping(Graph &graph,
                             Tensor partials,
                             Tensor reduced,
                             const std::vector<
                               std::vector<Interval>
                             > &reducedMapping) {
  const auto &target = graph.getTarget();
  const auto partialType = partials.elementType();
  const auto partialVectorWidth = target.getTypeSize(partialType);
  const auto reduceAtDstCost = estimateReduceAtDstCost(graph, partials,
                                                       reducedMapping);
  const auto grainSize = partialVectorWidth;
  const auto balancedReduceCost =
      estimateBalancedReduceCost(graph, partials, reduced, reducedMapping,
                                 grainSize);
  if (balancedReduceCost < reduceAtDstCost) {
    return poputil::calcLinearTileMapping(graph, reduced);
  }
  return reducedMapping;
}

static void
reduce(Graph &graph,
       Tensor partials,
       Tensor reduced, float k, bool isUpdate, bool isScale,
       Operation operation,
       const std::vector<
         std::vector<Interval>
       > &reduceVertexMapping,
       ComputeSet reduceCS) {
  // can't have both scale and update
  assert(!(isScale && isUpdate));

  assert(partials[0].shape() == reduced.shape());
  if (partials.dim(0) == 0) {
    zero(graph, reduced, reduceVertexMapping, reduceCS);
    return;
  }
  if (partials.dim(0) == 1) {
    cast(graph, partials[0], reduced, reduceCS);
    return;
  }
  const auto partialType = partials.elementType();
  const auto reducedType = reduced.elementType();
  const auto tilesPerInZGroup = partials.dim(0);
  auto flatPartials =
      partials.reshape({tilesPerInZGroup,
                        partials.numElements() / tilesPerInZGroup});
  auto flatReduced = reduced.flatten();
  const auto &target = graph.getTarget();
  std::string vertexName = "popops::" + getVertexStr(operation);
  if (isUpdate && operation == Operation::ADD) {
    vertexName = "popops::ReduceAddUpdate";
  }
  if (isScale && operation == Operation::ADD) {
    vertexName = "popops::ReduceAddScale";
  }

  // Accumulate the partial sums.
  const auto numUsedTiles =  reduceVertexMapping.size();
  assert(numUsedTiles <= target.getNumTiles());
  for (unsigned tile = 0; tile != numUsedTiles; ++tile) {
    const auto &tileRegions = reduceVertexMapping[tile];
    unsigned vectorWidth = target.getVectorWidth(partialType);
    const auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileRegions, vectorWidth);
    for (const auto &regions : vertexRegions) {
      const auto v = graph.addVertex(reduceCS,
                                     templateVertex(vertexName,
                                                    reducedType,
                                                    partialType));
      graph.setFieldSize(v["out"], regions.size());
      graph.setFieldSize(v["partials"], regions.size() * tilesPerInZGroup);
      if (isUpdate || isScale)
        graph.setInitialValue(v["k"], k);
      graph.setTileMapping(v, tile);
      const auto numRegions = regions.size();
      for (unsigned i = 0; i != numRegions; ++i) {
        const auto &region = regions[i];
        const auto regionBegin = region.begin();
        const auto regionEnd = region.end();
        auto out = flatReduced.slice(regionBegin, regionEnd);
        graph.connect(v["out"][i], out);
        for (unsigned j = 0; j != tilesPerInZGroup; ++j) {
          graph.connect(
            v["partials"][i * tilesPerInZGroup + j],
            flatPartials[j].slice(regionBegin, regionEnd)
          );
        }
      }
    }
  }
}

void
reduce(Graph &graph,
       Tensor partials,
       Tensor reduced,
       const std::vector<
         std::vector<Interval>
       > &reduceVertexMapping,
       ComputeSet reduceCS) {
  reduce(graph, partials, reduced, 1, false, false, Operation::ADD,
         reduceVertexMapping, reduceCS);
}

void
reduceByDstMapping(Graph &graph,
                   Tensor partials,
                   Tensor reduced,
                   const std::vector<
                     std::vector<Interval>
                   > &reducedMapping,
                   ComputeSet reduceCS) {
  if (partials.dim(0) < 2) {
    reduce(graph, partials, reduced, reducedMapping, reduceCS);
  }
  const auto reduceVertexMapping = determineReduceVertexMapping(graph,
                                                                partials,
                                                                reduced,
                                                                reducedMapping);
  return reduce(graph, partials, reduced, reduceVertexMapping, reduceCS);
}


Tensor reduce(poplar::Graph &graph,  poplar::Tensor in,
              poplar::program::Sequence &prog,
              const std::string &debugPrefix) {
  const auto numAddends = in.dim(0);
  const auto resultSize = in.dim(1);
  const auto dType = in.elementType();
  const auto out = graph.addVariable(dType, {resultSize},
                                     debugPrefix + "/Reduced");
  poputil::mapTensorLinearly(graph, out);

  // If batch size is 1 then no reduction is required.
  if (numAddends == 1) {
    prog.add(Copy(in[0], out));
    return out;
  }

  const auto cs = graph.addComputeSet(debugPrefix + "/Reduce");
  reduce(graph, in, out, graph.getTileMapping(out), cs);
  prog.add(Execute(cs));
  return out;
}

void reduceAcc(Graph &graph, Tensor out, float k, Tensor in,
               Sequence &prog, const std::string &debugPrefix) {
  auto reduceMapping = graph.getTileMapping(out);
  if (in.dim(0) >= 2) {
    reduceMapping = determineReduceVertexMapping(graph, in, out, reduceMapping);
  }
  const auto cs = graph.addComputeSet(debugPrefix + "/Reduce");
  reduce(graph, in, out, k, true, false, Operation::ADD, reduceMapping, cs);
  prog.add(Execute(cs));
}


Tensor reduceScale(Graph &graph, float k, Tensor &in,
                   const Type &outType,
                   Sequence &prog, const std::string &debugPrefix) {
  const auto numAddends = in.dim(0);
  const auto resultSize = in.dim(1);
  const auto dType = in.elementType();

  // If batch size is 1 then no reduction is required.
  if (numAddends == 1) {
    Tensor B = graph.addConstant<float>(outType, in.shape(), k);
    Tensor A = in;

    // TODO: Remove the cast when standard operators have output type as an arg
    if (dType != outType) {
      A = cast(graph, A, outType, prog, debugPrefix + "/ReduceScale");
    }
    return popops::mul(graph, A, B, prog, debugPrefix);
  }

  const auto out = graph.addVariable(outType, {resultSize},
                                     debugPrefix + "/ReducedScaled");
  poputil::mapTensorLinearly(graph, out);

  const auto reduceVertexMapping =
      determineReduceVertexMapping(graph, in, out, graph.getTileMapping(out));
  const auto cs = graph.addComputeSet(debugPrefix + "/ReduceScale");
  reduce(graph, in, out, k, false, true, Operation::ADD, reduceVertexMapping,
         cs);
  prog.add(Execute(cs));
  return out;
}

Tensor
reduce(Graph &graph, const Tensor &A_, const std::vector<std::size_t> &dims,
       Operation operation, Sequence &prog,const std::string &debugPrefix) {
  auto orderedDims = dims;
  std::sort(orderedDims.begin(), orderedDims.end());
  orderedDims.erase(std::unique(orderedDims.begin(), orderedDims.end()),
                    orderedDims.end());
  if (orderedDims.size() == 0) {
    auto out = graph.clone(A_);
    prog.add(Copy(A_, out));
    return out;
  }

  if (orderedDims.size() > A_.rank()) {
    throw poputil::poplib_error("Dimensions to reduce must be at least 1 "
                               "smaller than the rank of input tensor");
  }

  // Add a dimension of 1 if sizes match
  Tensor A;
  const bool fullRank = orderedDims.size() == A_.rank();
  if (fullRank) {
    std::vector<std::size_t> newShape = A_.shape();
    newShape.push_back(1);
    A = A_.reshape({newShape});
  } else {
    A = A_;
  }

  std::vector<unsigned> permutation;
  std::size_t numElements = 1ULL;
  for (const auto &i : orderedDims) {
    numElements *= A.dim(i);
    permutation.push_back(i);
  }

  std::vector<std::size_t> reshapeDims = { numElements };
  std::vector<std::size_t> outputDims;

  // find dimshuffle order and reshape dims
  for (std::size_t i = 0; i != A.rank(); ++i) {
    if (std::find(permutation.begin(), permutation.end(), i)
        == permutation.end()) {
      permutation.push_back(i);
      reshapeDims.push_back(A.dim(i));
      outputDims.push_back(A.dim(i));
    }
  }

  const std::string vName = debugPrefix + "/" + getVertexStr(operation);

  const auto aShuffled = A.dimShuffle({permutation}).reshape({reshapeDims});
  const auto outType = getOutputType(operation, A.elementType());
  auto out = graph.clone(outType, aShuffled[0]);

  /* No reduction to be done if number of elements to reduce is 1 */
  if (numElements == 1) {
    prog.add(Copy(aShuffled[0], out));
    return out;
  }

  const auto reduceVertexMapping =
      determineReduceVertexMapping(graph, aShuffled, out,
                                   graph.getTileMapping(out));

  const auto cs = graph.addComputeSet(vName);
  reduce(graph, aShuffled, out, 1, false, false, operation,
         reduceVertexMapping, cs);

  prog.add(Execute(cs));
  return  orderedDims.size() == A_.rank() ? out[0] : out;
}

} // end namespace popops
