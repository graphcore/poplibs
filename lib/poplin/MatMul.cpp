#include "poplin/MatMul.hpp"

#include "MatMulPlan.hpp"
#include "PerformanceEstimation.hpp"
#include "popstd/VertexTemplates.hpp"
#include "popstd/TileMapping.hpp"
#include "popstd/exceptions.hpp"
#include "popreduce/Reduce.hpp"
#include "popstd/Util.hpp"
#include "util/gcd.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace poplin {


static void applyTensorMapping(poplar::Graph &graph, poplar::Tensor t,
                               const std::vector<unsigned> &mapping) {
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  assert(mapping.size() == numTiles + 1);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    graph.setTileMapping(t.flatten().slice(mapping[tile], mapping[tile + 1]),
                         tile);
  }
}


static std::vector<unsigned>
computeActivationsMapping(const poplar::Graph &graph,
                          const std::string &actType,
                          const std::vector<std::size_t> &shape,
                          unsigned batchNum, unsigned batchSize) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numActivations = std::accumulate(shape.begin(), shape.end(),
                                              1UL,
                                              std::multiplies<std::size_t>());
  const auto actTypeSize = actType == "float" ? 4 : 2;
  unsigned grainSize = actType == "float" ? deviceInfo.getFloatVectorWidth() :
                                            deviceInfo.getHalfVectorWidth();
  // Limit the minimum number of activation bytes per tile to reduce the
  // amount of exchange code. Increasing this constant reduces exchange code
  // size and increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile =
    (minBytesPerTile + actTypeSize - 1) / minBytesPerTile;
  unsigned beginTile, endTile;
  const auto rank = shape.size();
  if (rank == 1) {
    beginTile = 0;
    endTile = numTiles;
  } else {
    assert(rank == 4);
    const unsigned chansPerGroup = shape[3];
    // The grain size is chosen to avoid splitting the tensor at a point
    // that will require the incoming pointer to be changed if the messages from
    // the source tiles are received in the wrong order. The convolution layers
    // access elements in groups of chansPerGroup. Other layers (e.g.
    // FwdNonLinearity) flatten the tensor and access elements in groups of the
    // vector width. We don't know which layer comes next so hedge our bets by
    // using least common multiple of the vector width and chansPerGroup.
    grainSize = lcm(grainSize, chansPerGroup);
    const auto batchElemsPerTile = (batchSize + numTiles - 1) / numTiles;
    const auto numBatchGroups =
        (batchSize + batchElemsPerTile - 1) / batchElemsPerTile;
    const auto tilesPerBatchGroup =
        numTiles / numBatchGroups;
    beginTile = batchNum / batchElemsPerTile * tilesPerBatchGroup;
    endTile = beginTile + tilesPerBatchGroup;
  }
  const auto numBatchTiles = endTile - beginTile;
  std::vector<unsigned> mapping;
  mapping.resize(numTiles + 1);
  std::vector<poplar::Interval<std::size_t>> regions = {
    {0, numActivations}
  };
  const auto perTileRegions =
      splitRegions(regions, grainSize, numBatchTiles, minElementsPerTile);
  for (unsigned tile = beginTile; tile != numTiles; ++tile) {
    if (tile - beginTile < perTileRegions.size() &&
        !perTileRegions[tile - beginTile].empty()) {
      assert(perTileRegions[tile - beginTile].size() == 1);
      const auto &region = perTileRegions[tile - beginTile].front();
      assert(mapping[tile] == region.begin());
      mapping[tile + 1] = region.end();
    } else {
      mapping[tile + 1] = mapping[tile];
    }
  }
  assert(mapping[endTile] == numActivations);
  return mapping;
}

static std::vector<unsigned>
computeActivationsMapping(const poplar::Graph &graph, poplar::Tensor act,
                          unsigned batchNum, unsigned batchSize) {
  return computeActivationsMapping(graph, act.elementType(),
                                   act.shape(), batchNum, batchSize);
}

static void mapActivations(poplar::Graph &graph, poplar::Tensor act) {
  poplar::Tensor actExt = act;
  if (actExt.rank() == 4) {
    // In cases when there is no channel grouping, extend tensor such that
    // number of groups is 1
    actExt = actExt.reshape({act.dim(0), act.dim(1), act.dim(2), 1, act.dim(3)})
                   .dimShuffle({0, 3, 1, 2, 4});
  }
  auto batchSize = actExt.dim(0);
  for (unsigned i = 0; i != batchSize; ++i) {
    auto actMapping = computeActivationsMapping(graph, actExt[i], i, batchSize);
    applyTensorMapping(graph, actExt[i], actMapping);
  }
}



static
void mapInput(Graph &graph, Tensor A,
              const std::vector<std::size_t> &bShape,
              const MatMulOptions &options) {
  const auto dType = A.elementType();
  const auto &plan = getPlan(graph, dType, A.shape(), bShape, options);
  const auto bCols = bShape[1];
  const auto outMapping =
      computeActivationsMapping(graph, dType, {A.dim(0)}, 0, bCols);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto prevSize = A.dim(1);
  const auto numCols = prevSize;
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
  const auto &ipuPartition = plan.ipuPartition;
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    const auto ipuBeginRow = outMapping[ipu * tilesPerIPU];
    const auto ipuEndRow = outMapping[(ipu + 1) * tilesPerIPU];
    const auto ipuRows = ipuEndRow - ipuBeginRow;
    for (unsigned tileY = 0; tileY != ipuPartition.tilesPerColumn; ++tileY) {
      const auto tileRowBegin = ipuBeginRow   + (tileY * ipuRows) /
          ipuPartition.tilesPerColumn;
      const auto tileRowEnd = ipuBeginRow + ((tileY + 1) * ipuRows) /
          ipuPartition.tilesPerColumn;
      if (tileRowBegin == tileRowEnd)
        continue;
      for (unsigned tileX = 0; tileX != ipuPartition.tilesPerRow; ++tileX) {
        const auto tile = ipu * tilesPerIPU +
            tileY * ipuPartition.tilesPerRow +
            tileX;
        const auto j = tileX;
        const auto beginElement =
            (numCols * j) / ipuPartition.tilesPerRow;
        const auto endElement =
            (numCols * (j + 1)) / ipuPartition.tilesPerRow;
        if (beginElement == endElement)
          continue;
        graph.setTileMapping(A.slice({tileRowBegin, beginElement},
                                     {tileRowEnd, endElement}),
                             tile);
      }
    }
  }
}

poplar::Tensor
createMatMulInputA(poplar::Graph &graph,
                   const std::string &type,
                   const std::vector<std::size_t> &aShape,
                   const poplar::Tensor &B,
                   const std::string &name,
                   const MatMulOptions &options) {
  auto A = graph.addTensor(type, aShape, name);
  mapInput(graph, A, B.shape(), options);
  return A;
}

static Tensor
matMul1(Graph &graph, Tensor A, Tensor B, Sequence &prog,
        const std::string &debugPrefix, const MatMulOptions &options) {
  const auto bCols = B.dim(1);
  const auto dType = B.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto layerName = debugPrefix
                         + "/MatMul" + std::to_string(A.dim(0)) + "x"
                         + std::to_string(A.dim(1)) + "x"
                         + std::to_string(B.dim(1));
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto aRows = A.dim(0);
  const auto aCols = A.dim(1);
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
  assert(dType == "float" || dType == "half");
  const auto &plan = getPlan(graph, dType, A.shape(), B.shape(), options);
  const auto &ipuPartition = plan.ipuPartition;
  ComputeSet dotProductCS = graph.addComputeSet(layerName + "/DotProd");
  prog.add(Execute(dotProductCS));
  ComputeSet reduceCS = graph.addComputeSet(layerName + "/Reduce");
  prog.add(Execute(reduceCS));
  Tensor out = graph.addTensor(dType, {bCols, aRows}, layerName + "/Out");
  mapActivations(graph, out);
  // Iterate through the batch add to the same compute set
  // (i.e. execute the batch in parallel).
  for (unsigned b = 0; b < bCols; ++b) {
    const auto &activationsOutMapping =
      computeActivationsMapping(graph, out[b], b, bCols);
    Tensor partials = graph.addTensor("float", {ipuPartition.tilesPerRow,
                                                aRows},
                                      "partials");
    for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
      const auto ipuBeginRow = activationsOutMapping[ipu * tilesPerIPU];
      const auto ipuEndRow = activationsOutMapping[(ipu + 1) * tilesPerIPU];
      const auto ipuRows = ipuEndRow - ipuBeginRow;
      for (unsigned tileY = 0; tileY != ipuPartition.tilesPerColumn; ++tileY) {
        const auto tileRowBegin = ipuBeginRow + (tileY * ipuRows) /
            ipuPartition.tilesPerColumn;
        const auto tileRowEnd = ipuBeginRow + ((tileY + 1) * ipuRows) /
            ipuPartition.tilesPerColumn;
        if (tileRowBegin == tileRowEnd)
          continue;
        for (unsigned tileX = 0; tileX != ipuPartition.tilesPerRow; ++tileX) {
          const auto tile = ipu * tilesPerIPU +
              tileY * ipuPartition.tilesPerRow +
              tileX;
          const auto j = tileX;
          const auto beginElement =
              (aCols * j) / ipuPartition.tilesPerRow;
          const auto endElement =
              (aCols * (j + 1)) / ipuPartition.tilesPerRow;
          if (beginElement == endElement)
            continue;
          for (unsigned i = tileRowBegin; i != tileRowEnd; ++i) {
            Tensor partialIn = B.slice({beginElement, b},
                                       {endElement, b + 1})
                                .reshape({endElement - beginElement});
            Tensor partialWeights = A[i].slice(beginElement, endElement);
            auto v =
                graph.addVertex(dotProductCS,
                                templateVertex("poplin::MatMul1Partial",
                                               dType),
                                {{"in", partialIn},
                                 {"weights", partialWeights},
                                 {"out", partials[j][i]}});
            graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
            graph.setTileMapping(partials[j][i], tile);
            graph.setTileMapping(v, tile);
          }
        }
      }
    }
    popreduce::reduce(graph, partials, out[b], graph.getTileMapping(out[b]),
                      reduceCS);
  }
  return out;
}

Tensor
matMul2(Graph &graph, Tensor A,  Tensor B,
        Sequence &prog, const std::string &debugPrefix,
        const MatMulOptions &options) {
  const auto dType = A.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto aRows = A.dim(0);
  const auto bCols = static_cast<unsigned>(B.dim(1));
  const auto layerName = debugPrefix
                         + "/MatMul" + std::to_string(A.dim(0)) + "x"
                         + std::to_string(A.dim(1)) + "x"
                         + std::to_string(B.dim(1));
  const auto &plan = getPlan(graph, dType, B.shape(),
                             A.transpose().shape(), options);
  const auto &ipuPartition = plan.ipuPartition;
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
  auto cs = graph.addComputeSet(layerName + "/DotProd");
  auto reduceCS = graph.addComputeSet(layerName + "/Reduce");
  Tensor out = graph.addTensor(dType, {aRows, bCols}, debugPrefix + "/Out");
  mapActivations(graph, out);
  for (unsigned b = 0; b < aRows; ++b) {
    Tensor partials =
        graph.addTensor("float",
                        {numIPUs, ipuPartition.tilesPerColumn, bCols},
                        "partials");
    auto outGradientMapping = computeActivationsMapping(graph, A[b],
                                                        b, aRows);
    for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
      const auto ipuBeginRow = outGradientMapping[ipu * tilesPerIPU];
      const auto ipuEndRow = outGradientMapping[(ipu + 1) * tilesPerIPU];
      const auto ipuRows = ipuEndRow - ipuBeginRow;

      for (unsigned tileY = 0; tileY != ipuPartition.tilesPerColumn; ++tileY) {
        const auto tileRowBegin = ipuBeginRow + (tileY * ipuRows) /
                                  ipuPartition.tilesPerColumn;
        const auto tileRowEnd = ipuBeginRow + ((tileY + 1) * ipuRows) /
                                ipuPartition.tilesPerColumn;

        for (unsigned tileX = 0; tileX != ipuPartition.tilesPerRow; ++tileX) {
          const auto tile = ipu * tilesPerIPU +
                            tileY * ipuPartition.tilesPerRow +
                            tileX;
          const auto j = tileY;
          const auto beginElement =
              (bCols * tileX) / ipuPartition.tilesPerRow;
          const auto endElement =
              (bCols * (tileX + 1)) / ipuPartition.tilesPerRow;
          if (beginElement == endElement)
            continue;
          const auto vectorWidth =
              dType == "float" ? deviceInfo.getFloatVectorWidth() :
                                 deviceInfo.getHalfVectorWidth();
          for (unsigned i = beginElement; i < endElement; i += vectorWidth) {
            const auto vectorNumElements = std::min(endElement - i,
                                                    vectorWidth);

            Tensor outWindow = partials[ipu][j].slice(i, i + vectorNumElements);
            if (tileRowBegin == tileRowEnd) {
              auto vZ =
                  graph.addVertex(cs, templateVertex("popstd::Zero", "float"));
              graph.setInitialValue(vZ["dataPathWidth"],
                                    deviceInfo.dataPathWidth);

              graph.connect(vZ["out"], outWindow);
              graph.setTileMapping(vZ, tile);
            } else {
              auto w = B.slice({tileRowBegin, i},
                               {tileRowEnd, i + vectorNumElements});
              Tensor inWindow = A[b].slice(tileRowBegin, tileRowEnd);
              auto v = graph.addVertex(
                            cs,
                            templateVertex("poplin::MatMul2", dType),
                            {{"in", inWindow},
                             {"weights", w},
                             {"out", outWindow},
                             });
              graph.setTileMapping(v, tile);
            }
            graph.setTileMapping(outWindow, tile);
          }
        }
      }
    }
    popreduce::reduce(graph,
                      partials.reshape({numIPUs * ipuPartition.tilesPerColumn,
                                        bCols}),
                      out[b],
                      graph.getTileMapping(out[b]),
                      reduceCS);
  }
  prog.add(Execute(cs));
  prog.add(Execute(reduceCS));
  return out;
}

static
void matMul3(Graph &graph,
             Tensor dst, bool update, float K,
             Tensor A, Tensor B,
             Sequence &prog,
             const std::string &debugPrefix,
             const MatMulOptions &options) {
  const auto dType = B.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto aCols = A.dim(1);
  const auto aRows = A.dim(0);
  const auto &activationsOutMapping =
      computeActivationsMapping(graph,
                                A.slice({0, 0}, {aRows, 1}).flatten(),
                                0, aCols);
  const auto opName = update ? "MatMulAcc" : "MatMul";
  const auto csName = debugPrefix
                         + "/" + opName + std::to_string(A.dim(0)) + "x"
                         + std::to_string(A.dim(1)) + "x"
                         + std::to_string(B.dim(1));
  auto cs = graph.addComputeSet(csName);
  const auto bCols = B.dim(1);
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
  const auto &plan = getPlan(graph, dType, {A.dim(0), B.dim(1)},
                             {A.dim(1), B.dim(0)}, options);
  const auto &ipuPartition = plan.ipuPartition;
  // Update the weights.
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    const auto ipuBeginRow = activationsOutMapping[ipu * tilesPerIPU];
    const auto ipuEndRow = activationsOutMapping[(ipu + 1) * tilesPerIPU];
    const auto ipuRows = ipuEndRow - ipuBeginRow;
    for (unsigned tileY = 0; tileY != ipuPartition.tilesPerColumn; ++tileY) {
      const auto tileRowBegin = ipuBeginRow + (tileY * ipuRows) /
                                ipuPartition.tilesPerColumn;
      const auto tileRowEnd = ipuBeginRow + ((tileY + 1) * ipuRows) /
                              ipuPartition.tilesPerColumn;
      if (tileRowBegin == tileRowEnd)
        continue;
      for (unsigned tileX = 0; tileX != ipuPartition.tilesPerRow; ++tileX) {
        const auto tile = ipu * tilesPerIPU +
                          tileY * ipuPartition.tilesPerRow +
                          tileX;
        const auto j = tileX;
        const auto beginElement =
            (bCols * j) / ipuPartition.tilesPerRow;
        const auto endElement =
            (bCols * (j + 1)) / ipuPartition.tilesPerRow;
        if (beginElement == endElement)
          continue;
        for (unsigned i = tileRowBegin; i != tileRowEnd; ++i) {
          auto vDst = dst[i].slice(beginElement, endElement);
          auto bWindow = B.slice({0, beginElement}, {aCols, endElement});
          auto aWindow = A.slice({i, 0}, {i + 1, aCols})
                          .flatten();
          auto vertexTypeName = update ? "poplin::MatMul3Update"
                                       : "poplin::MatMul3";
          auto vertexType = templateVertex(vertexTypeName, dType);
          auto v = graph.addVertex(cs, vertexType,
                                   {{"d", aWindow},
                                    {"dst", vDst},
                                    {"in", bWindow}});
          if (update)
            graph.setInitialValue(v["K"], K);

          graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
          graph.setTileMapping(v, tile);
        }
      }
    }
  }

  prog.add(Execute(cs));
  return;
}

Tensor
matMul3(Graph &graph, Tensor A, Tensor B,
        Sequence &prog,
        const std::string &debugPrefix,
        const MatMulOptions &options) {
  const auto dType = B.elementType();
  const auto aRows = A.dim(0);
  const auto bCols = B.dim(1);
  const auto out = graph.addTensor(dType, {aRows, bCols},
                                              debugPrefix + "/MatMulOut");
  mapInput(graph, out, {bCols, A.dim(1)}, options);
  matMul3(graph, out, false, 0, A, B, prog, debugPrefix, options);
  return out;
}

static
bool isContiguous(const Graph &graph, const poplar::Tensor &t, unsigned dim) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numWorkers =
      deviceInfo.getNumTiles() * deviceInfo.numWorkerContexts;
  const auto dType = t.elementType();
  unsigned vectorWidth = 4;
  if (dType == "float") {
    vectorWidth = deviceInfo.getFloatVectorWidth();
  } else if (dType == "half") {
    vectorWidth = deviceInfo.getHalfVectorWidth();
  }
  unsigned sampleWidth = t.dim(dim) / numWorkers;
  sampleWidth = std::max(sampleWidth, vectorWidth);
  sampleWidth = std::min(sampleWidth, (unsigned) t.dim(dim));
  std::vector<std::size_t> begin, end;
  for (unsigned i = 0; i < t.rank(); ++i) {
    begin.push_back(0);
    if (i == dim)
      end.push_back(sampleWidth);
    else
      end.push_back(1);
  }
  // TODO: Currently a heuristic is used to sample one vector in the axis
  // to be checked and infer that it is contiguous across the whole axis.
  // A more robust analysis would look at more samples (or the whole axis) and
  // make a decision based on that.
  auto slice = t.slice(begin, end);
  return slice.getContiguousRegions().size() == 1;
}

static poplar::Tensor
matMul(poplar::Graph &graph, poplar::Tensor &C, bool isUpdate, float k,
       poplar::Tensor A, poplar::Tensor B,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix,
       const MatMulOptions &options) {
  if (A.dim(1) != B.dim(0)) {
    throw popstd::poplib_error("Second dimension of first operand to matrix "
                               "multiplication does not match first dimension "
                               "of second operand.");
  }
  if (A.dim(1) == 1) {
    // Outer Product
    if (isUpdate) {
      matMul3(graph, C, true, k, A, B, prog, debugPrefix, options);
      return C;
    }
    if (options.outputIsRowContiguous) {
      return matMul3(graph, B.transpose(), A.transpose(), prog,
                     debugPrefix, options).transpose();
    } else {
      return matMul3(graph, A, B, prog, debugPrefix, options);
    }
  }
  auto aIsVector = A.dim(0) == 1;
  auto bIsVector = B.dim(1) == 1;
  auto aRowCont = isContiguous(graph, A, 0);
  auto aColCont = isContiguous(graph, A, 1);
  auto bRowCont = isContiguous(graph, B, 0);
  auto bColCont = isContiguous(graph, B, 1);
  if (bIsVector && !aIsVector) {
    if (isUpdate) {
        throw popstd::poplib_error("Matrix * vector matMulAcc "
                                   "not implemented");
    }
    if (aColCont && options.outputIsRowContiguous) {
      return matMul1(graph, A, B, prog, debugPrefix, options).transpose();
    } else if (aColCont && !options.outputIsRowContiguous) {
      throw popstd::poplib_error("Matrix (col contiguous) * vector for "
                                 "col contiguous output not implemented");
    } else if (options.outputIsRowContiguous) {
      return matMul2(graph, B.transpose(), A.transpose(), prog,
                     debugPrefix, options).transpose();
    } else {
      throw popstd::poplib_error("Matrix (row contiguous) * vector for "
                                 "col contiguous output not implemented");
    }
  }
  if (aIsVector && !bIsVector) {
    if (isUpdate) {
      throw popstd::poplib_error("Vector * matrix matMulAcc "
                                 "not implemented");
    }
    if (bColCont && !options.outputIsRowContiguous) {
      return matMul2(graph, A, B, prog, debugPrefix, options);
    } else if (bColCont && options.outputIsRowContiguous) {
      throw popstd::poplib_error("Vector * matrix (col contiguous) "
                                 "for row contiguous output not implemented");
    } else if (!options.outputIsRowContiguous) {
      return matMul1(graph, B.transpose(), A.transpose(), prog,
                     debugPrefix, options);
    } else {
      throw popstd::poplib_error("Vector * matrix (row contiguous) "
                                 "for row contiguous output not implemented");
    }
  }
  if (aIsVector && bIsVector) {
    return matMul1(graph, A, B, prog,
                   debugPrefix, options);
  }
  if (aColCont && bRowCont) {
    if (isUpdate) {
        throw popstd::poplib_error("Matrix (col contig) * matrix (row contig) "
                                   "matMulAcc not implemented");
    }
    if (!options.outputIsRowContiguous) {
      return matMul1(graph, B.transpose(), A.transpose(), prog, debugPrefix,
                     options);
    } else {
      return matMul1(graph, A, B, prog, debugPrefix, options);
    }
  }
  if (aColCont && bColCont) {
    if (isUpdate) {
      throw popstd::poplib_error("Matrix (col contig) * matrix (col contig) "
                                 "matMulAcc not implemented");
    }
    if (options.outputIsRowContiguous) {
      throw popstd::poplib_error("Matrix (col contig) * matrix (col contig) "
                                 "for row contiguous output not implemented");
    } else {
      return matMul2(graph, A, B, prog, debugPrefix, options);
    }
  }
  if (aRowCont && bColCont) {
    if (isUpdate) {
      matMul3(graph, C, true, k, A, B, prog, debugPrefix, options);
      return C;
    }
    if (options.outputIsRowContiguous) {
      return matMul3(graph, B.transpose(), A.transpose(), prog,
                     debugPrefix, options).transpose();
    } else {
      return matMul3(graph, A, B, prog, debugPrefix, options);
    }
  }
  throw popstd::poplib_error("matmul for input layout not implemented yet");
}

poplar::Tensor
matMul(poplar::Graph &graph,
       poplar::Tensor A, poplar::Tensor B,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix,
       const MatMulOptions &options) {
  Tensor C;
  return matMul(graph, C, false, 0, A, B, prog, debugPrefix, options);
}

void
matMulAcc(poplar::Graph &graph, poplar::Tensor C, float k,
          poplar::Tensor A, poplar::Tensor B,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix,
          const MatMulOptions &options) {
  (void) matMul(graph, C, true, k, A, B, prog, debugPrefix, options);
}

}
