// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/exceptions.hpp"
#include <cmath>
#include <popnn/experimental/ROIAlign.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>

namespace popnn::experimental {

/** Calculate a grouping of the 'rois' dimension which is the number of rois for
 * which to calculate grads in parallel at a time to saturate workers on the
 * device.
 *
 * \param rois              Number of input rois.
 * \param channel           Channel of input feature.
 * \param tile              Number of tiles.
 * \param binGridH          The number of points used for interpolation in each
 *                          bin in the height direction.
 * \param binGridW          The number of points used for interpolation in each
 *                          bin in the width direction.
 */
static long unsigned int
calcGroup(long unsigned int numWorkers, long unsigned int rois,
          long unsigned int channel, long unsigned int tile,
          long unsigned binGridH, long unsigned binGridW) {
  long unsigned int nGroup =
      ceilf((float)numWorkers * tile / (channel * binGridH * binGridW));
  long unsigned int resCroup = std::min(rois, nGroup);
  return resCroup;
}

poplar::Tensor
roiAlignFwd(poplar::Graph &graph, poplar::program::Sequence &prog,
            poplar::Tensor &bottomData, poplar::Tensor &bottomRois,
            poplar::Tensor &bottomBatchIndex, const roiAlignParams &params,
            const poplar::DebugContext &debugContext) {
  if (bottomData.rank() != 4) {
    throw poputil::poplibs_error("bottomData must be a 4-dimensional tensor");
  }
  if (bottomRois.rank() != 2) {
    throw poputil::poplibs_error("bottomRois must be a 2-dimensional tensor");
  }
  if (bottomRois.dim(1) != 4) {
    throw poputil::poplibs_error("bottomRois.dim(1) must have 4 elements");
  }
  if (bottomBatchIndex.rank() != 1) {
    throw poputil::poplibs_error(
        "bottomBatchIndex must be a 1-dimensional tensor");
  }
  if (params.samplingRatio_ == 0) {
    throw poputil::poplibs_error("samplingRatio must be greater than 0");
  }
  if (params.spatialScale_ <= 0) {
    throw poputil::poplibs_error("spatialScale must be greater than 0");
  }

  long unsigned int batchSize = bottomData.shape()[0];
  long unsigned int channels = bottomData.shape()[1];
  long unsigned int height = bottomData.shape()[2];
  long unsigned int width = bottomData.shape()[3];
  long unsigned int numRois = bottomRois.shape()[0];

  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(bottomData, bottomRois, bottomBatchIndex, params.samplingRatio_,
              params.alignedHeight_, params.alignedWidth_, params.spatialScale_,
              params.aligned_));
  poplar::DebugNameAndId dnai = {di};
  poplibs_support::logging::popnn::info("roiAlignFwd bottomData={}, name={}",
                                        bottomData.shape(),
                                        dnai.getPathName() + "bottomData");
  poplibs_support::logging::popnn::info("roiAlignFwd inputRois={}, name={}",
                                        bottomRois.shape(),
                                        dnai.getPathName() + "inputRois");
  poplibs_support::logging::popnn::info(
      "roiAlignFwd bottomBatchIndex={}, name={}", bottomBatchIndex.shape(),
      dnai.getPathName() + "bottomBatchIndex");

  auto dataType = bottomData.elementType();
  poplar::Tensor topData = graph.addVariable(
      dataType,
      {numRois, channels, params.alignedHeight_, params.alignedWidth_},
      {dnai, "topData"});
  long unsigned int binGridH = params.samplingRatio_;
  long unsigned int binGridW = params.samplingRatio_;
  poplar::Tensor topDataGrid =
      graph.addVariable(dataType,
                        {numRois, channels, binGridH, binGridW,
                         params.alignedHeight_, params.alignedWidth_},
                        {dnai, "topDataGrid"});
  poplar::Tensor batchBottomData = graph.addVariable(
      dataType, {batchSize, height * width}, {dnai, "batchBottomData"});
  poplar::Tensor count = graph.addConstant<long unsigned int>(
      dataType, {1}, {binGridH * binGridW}, {dnai, "count"});
  graph.setTileMapping(count, 0);

  poplar::ComputeSet roiAlignCS = graph.addComputeSet({dnai, "roiAlignCS"});
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  unsigned int tile_index = 0;
  for (auto i = 0u; i < numRois; ++i) {
    for (auto c = 0u; c < channels; ++c) {
      for (auto iy = 0u; iy < binGridH; iy++) {
        for (auto ix = 0u; ix < binGridW; ix++) {
          unsigned tile = tile_index % numTiles;
          graph.setTileMapping(batchBottomData, tile);
          batchBottomData = bottomData.slice(c, c + 1, 1)
                                .reshape({batchSize, height * width});
          graph.setTileMapping(topData[i][c], tile);
          graph.setTileMapping(topDataGrid[i][c][iy][ix], tile);
          poplar::VertexRef roiAlignVertex = graph.addVertex(
              roiAlignCS,
              poputil::templateVertex("popnn::ROIAlignForward", dataType),
              {{"bottom_data", batchBottomData},         // Input
               {"bottom_rois", bottomRois[i].flatten()}, // Input
               {"top_data_grid", topDataGrid[i][c][iy][ix].flatten()},
               {"batch_index", bottomBatchIndex},
               {"iter", i}});
          graph.setInitialValue(roiAlignVertex["spatial_scale"],
                                params.spatialScale_);
          graph.setInitialValue(roiAlignVertex["height"], height);
          graph.setInitialValue(roiAlignVertex["width"], width);
          graph.setInitialValue(roiAlignVertex["aligned_height"],
                                params.alignedHeight_);
          graph.setInitialValue(roiAlignVertex["aligned_width"],
                                params.alignedWidth_);
          graph.setInitialValue(roiAlignVertex["iy"], iy);
          graph.setInitialValue(roiAlignVertex["ix"], ix);
          graph.setInitialValue(roiAlignVertex["bin_grid_h"], binGridH);
          graph.setInitialValue(roiAlignVertex["bin_grid_w"], binGridW);
          graph.setInitialValue(roiAlignVertex["aligned"], params.aligned_);
          graph.setTileMapping(roiAlignVertex, tile);
          tile_index++;
        }
      }
    }
  }
  popops::fill(graph, topDataGrid, prog, 0.0f);
  prog.add(poplar::program::Execute(roiAlignCS));
  topData =
      popops::reduce(graph, topDataGrid, dataType, {2, 3},
                     popops::Operation::ADD, prog, {dnai, "topDataReduce"});
  topData = popops::div(graph, topData, count, prog, {dnai, "topDataDiv"});
  di.addOutputs({{"forwardResult", poputil::toProfileValue(topData)}});
  return topData;
}

poplar::Tensor
roiAlignInputGradient(poplar::Graph &graph, poplar::program::Sequence &prog,
                      poplar::Tensor &bottomData, poplar::Tensor &bottomRois,
                      poplar::Tensor &bottomBatchIndex,
                      poplar::Tensor &topDataGrad, const roiAlignParams &params,
                      const poplar::DebugContext &debugContext) {
  if (bottomData.rank() != 4) {
    throw poputil::poplibs_error("bottomData must be a 4-dimensional tensor");
  }
  if (bottomRois.rank() != 2) {
    throw poputil::poplibs_error("bottomRois must be a 2-dimensional tensor");
  }
  if (bottomRois.dim(1) != 4) {
    throw poputil::poplibs_error("bottomRois.dim(1) must have 4 elements");
  }
  if (bottomBatchIndex.rank() != 1) {
    throw poputil::poplibs_error(
        "bottomBatchIndex must be a 1-dimensional tensor");
  }
  if (params.samplingRatio_ == 0) {
    throw poputil::poplibs_error("samplingRatio must be greater than 0");
  }
  if (params.spatialScale_ <= 0) {
    throw poputil::poplibs_error("spatialScale must be greater than 0");
  }
  if (topDataGrad.rank() != 4) {
    throw poputil::poplibs_error("topDataGrad must be a 4-dimensional tensor");
  }

  long unsigned int batchSize = bottomData.shape()[0];
  long unsigned int channels = bottomData.shape()[1];
  long unsigned int height = bottomData.shape()[2];
  long unsigned int width = bottomData.shape()[3];
  long unsigned int numRois = bottomRois.shape()[0];

  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(bottomData, bottomRois, bottomBatchIndex, topDataGrad,
              params.samplingRatio_, params.alignedHeight_,
              params.alignedWidth_, params.spatialScale_, params.aligned_));
  poplar::DebugNameAndId dnai = {di};
  poplibs_support::logging::popnn::info("roiAlignBwd bottomData={}, name={}",
                                        bottomData.shape(),
                                        dnai.getPathName() + "bottomData");
  poplibs_support::logging::popnn::info("roiAlignBwd inputRois={}, name={}",
                                        bottomRois.shape(),
                                        dnai.getPathName() + "inputRois");
  poplibs_support::logging::popnn::info(
      "roiAlignBwd bottomBatchIndex={}, name={}", bottomBatchIndex.shape(),
      dnai.getPathName() + "bottomBatchIndex");
  poplibs_support::logging::popnn::info("roiAlignBwd topDataGrad={}, name={}",
                                        topDataGrad.shape(),
                                        dnai.getPathName() + "topDataGrad");

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numWorkers = target.getNumWorkerContexts();
  auto dataType = topDataGrad.elementType();
  long unsigned int binGridH = params.samplingRatio_;
  long unsigned int binGridW = params.samplingRatio_;
  long unsigned int group =
      calcGroup(numWorkers, numRois, channels, numTiles, binGridH, binGridW);

  std::vector<poplar::ComputeSet> mapCS;
  for (unsigned int i = 0; i <= numRois / group; i++) {
    mapCS.push_back(
        graph.addComputeSet({dnai, "roiAlignGradCS_" + std::to_string(i)}));
  }
  poplar::Tensor bottomBuffGrad = graph.addVariable(
      dataType, {batchSize, channels, group, binGridH, binGridW, height, width},
      {dnai, "bottomBuffGrad"});
  for (auto c = 0u; c < channels; ++c) {
    for (auto i = 0u; i < numRois; i += group) {
      for (auto g = 0u; g < group && i + g < numRois; g++) {
        for (auto iy = 0u; iy < binGridH; ++iy) {
          for (auto ix = 0u; ix < binGridW; ++ix) {
            unsigned tile = (c * group + g) % numTiles;
            for (unsigned int b = 0u; b < batchSize; b++) {
              graph.setTileMapping(bottomBuffGrad[b][c][g][iy][ix], tile);
            }
            poplar::Tensor bottomBuffGradVertex =
                bottomBuffGrad.slice(c, c + 1, 1)
                    .slice(g, g + 1, 2)
                    .slice(iy, iy + 1, 3)
                    .slice(ix, ix + 1, 4)
                    .reshape({batchSize, height * width});
            poplar::VertexRef roiAlignVertex = graph.addVertex(
                mapCS[i / group],
                poputil::templateVertex("popnn::ROIAlignBackward", dataType),
                {{"top_diff", topDataGrad[i + g][c].flatten()},
                 {"bottom_rois", bottomRois[i + g].flatten()},
                 {"bottom_diff", bottomBuffGradVertex},
                 {"batch_index", bottomBatchIndex},
                 {"iter", i},
                 {"group", g}});
            graph.setInitialValue(roiAlignVertex["spatial_scale"],
                                  params.spatialScale_);
            graph.setInitialValue(roiAlignVertex["height"], height);
            graph.setInitialValue(roiAlignVertex["width"], width);
            graph.setInitialValue(roiAlignVertex["aligned_height"],
                                  params.alignedHeight_);
            graph.setInitialValue(roiAlignVertex["aligned_width"],
                                  params.alignedWidth_);
            graph.setInitialValue(roiAlignVertex["ix"], ix);
            graph.setInitialValue(roiAlignVertex["iy"], iy);
            graph.setInitialValue(roiAlignVertex["bin_grid_h"], binGridH);
            graph.setInitialValue(roiAlignVertex["bin_grid_w"], binGridW);
            graph.setInitialValue(roiAlignVertex["aligned"], params.aligned_);
            graph.setTileMapping(roiAlignVertex, tile);
          }
        }
      }
    }
  }
  popops::fill(graph, bottomBuffGrad, prog, 0.0f);
  for (unsigned int i = 0; i < mapCS.size(); i++) {
    prog.add(poplar::program::Execute(mapCS[i]));
  }
  poplar::Tensor bottomDiff =
      popops::reduce(graph, bottomBuffGrad, dataType, {2, 3, 4},
                     popops::Operation::ADD, prog, {dnai, "gradCSReduce"});
  di.addOutputs({{"backwardResult", poputil::toProfileValue(bottomDiff)}});
  return bottomDiff;
}
} // namespace popnn::experimental