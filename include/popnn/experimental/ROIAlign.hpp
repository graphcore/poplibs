// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popnn_ROIAlign_hpp
#define popnn_ROIAlign_hpp
#include <poplar/Program.hpp>

namespace popnn::experimental {

/** parameters of ROIAlign
 *
 * \param samplingRatio_            Number of sampling points in the
 *                                  interpolation grid used to compute the
 *                                  output value of each pooled output bin.
 * \param alignedHeight_            Pooled output Y's height.
 * \param alignedWidth_             Pooled output X's height.
 * \param aligned_                  Whether to use the value 'half_pixel' to
 *                                  pixel shift the input coordinates by -0.5.
 * \param spatialScale_             Multiplicative spatial scale factor to
 *                                  translate ROI coordinates from their input
 *                                  spatial scale to the scale used when
 *                                  pooling, i.e., spatial scale of the input
 *                                  feature map X relative to the input image.
 */
struct roiAlignParams {
  long unsigned int samplingRatio_, alignedHeight_, alignedWidth_;
  bool aligned_;
  float spatialScale_;
  roiAlignParams(long unsigned int samplingRatio, long unsigned int poolH,
                 long unsigned int poolW, bool aligned, float spatialScale)
      : samplingRatio_{samplingRatio}, alignedHeight_{poolH},
        alignedWidth_{poolW}, aligned_{aligned}, spatialScale_{spatialScale} {}
};

/** Forward computation of ROIAlign.
 *
 * \param graph                     The graph to add the operation to.
 * \param prog                      The sequence to add the operation to.
 * \param bottomData                Input data tensor from the previous
 *                                  operator; 4-D feature map of shape (N, C, H,
 *                                  W), where N is the batch size, C is the
 *                                  number of channels, and H and W are the
 *                                  height and the width of the data.
 * \param bottomRois                RoIs (Regions of Interest) to pool over;
 *                                  rois is 2-D input of shape (num_rois, 4)
 *                                  given as [[x1, y1, x2, y2], ...]. The RoIs'
 *                                  coordinates are in the coordinate system of
 *                                  the input image. Each coordinate set has a
 *                                  1:1 correspondence with 'bottomBatchIndex'
 *                                  input.
 * \param bottomBatchIndex          1-D tensor of shape (numRois,) with each
 *                                  element denoting the index of the
 *                                  corresponding image in the batch.
 * \param params                    The configuration parameters of ROIAlign.
 * \param debugContext              Optional debug information.
 *
 * \return                          RoI pooled output Y, 4-D tensor of
 *                                  shape (numRois, channels,
 *                                  alignedHeight_, alignedWidth_). The
 *                                  r-th batch element Y[r-1] is a
 *                                  pooled feature map corresponding to
 *                                  the r-th RoI X[r-1].
 */
poplar::Tensor
roiAlignFwd(poplar::Graph &graph, poplar::program::Sequence &prog,
            poplar::Tensor &bottomData, poplar::Tensor &bottomRois,
            poplar::Tensor &bottomBatchIndex, const roiAlignParams &params,
            const poplar::DebugContext &debugContext = {});

/** Backward computation of ROIAlign.
 *
 * \param graph                     The graph to add the operation to.
 * \param prog                      The sequence to add the operation to.
 * \param bottomData                Input data tensor from the forward pass;
 *                                  4-D feature map of shape (N, C, H,
 *                                  W), where N is the batch size, C is the
 *                                  number of channels, and H and W are the
 *                                  height and the width of the data.
 * \param bottomRois                RoIs (Regions of Interest) to pool over;
 *                                  rois is 2-D input of shape (num_rois, 4)
 *                                  given as [[x1, y1, x2, y2], ...]. The RoIs'
 *                                  coordinates are in the coordinate system of
 *                                  the input image. Each coordinate set has a
 *                                  1:1 correspondence with 'bottomBatchIndex'
 *                                  input.
 * \param bottomBatchIndex          1-D tensor of shape (numRois,) with each
 *                                  element denoting the index of the
 *                                  corresponding image in the batch.
 * \param topDataGrad               The gradients at the output of the ROIAlign
 *                                  operation.
 * \param params                    The configuration parameters of
 *                                  ROIAlign.
 * \param debugContext              Optional debug information.
 *
 * \return                          Gradient at the input to the ROIAlign
 *                                  operation Y_grad, 4-D tensor of shape
 *                                  (batchSize, channels, height, width). It's
 *                                  consistent with bottomData.
 */
poplar::Tensor
roiAlignInputGradient(poplar::Graph &graph, poplar::program::Sequence &prog,
                      poplar::Tensor &bottomData, poplar::Tensor &bottomRois,
                      poplar::Tensor &bottomBatchIndex,
                      poplar::Tensor &topDataGrad, const roiAlignParams &params,
                      const poplar::DebugContext &debugContext = {});

} // namespace popnn::experimental

#endif // popnn_ROIAlign_hpp