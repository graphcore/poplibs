// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ConvProgramTree.hpp"
#include "ConvolutionInternal.hpp"
#include <poputil/VarStructure.hpp>

namespace poplin {

std::vector<poputil::GroupingInfo>
determinePreprocessedGroupingFromPlan(const ConvParams &params,
                                      const Plan &plan, unsigned level);

void swapOperands(ConvParams &params, boost::optional<poplar::Tensor> &acts,
                  boost::optional<poplar::Tensor> &weights);

bool expandDimTransformIsViewOnly(const ConvParams &params, unsigned dim);

void expandSpatialDim(ConvParams &params, unsigned dim, poplar::Graph &graph,
                      boost::optional<poplar::Tensor> &acts,
                      boost::optional<poplar::Tensor> &weights,
                      const std::string &debugPrefix);

void expandSpatialDim(ConvParams &params, unsigned dim);

void expandSpatialDims(ConvParams &params, Plan &plan, unsigned level,
                       poplar::Graph &graph,
                       boost::optional<poplar::Tensor> &acts,
                       boost::optional<poplar::Tensor> &weights,
                       ConvProgramTree::TransformPreProgram *rearrangeProg,
                       bool rearrangeActs = false,
                       bool rearrangeWeights = false,
                       const std::string &debugPrefix = "");

poplar::Tensor flattenDims(poplar::Tensor t, unsigned from, unsigned to);

poplar::Tensor unflattenDims(poplar::Tensor t, unsigned from, unsigned to,
                             unsigned fromSize);

void doFlatten(const std::vector<unsigned> &dimsToFlatten,
               boost::optional<poplar::Tensor> &acts,
               std::vector<std::size_t> &spatialDims, std::size_t &batchSize);

poplar::Tensor dilate(poplar::Graph &graph, const poplar::Tensor &t,
                      unsigned dilationFactor, unsigned dim,
                      const std::string &debugPrefix);

poplar::Tensor dilateWithNearestNeighbour(const poplar::Tensor &t,
                                          unsigned dilationFactor,
                                          unsigned dim);

} // namespace poplin
