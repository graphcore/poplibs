// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "poplin/ConvParams.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <popnn/Pooling.hpp>

namespace popnn {
namespace pooling {

void createPoolingVertex(poplar::Graph &graph, const PoolParams &params,
                         const poplar::Tensor &prevAct,
                         const poplar::Tensor &nextAct,
                         poplar::program::Sequence &prog);

poplin::ConvParams makeConvParams(const PoolParams &poolParams);

} // namespace pooling
} // namespace popnn
