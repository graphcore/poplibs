// Copyright (c) Graphcore Ltd, All rights reserved.
// This file just includes all of the public Poplibs headers and is compiled
// in C++11 mode, so that we know that we haven't accidentally added
// C++14 features in the API.

#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/MeshGrid.hpp>
#include <poplin/Norms.hpp>
#include <poplin/codelets.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/GroupNorm.hpp>
#include <popnn/InstanceNorm.hpp>
#include <popnn/LayerNorm.hpp>
#include <popnn/Loss.hpp>
#include <popnn/Lstm.hpp>
#include <popnn/LstmDef.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popnn/NonLinearityDefUtil.hpp>
#include <popnn/Norms.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/PoolingDef.hpp>
#include <popnn/Recurrent.hpp>
#include <popnn/SpatialSoftMax.hpp>
#include <popnn/codelets.hpp>
#include <popops/AllTrue.hpp>
#include <popops/Cast.hpp>
#include <popops/CircBuf.hpp>
#include <popops/Collectives.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/EncodingConstants.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Gather.hpp>
#include <popops/Operation.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Scatter.hpp>
#include <popops/Sort.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <popsolver/Model.hpp>
#include <popsolver/Variable.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/GraphFunction.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

int main() {}
