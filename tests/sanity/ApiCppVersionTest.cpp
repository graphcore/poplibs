// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ApiCppVersionTest
// This file just includes all of the public Poplibs headers and is compiled
// in C++11 mode, so that we know that we haven't accidentally added
// C++14 features in the API. It is also compiled with `-Wall -Wextra pedantic`
// to catch any compilation erros that users of the API might see if they have
// stricter warnings on their projects than we do.

#include <popfloat/experimental/CastToGfloat.hpp>
#include <popfloat/experimental/CastToHalf.hpp>
#include <popfloat/experimental/GfloatExpr.hpp>
#include <popfloat/experimental/GfloatExprUtil.hpp>
#include <popfloat/experimental/codelets.hpp>
#include <poplin/Cholesky.hpp>
#include <poplin/ConvParams.hpp>
#include <poplin/ConvPreplan.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/FullyConnected.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/MeshGrid.hpp>
#include <poplin/MultiConvolution.hpp>
#include <poplin/Norms.hpp>
#include <poplin/TriangularSolve.hpp>
#include <poplin/codelets.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/CTCInference.hpp>
#include <popnn/CTCLoss.hpp>
#include <popnn/CTCPlan.hpp>
#include <popnn/GroupNorm.hpp>
#include <popnn/Gru.hpp>
#include <popnn/GruDef.hpp>
#include <popnn/InstanceNorm.hpp>
#include <popnn/LayerNorm.hpp>
#include <popnn/LogSoftmax.hpp>
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
#include <popnn/Rnn.hpp>
#include <popnn/SpatialSoftMax.hpp>
#include <popnn/codelets.hpp>
#include <popops/AllTrue.hpp>
#include <popops/Cast.hpp>
#include <popops/CircBuf.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ElementWiseUtil.hpp>
#include <popops/Encoding.hpp>
#include <popops/EncodingConstants.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Fill.hpp>
#include <popops/Gather.hpp>
#include <popops/GatherStatistics.hpp>
#include <popops/HostSliceTensor.hpp>
#include <popops/Loop.hpp>
#include <popops/NaN.hpp>
#include <popops/NormaliseImage.hpp>
#include <popops/Operation.hpp>
#include <popops/OperationDef.hpp>
#include <popops/OperationDefUtil.hpp>
#include <popops/Pad.hpp>
#include <popops/PerformanceEstimation.hpp>
#include <popops/Rearrange.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Scatter.hpp>
#include <popops/SelectScalarFromRows.hpp>
#include <popops/SequenceSlice.hpp>
#include <popops/Sort.hpp>
#include <popops/SortOrder.hpp>
#include <popops/TopK.hpp>
#include <popops/UpdateScalarInRows.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <popsparse/Embedding.hpp>
#include <popsparse/FullyConnected.hpp>
#include <popsparse/FullyConnectedParams.hpp>
#include <popsparse/MatMul.hpp>
#include <popsparse/MatMulParams.hpp>
#include <popsparse/PlanningCache.hpp>
#include <popsparse/SparsePartitioner.hpp>
#include <popsparse/SparseStorageFormats.hpp>
#include <popsparse/SparseTensor.hpp>
#include <popsparse/SparsityParams.hpp>
#include <popsparse/codelets.hpp>
#include <popsparse/experimental/BlockSparse.hpp>
#include <popsparse/experimental/BlockSparseMatMul.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/GraphFunction.hpp>
#include <poputil/cyclesTables.hpp>
#include <poputil/exceptions.hpp>
// This header is actually C++17 and is used by GCL. Ideally this should be
// moved to GCCS.
// #include <poputil/OptionParsing.hpp>
#include <poputil/TensorMetaData.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VarStructure.hpp>
#include <poputil/VertexTemplates.hpp>

int main() {}
