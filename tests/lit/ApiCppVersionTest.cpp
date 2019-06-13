// This file just includes all of the public Poplar headers and is compiled
// in C++11 mode, so that we know that we haven't accidentally added
// C++14 features in the API.

// RUN: c++ -std=c++11 %s -lpoplar -o%t

#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/Norms.hpp>
#include <poplin/ChannelOps.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/MeshGrid.hpp>
#include <poplin/codelets.hpp>
#include <poplin/Convolution.hpp>
#include <popsys/CSRFunctions.hpp>
#include <popsys/CycleCount.hpp>
#include <popsys/CycleStamp.hpp>
#include <popsys/codelets.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Operation.hpp>
#include <popops/Reduce.hpp>
#include <popops/Cast.hpp>
#include <popops/Zero.hpp>
#include <popops/AllTrue.hpp>
#include <popops/Pad.hpp>
#include <popops/EncodingConstants.hpp>
#include <popops/Encoding.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Scatter.hpp>
#include <popops/PopopsChannelOps.hpp>
#include <popops/Collectives.hpp>
#include <popops/Expr.hpp>
#include <popops/CircBuf.hpp>
#include <popops/Sort.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <popops/Gather.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/GraphFunction.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/Broadcast.hpp>
#include <popsolver/Model.hpp>
#include <popsolver/Variable.hpp>
#include <popnn/LayerNorm.hpp>
#include <popnn/NonLinearityDefUtil.hpp>
#include <popnn/InstanceNorm.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/Lstm.hpp>
#include <popnn/SpatialSoftMax.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popnn/Norms.hpp>
#include <popnn/LstmDef.hpp>
#include <popnn/GroupNorm.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/PoolingDef.hpp>
#include <popnn/Recurrent.hpp>
#include <popnn/Loss.hpp>
#include <popnn/codelets.hpp>
#include <popnn/Pooling.hpp>

int main() {}
