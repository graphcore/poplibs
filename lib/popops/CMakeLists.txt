include(GNUInstallDirs)

find_package(spdlog 1.8.0 REQUIRED)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popops SHARED
  AllTrue.cpp
  BitonicTopK.cpp
  BitonicTopK.hpp
  Cast.cpp
  CircBuf.cpp
  TensorCollectives.cpp
  CollectiveTypes.cpp
  codelets.cpp
  DynamicSlice.cpp
  DynamicSliceInternal.hpp
  ElementWise.cpp
  ElementWiseInternal.hpp
  ElementWiseInternal.cpp
  ElementWiseUtil.cpp
  Encoding.cpp
  Expr.cpp
  ExpressionGenerator.cpp
  ExpressionGenerator.hpp
  ExprOpUtil.cpp
  ExprOpUtil.hpp
  Fill.cpp
  Gather.cpp
  GatherInternal.cpp
  GatherStatistics.cpp
  HistogramPerformanceEstimation.cpp
  HistogramPerformanceEstimation.hpp
  HostSliceTensor.cpp
  NaN.cpp
  NormaliseImage.cpp
  Loop.cpp
  Operation.cpp
  Pad.cpp
  Padder.cpp
  PerformanceEstimation.cpp
  popopsCycleEstimators.cpp
  Rearrange.cpp
  ScaledAdd.cpp
  Scatter.cpp
  SelectScalarFromRows.cpp
  SequenceSlice.cpp
  Sort.cpp
  SortOrder.cpp
  SparseUtils.cpp
  TopK.cpp
  UpdateScalarInRows.cpp
  VarianceToOrFromInvStdDev.cpp
  Zero.cpp
  ${CMAKE_SOURCE_DIR}/include/popops/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ScaledAdd.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/AllTrue.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Cast.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/CircBuf.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/TensorCollectives.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/DynamicSlice.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ElementWise.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ElementWiseUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Encoding.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/EncodingConstants.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Expr.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ExprOp.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Fill.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/GatherStatistics.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/NaN.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/NormaliseImage.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Loop.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Operation.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/OperationDef.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/OperationDefUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Pad.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Rearrange.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/SequenceSlice.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Reduce.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/SortOrder.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/TopK.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Zero.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/PerformanceEstimation.hpp

  reduction/ComputeSetList.cpp
  reduction/ComputeSetList.hpp
  reduction/CycleEstimationFunctions.cpp
  reduction/CycleEstimationFunctions.hpp
  reduction/IntermediatePartials.cpp
  reduction/IntermediatePartials.hpp
  reduction/IntermediatePartialsUtil.cpp
  reduction/IntermediatePartialsUtil.hpp
  reduction/Reduction.cpp
  reduction/Reduction.hpp
  reduction/ReductionConnection.cpp
  reduction/ReductionConnection.hpp
  reduction/ReductionIntrospection.cpp
  reduction/ReductionIntrospection.hpp
  reduction/ReductionPlan.cpp
  reduction/ReductionPlan.hpp
  reduction/ReductionStages.cpp
  reduction/ReductionStages.hpp
  reduction/ReductionVertex.hpp
  reduction/RegionWrapping.cpp
  reduction/RegionWrapping.hpp
)

target_link_libraries(popops
  PUBLIC
    poplar poputil ${CMAKE_DL_LIBS}
  PRIVATE
    TBB::tbb
    poplibs_support
    popsolver
    Boost::boost
    Boost::filesystem
    Boost::system
    spdlog::spdlog_header_only
)

target_include_directories(popops
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(codelet_asm_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/BroadcastSelect.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastCommon.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastFromInt8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastHalfFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastToInt8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Clamp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ContinuousReductionAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ContinuousReductionNoAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CompareAndSwapAtDistanceKeyVal_float_unsigned.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/EncodeOneHot.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ForLoopCounter.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/HasNaN.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/NormaliseImage.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/MultiSlice.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/MultiUpdateOp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionSpecial01Acc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionSpecial01NoAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionSpecial2.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionSpecial3.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2DXminusaXPlusbY_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_float_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_half_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_integral.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddSupervisor_float_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddSupervisor_fp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddSupervisor_integral.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd_half_half_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddaXPlusbY_mixed.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/SliceCopyFunction.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Transpose.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/VarianceConversion2D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/VarianceConversion1D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Fill.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/binaryOps2D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/binaryOps1D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/unaryOps2D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/unaryOps1D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/dynamicSlice.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/dynamicSlice1D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/dynamicSlice1D_8bit.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/elementwiseStubs.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_bool.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_int_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorInnerAdd_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorInnerAdd_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorInnerDiv_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorInnerDiv_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorInnerMul_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorInnerMul_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorOuterOps1D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/histogram1DByLimit.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/histogram1DByData.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/exp_inner_loop.S    
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/histogram2D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/unary2DNonLinearity.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/unaryOps1DNonLinearity.S
)

add_gp_library(
  NAME popops
  ASM_SOURCES
    ${codelet_asm_sources}
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastVectorInner2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastVectorInner2DInPlace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastVectorInner1DInPlace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastVectorInner1D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CircBufIncrIndex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CircOffset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CompareAndSwapAtDistance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/DynamicSlice1D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/DynamicSlice2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/DynamicUpdateSlice1D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/DynamicUpdateSlice2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/EncodeOneHot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/EncodeOneHotCustomValues.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HasNaN.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HeapSortVertex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HeapSortVertexKV.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Iota.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MultiSlice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MultiUpdate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MultiUpdateOp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/NormaliseImage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledContinuousReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromInterval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromIntervals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromRowsInColumns.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SeqSlice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Transpose2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Transpose1D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateColumnsDEC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateIntervalDEC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateIntervalsDEC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/broadcastCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseBinaryCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseMiscCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseScaledAddCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseUnaryCodelets.cpp
  HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastCommon.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionsCommon.h.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/unaryOpsOps.h.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/workDivision.h.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/SliceCopyFunction_8bit.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReduce.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HeapSort.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceCodelets.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectScalarFromRows.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elementwiseCodelets.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseBinaryOps.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/histogramCommon.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MultiUpdateOp.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/util.hpp
    ${CMAKE_SOURCE_DIR}/include/poplibs_support/ExternalCodelet.hpp
    ${CMAKE_SOURCE_DIR}/include/poplibs_support/LogArithmetic.hpp
    ${CMAKE_SOURCE_DIR}/include/popops/EncodingConstants.hpp
    ${CMAKE_SOURCE_DIR}/include/popops/ExprOp.hpp
)

install(TARGETS popops
        EXPORT popops
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popops
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popops
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popops
        FILE popops-targets.cmake
        COMPONENT popops)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popops
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popops)
