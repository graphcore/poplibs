include(GNUInstallDirs)

find_package(Boost 1.65.1 COMPONENTS filesystem system REQUIRED)

get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popops SHARED
  AllTrue.cpp
  Cast.cpp
  CircBuf.cpp
  Collectives.cpp
  ReplicatedCollectives.cpp
  codelets.cpp
  DynamicSlice.cpp
  DynamicSliceInternal.hpp
  ElementWise.cpp
  ElementWiseUtil.cpp
  Encoding.cpp
  Expr.cpp
  ExpressionGenerator.cpp
  ExpressionGenerator.hpp
  ExprOpUtil.cpp
  ExprOpUtil.hpp
  Gather.cpp
  GatherInternal.cpp
  HostSliceTensor.cpp
  NaN.cpp
  Operation.cpp
  Pad.cpp
  Padder.cpp
  popopsCycleEstimators.cpp
  Rearrange.cpp
  ScaledAdd.cpp
  Scatter.cpp
  SelectScalarFromRows.cpp
  Sort.cpp
  SparseUtils.cpp
  UpdateScalarInRows.cpp
  VarianceToOrFromInvStdDev.cpp
  Zero.cpp
  ${CMAKE_SOURCE_DIR}/include/popops/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ScaledAdd.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/AllTrue.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Cast.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/CircBuf.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Collectives.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/DynamicSlice.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ElementWise.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ElementWiseUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Encoding.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/EncodingConstants.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Expr.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ExprOp.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/NaN.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Operation.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Pad.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Rearrange.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Reduce.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Zero.hpp

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
  reduction/ReductionDebug.hpp
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
    TBB::TBB
    poplibs_support
    popsolver
    Boost::boost
    Boost::filesystem
    Boost::system
)

target_include_directories(popops
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(codelet_asm_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/AddToChannel_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/AddToChannel_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/BroadcastSelect.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Cast.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ChannelMul_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ChannelMul_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Clamp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ContinuousReductionAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ContinuousReductionNoAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/EncodeOneHot.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/MultiSlice.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/MultiUpdateAdd.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionSpecial01Acc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionSpecial01NoAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionSpecial2.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionSpecial3.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionsCommon.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2DXminusaXPlusbY_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_float_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_integral.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddSupervisor_float_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddSupervisor_fp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddSupervisor_integral.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd_half_half_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/SliceCopyFunction.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Transpose.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/VarianceConversion2D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/VarianceConversionSupervisor.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Zero.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/binaryOps.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/binarySupervisorOps.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/dynamicSlice.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/dynamicSliceSupervisor.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/elementwiseStubs.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_bool.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_int_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorOuterSupervisorOps.S
)

add_gp_library(
  NAME popops
  ASM_SOURCES
    ${codelet_asm_sources}
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastVectorInner2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastVectorInner2DInPlace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastVectorInnerInPlaceSupervisor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastVectorInnerSupervisor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CircBufIncrIndex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CircOffset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/DynamicSlice1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/DynamicSlice2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/DynamicUpdateSlice1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/DynamicUpdateSlice2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/EncodeOneHot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/EncodeOneHotCustomValues.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HasNaN.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HeapSortVertex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HeapSortVertexKV.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Iota.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MultiSlice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MultiUpdate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MultiUpdateAdd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledContinuousReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromInterval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromIntervals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromRowsInColumns.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Transpose2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/TransposeSupervisor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateColumnsDEC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateIntervalDEC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateIntervalsDEC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/broadcastCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseBinaryCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseMiscCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseUnaryCodelets.cpp
  HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReduce.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ElementOp.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HeapSort.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceCodelets.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectScalarFromRows.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/util.hpp
    ${CMAKE_SOURCE_DIR}/include/poplibs_support/ExternalCodelet.hpp
    ${CMAKE_SOURCE_DIR}/include/poplibs_support/TileConstants.hpp
    ${CMAKE_SOURCE_DIR}/include/popops/EncodingConstants.hpp
    ${CMAKE_SOURCE_DIR}/include/popops/ExprOp.hpp
    ${CMAKE_SOURCE_DIR}/include/popops/elementwiseCodelets.hpp
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
