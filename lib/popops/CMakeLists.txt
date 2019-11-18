include(GNUInstallDirs)

find_package(Boost 1.65.1 REQUIRED)

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
  NaN.cpp
  Operation.cpp
  Pad.cpp
  Padder.cpp
  popopsCycleEstimators.cpp
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
    poplar poputil popsolver ${CMAKE_DL_LIBS}
  PRIVATE
    TBB::TBB
    poplibs_support Boost::boost
)

target_include_directories(popops
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(codelet_asm_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Clamp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Cast.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ContinuousReductionFloatFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ContinuousReductionFloatHalf.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ContinuousReductionHalfFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ContinuousReductionHalfHalf.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Zero.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/dynamicSlice.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/dynamicSliceSupervisor.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/EncodeOneHot.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/MultiUpdateAdd.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd_half_half_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAdd2D_integral.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddSupervisor_fp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ScaledAddSupervisor_integral.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_bool.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/select_int_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/BroadcastSelect.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReductionsCommon.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReducePartialsEqualSizeAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReducePartialsEqualSizeNoAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/elementwiseStubs.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/AddToChannel_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/AddToChannel_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ChannelMul_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ChannelMul_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/binaryOps.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/binarySupervisorOps.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/vectorOuterSupervisorOps.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/VarianceConversion2D.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/VarianceConversionSupervisor.S
)

foreach(ReduceOp ReduceAdd ReduceSquareAdd)
  foreach(IS_UPDATE 0 1)
    if("${ReduceOp}" STREQUAL "ReduceAdd")
      set(INSTRUCTION "acc")
    elseif("${ReduceOp}" STREQUAL "ReduceSquareAdd")
      set(INSTRUCTION "sqacc")
    endif()
    set(OUTPUT_FILE
        ${CMAKE_CURRENT_BINARY_DIR}/codelets/${ReduceOp}_${IS_UPDATE}.S)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Reductions.S
                   ${OUTPUT_FILE})
    list(APPEND codelet_asm_sources ${OUTPUT_FILE})
  endforeach()
endforeach()

foreach(ReduceOp ReduceMul ReduceMax ReduceMin)
  if("${ReduceOp}" STREQUAL "ReduceMul")
    set(INSTRUCTION "mul")
    set(INIT_FLOAT "0x3f800000") # 1.0 in float format
    set(INIT_HALF "0x3c003c00") # 1.0 half format broadcast
  elseif("${ReduceOp}" STREQUAL "ReduceMax")
    set(INSTRUCTION "max")
    set(INIT_FLOAT "0xff800000") # -inf in float format
    # tile treats inf as NAN for halves so must use representable number
    set(INIT_HALF  "0xfbfffbff") # smallest half broadcast
  elseif("${ReduceOp}" STREQUAL "ReduceMin")
    set(INSTRUCTION "min")
    set(INIT_FLOAT "0x7f800000") # inf in float format
    set(INIT_HALF  "0x7bff7bff") # largest half broadcast
  endif()
  set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/codelets/${ReduceOp}.S)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Reductions_4_vectorised.S
                 ${OUTPUT_FILE})
  list(APPEND codelet_asm_sources ${OUTPUT_FILE})
endforeach()

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
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReducePartialsEqualSize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledContinuousReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledReducePartialsEqualSize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromInterval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromIntervals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectFromRowsInColumns.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateColumnsDEC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateIntervalDEC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/UpdateIntervalsDEC.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseUnaryCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseBinaryCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseMiscCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/broadcastCodelets.cpp
  HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReduce.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ElementOp.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/HeapSort.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceCodelets.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReducePartialsEqualSize.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectScalarFromRows.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/util.hpp
    ${CMAKE_SOURCE_DIR}/include/poplibs_support/ExternalCodelet.hpp
    ${CMAKE_SOURCE_DIR}/include/poplibs_support/TileConstants.hpp
    ${CMAKE_SOURCE_DIR}/include/popops/elementwiseCodelets.hpp
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
