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
  ElementWise.cpp
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
  PerformanceEstimation.hpp
  PopopsChannelOps.cpp
  popopsCycleEstimators.cpp
  ScaledAdd.cpp
  Scatter.cpp
  SelectScalarFromRows.cpp
  Sort.cpp
  SparseUtils.cpp
  UpdateScalarInRows.cpp
  Zero.cpp
  ${CMAKE_SOURCE_DIR}/include/popops/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/PopopsChannelOps.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ScaledAdd.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/AllTrue.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Cast.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/CircBuf.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Collectives.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/DynamicSlice.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ElementWise.hpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Clamp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Cast.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReductionFloatFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReductionFloatHalf.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReductionHalfFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ContinuousReductionHalfHalf.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Zero.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/dynamicSlice.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/dynamicSliceSupervisor.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/EncodeOneHot.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MultiUpdateAdd.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledAdd2D_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledAdd2D_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledAdd2D_integral.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledAddSupervisor_fp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ScaledAddSupervisor_integral.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/select_bool.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/select_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/select_int_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/BroadcastSelect.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReductionsCommon.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReducePartialsEqualSizeAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReducePartialsEqualSizeNoAcc.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elementwiseStubs.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/AddToChannel_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/AddToChannel_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ChannelMul_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ChannelMul_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/binaryOps.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/binarySupervisorOps.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/vectorOuterSupervisorOps.S
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
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/codelets/Reductions.S
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
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/codelets/Reductions_4_vectorised.S
                 ${OUTPUT_FILE})
  list(APPEND codelet_asm_sources ${OUTPUT_FILE})
endforeach()

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/channelOpCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/dynamicSliceCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseUnaryCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseBinaryCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elemwiseMiscCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/broadcastCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/encodingCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/miscCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/nanCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/reduceCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/heapSort.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/selectScalarFromRowsCodelets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/updateScalarInRowsCodelets.cpp
)

# Header files used by the codelets. This is needed as we will not be getting
# automatic dependency generation for codelet .cpp (or .S) files.
# TODO: T10282 - fix broken dependencies with poplar's headers.
set(codelet_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/util.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/EncodingConstants.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/elementwiseCodelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ExprOp.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/ExternalCodelet.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/TileConstants.hpp
)

set (codelet_all_sources ${codelet_cpp_sources} ${codelet_asm_sources})

# Each .S or .cpp file will be compiled individually into a .gp file.
# The file names of all .gp files will be accumulated in this list.
set (codelet_all_gp_files)

# A rule ('custom command') to compile each .S or .cpp codelet source file.
foreach(source ${codelet_all_sources})
  get_filename_component(filename ${source} NAME)
  set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/codelets/${filename}.gp)
  list(APPEND codelet_all_gp_files ${OUT_FILE})
  add_custom_command(
    OUTPUT
      ${OUT_FILE}
    COMMAND
      ${POPC_EXECUTABLE} ${POPC_FLAGS}
      -DNDEBUG
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
      -I ${ARCH_MAN_PATH}/include
      -o ${OUT_FILE}
      ${source}
    DEPENDS
      ${source}
      ${codelet_headers}
      popc_bin
  )
endforeach()

# A rule to link all .gp files into 'popops.gp'
add_custom_command(
  OUTPUT
    popops.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o popops.gp
    ${codelet_all_gp_files}
  DEPENDS
    ${codelet_all_gp_files}
    popc_bin
)

# The .cpp files will be compiled individually into a .gp file for the
# 'C/C++ only' build. All file names of .gp files thusly generated will be
# accumulated here.
set (codelet_cpp_gp_files)

# A rule ('custom command') for each .cpp codelet source file. Note that
# compiler options are different from those in the 'all sources' rule above.
foreach(source ${codelet_cpp_sources})
  get_filename_component(filename ${source} NAME)
  set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/codelets/${filename}_c.gp)
  list(APPEND codelet_cpp_gp_files ${OUT_FILE})
  add_custom_command(
    OUTPUT
      ${OUT_FILE}
    COMMAND
      ${POPC_EXECUTABLE} ${POPC_FLAGS}
      -DPOPLIBS_DISABLE_ASM_CODELETS
      -DENABLE_POPLAR_RUNTIME_CHECKS
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
      -I ${ARCH_MAN_PATH}/include
      -o ${OUT_FILE}
      ${source}
    DEPENDS
      ${source}
      ${codelet_headers}
      popc_bin
  )
endforeach()

# A rule to link all .gp files into the 'C/C++ only' build ('popops_c.gp').
add_custom_command(
  OUTPUT
    popops_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o popops_c.gp
    ${codelet_cpp_gp_files}
  DEPENDS
    ${codelet_cpp_gp_files}
    popc_bin
)

add_custom_target(popops_codelets DEPENDS popops.gp popops_c.gp
                  SOURCES ${codelet_cpp_sources} ${codelet_asm_sources}
                          ${codelet_headers})
add_dependencies(popops popops_codelets)

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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popops.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popops)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popops_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popops)
