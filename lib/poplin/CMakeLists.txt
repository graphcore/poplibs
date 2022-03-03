include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)


add_library(poplin SHARED
  CanonicalConvParams.hpp
  Cholesky.cpp
  codelets.cpp
  ConvModel.cpp
  ConvModel.hpp
  Convolution.cpp
  ConvOptions.cpp
  ConvOptions.hpp
  ConvParams.cpp
  ConvPartialsStridesPacking.cpp
  ConvPartialsStridesPacking.hpp
  ConvPlan.cpp
  ConvPlan.hpp
  ConvPlanTypes.hpp
  ConvPreplan.cpp
  ConvReduce.cpp
  ConvReduce.hpp
  ConvReducePlan.cpp
  ConvReducePlan.hpp
  ConvTransforms.cpp
  ConvTransforms.hpp
  ConvUtil.cpp
  ConvUtilInternal.cpp
  ConvUtilInternal.hpp
  ConvValidation.cpp
  ConvValidation.hpp
  ConvVertexType.cpp
  ConvVertexType.hpp
  ConvVertices.cpp
  ConvVertices.hpp
  ExchangeEstimator.hpp
  FullyConnected.cpp
  MatMul.cpp
  MatMulInternal.hpp
  MeshGrid.cpp
  MultiConvolution.cpp
  Norms.cpp
  PerformanceEstimation.hpp
  PlanningCache.hpp
  PlanningObjective.hpp
  poplinCycleEstimators.cpp
  TriangularSolve.cpp
  Winograd.cpp
  Winograd.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/Cholesky.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/Convolution.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/ConvPreplan.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/ConvUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/FullyConnected.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/MatMul.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/MeshGrid.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/Norms.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/TriangularSolve.hpp
)

target_link_libraries(poplin
  PUBLIC
    poplar popops poputil
  PRIVATE
    poplibs_support
    popsolver
    Boost::boost
    TBB::tbb
    spdlog::spdlog_header_only
)

target_include_directories(poplin
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(IPU1_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_half_half.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_float_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_supervisor.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_zero_output.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_supervisor.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_float_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_half_half.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReduceAdd.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReduceAddSingleInput.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_vmac_half_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_vmac_half_half.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_vmac_half_half_8.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_vmac_half_half_16.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_half_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_float_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_half_half.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_half_half_non_limited.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_half_float_non_limited.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_float_float_non_limited.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ConvPartial1x4SLIC_half_half_8.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/OuterProduct.S
)
# These contain source for both IPU1 and IPU2.  Internal to the source file
# IPU2 is conditionally assembled by checking __IPU_ARCH_VERSION__
set(IPU1_AND_IPU2_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_half_float_8_and_half_half_h16.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ConvPartial1x4SLIC_half_float_8_and_half_half_16.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ConvPartial1x4SLIC_half_float_8_and_half_half_16_cgpg_gt4.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_half_float_8_and_half_half_16.S
)
set(IPU2_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_float_float_16.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_float_float_16.S
)

set(ARCH_SOURCES ${IPU1_SOURCES})
list(APPEND ARCH_SOURCES ${IPU1_AND_IPU2_SOURCES})
if("ipu2" IN_LIST POPLIBS_ENABLED_IPU_ARCH_NAMES)
  list(APPEND ARCH_SOURCES ${IPU2_SOURCES})
endif()

add_gp_library(
  NAME poplin
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Cholesky.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartial1x1Out.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartial1xNSLIC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartialHorizontalMac.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartialVerticalMac.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartialnx1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/InverseStdDeviation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/OuterProduct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceAdd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/TriangularSolve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdConvComplete.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdDataTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdInverseTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdKernelTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdPartials.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdReduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConvPartialsStridesPacking.cpp

  ASM_SOURCES
    ${ARCH_SOURCES}

  HEADERS
    PerformanceEstimation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/convCastSupport.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Dot.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/inlineAssemblerConv.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/inlineAssemblerSLIC.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/inlineAssemblerSLICStride2.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_vmac_common.h.S
)

install(TARGETS poplin
        EXPORT poplin
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poplin
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT poplin
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poplin
        FILE poplin-targets.cmake
        COMPONENT poplin)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/poplin
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poplin)
