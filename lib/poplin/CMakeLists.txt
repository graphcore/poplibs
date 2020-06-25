include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)


add_library(poplin SHARED
  codelets.cpp
  poplinCycleEstimators.cpp
  ConvOptions.hpp
  ConvOptions.cpp
  Convolution.cpp
  ConvParams.cpp
  CanonicalConvParams.hpp
  ConvPlan.hpp
  ConvPlan.cpp
  ConvReduce.hpp
  ConvReduce.cpp
  ConvReducePlan.hpp
  ConvReducePlan.cpp
  ConvUtil.cpp
  ConvUtilInternal.cpp
  ConvUtilInternal.hpp
  ConvValidation.hpp
  ConvValidation.cpp
  CreateConvPartialVertex.hpp
  MatMul.cpp
  MultiConvolution.cpp
  FullyConnected.cpp
  MatMulInternal.hpp
  MeshGrid.cpp
  Norms.cpp
  PerformanceEstimation.hpp
  Winograd.cpp
  Winograd.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/Convolution.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/ConvUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/MatMul.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/FullyConnected.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/MeshGrid.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/Norms.hpp
)

target_link_libraries(poplin
  PUBLIC
    poplar popops poputil
  PRIVATE
    poplibs_support
    popsolver
    Boost::boost
    TBB::TBB
)

target_include_directories(poplin
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

add_gp_library(
  NAME poplin
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartial1x1Out.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartial1x4SLIC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartialHorizontalMac.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ConvPartialnx1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/InverseStdDeviation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/OuterProduct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceAdd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdConvComplete.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdDataTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdInverseTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdKernelTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdPartials.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/WgdReduce.cpp
  ASM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_supervisor.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_float_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_float_float_16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_half_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_1x1_half_float_8_and_half_half_h16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_zero_output.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_supervisor.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_float_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_float_float_16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_half_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_partial_nx1_half_float_8_and_half_half_16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ConvPartial1x4SLIC_half_float_8_and_half_half_16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ConvPartial1x4SLIC_half_half_8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReduceAdd.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/ReduceAddSingleInput.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_half_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_half_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/conv_hzmac_float_float.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/OuterProduct.S
  HEADERS
    PerformanceEstimation.hpp
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
