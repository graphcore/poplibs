include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popnn SHARED
  codelets.cpp
  BatchNorm.cpp
  GroupNorm.cpp
  Loss.cpp
  Lstm.cpp
  Gru.cpp
  Norms.cpp
  NormsInternal.cpp
  NormsInternal.hpp
  Pooling.cpp
  PoolOptions.hpp
  PoolPlan.cpp
  PoolPlan.hpp
  PoolVertices.cpp
  PoolVertices.hpp
  popnnCycleEstimators.cpp
  NonLinearity.cpp
  PerformanceEstimation.hpp
  Recurrent.cpp
  SpatialSoftMax.cpp
  ${CMAKE_SOURCE_DIR}/include/popnn/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/BatchNorm.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/GroupNorm.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/InstanceNorm.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/LayerNorm.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Loss.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/LstmDef.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Lstm.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/GruDef.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Gru.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/NonLinearityDef.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/NonLinearityDefUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/NonLinearity.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Norms.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/PoolingDef.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Pooling.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Recurrent.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/SpatialSoftMax.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/Compiler.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/gcd.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/VectorUtils.hpp
)

target_link_libraries(popnn
  PUBLIC
    poplar poplin poputil ${CMAKE_DL_LIBS}
  PRIVATE
    poplibs_support Boost::boost
    TBB::TBB
)

target_include_directories(popnn
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(codelet_asm_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/LossTransform.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Pooling.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/NonLinearity2D_gelu.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/NonLinearityGrad2D_gelu.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/NonLinearitySupervisor_gelu.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/NonLinearityGradSupervisor_gelu.S
)

foreach(MIN_OR_MAX "Max" "Min")
  foreach(VARIANT Gather Sparse)
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/codelets/Reduce${MIN_OR_MAX}Class${VARIANT}.S")

    if ("${MIN_OR_MAX}" STREQUAL "Max")
      set(COMPARE_OP "cmpgt")
    else()
      set(COMPARE_OP "cmplt")
    endif()

    string(TOLOWER ${MIN_OR_MAX} MIN_OR_MAX_LOWER)
    configure_file("codelets/asm/ReduceClass${VARIANT}.S" ${VARIANT_SRC})
    list(APPEND codelet_asm_sources ${VARIANT_SRC})
  endforeach()
endforeach()


# Generate non-linearity assembly implementation variants
foreach(VERTEX_TYPE Supervisor 2D GradSupervisor Grad2D)
  set(VARIANT_TEMPLATE "codelets/asm/NonLinearity${VERTEX_TYPE}.S.in")
  foreach(NL_TYPE sigmoid relu tanh)
    set(VARIANT_NAME "NonLinearity${VERTEX_TYPE}_${NL_TYPE}")
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/codelets/${VARIANT_NAME}.S")
    string(TOUPPER "${NL_TYPE}" NL_TYPE_UPPER)
    configure_file(${VARIANT_TEMPLATE} ${VARIANT_SRC})
    list(APPEND codelet_asm_sources ${VARIANT_SRC})
  endforeach()
endforeach()

add_gp_library(
  NAME popnn
  ASM_SOURCES
    ${codelet_asm_sources}
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CalcAccuracy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/LossCrossEntropyTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/LossSumSquaredTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MaxPooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MaxPoolingGrad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MaxPoolingGradientScale.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/NonLinearity2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/NonLinearityGrad2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/NonLinearityGradSupervisor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/NonLinearitySupervisor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMaxClassGather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMaxClassSparse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMaxNClassGather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMaxNClassSparse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMinClassGather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMinClassSparse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SelectiveScaling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SumPooling.cpp
  HEADERS
    PerformanceEstimation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MinHeapView.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/NonLinearity.hpp
    ${CMAKE_SOURCE_DIR}/include/popops/EncodingConstants.hpp
)

install(TARGETS popnn
        EXPORT popnn
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popnn
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popnn
        FILE popnn-targets.cmake
        COMPONENT popnn)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popnn
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popnn)
