include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popnn SHARED
  codelets.cpp
  BatchNorm.cpp
  GroupNorm.cpp
  Loss.cpp
  Lstm.cpp
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
    poplar poplin poputil poplin ${CMAKE_DL_LIBS}
  PRIVATE
    poplibs_support Boost::boost
)

target_include_directories(popnn
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

# Search for arch_man path
foreach(path ${CMAKE_PREFIX_PATH})
    if(EXISTS "${path}/colossus/include/tileimplconsts.h")
      set(ARCH_MAN_PATH ${path})
    endif()
endforeach(path)

set(codelet_asm_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/LossTransform.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Pooling.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMaxClassGather.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMaxClassSparse.S
)

# Generate non-linearity assembly implementation variants
foreach(VERTEX_TYPE Supervisor 2D GradSupervisor Grad2D)
  set(VARIANT_TEMPLATE "codelets/NonLinearity${VERTEX_TYPE}.S.in")
  foreach(NL_TYPE sigmoid relu tanh)
    set(VARIANT_NAME "NonLinearity${VERTEX_TYPE}_${NL_TYPE}")
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/codelets/${VARIANT_NAME}.S")
    string(TOUPPER "${NL_TYPE}" NL_TYPE_UPPER)
    configure_file(${VARIANT_TEMPLATE} ${VARIANT_SRC})
    list(APPEND codelet_asm_sources ${VARIANT_SRC})
  endforeach()
endforeach()

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popnnCodelets.cpp
)
set(codelet_headers
  PerformanceEstimation.hpp
)

add_custom_command(
  OUTPUT
    popnn.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${ARCH_MAN_PATH}/colossus/include
    -o popnn.gp
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_command(
  OUTPUT
    popnn_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS} -DPOPLIBS_DISABLE_ASM_CODELETS
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -o popnn_c.gp
    ${codelet_cpp_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_target(popnn_codelets DEPENDS popnn.gp popnn_c.gp
                  SOURCES ${codelets_cpp_sources} ${codelet_asm_sources})
add_dependencies(popnn popnn_codelets)

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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popnn.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popnn_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn)
