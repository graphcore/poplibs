include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

find_package(Boost REQUIRED)

add_library(poplin SHARED
  codelets.cpp
  poplinCycleEstimators.cpp
  ChannelOps.cpp
  ChannelOps.hpp
  Convolution.cpp
  ConvPlan.cpp
  ConvReduce.hpp
  ConvReduce.cpp
  ConvUtil.cpp
  ConvUtilInternal.cpp
  ConvUtilInternal.hpp
  ConvValidation.hpp
  ConvValidation.cpp
  MatMul.cpp
  PerformanceEstimation.hpp
  Winograd.cpp
  Winograd.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/Convolution.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/ConvUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/MatMul.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/internal/ConvPlan.hpp
)

# Search for the runtime and arch man
foreach(path ${CMAKE_PREFIX_PATH})
    if(EXISTS "${path}/colossus/include/tileimplconsts.h")
      set(ARCH_MAN_PATH ${path})
    endif()
endforeach(path)

target_link_libraries(poplin
  PUBLIC
    poplar popops poputil popsolver
  PRIVATE
    poplibs_support Boost::boost
)

target_include_directories(poplin
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

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/poplinCodelets.cpp
)
set(codelet_asm_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/conv_partial_1x1_half_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/conv_partial_nx1_half_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/AddToChannel_half.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/AddToChannel_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ChannelMul_half.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ChannelMul_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceAdd.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/conv_hzmac_half_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Transpose2d.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/OuterProduct.S
)
set(codelet_headers
  PerformanceEstimation.hpp
)

add_custom_command(
  OUTPUT
    poplin.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${ARCH_MAN_PATH}/colossus/include
    -o poplin.gp
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
    poplin_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS} -DPOPLIBS_DISABLE_ASM_CODELETS
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -o poplin_c.gp
    ${codelet_cpp_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_target(graph_program DEPENDS poplin.gp poplin_c.gp
                  SOURCES ${codelet_cpp_source} ${codelet_asm_sources})
add_dependencies(poplin graph_program)

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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/poplin.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poplin)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/poplin_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poplin)
