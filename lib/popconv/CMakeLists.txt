include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

find_package(Boost REQUIRED)

add_library(popconv SHARED
  codelets.cpp
  popconvCycleEstimators.cpp
  Convolution.cpp
  ConvPlan.cpp
  ConvReduce.hpp
  ConvReduce.cpp
  ConvUtil.cpp
  ConvValidation.hpp
  ConvValidation.cpp
  PerformanceEstimation.hpp
  Winograd.cpp
  Winograd.hpp
  ${CMAKE_SOURCE_DIR}/include/popconv/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popconv/Convolution.hpp
  ${CMAKE_SOURCE_DIR}/include/popconv/ConvUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/popconv/internal/ConvPlan.hpp
)

# Search for the runtime and arch man
foreach(path ${CMAKE_PREFIX_PATH})
    if(EXISTS "${path}/colossus/include/tileimplconsts_tommy.h")
      set(ARCH_MAN_PATH ${path})
    endif()
endforeach(path)

target_link_libraries(popconv
  PUBLIC
    poplar popops poputil popsolver ${CMAKE_DL_LIBS}
  PRIVATE
    Boost::boost
)

target_include_directories(popconv
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popconvCodelets.cpp
)
set(codelet_asm_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/conv_partial_1x1_half_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/conv_partial_nx1_half_float.S
)
set(codelet_headers
  PerformanceEstimation.hpp
)

add_custom_command(
  OUTPUT
    popconv.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${ARCH_MAN_PATH}/colossus/include
    -o popconv.gp
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
    popconv_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS} -DPOPLIBS_DISABLE_ASM_CODELETS
    -I ${CMAKE_CURRENT_SOURCE_DIR}   
    -o popconv_c.gp
    ${codelet_cpp_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_target(graph_program DEPENDS popconv.gp popconv_c.gp
                  SOURCES ${codelet_cpp_source} ${codelet_asm_sources})
add_dependencies(popconv graph_program)

install(TARGETS popconv
        EXPORT popconv
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popconv
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popconv
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popconv
        FILE popconv-targets.cmake
        COMPONENT popconv)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popconv
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popconv)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popconv.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popconv)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popconv_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popconv)
