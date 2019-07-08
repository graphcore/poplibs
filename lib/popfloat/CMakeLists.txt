include(GNUInstallDirs)

get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popfloat SHARED
  GfloatExprUtil.cpp
  codelets.cpp
  popfloatCycleEstimators.cpp
  CastToGfloat.cpp
  ${CMAKE_SOURCE_DIR}/include/popfloat/GfloatExprUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/popfloat/GfloatExpr.hpp
  ${CMAKE_SOURCE_DIR}/include/popfloat/CastToGfloat.hpp

  codelets/popfloatUtils.hpp
)

target_link_libraries(popfloat
  PUBLIC
    poplar poputil popops ${CMAKE_DL_LIBS}
)

target_include_directories(popfloat
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(codelet_asm_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGF16ToFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastFloatToGF16Param.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastFloatToGF16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGF16ToFloatParam.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGF8ToHalf.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGF8ToHalfParam.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastHalfToGF8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastHalfToGF8Param.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16Sr.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16Param.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32Sr.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32Param.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/PackedGfloatParams.S
)

set(codelet_cpp_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popfloatCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popfloatParamCodelets.cpp
)

set(codelet_headers
  # Add any header files used here
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popfloatUtils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popfloatCodelets.hpp
)

add_custom_command(
  OUTPUT
    popfloat.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -DNDEBUG
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
    -o popfloat.gp
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
    popfloat_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -DPOPLIBS_DISABLE_ASM_CODELETS
    -DENABLE_POPLAR_RUNTIME_CHECKS
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
    -o popfloat_c.gp
    ${codelet_cpp_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_target(popfloat_codelets DEPENDS popfloat.gp popfloat_c.gp
                  SOURCES ${codelet_cpp_sources} ${codelet_asm_sources})
add_dependencies(popfloat popfloat_codelets)

install(TARGETS popfloat
        EXPORT popfloat
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popfloat
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popfloat
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popfloat
        FILE popfloat-targets.cmake
        COMPONENT popfloat)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popfloat
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popfloat)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popfloat.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popfloat)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popfloat_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popfloat)
