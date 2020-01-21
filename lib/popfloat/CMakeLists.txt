include(GNUInstallDirs)

get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popfloat SHARED
  GfloatExprUtil.cpp
  codelets.cpp
  popfloatCycleEstimators.cpp
  CastToGfloat.cpp
  CastToHalf.cpp
  ${CMAKE_SOURCE_DIR}/include/popfloat/experimental/GfloatExprUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/popfloat/experimental/GfloatExpr.hpp
  ${CMAKE_SOURCE_DIR}/include/popfloat/experimental/CastToGfloat.hpp
  ${CMAKE_SOURCE_DIR}/include/popfloat/experimental/CastToHalf.hpp
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

add_gp_library(
  NAME popfloat
  ASM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastGF16ToFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastFloatToGF16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastGF8ToHalf.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastHalfToGF8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastGF8ToFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastFloatToGF8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastToGfloat16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastToGfloat16Sr.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastToGfloat16Param.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastToGfloat32.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastToGfloat32Sr.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/CastToGfloat32Param.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/PackedGfloatParams.S
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastFloatToGf16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastFloatToGf8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGf16ToFloat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGf8ToFloat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGf8ToHalf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastHalfToGf8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16InPlace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16Param.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16Sr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16SrInPlace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32InPlace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32Param.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32Sr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32SrInPlace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/PackedGfloatParams.cpp
  HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/GfloatConst.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popfloatUtils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popfloatCodelets.hpp
)

install(TARGETS popfloat
        EXPORT popfloat
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popfloat
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popfloat
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popfloat
        FILE popfloat-targets.cmake
        COMPONENT popfloat)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popfloat/experimental
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/popfloat
        COMPONENT popfloat)
