include(GNUInstallDirs)

get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popfloat SHARED
  GfloatExprUtil.cpp
  codelets.cpp
  popfloatCycleEstimators.cpp
  CastToGfloat.cpp
  CastToHalf.cpp
  ${CMAKE_SOURCE_DIR}/include/experimental/popfloat/GfloatExprUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/experimental/popfloat/GfloatExpr.hpp
  ${CMAKE_SOURCE_DIR}/include/experimental/popfloat/CastToGfloat.hpp
  ${CMAKE_SOURCE_DIR}/include/experimental/popfloat/CastToHalf.hpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGF16ToFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastFloatToGF16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGF8ToHalf.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastHalfToGF8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastGF8ToFloat.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastFloatToGF8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16Sr.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat16Param.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32Sr.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/CastToGfloat32Param.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/PackedGfloatParams.S
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popfloatCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popfloatParamCodelets.cpp
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

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/experimental/popfloat
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/experimental
        COMPONENT popfloat)
