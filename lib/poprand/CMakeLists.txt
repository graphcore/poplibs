include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(poprand SHARED
  codelets.cpp
  RandomGen.cpp
  poprandCycleEstimators.cpp
  ${CMAKE_SOURCE_DIR}/include/poprand/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poprand/RandomGen.hpp
)

target_link_libraries(poprand
  PUBLIC
    poplar poputil ${CMAKE_DL_LIBS}
  PRIVATE
    poplibs_support popops
    Boost::boost
    spdlog::spdlog_header_only
)

target_include_directories(poprand
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

add_gp_library(
  NAME poprand
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Bernoulli.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Dropout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Normal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/SetSeed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/TruncatedNormal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Uniform.cpp
  ASM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Bernoulli.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Uniform.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Normal.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/TruncatedNormal.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Dropout.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/Seeds.S
  HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/asm/poprandCommon.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/RandomUtils.hpp
)

install(TARGETS poprand
        EXPORT poprand
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprand
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT poprand
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poprand
        FILE poprand-targets.cmake
        COMPONENT poprand)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/poprand
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poprand)
