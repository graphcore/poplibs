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
    poplibs_support
    Boost::boost
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
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/poprandCodelets.cpp
  ASM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Bernoulli.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Uniform.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Normal.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/TruncatedNormal.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Dropout.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Seeds.S
  HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/poprandCommon.inc
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
