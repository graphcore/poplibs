include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popconv SHARED
  codelets.cpp
  Convolution.cpp
  ConvPlan.cpp
  ConvPlan.hpp
  ConvUtil.cpp
  ConvValidation.hpp
  ConvValidation.cpp
  PerformanceEstimation.hpp
  Winograd.cpp
  Winograd.hpp
  ${CMAKE_SOURCE_DIR}/include/popconv/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popconv/Convolution.hpp
  ${CMAKE_SOURCE_DIR}/include/popconv/ConvUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/util/Compiler.hpp
  ${CMAKE_SOURCE_DIR}/include/util/gcd.hpp
)

target_link_libraries(popconv poplar popreduce popstd popsolver)

target_include_directories(popconv
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(POPC_FLAGS -I ${CMAKE_CURRENT_SOURCE_DIR} -I ${CMAKE_SOURCE_DIR}/include)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  list(APPEND POPC_FLAGS -g)
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
  list(APPEND POPC_FLAGS -O3 -g)
else()
  list(APPEND POPC_FLAGS -O3)
endif()

add_custom_command(
  OUTPUT
    popconv.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o popconv.gp
    ${CMAKE_CURRENT_SOURCE_DIR}/popconvCodelets.cpp
  DEPENDS
    popconvCodelets.cpp
    PerformanceEstimation.hpp
)

add_custom_target(graph_program DEPENDS popconv.gp SOURCES popconvCodelets.cpp)
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
