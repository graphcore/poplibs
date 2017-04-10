include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(poplin SHARED
  codelets.cpp
  MatMul.cpp
  MatMulPlan.cpp
  MatMulPlan.hpp
  PerformanceEstimation.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/MatMul.hpp
)

target_link_libraries(poplin poplar popconv popreduce popstd)

target_include_directories(poplin
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
    poplin.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o poplin.gp
    ${CMAKE_CURRENT_SOURCE_DIR}/poplinCodelets.cpp
  DEPENDS
    poplinCodelets.cpp
    PerformanceEstimation.hpp
)

add_custom_target(poplin_codelets DEPENDS poplin.gp SOURCES poplinCodelets.cpp)
add_dependencies(poplin poplin_codelets)

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
