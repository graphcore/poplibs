include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popops SHARED
  Add.cpp
  AllTrue.cpp
  Cast.cpp
  CircBuf.cpp
  codelets.cpp
  DynamicSlice.cpp
  ElementWise.cpp
  ElementWiseDef.hpp
  HadamardProduct.cpp
  Pad.cpp
  PerformanceEstimation.hpp
  popopsCycleEstimators.cpp
  Reduce.cpp
  SubtractFrom.cpp
  Zero.cpp
  ${CMAKE_SOURCE_DIR}/include/popops/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Add.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/AllTrue.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Cast.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/CircBuf.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/DynamicSlice.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ElementWise.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/HadamardProduct.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Pad.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Reduce.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/SubtractFrom.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Zero.hpp
)

target_link_libraries(popops poplar poputil)

target_include_directories(popops
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

set(codelet_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/allCodelets.cpp
  )

add_custom_command(
  OUTPUT
    popops.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o popops.gp
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
    ${codelet_sources}
  DEPENDS
    PerformanceEstimation.hpp
    ${codelet_sources}
    popc_bin
)

add_custom_target(popops_codelets DEPENDS popops.gp
                  SOURCES ${codelet_sources})
add_dependencies(popops popops_codelets)

install(TARGETS popops
        EXPORT popops
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popops
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popops
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popops
        FILE popops-targets.cmake
        COMPONENT popops)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popops
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popops)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popops.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popops)
