include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(poprand SHARED
  codelets.cpp
  RandomGen.cpp
  poprandCycleEstimators.cpp
  ${CMAKE_SOURCE_DIR}/include/poprand/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poprand/RandomGen.hpp
)

target_link_libraries(poprand poplar poputil ${CMAKE_DL_LIBS})

target_include_directories(poprand
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
    poprand.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o poprand.gp
    ${CMAKE_CURRENT_SOURCE_DIR}/poprandCodelets.cpp
  DEPENDS
    poprandCodelets.cpp
    popc_bin
)

add_custom_target(poprand_codelets DEPENDS poprand.gp
                  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/poprandCodelets.cpp)
add_dependencies(poprand poprand_codelets)

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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/poprand.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprand)
