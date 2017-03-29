include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popreduce SHARED
  codelets.cpp
  Reduce.cpp
  PerformanceEstimation.hpp
  ${CMAKE_SOURCE_DIR}/include/popreduce/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popreduce/Reduce.hpp
)

target_link_libraries(popreduce poplar popstd)

target_include_directories(popreduce
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
    popreduce.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o popreduce.gp
    ${CMAKE_CURRENT_SOURCE_DIR}/popreduceCodelets.cpp
  DEPENDS
    PerformanceEstimation.hpp
    popreduceCodelets.cpp
)

add_custom_target(popreduce_codelets DEPENDS popreduce.gp
                  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/popreduceCodelets.cpp)
add_dependencies(popreduce popreduce_codelets)

install(TARGETS popreduce
        EXPORT popreduce
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popreduce
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popreduce DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popreduce)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popreduce
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popreduce)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popreduce.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popreduce)
