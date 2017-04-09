include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popstd SHARED
  ActivationMapping.cpp
  Add.cpp
  Cast.cpp
  codelets.cpp
  GraphFunction.cpp
  HadamardProduct.cpp
  Pad.cpp
  Regroup.cpp
  Util.cpp
  Zero.cpp
  ${CMAKE_SOURCE_DIR}/include/popstd/ActivationMapping.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/Add.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/Cast.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/exceptions.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/GraphFunction.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/HadamardProduct.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/Pad.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/Regroup.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/Util.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/VertexTemplates.hpp
  ${CMAKE_SOURCE_DIR}/include/popstd/Zero.hpp
)

target_link_libraries(popstd poplar)

target_include_directories(popstd
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
    popstd.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o popstd.gp
    ${CMAKE_CURRENT_SOURCE_DIR}/popstdCodelets.cpp
  DEPENDS
    popstdCodelets.cpp
)

add_custom_target(popstd_codelets DEPENDS popstd.gp
                  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/popstdCodelets.cpp)
add_dependencies(popstd popstd_codelets)

install(TARGETS popstd
        EXPORT popstd
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popstd
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popstd DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popstd)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popstd
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popstd)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popstd.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popstd)
