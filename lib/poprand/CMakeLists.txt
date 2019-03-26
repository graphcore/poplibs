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
)

target_include_directories(poprand
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

# Search for arch_man path
foreach(path ${CMAKE_PREFIX_PATH})
  if(EXISTS "${path}/include/colossus/tileimplconsts.h")
    set(ARCH_MAN_PATH ${path})
  endif()
endforeach(path)

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/poprandCodelets.cpp
)

set(codelet_asm_sources
  # Assembly to be added
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Seeds.S
)
set(codelet_headers
  # Add any header files used here
)

add_custom_command(
  OUTPUT
    poprand.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -DNDEBUG
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
    -I ${ARCH_MAN_PATH}/include
    -o poprand.gp
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_command(
  OUTPUT
    poprand_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -DPOPLIBS_DISABLE_ASM_CODELETS
    -DENABLE_POPLAR_RUNTIME_CHECKS
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
    -I ${ARCH_MAN_PATH}/include
    -o poprand_c.gp
    ${codelet_cpp_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_target(poprand_codelets DEPENDS poprand.gp poprand_c.gp
                  SOURCES ${codelet_cpp_sources} ${codelet_asm_sources})
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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/poprand_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprand)
