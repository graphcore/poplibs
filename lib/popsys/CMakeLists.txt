include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

find_package(Boost REQUIRED)

add_library(popsys SHARED
  codelets.cpp
  CycleCount.cpp
  popsysCycleEstimators.cpp
  ${CMAKE_SOURCE_DIR}/include/popsys/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popsys/CycleCount.hpp
)

# Search for the runtime and arch man
foreach(path ${CMAKE_PREFIX_PATH})
    if(EXISTS "${path}/colossus/include/tileimplconsts_tommy.h")
      set(ARCH_MAN_PATH ${path})
    endif()
endforeach(path)

target_link_libraries(popsys
  PUBLIC
    poplar poputil ${CMAKE_DL_LIBS}
  PRIVATE
    poplibs_support Boost::boost
)

target_include_directories(popsys
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

foreach(CSR 0 96)
  set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/codelets/GetCSR_${CSR}.S)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/codelets/GetCSR.S
                 ${OUTPUT_FILE})
  list(APPEND SYS_VERTICES ${OUTPUT_FILE})
endforeach()

set(codelet_asm_sources
  ${SYS_VERTICES}
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/GetCycleCount.S
)

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/PopsysCodelets.cpp
)

add_custom_command(
  OUTPUT
    popsys.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${ARCH_MAN_PATH}/colossus/include
    -o popsys.gp
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
    ${codelet_headers}
    popc_bin
)

# as many of these vertices require to being run on the supervisor these must
# not be compiled as the compiler can't do this. Hence creating the _c with
# same flags
add_custom_command(
  OUTPUT
    popsys_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${ARCH_MAN_PATH}/colossus/include
    -o popsys_c.gp
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_target(popsys_codelets DEPENDS popsys.gp popsys_c.gp
                  SOURCES ${codelet_cpp_source} ${codelet_asm_sources})
add_dependencies(popsys popsys_codelets)

install(TARGETS popsys
        EXPORT popsys
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popsys
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popsys
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popsys
        FILE popsys-targets.cmake
        COMPONENT popsys)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popsys
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popsys)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popsys.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popsys)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popsys_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popsys)
