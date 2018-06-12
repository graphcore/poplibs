include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

# Search for the runtime and arch man
foreach(path ${CMAKE_PREFIX_PATH})
    if(EXISTS "${path}/lib/graphcore/include/stddef.h")
      set(RUNTIME_PATH ${path})
    endif()
    if(EXISTS "${path}/colossus/include/tileimplconsts_tommy.h")
      set(ARCH_MAN_PATH ${path})
    endif()
endforeach(path)

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/popnnCodelets.cpp
)
set(codelet_asm_sources
)

add_custom_command(
  OUTPUT
    popnn.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_SOURCE_DIR}/include
    -I ${ARCH_MAN_PATH}/colossus/include
    -target cpu,ipu0
    -o popnn.gp
    ${codelet_cpp_sources} ${codelet_asm_sources}
  DEPENDS
    ${codelet_cpp_sources} ${codelet_asm_sources}
    popc_bin
)

add_custom_target(popnn_codelets ALL DEPENDS popnn.gp
                  SOURCES ${codelet_cpp_sources} ${codelet_asm_sources})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popnn.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn_codelets)

