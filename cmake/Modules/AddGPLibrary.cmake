# function that creates a GP library that contains ipu and cpu targets. It
# builds each source file and each target in a separate partial gp file and then
# link them all together into the final library to allow for maximal parallelism
# during build time.
#
# this function requires the following global variables to exist (all of which
# are defined in the top level CMakeLists.txt):
#   - POPLIBS_ENABLED_IPU_ARCH_NAMES
#   - POPLIBS_ENABLE_CPP_CODELETS_FOR_TARGETS
#   - POPC_EXECUTABLE
#   - POPC_FLAGS

function(add_gp_library)
  cmake_parse_arguments(CODELET "" "NAME" "ASM_SOURCES;CPP_SOURCES;HEADERS" ${ARGN})

  set(LIST_OF_TARGETS_JSON_TMP "")
  foreach(variant ${POPLIBS_ENABLED_IPU_ARCH_NAMES})
    list(APPEND LIST_OF_TARGETS_JSON_TMP "\"${variant}\"")
    if (${variant} IN_LIST POPLIBS_ENABLE_CPP_CODELETS_FOR_TARGETS)
      list(APPEND LIST_OF_TARGETS_JSON_TMP "{\"name\":\"${variant}:cpp\",\"arch\":\"${variant}\",\"compile-flags\":[\"-DPOPLIBS_DISABLE_ASM_CODELETS\",\"-DENABLE_POPLAR_RUNTIME_CHECKS\"]}")
    endif()
  endforeach()

  # Break all targets into separate JSON lists to allow build
  # them separately
  set(LIST_ALL_TARGETS_JSON "")
  foreach(TARGET IN LISTS LIST_OF_TARGETS_JSON_TMP)
    list(APPEND LIST_ALL_TARGETS_JSON "'[${TARGET}]'")
  endforeach()

  # To build all ASM sources combine POPLIBS_ENABLED_IPU_ARCH_NAMES list into
  # a string of targets. Could use JSON string here as well but for simplicity
  # use just a comma separaed string
  string(REPLACE ";" "," ASM_TARGETS "${POPLIBS_ENABLED_IPU_ARCH_NAMES}")

  # For the final step to combine all gp files it requries to JSON formated
  # string of all targets
  string(REPLACE ";" "," STRING_ALL_TARGETS_JSON_TMP "${LIST_OF_TARGETS_JSON_TMP}")
  set(STRING_ALL_TARGETS_JSON "'[${STRING_ALL_TARGETS_JSON_TMP}]'")

  set(COMMAND
    ${CMAKE_COMMAND} -E env ${POPC_ENVIRONMENT}
    ${POPC_EXECUTABLE}
    ${POPC_FLAGS}
    -DNDEBUG
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets
  )

  set(PARTIAL_OUTPUTS)

  # compile each C++ file in it's own gp file so that we don't have to rebuild
  # the entire library whenever one of those files has changed. for now we
  # add all of the headers as dependencies to all of the partial gp files. a
  # future improvement would be to only pass the required headers to each one.
  #
  # TODO: T10282 Fix dependencies with poplar's headers.
  foreach(CPP_SOURCE ${CODELET_CPP_SOURCES})
    get_filename_component(FILE ${CPP_SOURCE} NAME_WE)
    foreach(TARGET IN LISTS LIST_ALL_TARGETS_JSON)
      # Ideally we want to extract target name and add it into a filename
      # but for now use SHA1 for the file names
      string(SHA1 MAGIC_NUMBER ${TARGET})
      set(PARTIAL_GP_NAME "${CODELET_NAME}_${FILE}_${MAGIC_NUMBER}.gp")
      add_custom_command(
        OUTPUT
          ${PARTIAL_GP_NAME}
        COMMAND
          ${COMMAND}
          -o ${PARTIAL_GP_NAME}
          --target ${TARGET}
          ${CPP_SOURCE}
        DEPENDS
          ${CPP_SOURCE}
          ${CODELET_HEADERS}
          popc_bin
      )
      list(APPEND PARTIAL_OUTPUTS ${PARTIAL_GP_NAME})
    endforeach()
  endforeach()

  # compile all the assembly into a separate partial gp object.
  set(ASM_GP_NAME "${CODELET_NAME}_asm.gp")
  add_custom_command(
    OUTPUT
      ${ASM_GP_NAME}
    COMMAND
      ${COMMAND}
      -o ${ASM_GP_NAME}
      --target ${ASM_TARGETS}
      ${CODELET_ASM_SOURCES}
    DEPENDS
      ${CODELET_ASM_SOURCES}
      ${CODELET_HEADERS}
      popc_bin
  )
  list(APPEND PARTIAL_OUTPUTS ${ASM_GP_NAME})

  # compile all of the partial gp files into the actual final library objects
  set(NAME "${CODELET_NAME}.gp")
  add_custom_command(
    OUTPUT
      ${NAME}
    COMMAND
      ${COMMAND}
      -o ${NAME}
      --target ${STRING_ALL_TARGETS_JSON}
      ${PARTIAL_OUTPUTS}
    DEPENDS
      ${PARTIAL_OUTPUTS}
      popc_bin
  )
  set(OUTPUTS ${NAME})


  add_custom_target(${NAME}_codelets ALL DEPENDS ${OUTPUTS}
    SOURCES
      ${CODELET_CPP_SOURCES}
      ${CODELET_ASM_SOURCES}
      ${CODELET_HEADERS}
  )

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${NAME}
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT ${CODELET_NAME})

endfunction()
