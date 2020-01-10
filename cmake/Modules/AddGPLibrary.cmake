# function that creates a GP library that contains ipu and cpu targets. It
# builds each source file and each target in a separate partial gp file and then
# link them all together into the final library to allow for maximal parallelism
# during build time.
#
# this function requires the following global variables to exist (all of which
# are defined in the top level CMakeLists.txt):
#   - DEFAULT_TEST_VARIANTS
#   - POPC_EXECUTABLE
#   - POPC_FLAGS
function(add_gp_library)
  cmake_parse_arguments(CODELET "" "NAME" "ASM_SOURCES;CPP_SOURCES;HEADERS" ${ARGN})
  set(IPU_TARGETS ${ENABLED_IPU_ARCH_NAMES})
  set(IPU_TARGETS_COMMA_SEPARATED ${ENABLED_IPU_ARCH_NAMES_COMMA_SEPARATED})

  # we don't build the _c.gp files if we are not planning to run any of the
  # {Sim,Hw,*}:cpp tests. for the time being poplibs does not have any tests that
  # expliticly force this variant and therefore we can do this. in the future if
  # that changes we will have to build the _c libraries that are used by any of
  # those tests unconditionally. the reason we do this is to improve the
  # compile time for debug builds.
  foreach(TEST_VARIANT IN LISTS DEFAULT_TEST_VARIANTS)
    if (TEST_VARIANT MATCHES ".*:cpp")
      set(BUILD_CPP_CODELETS TRUE)
    endif()
  endforeach()

  set(COMMAND
    ${CMAKE_COMMAND} -E env ${POPC_ENVIRONMENT}
    ${POPC_EXECUTABLE}
    ${POPC_FLAGS}
    -DNDEBUG
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets
  )

  set(PARTIAL_OUTPUTS)
  set(CPP_PARTIAL_OUTPUTS)

  # compile each C++ file in it's own gp file so that we don't have to rebuild
  # the entire library whenever one of those files has changed. for now we
  # add all of the headers as dependencies to all of the partial gp files. a
  # future improvement would be to only pass the required headers to each one.
  #
  # TODO: T10282 Fix dependencies with poplar's headers.
  foreach(CPP_SOURCE ${CODELET_CPP_SOURCES})
    get_filename_component(FILE ${CPP_SOURCE} NAME_WE)

    # build each target in parallel and link together at the end.
    foreach(TARGET cpu ${IPU_TARGETS})
      set(PARTIAL_GP_NAME "${CODELET_NAME}_${FILE}_${TARGET}.gp")
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

      if(BUILD_CPP_CODELETS)
        set(CPP_PARTIAL_GP_NAME "${CODELET_NAME}_${FILE}_${TARGET}_c.gp")
        add_custom_command(
          OUTPUT ${CPP_PARTIAL_GP_NAME}
          COMMAND
            ${COMMAND}
            -o ${CPP_PARTIAL_GP_NAME}
            --target ${TARGET}
            ${CPP_SOURCE}
            -DPOPLIBS_DISABLE_ASM_CODELETS
            -DENABLE_POPLAR_RUNTIME_CHECKS
          DEPENDS
            ${CPP_SOURCE}
            ${CODELET_HEADERS}
            popc_bin
        )
        list(APPEND CPP_PARTIAL_OUTPUTS ${CPP_PARTIAL_GP_NAME})
      endif()
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
      --target ${IPU_TARGETS_COMMA_SEPARATED}
      ${CODELET_ASM_SOURCES}
    DEPENDS
      ${CODELET_ASM_SOURCES}
      ${CODELET_HEADERS}
      popc_bin
  )
  list(APPEND PARTIAL_OUTPUTS ${ASM_GP_NAME})

  # compile all of the partial gp files into the actual final library objects
  set(NAME "${CODELET_NAME}.gp")
  set(CPP_NAME "${CODELET_NAME}_c.gp")

  add_custom_command(
    OUTPUT
      ${NAME}
    COMMAND
      ${COMMAND}
      -o ${NAME}
      --target cpu,${IPU_TARGETS_COMMA_SEPARATED}
      ${PARTIAL_OUTPUTS}
    DEPENDS
      ${PARTIAL_OUTPUTS}
      popc_bin
  )
  set(OUTPUTS ${NAME})

  if(BUILD_CPP_CODELETS)
    add_custom_command(
      OUTPUT
        ${CPP_NAME}
      COMMAND
        ${COMMAND}
        -o ${CPP_NAME}
        --target cpu,${IPU_TARGETS_COMMA_SEPARATED}
        ${CPP_PARTIAL_OUTPUTS}
      DEPENDS
        ${CPP_PARTIAL_OUTPUTS}
        popc_bin
    )
    list(APPEND OUTPUTS ${CPP_NAME})
  endif()

  add_custom_target(${NAME}_codelets ALL DEPENDS ${OUTPUTS}
    SOURCES
      ${CODELET_CPP_SOURCES}
      ${CODELET_ASM_SOURCES}
      ${CODELET_HEADERS}
  )

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${NAME}
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT ${CODELET_NAME})

  if(BUILD_CPP_CODELETS)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CPP_NAME}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${CODELET_NAME})
  endif()
endfunction()
