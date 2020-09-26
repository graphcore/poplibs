find_package(PythonInterp 3)

include(GNUInstallDirs)

function(add_tool name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} poprand popnn poplin popops popfloat poputil)
  if(TARGET popsparse)
    target_link_libraries(${name} popsparse)
  endif()
endfunction()

add_tool(single_conv_layer single_conv_layer.cpp)
target_link_libraries(single_conv_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(multi_conv_layer multi_conv_layer.cpp)
target_link_libraries(multi_conv_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(oct_conv_layer oct_conv_layer.cpp)
target_link_libraries(oct_conv_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(fully_connected_layer fully_connected_layer.cpp)
target_link_libraries(fully_connected_layer
                      poplibs_support poplibs_test
                      Boost::program_options)


add_tool(lstm_layer lstm_layer.cpp)
target_link_libraries(lstm_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(gru_layer gru_layer.cpp)
target_link_libraries(gru_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

if(TARGET popsparse)
  add_tool(bs_matmul_test bs_matmul_test.cpp)
  target_link_libraries(bs_matmul_test
    poplibs_support poplibs_test
    Boost::program_options)
endif()

add_tool(pooling_layer pooling_layer.cpp)
target_link_libraries(pooling_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(general_matrix_multiply general_matrix_multiply.cpp)
target_link_libraries(general_matrix_multiply
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(cast_to_gfloat cast_to_gfloat.cpp)
target_link_libraries(cast_to_gfloat
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(cast_to_gfloat_sr cast_to_gfloat_sr.cpp)
target_link_libraries(cast_to_gfloat_sr
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(rnn_layer rnn_layer.cpp)
target_link_libraries(rnn_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(reduce_op reduce_op.cpp)
target_link_libraries(reduce_op
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(collectives collectives.cpp)
target_link_libraries(collectives
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(replicated_collectives replicated_collectives.cpp)
target_link_libraries(replicated_collectives
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(random_generator random_generator.cpp)
target_link_libraries(random_generator
                     poplibs_support poplibs_test
                     Boost::program_options)

add_tool(embedding_layer embedding_layer.cpp)
target_link_libraries(embedding_layer
                      poplibs_support
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(norm_layer norm_layer.cpp)
target_link_libraries(norm_layer
                      poplibs_support poplibs_test
                      Boost::program_options)


if (TARGET popsparse)
  add_tool(sparse_fc_layer sparse_fc_layer.cpp)
  target_link_libraries(sparse_fc_layer
                        poplibs_support
                        poplibs_support poplibs_test
                        Boost::program_options)

  add_tool(sparse_matmul sparse_matmul.cpp)
  target_link_libraries(sparse_matmul
                        poplibs_support
                        poplibs_test
                        Boost::program_options)

  add_tool(sparse_embedding_tied_matmul sparse_embedding_tied_matmul.cpp)
  target_link_libraries(sparse_embedding_tied_matmul
                        poplibs_support
                        poplibs_test
                        Boost::program_options)
endif()


# TODO: Shall be reused from poplar/CMakeLists.
# Remove any requested variants not present in ${ENABLED_TEST_VARIANTS}
# If no variants were requested filtered ${DEFAULT_TEST_VARIANTS} are returned.
function(sanitise_variants variant_name)
  set(variants ${${variant_name}})
  set(old_variants "${variants}")
  if (NOT old_variants)
    set(old_variants ${DEFAULT_TEST_VARIANTS})
  endif()
  set(new_variants "")
  # Only allow each variant if it is enabled:
  foreach(variant ${old_variants})
    if (${variant} IN_LIST ENABLED_TEST_VARIANTS)
      list(APPEND new_variants ${variant})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES new_variants)
  set(${variant_name} "${new_variants}" PARENT_SCOPE)
endfunction()

function(add_python_tool_test)
  cmake_parse_arguments(add_python_tool_test "" "NAME"
                        "COMMAND;VARIANTS"
                        "${ARGN}")

  sanitise_variants(add_python_tool_test_VARIANTS DEFAULT_TEST_VARIANTS ENABLED_TEST_VARIANTS)

  if (add_python_tool_test_VARIANTS)
    if (${add_python_tool_test_VARIANTS} STREQUAL "Hw" AND NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
      add_test(NAME ${add_python_tool_test_NAME}
        COMMAND ${add_python_tool_test_COMMAND}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})

      # Make sure tests that use physical IPUs only run if an appropriate
      # number were available according to the relevant test fixture:
      set_tests_properties(${add_python_tool_test_NAME}
        PROPERTIES FIXTURES_REQUIRED IpuIsAvailable)

      set_tests_properties(${add_python_tool_test_NAME} PROPERTIES LABELS "python_tool")
    endif()
  endif()

endfunction()

add_python_tool_test(
    NAME transformsTool
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tools/transforms.py
            --ci-test
            --remove-files
            --test-binary $<TARGET_FILE:single_conv_layer>
    VARIANTS "Hw")
