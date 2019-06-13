# Add a lit test. This lacks specialisations for device type because tests for
# different targets can be easily specified inline in the same test with
# FileCheck. This function also passes along any paths or variables to the test
# to let it build codelets.
function(add_lit_test name)
  cmake_parse_arguments(test_args "" "FILE" "" "FILE;${ARGN}")
  set(test_file "${CMAKE_CURRENT_SOURCE_DIR}/${test_args_FILE}")

  message(STATUS "Adding lit test '${name}'")
  add_test(NAME "${name}"
           COMMAND gc-lit "${test_file}"
           "--param" "test_dir=${CMAKE_CURRENT_BINARY_DIR}" "-vv")
  set_tests_properties(${name} PROPERTIES
    FAIL_REGULAR_EXPRESSION "warning: input '.*' contained no tests")
endfunction()

add_lit_test(ApiCppVersionTest ApiCppVersionTest.cpp)
