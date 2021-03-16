find_package(PythonInterp 3)

#Controls verbose debug output on creating test cases
set(DEBUG_TEST_GENERATION FALSE)

# Function to add a multi-target test executable
function(add_multi_target_test_executable name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/tests
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poplin,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name}
    popfloat poprand popnn poputil poplar poplibs_support poplibs_test
    Boost::program_options ${CMAKE_THREAD_LIBS_INIT})
  if(TARGET popsparse)
    target_link_libraries(${name} popsparse)
  endif()
endfunction()

# Remove any requested variants not present in ${ENABLED_TEST_VARIANTS}
# If no variants were requested filtered ${DEFAULT_TEST_VARIANTS} are returned.
# Raise a fatal error if any of the variants are not in ${ALL_POSSIBLE_TEST_VARIANTS}.
function(sanitise_variants variant_name test_ipumodel_always labels)
  set(variants ${${variant_name}})
  set(old_variants "${variants}")
  if (NOT old_variants)
    set(old_variants ${DEFAULT_TEST_VARIANTS})
  endif()

  # Check that all specified variants are supported.
  foreach(variant ${old_variants})
    if (NOT ${variant} IN_LIST ALL_POSSIBLE_TEST_VARIANTS)
      message(FATAL_ERROR "Unrecognised variant: ${variant}")
    endif()
  endforeach()

  set(codelet_label "false")
  foreach(label ${labels})
    if(${label} STREQUAL "codelet")
      set(codelet_label "true")
    endif()
  endforeach()

  # If a Sim1 or Sim2 variant exists, exclude the IpuModel1 or IpuModel2 variant
  # to reduce test time. This can be overridden if required for testing
  SET(test_ipumodel "true")
  SET(test_ipumodel2 "true")
  if(${codelet_label} STREQUAL "true")
    if(NOT ${test_ipumodel_always})
      foreach(variant ${old_variants})
        if(${variant} STREQUAL "Sim1")
          SET(test_ipumodel "false")
        endif()
        if(${variant} STREQUAL "Sim2")
          SET(test_ipumodel2 "false")
        endif()
      endforeach()
    endif()
  endif()

  set(new_variants "")
  set(enabled_variants "${ENABLED_TEST_VARIANTS}")

  # Only allow each variant if it is enabled:
  # Tests labelled with VARIANT Hw1 or Hw2 are included if ENABLED_TEST_VARIANTS contains Hw
  # They are attached to a specific fixture test that checks the hardware architecture
  foreach(variant ${old_variants})
    if (${variant} MATCHES "^Hw")
      set(variant_to_match "Hw")
    else()
      set(variant_to_match ${variant})
    endif()
    set(add_variant "true")
    if(${variant} STREQUAL "IpuModel1" AND ${test_ipumodel} STREQUAL "false")
      set(add_variant "false")
    endif()
    if(${variant} STREQUAL "IpuModel2" AND ${test_ipumodel2} STREQUAL "false")
      set(add_variant "false")
    endif()
    if(${add_variant} STREQUAL "true")
      if (${variant_to_match} IN_LIST enabled_variants)
        list(APPEND new_variants ${variant})
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES new_variants)
  set(${variant_name} "${new_variants}" PARENT_SCOPE)
endfunction()

function(add_test_executable name)
  add_executable(${name} ${ARGN})
  if (DEFINED TEST_REQUIRES_TEST_TARGET)
    set_property(TARGET ${name}
                 APPEND_STRING PROPERTY COMPILE_FLAGS "-DTEST_WITH_TARGET")
  else()
    set_property(TARGET ${name}
                 APPEND_STRING PROPERTY COMPILE_FLAGS "-DTEST_WITHOUT_TARGET")
  endif()
  target_compile_options(${name} PRIVATE -Wno-deprecated-declarations)
  target_include_directories(${name}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/tests
      ${CMAKE_SOURCE_DIR}/lib
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poplin,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} popfloat poprand popnn poputil
    poplibs_support poplibs_test poplar Boost::unit_test_framework
    Boost::timer Boost::system Boost::program_options
    ${CMAKE_THREAD_LIBS_INIT})
  if(TARGET popsparse)
    target_link_libraries(${name} popsparse)
  endif()
  if (NOT Boost_UNIT_TEST_FRAMEWORK_LIBRARY MATCHES "\\.a$")
    target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  endif()
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts the "target" part storing it in the second argument.
function(extract_target variant target)
  string(REPLACE ":" ";" vspec ${${variant}})
  list(GET vspec 0 HEAD)
  set(${target} ${HEAD} PARENT_SCOPE)
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts the "runconfig" part storing it in the second argument.
function(extract_runconfig variant run_config)
  string(REPLACE ":" ";" vspec ${${variant}})
  list(LENGTH vspec vlen)
  if (${vlen} EQUAL "1")
    set(${run_config} "default" PARENT_SCOPE)
  else()
    list(GET vspec 1 config)
    set(${run_config} ${config} PARENT_SCOPE)
  endif()
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts a list containing just the "target" parts. Where targets are listed
# with multiple runconfigs in the original list the duplicates are removed
# from the returned list.
function(extract_targets variants targets)
  set(extracted_targets "")
  foreach(variant ${${variants}})
    string(REPLACE ":" ";" vspec ${variant})
    list(GET vspec 0 HEAD)
    list(APPEND extracted_targets ${HEAD})
  endforeach()
  list(REMOVE_DUPLICATES extracted_targets)
  set(${targets} ${extracted_targets} PARENT_SCOPE)
endfunction()

# Append the parent directories as labels, e.g tests/popops/codelet/StdOpTest.cpp
# gets two additional labels of popops and codelet. However if a test is not in a
# sub-directory then don't add the label, e.g. tests/Foo.cpp does not get a label.
# This function also appends the concatenation of all the labels.
function (add_parent_dir_labels labels_name)

  # Build the list of labels from the directories in `CMAKE_CURRENT_SOURCE_DIR`,
  # e.g: tests/popops/codelet/StdOpTest.cpp --> codelet;popops
  set(new_labels "")
  set(directory ${CMAKE_CURRENT_SOURCE_DIR})
  get_filename_component(parent_dir ${directory} NAME)

  while (NOT ${parent_dir} STREQUAL tests)

    list(APPEND new_labels ${parent_dir})

    get_filename_component(directory ${directory} DIRECTORY)
    get_filename_component(parent_dir ${directory} NAME)

  endwhile()

  # Reorder the labels so they match the order in the path,
  # e.g: codelet;popops --> popops;codelet
  list(REVERSE new_labels)

  # Append the new labels to the existing labels,
  # e.g: FooLabel --> FooLabel;popops;codelet
  set(labels ${${labels_name}})
  list(APPEND labels ${new_labels})

  # Append all the labels concatenated together so that it's possible to filter
  # on the logical AND of all the labels. E.g. `-L popops&codelets` to just run
  # the codelets in popops, and not everything labelled by popops or codelets,
  # e.g: FooLabel;popops;codelet --> FooLabel;popops;codelet;popops&codelet
  list(LENGTH new_labels new_labels_length)
  if (${new_labels_length} GREATER 1)
    string(REPLACE ";" "&" new_labels "${new_labels}")
    list(APPEND labels ${new_labels})
  endif()

  set(${labels_name} "${labels}" PARENT_SCOPE)
endfunction()

# Add a test to run with the specified VARIANTS (or all enabled VARIANTS if
# none are specified). The first parameter is the test name, the following are
# source files, and finally everything after VARIANTS are variant
# specifications.
#
# A variation specification must be of the form "target:[config]"
#
# Currently supported targets are "Cpu;IpuModel1;IpuModel2;Sim1;Sim2;Hw;Hw1;Hw2"
# Hw variants are attached to specific fixture tests that check the availability of hardware
# Hw variant works with any architecture version, while Hw1 and Hw2 require particular ones
# Note that ENABLED_TEST_VARIANTS (AND DEFAULT_TEST_VARIANTS) do not specify particular hardware versions
# Hw1 and Hw2 variants are enabled if ENABLED_TEST_VARIANTS contains Hw
# and will only execute if the present hardware complies with the requested architecture version
# Note that the Cpu target does not support multi-tile tests
# Currently supported configs are "cpp;default" which specify the codelet types
function(add_unit_test name)
  cmake_parse_arguments(add_unit_test "" "FILES" "VARIANTS;SUITES;LABELS;TEST_IPUMODEL;" "FILES;${ARGN}")
  sanitise_variants(add_unit_test_VARIANTS add_unit_test_TEST_IPUMODEL add_unit_test_LABELS)
  if (NOT add_unit_test_VARIANTS)
    # When all variants are disabled don't even generate the test exe as
    # we don't know whether it needs a target
    return()
  endif()
  set(FILES "${add_unit_test_FILES}")
  extract_targets(add_unit_test_VARIANTS TARGETS)

  if(DEBUG_TEST_GENERATION)
    message(STATUS
            "Adding test '${name}' with variants: ${add_unit_test_VARIANTS}")
  endif()
  if (NOT TARGETS STREQUAL "NoTarget")
    set(TEST_REQUIRES_TEST_TARGET TRUE)
  endif()
  add_test_executable(${name} ${FILES})

  add_parent_dir_labels(add_unit_test_LABELS)
  list(APPEND add_unit_test_LABELS unit_test)

  foreach(VARIANT ${add_unit_test_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    if (${TEST_TARGET} STREQUAL "Hw1" AND NOT DEFINED hasHw1Test)
        set (hasHw1Test true PARENT_SCOPE)
        add_ipu_hw_test(1)
    endif()
    if (${TEST_TARGET} STREQUAL "Hw2" AND NOT DEFINED hasHw2Test)
        set (hasHw2Test true PARENT_SCOPE)
        add_ipu_hw_test(2)
    endif()

    macro(add_unit_test_impl)
      set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")

      set(test_suite "")
      if (SUITE)
        string(APPEND test_name "_${SUITE}")
        string(APPEND test_suite "--run_test=${SUITE}/*")
      endif()

      if (TEST_TARET STREQUAL "NoTarget")
      add_test(NAME "${test_name}"
        COMMAND ${name} ${test_suite}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
      else()
        add_test(NAME "${test_name}"
          COMMAND ${name} ${test_suite} -- --device-type ${TEST_TARGET}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
      endif()
      if (add_unit_test_LABELS)
        set_tests_properties(${test_name}
            PROPERTIES LABELS "${add_unit_test_LABELS}")
      endif()

      set(test_env ${TEST_ENVIRONMENT})
      if (${TEST_CONFIG} STREQUAL "cpp")
        list(APPEND test_env "POPLIBS_CODELET_SUFFIX=_c")
      endif()
      set_tests_properties(${test_name} PROPERTIES
                           ENVIRONMENT "${test_env}")

      # Make sure tests that use physical IPUs only run if an appropriate
      # number were available according to the relevant test fixture:
      if (${TEST_TARGET} STREQUAL "Hw1")
        set_tests_properties(${test_name}
          PROPERTIES FIXTURES_REQUIRED Ipu1IsAvailable)
      elseif (${TEST_TARGET} STREQUAL "Hw2")
        set_tests_properties(${test_name}
          PROPERTIES FIXTURES_REQUIRED Ipu2IsAvailable)
      elseif (${TEST_TARGET} STREQUAL "Hw")
        set_tests_properties(${test_name}
          PROPERTIES FIXTURES_REQUIRED IpuIsAvailable)
      endif()
    endmacro()

    if (add_unit_test_SUITES)
      foreach(SUITE ${add_unit_test_SUITES})
        add_unit_test_impl()
      endforeach()
    else()
      add_unit_test_impl()
    endif()
  endforeach()
endfunction()

# add a test to run on the specified VARIANTS
function(add_multitarget_test)
  cmake_parse_arguments(add_multitarget_test "ALLOW_SKIP" "NAME"
                        "VARIANTS;COMMAND;LABELS;TEST_IPUMODEL;"
                        "${ARGN}")
  sanitise_variants(add_multitarget_test_VARIANTS add_multitarget_test_TEST_IPUMODEL
                    add_multitarget_test_LABELS)
  set(name ${add_multitarget_test_NAME})
  if (NOT add_multitarget_test_NAME)
    message(FATAL_ERROR "add_multitarget_test requires a NAME to be specified")
  endif()

  if(DEBUG_TEST_GENERATION)
    message(STATUS
      "Adding multi target test '${name}' with "
      "variants: ${add_multitarget_test_VARIANTS}")
  endif()

  add_parent_dir_labels(add_multitarget_test_LABELS)

  # Also append the name of the tool to the labels.
  list(GET add_multitarget_test_COMMAND 0 tool)
  get_filename_component(tool ${tool} NAME)
  list(APPEND add_multitarget_test_LABELS ${tool})

  foreach(VARIANT ${add_multitarget_test_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")
    add_test(NAME "${test_name}"
      COMMAND ${add_multitarget_test_COMMAND} "--device-type=${TEST_TARGET}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
    if (add_multitarget_test_ALLOW_SKIP)
      set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    set(test_env ${TEST_ENVIRONMENT})
    if (${TEST_CONFIG} STREQUAL "cpp")
      list(APPEND test_env "POPLIBS_CODELET_SUFFIX=_c")
    endif()

    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "${test_env}")

    if (${TEST_TARGET} STREQUAL "Hw")
      # Make sure tests that use physical IPUs only run if an appropriate
      # number were available according to the relevant test fixture:
      set_tests_properties(${test_name}
        PROPERTIES FIXTURES_REQUIRED IpuIsAvailable)
    endif()

    if (add_multitarget_test_LABELS)
      set_tests_properties(${test_name}
        PROPERTIES LABELS "${add_multitarget_test_LABELS}")
    endif()
  endforeach()
endfunction()

# Make an alias for IpuModel that can be used to denote tests that specifically
# should not be run on Sim variants because the test times out if run on the
# simulator:
set(TimesOutOnSim "${IPUMODEL_VARIANTS};Hw")

# Make an alias for IpuModel that can be used to denote tests that specifically
# should not be run on the Sim:cpp variant due to test timeouts:
set(TimesOutOnSimCpp "${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw")

# Tests
add_subdirectory(popfloat)
add_subdirectory(poplibs_support)
add_subdirectory(poplin)
add_subdirectory(popnn)
add_subdirectory(popops)
add_subdirectory(poprand)
add_subdirectory(popsolver)

if(TARGET popsparse)
  add_subdirectory(popsparse)
endif()
add_subdirectory(poputil)
add_subdirectory(sanity)
