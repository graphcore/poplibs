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
function(sanitise_variants variant_name)
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

  set(new_variants "")
  # Only allow each variant if it is enabled:
  # Tests labelled with VARIANT Hw1 or Hw2 are included if ENABLED_TEST_VARIANTS contains Hw
  # They are attached to a specific fixture test that checks the hardware architecture
  foreach(variant ${old_variants})
    if (${variant} MATCHES "^Hw")
      set(variant_to_match "Hw")
    else()
      set(variant_to_match ${variant})
    endif()
    if (${variant_to_match} IN_LIST ENABLED_TEST_VARIANTS)
      list(APPEND new_variants ${variant})
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
# Currently supported targets are "Cpu;IpuModel;IpuModel2;Sim;Sim2;Hw;Hw1;Hw2"
# Hw variants are attached to specific fixture tests that check the availability of hardware
# Hw variant works with any architecture version, while Hw1 and Hw2 require particular ones
# Note that ENABLED_TEST_VARIANTS (AND DEFAULT_TEST_VARIANTS) do not specify particular hardware versions
# Hw1 and Hw2 variants are enabled if ENABLED_TEST_VARIANTS contains Hw
# and will only execute if the present hardware complies with the requested architecture version
# Note that the Cpu target does not support multi-tile tests
# Currently supported configs are "cpp;default" which specify the codelet types
function(add_unit_test name)
  cmake_parse_arguments(add_unit_test "" "FILES" "VARIANTS;SUITES;LABELS" "FILES;${ARGN}")
  sanitise_variants(add_unit_test_VARIANTS)
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
                        "VARIANTS;COMMAND;LABELS"
                        "${ARGN}")
  sanitise_variants(add_multitarget_test_VARIANTS)
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

function(add_benchmark)
  cmake_parse_arguments(BENCHMARK "" "NAME;BINARY;PARALLEL_LEVEL" "PARAMS" "${ARGN}")

  if (NOT DEFINED BENCHMARK_PARALLEL_LEVEL)
    set(BENCHMARK_LABEL "benchmarks")
  elseif(BENCHMARK_PARALLEL_LEVEL EQUAL 4)
    set(BENCHMARK_LABEL "benchmarks_j4")
  elseif(BENCHMARK_PARALLEL_LEVEL EQUAL 8)
    set(BENCHMARK_LABEL "benchmarks_j8")
  else()
    message(FATAL_ERROR "Cannot set a parallel level that is not 4 or 8 as "
                        "buildbot is currently only configured to split the "
                        "tasks at that granularity.")
  endif()

  # common args used by all the benchmarks
  list(APPEND BENCHMARK_PARAMS --ignore-data --use-unstable-format)

  add_multitarget_test(
    NAME "${BENCHMARK_NAME}_benchmark"
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tools/bench.py
            --name ${BENCHMARK_NAME}
            --config default
            --expected_csv ${CMAKE_SOURCE_DIR}/tests/benchmark_results.csv
            ${BENCHMARK_BINARY} ${BENCHMARK_PARAMS}
    LABELS ${BENCHMARK_LABEL}
    VARIANTS "${IPUMODEL_VARIANTS}")
endfunction()

function(add_conv_benchmark)
  cmake_parse_arguments(CONV_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}" )

  # common arguments for all conv benchmarks
  list(APPEND CONV_BENCHMARK_PARAMS --bias 0)
  add_benchmark(
      NAME ${CONV_BENCHMARK_NAME}
      PARALLEL_LEVEL ${CONV_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:single_conv_layer>
      PARAMS ${CONV_BENCHMARK_PARAMS})
endfunction()

function(add_oct_conv_benchmark)
  cmake_parse_arguments(OCT_CONV_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}" )

  # common arguments for all oct conv benchmarks
  list(APPEND OCT_CONV_BENCHMARK_PARAMS --pool-optimize-for-speed=true)
  add_benchmark(
      NAME ${OCT_CONV_BENCHMARK_NAME}
      PARALLEL_LEVEL ${OCT_CONV_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:oct_conv_layer>
      PARAMS ${OCT_CONV_BENCHMARK_PARAMS})
endfunction()

function(add_fc_benchmark)
  cmake_parse_arguments(FC_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}")

  # common arguments for all fc benchmarks
  list(APPEND FC_BENCHMARK_PARAMS --bias 0)
  add_benchmark(
      NAME ${FC_BENCHMARK_NAME}
      PARALLEL_LEVEL ${FC_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:fully_connected_layer>
      PARAMS ${FC_BENCHMARK_PARAMS})
endfunction()

function(add_embedding_benchmark)
  cmake_parse_arguments(EMBED_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}")

  add_benchmark(
      NAME ${EMBED_BENCHMARK_NAME}
      PARALLEL_LEVEL ${EMBED_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:embedding_layer>
      PARAMS ${EMBED_BENCHMARK_PARAMS})
endfunction()

function(add_gemm_benchmark)
  cmake_parse_arguments(GEMM_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}")

  add_benchmark(
      NAME ${GEMM_BENCHMARK_NAME}
      PARALLEL_LEVEL ${GEMM_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:general_matrix_multiply>
      PARAMS ${GEMM_BENCHMARK_PARAMS})
endfunction()

function(add_reduction_benchmark)
  cmake_parse_arguments(REDUCTION_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}" )

  add_benchmark(
      NAME ${REDUCTION_BENCHMARK_NAME}
      PARALLEL_LEVEL ${REDUCTION_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:reduce_op>
      PARAMS ${REDUCTION_BENCHMARK_PARAMS})
endfunction()

function(add_pooling_benchmark)
  cmake_parse_arguments(POOLING_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}" )

  add_benchmark(
      NAME ${POOLING_BENCHMARK_NAME}
      PARALLEL_LEVEL ${POOLING_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:pooling_layer>
      PARAMS ${POOLING_BENCHMARK_PARAMS})
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

# Benchmarks
if (PYTHONINTERP_FOUND AND PYTHON_VERSION_MAJOR EQUAL 3)

  # Some of the following benchmarks use quite a lot of host memory, too much
  # in fact for a buildslave (which currently have 32GB RAM and 16 cores) to
  # parallelise across the number of cores for all of these benchmarks. In this
  # case an optional PARALLEL_LEVEL argument is added that can specify that this
  # test must only be run with a -j flag of either 2, 4 or 8. If this option
  # does not exist then the buildbot will run the benchmark at a parallel
  # level of `nproc`.

  # These numbers must match those of bench.py
  set(GEMM_BENCHMARK_MS 200 600 1000)
  set(GEMM_BENCHMARK_KS 64 256 512)
  set(GEMM_BENCHMARK_NS 10000 20000 30000)

  # Matrix multiplication benchmarks with regression testing.
  foreach(m ${GEMM_BENCHMARK_MS})
    foreach(k ${GEMM_BENCHMARK_KS})
      foreach(n ${GEMM_BENCHMARK_NS})
        set(NAME "gemm_${m}x${k}x${n}")

        unset(GEMM_EXTRA_ARGS)
        if(NAME STREQUAL "gemm_1000x256x30000" OR
           NAME STREQUAL "gemm_1000x512x30000")
          set(GEMM_EXTRA_ARGS PARALLEL_LEVEL 8)
        endif()

        add_gemm_benchmark(NAME ${NAME} PARAMS --m ${m} --k ${k} --n ${n} ${GEMM_EXTRA_ARGS})
      endforeach()
    endforeach()
  endforeach()

  # resnet50 conv layer benchmarks (batch size 4). each name is the enigma name
  # the first time that layer is seen.
  add_conv_benchmark(NAME resnet50_tr_bs4_cnv
          PARAMS --field {224,224}
                 --kernel-size 7
                 --stride 2
                 --padding 3
                 --input-channels 4
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm64L0_projection
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm64L0
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm64L0A1
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 64
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm64L1A0
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L0_projection
          PARAMS --field {56,56}
                 --kernel-size 1
                 --stride 2
                 --input-channels 256
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L0A0
          PARAMS --field {56,56}
                 --kernel-size 1
                 --stride 2
                 --input-channels 256
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L0A1
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L0A2
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 128
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L1A0
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L0_projection
          PARAMS --field {28,28}
                 --kernel-size 1
                 --stride 2
                 --input-channels 512
                 --output-channels 1024
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L0A0
          PARAMS --field {28,28}
                 --kernel-size 1
                 --stride 2
                 --input-channels 512
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L0A1
          PARAMS --field {14,14}
                 --kernel-size 3
                 --padding 1
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L0A2
          PARAMS --field {14,14}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 1024
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L1A0
          PARAMS --field {14,14}
                 --kernel-size 1
                 --input-channels 1024
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L0_projection
          PARAMS --field {14,14}
                 --kernel-size 1
                 --stride 2
                 --input-channels 1024
                 --output-channels 2048
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L0A0
          PARAMS --field {14,14}
                 --kernel-size 1
                 --stride 2
                 --input-channels 1024
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L0A1
          PARAMS --field {7,7}
                 --kernel-size 3
                 --padding 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L0A2
          PARAMS --field {7,7}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 2048
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L1A0
          PARAMS --field {7,7}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 512
                 --batch-size 4)

  # resnet50 octave conv layer (alpha = 0.25) benchmarks (batch size 2). each
  # name is the enigma name the first time that layer is seen.
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_cnv_ain0_aout25
          PARAMS --field {224,224}
                 --kernel-size 7
                 --stride 2
                 --padding 3
                 --input-channels 4
                 --output-channels 64
                 --batch-size 4
                 --alpha-in 0
                 --alpha-out 0.25
          PARALLEL_LEVEL 8)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm64L0_projection_a25
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 256
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm64L0_a25
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 64
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm64L0A1_a25
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 64
                 --output-channels 64
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm64L1A0_a25
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 64
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm128L0_projection_a25
          PARAMS --field {56,56}
                 --kernel-size 1
                 --stride 2
                 --input-channels 256
                 --output-channels 512
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm128L0A0_a25
          PARAMS --field {56,56}
                 --kernel-size 1
                 --stride 2
                 --input-channels 256
                 --output-channels 128
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm128L0A1_a25
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm128L0A2_a25
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 128
                 --output-channels 512
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm128L1A0_a25
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 128
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm256L0_projection_a25
          PARAMS --field {28,28}
                 --kernel-size 1
                 --stride 2
                 --input-channels 512
                 --output-channels 1024
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm256L0A0_a25
          PARAMS --field {28,28}
                 --kernel-size 1
                 --stride 2
                 --input-channels 512
                 --output-channels 256
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm256L0A1_a25
          PARAMS --field {16,16}
                 --kernel-size 3
                 --padding 1
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm256L0A2_a25
          PARAMS --field {16,16}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 1024
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm256L1A0_a25
          PARAMS --field {16,16}
                 --kernel-size 1
                 --input-channels 1024
                 --output-channels 256
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm512L0_projection_a25
          PARAMS --field {16,16}
                 --kernel-size 1
                 --stride 2
                 --input-channels 1024
                 --output-channels 2048
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm512L0A0_a25
          PARAMS --field {16,16}
                 --kernel-size 1
                 --stride 2
                 --input-channels 1024
                 --output-channels 512
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm512L0A1_a25
          PARAMS --field {8,8}
                 --kernel-size 3
                 --padding 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm512L0A2_a25
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 2048
                 --batch-size 4
                 --alpha 0.25)
  add_oct_conv_benchmark(NAME oct_resnet50_tr_bs4_bm512L1A0_a25
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 512
                 --batch-size 4
                 --alpha 0.25)

  # inception conv layer benchmarks (batch size 4). each name is the enigma name
  # the first time that layer is seen.
  add_conv_benchmark(NAME inception_tr_bs4_layer1
          PARAMS --field {299,299}
                 --kernel-size 3
                 --stride 2
                 --input-channels 4
                 --output-channels 32
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME inception_tr_bs4_layer3
          PARAMS --field {149,149}
                 --kernel-size 3
                 --input-channels 32
                 --output-channels 32
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME inception_tr_bs4_layer5
          PARAMS --field {147,147}
                 --kernel-size 3
                 --padding 1
                 --input-channels 32
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME inception_tr_bs4_layer8
          PARAMS --field {147,147}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 80
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME inception_tr_bs4_layer10
          PARAMS --field {73,73}
                 --kernel-size 3
                 --input-channels 80
                 --output-channels 192
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME inception_tr_bs4_i1_a1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 192
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_b1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 192
                 --output-channels 48
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_b5x5
          PARAMS --field {35,35}
                 --kernel-size 5
                 --padding 2
                 --input-channels 48
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_db1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 192
                 --output-channels 32
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_c3x3a
          PARAMS --field {35,35}
                 --kernel-size 3
                 --padding 1
                 --input-channels 64
                 --output-channels 96
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_c3x3b
          PARAMS --field {35,35}
                 --kernel-size 3
                 --padding 1
                 --input-channels 96
                 --output-channels 96
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i2_a1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i2_b1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 48
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i3_a1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 288
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i3_b1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 288
                 --output-channels 48
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i4_a3x3
          PARAMS --field {35,35}
                 --kernel-size 3
                 --stride 2
                 --input-channels 288
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i4_b3x3b
          PARAMS --field {35,35}
                 --kernel-size 3
                 --stride 2
                 --input-channels 96
                 --output-channels 96
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_a1x1
          PARAMS --field {17,17}
                 --kernel-size 1
                 --input-channels 768
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_b1x1
          PARAMS --field {17,17}
                 --kernel-size 1
                 --input-channels 768
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_b1x7
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_c7x1a
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_b7x1
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 128
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_c1x7b
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 128
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_b1x1
          PARAMS --field {17,17}
                 --kernel-size 1
                 --input-channels 768
                 --output-channels 160
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_b1x7
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 160
                 --output-channels 160
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_c7x1a
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 128
                 --output-channels 160
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_b7x1
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 160
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_c7x1b
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 160
                 --output-channels 160
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_c1x7b
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 160
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i7_b1x7
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 192
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i7_b7x1
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 192
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i8_a3x3
          PARAMS --field {17,17}
                 --kernel-size 3
                 --stride 2
                 --input-channels 192
                 --output-channels 320
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i8_b3x3
          PARAMS --field {17,17}
                 --kernel-size 3
                 --stride 2
                 --input-channels 192
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_a1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 1280
                 --output-channels 320
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_c1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 1280
                 --output-channels 448
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_ba1x3
          PARAMS --field {8,8}
                 --kernel-size {1,3}
                 --padding {0,1}
                 --input-channels 384
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_bb3x1
          PARAMS --field {8,8}
                 --kernel-size {3,1}
                 --padding {1,0}
                 --input-channels 384
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_d1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 1280
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_c3x3
          PARAMS --field {8,8}
                 --kernel-size 3
                 --padding 1
                 --input-channels 448
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i10_a1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 320
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i10_b1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i10_c1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 448
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i10_d1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 192
                 --batch-size 4)

  # vgg16 conv layer benchmarks (batch size 4). each name is the enigma name
  # the first time that layer is seen.
  add_conv_benchmark(NAME vgg16_tr_bs4_v1L0
          PARAMS --field {224,224}
                 --kernel-size 3
                 --padding 1
                 --input-channels 4
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 4)
# TODO: T17350 re-enable this after investigation
#  add_conv_benchmark(NAME vgg16_tr_bs4_v1L1
#          PARAMS --field {224,224}
#                 --kernel-size 3
#                 --padding 1
#                 --input-channels 64
#                 --output-channels 64
#                 --batch-size 4
#          PARALLEL_LEVEL 2)
  add_conv_benchmark(NAME vgg16_tr_bs4_v2L0
          PARAMS --field {112,112}
                 --kernel-size 3
                 --padding 1
                 --input-channels 64
                 --output-channels 128
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME vgg16_tr_bs4_v2L1
          PARAMS --field {112,112}
                 --kernel-size 3
                 --padding 1
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME vgg16_tr_bs4_v3L0
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 128
                 --output-channels 256
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME vgg16_tr_bs4_v3L1
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME vgg16_tr_bs4_v4L0
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 256
                 --output-channels 512
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME vgg16_tr_bs4_v4L1
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME vgg16_tr_bs4_v5L0
          PARAMS --field {14,14}
                 --kernel-size 3
                 --padding 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4
          PARALLEL_LEVEL 8)

  # alexnet conv layer benchmarks (batch size 4). each name is the enigma name
  # the first time that layer is seen.
  add_conv_benchmark(NAME alexnet_tr_bs4_layer1
          PARAMS --field {224,224}
                 --kernel-size 11
                 --padding 3
                 --stride 4
                 --input-channels 4
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME alexnet_tr_bs4_layer4
          PARAMS --field {27,27}
                 --kernel-size 5
                 --padding 2
                 --input-channels 64
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME alexnet_tr_bs4_layer7
          PARAMS --field {13,13}
                 --kernel-size 3
                 --input-channels 192
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME alexnet_tr_bs4_layer9
          PARAMS --field {13,13}
                 --kernel-size 3
                 --input-channels 384
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME alexnet_tr_bs4_layer11
          PARAMS --field {13,13}
                 --kernel-size 3
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4)

  # mobilenet conv layer benchmarks (batch size 4). each name is derived from
  # the first time that layer is seen in the public_examples tensorflow model.
  add_conv_benchmark(NAME mobilenet_conv1_1
          PARAMS --field {224,224}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 4
                 --output-channels 32
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise
          PARAMS --field {112,112}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 32
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_conv_pw_1_1
          PARAMS --field {112,112}
                 --kernel-size 1
                 --input-channels 32
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_1
          PARAMS --field {112,112}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 64
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_conv_pw_2_1
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_2
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 128
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_conv_pw_3_1
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_3
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 128
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_4_1
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 128
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_4
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 256
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_5_1
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_5
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 256
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_6_1
          PARAMS --field {14,14}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_6
          PARAMS --field {14,14}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 512
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_7_1
          PARAMS --field {14,14}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_11
          PARAMS --field {14,14}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 512
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_12_1
          PARAMS --field {7,7}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 1024
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_12
          PARAMS --field {7,7}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 1024
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_13_1
          PARAMS --field {7,7}
                 --kernel-size 1
                 --input-channels 1024
                 --output-channels 1024
                 --batch-size 4)

  add_conv_benchmark(NAME conv_5x200_1_in_100_out_bs1440
          PARAMS --field {15,200}
                 --kernel-size {5,200}
                 --input-channels 1
                 --output-channels 100
                 --batch-size 1440
          PARALLEL_LEVEL 8)

  # fc and embedding layer benchmarks (batch size 16). each name is derived from
  # the layer naming
  add_fc_benchmark(NAME fc_layer_1440x100x200
          PARAMS --batch-size 1440
                 --input-size 100
                 --output-size 200)
  add_fc_benchmark(NAME fc_layer_1440x200x400
          PARAMS --batch-size 1440
                 --input-size 200
                 --output-size 400)
  add_fc_benchmark(NAME fc_layer_16x1324x100
          PARAMS --batch-size 16
                 --input-size 1324
                 --output-size 100)
  add_fc_benchmark(NAME fc_layer_80_1324_100
          PARAMS --batch-size 80
                 --input-size 1324
                 --output-size 100)
  add_embedding_benchmark(NAME embedding_small
          PARAMS --shape {1000,200}
                 --num-indices 21600)
  add_embedding_benchmark(NAME embedding_vlarge
          PARAMS --shape {100000,200}
                 --num-indices 1440)
  add_embedding_benchmark(NAME embedding_vlarge_large_indices
          PARAMS --shape {100000,200}
                 --num-indices 11520)

  # Small fully connected layers at low batch sizes (half and float).
  foreach(type half float)
    foreach(bs 1 4)
      add_fc_benchmark(NAME fc_layer_${bs}_1000_1000_${type}
                      PARAMS --batch-size ${bs}
                             --input-size 1000
                             --output-size 1000
                             --data-type=${type})
      add_fc_benchmark(NAME fc_layer_${bs}_1000_5_${type}
                       PARAMS --batch-size ${bs}
                              --input-size 1000
                              --output-size 5
                              --data-type=${type})
    endforeach()
  endforeach()

  # bert fc layer benchmarks (batch size 1). based off of the following the from
  # the model in enigma:
  #   KQV concat FC (S , H) * (H, 3*H)
  #   Grouped (first dim is group)  (B * K, S, 64) * (B * K, 64, S)
  #   Attention projection FC  (S , H) * (H, H)
  #   FFN1 FC (S , H) * (H, FX)
  #   FFN2 FC (S , FX) * (FX, H)
  #   where
  #     H is the set (768, 1024)
  #     K is (12, 16) depending on H
  #     FX is (3072, 4096) depending on H
  #     S is an independent dimension and can be (128, 384, 512)
  set(BERT_H 768 1024)
  set(BERT_K 12 16)
  set(BERT_FX 3072 4096)
  set(BERT_S 128 384 512)

  foreach(idx 0 1)
    list(GET BERT_H ${idx} h)
    list(GET BERT_K ${idx} k)
    list(GET BERT_FX ${idx} fx)
    math(EXPR h3 "3 * ${h}")
    foreach(s ${BERT_S})
      add_fc_benchmark(NAME bert_kqv_${s}x${h}x${h3}
              PARAMS --input-size ${h}
                     --output-size ${h3}
                     --batch-size ${s})
      add_fc_benchmark(NAME bert_grouped_${k}x${s}x64x${s}
              PARAMS --input-size 64
                     --output-size ${s}
                     --batch-size ${s}
                     --num-groups ${k})
      add_fc_benchmark(NAME bert_proj_${s}x${h}x${h}
              PARAMS --input-size ${h}
                     --output-size ${s}
                     --batch-size ${h})
      add_fc_benchmark(NAME bert_ffn1_${s}x${h}x${fx}
              PARAMS --input-size ${h}
                     --output-size ${fx}
                     --batch-size ${s})
      add_fc_benchmark(NAME bert_ffn2_${s}x${fx}x${h}
              PARAMS --input-size ${fx}
                     --output-size ${h}
                     --batch-size ${s})
    endforeach()
  endforeach()

  # inception layer pooling benchmarks. each name is the enigma name
  # the first time that layer is seen.
  add_pooling_benchmark(NAME inception_tr_bs1_pool1
          PARAMS --field {147,147}
                 --kernel {3,3}
                 --channels 64
                 --stride 2
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8)
  add_pooling_benchmark(NAME inception_tr_bs1_pool2
          PARAMS --field {71,71}
                 --kernel {3,3}
                 --channels 192
                 --stride 2
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8)
  add_pooling_benchmark(NAME inception_tr_bs1_i1_dmaxpool
          PARAMS --field {35,35}
                 --kernel {3,3}
                 --channels 192
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i2_dmaxpool
          PARAMS --field {35,35}
                 --kernel {3,3}
                 --channels 256
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i3_dmaxpool
          PARAMS --field {35,35}
                 --kernel {3,3}
                 --channels 288
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i4_cmaxpool
          PARAMS --field {35,35}
                 --kernel {3,3}
                 --channels 288
                 --stride 2
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8)
  add_pooling_benchmark(NAME inception_tr_bs1_i5_dmaxpool
          PARAMS --field {17,17}
                 --kernel {3,3}
                 --channels 768
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i6_dmaxpool
          PARAMS --field {17,17}
                 --kernel {3,3}
                 --channels 768
                 --stride 2
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8)
  add_pooling_benchmark(NAME inception_tr_bs1_i9_dmax_pool
          PARAMS --field {8,8}
                 --kernel {3,3}
                 --channels 1280
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 1
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i10_dmax_pool
          PARAMS --field {8,8}
                 --kernel {3,3}
                 --channels 2048
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)

else()
  message(WARNING "Python 3 could not be found so several tests and benchmarks will not be run.")
endif()

# Based on Layer0/Attention/Z/Softmax/125/nonLinearityOutplace/Nonlinearity/SoftMax from BERT
add_reduction_benchmark(NAME bert_reduce_0
                        PARAMS --shape=16,4,16,32,8
                               --dims=1,3
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=MAX
                               --tiles-per-ipu=1024)

add_reduction_benchmark(NAME bert_reduce_1
                        PARAMS --shape=16,4,16,32,8
                               --dims=1,3
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1024)

# Based on Embedding/GroupNormalization/106/groupNormStatistics/Norm/statistics/mean/ConstScale from BERT
add_reduction_benchmark(NAME bert_reduce_2
                        PARAMS --shape=16,256,8,4
                               --dims=1,3
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1024)

# Based on Embedding/GroupNormalization/106/groupNormStatistics/Norm/statistics/power/ConstScale from BERT
add_reduction_benchmark(NAME bert_reduce_3
                        PARAMS --shape=16,256,8,4
                               --dims=1,3
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=SQUARE_ADD
                               --tiles-per-ipu=1024)

# Based on Layer0/FF/GroupNormalization/210/operandGrad/Norm/deltas/JointGammaDelta from BERT
add_reduction_benchmark(NAME bert_reduce_4
                        PARAMS --shape=128,16,8,16
                               --dims=0,2
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD)

# Based on cnv-gn/Fwd/Norm/statistics/power/ConstScale from Resnet50
add_reduction_benchmark(NAME resnet50_tr_bs1_cnv_reduce
                        PARAMS --shape=4,25088,8
                               --dims=1
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=SQUARE_ADD)

# Based on bm64L0A0-gn/Fwd/Norm/statistics/mean/ConstScale from Resnet50
add_reduction_benchmark(NAME resnet50_tr_bs1_bm64L0A0_reduce
                        PARAMS --shape=4,6272,8
                               --dims=1
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1200)

# Based on bm128L0A0-gn/Fwd/Norm/statistics/mean/ConstScale from Resnet50
add_reduction_benchmark(NAME resnet50_tr_bs1_bm128L0A0_reduce
                        PARAMS --shape=4,3136,8
                               --dims=1
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD)

# Based on bm128L0-gn/Fwd/Norm/statistics/mean/ConstScale from Resnet50
add_reduction_benchmark(NAME resnet50_tr_bs1_bm128L0_reduce
                        PARAMS --shape=4,12544,8
                               --dims=1
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD)
