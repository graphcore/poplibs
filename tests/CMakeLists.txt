find_package(Boost REQUIRED unit_test_framework timer system program_options)
find_package(PythonInterp REQUIRED)
find_package(Threads)


set(DEFAULT_TEST_VARIANTS "IpuModel" CACHE STRING
    "The device variants to run tests on when no variants are set explicitly")

set(ENABLED_TEST_VARIANTS "Cpu;IpuModel;Sim;Sim:cpp" CACHE STRING
    "Generate tests for these devices and variants")

add_subdirectory(popsolver)

# Remove any requested variants not present in ${ENABLED_TEST_VARIANTS}
# If no variants were requested all ${ENABLED_TEST_VARIANTS} are returned.
function(sanitise_variants variant_name)
  set(variants ${${variant_name}})
  set(old_variants "${variants}")
  if (NOT old_variants)
    set(new_variants ${DEFAULT_TEST_VARIANTS})
  else()
    set(new_variants "")
    # Only allow each variant if it is enabled:
    foreach(variant ${old_variants})
      if (${variant} IN_LIST ENABLED_TEST_VARIANTS)
        list(APPEND new_variants ${variant})
      endif()
    endforeach()
  endif()
  list(REMOVE_DUPLICATES new_variants)
  set(${variant_name} "${new_variants}" PARENT_SCOPE)
endfunction()

function(add_test_executable name)
    add_executable(${name} ${ARGN})
  set_property(TARGET ${name}
      APPEND_STRING PROPERTY
      COMPILE_FLAGS "-DTEST_TARGET=DeviceType::${TEST_TARGET}")
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_SOURCE_DIR}/tests
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:popconv,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} poprand popnn poputil
                        poplar
                        poplibs_test
                        Boost::unit_test_framework
                        Boost::timer
                        Boost::system
                        Boost::program_options
                        ${CMAKE_THREAD_LIBS_INIT})
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

# If running tests on Ipu hardware is enabled then add a test fixture that
# checks if a physical IPU device is present. Hw tests will list this test as a
# "required fixture" and they will therefore not be run if no Ipu device is
# present at run time (and the fixture will be listed as a failure):
extract_targets(ENABLED_TEST_VARIANTS ENABLED_TEST_TARGETS)
if ("Hw" IN_LIST ENABLED_TEST_TARGETS)
  add_test_executable(EnumerateDevices EnumerateDevices.cpp)
  add_test(NAME CheckIpuHwAvailable
           COMMAND EnumerateDevices
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
  set_tests_properties(CheckIpuHwAvailable
                       PROPERTIES FIXTURES_SETUP IpuIsAvailable)
endif()

# Add a test to run with the specified VARIANTS (or all enabled VARIANTS if
# none are specified). The first parameter is the test name, the following are
# source files, and finally everything after VARIANTS are variant
# specifications.
#
# A variation specification must be of the form "target:[config]"
#
# Currently supported targets are "Cpu;IpuModel;Sim;Hw"
# Note that the Cpu target does not support multi-tile tests
# Currently supported configs are "cpp;default" which specify the codelet types
function(add_popnn_unit_test name)
  cmake_parse_arguments(add_unit_test "" "FILES" "VARIANTS" "FILES;${ARGN}")
  sanitise_variants(add_unit_test_VARIANTS)
  set(FILES "${add_unit_test_FILES}")
  extract_targets(add_unit_test_VARIANTS TARGETS)
  message(STATUS
    "Adding test '${name}' with variants: ${add_unit_test_VARIANTS}")

  foreach(TEST_TARGET ${TARGETS})
    set(executable_name "${TEST_TARGET}_${name}")
    add_test_executable(${executable_name} ${FILES})
  endforeach()

  foreach(VARIANT ${add_unit_test_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")
    set(executable_name "${TEST_TARGET}_${name}")
    add_test(NAME "${test_name}"
      COMMAND ${executable_name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
    if (${TEST_CONFIG} STREQUAL "cpp")
      set_tests_properties(${test_name} PROPERTIES
                           ENVIRONMENT POPLIBS_CODELET_SUFFIX=_c)
    endif()
    if (${TEST_TARGET} STREQUAL "Hw")
      # Serialise tests that require a physical IPU using dummy resource
      # 'IpuDeviceAccess' and make sure those tests only run if an IPU was
      # found by the IpuIsAvailable test fixture:
      #message(STATUS "${test_name} requires an IPU: setting required fixture")
      set_tests_properties(${test_name}
        PROPERTIES RESOURCE_LOCK IpuDeviceAccess
                   FIXTURES_REQUIRED IpuIsAvailable)
    endif()
  endforeach()
endfunction()

# add a test to run on the specified VARIANTSs
function(add_multitarget_test)
  cmake_parse_arguments(add_multitarget_test "" "NAME"
                        "VARIANTS;COMMAND"
                        "${ARGN}")
  sanitise_variants(add_multitarget_test_VARIANTS)
  set(name ${add_multitarget_test_NAME})
  if (NOT add_multitarget_test_NAME)
    message(FATAL_ERROR "add_multitarget_test requires a NAME to be specified")
  endif()
  foreach(VARIANT ${add_multitarget_test_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")
    add_test(NAME "${test_name}"
      COMMAND ${add_multitarget_test_COMMAND} "--device-type=${TEST_TARGET}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
    if (${TEST_CONFIG} STREQUAL "cpp")
      set_tests_properties(${test_name} PROPERTIES
                           ENVIRONMENT POPLIBS_CODELET_SUFFIX=_c)
    endif()
    if (${TEST_TARGET} STREQUAL "Hw")
      # Serialise tests that require a physical IPU using dummy resource
      # 'IpuDeviceAccess' and make sure those tests only run if an IPU was
      # found by the IpuIsAvailable test fixture:
      #message(STATUS "${test_name} requires an IPU: setting required fixture")
      set_tests_properties(${test_name}
        PROPERTIES RESOURCE_LOCK IpuDeviceAccess
                   FIXTURES_REQUIRED IpuIsAvailable)
    endif()
  endforeach()
endfunction()

# Unit tests
add_popnn_unit_test(RandomGenTest RandomGenTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(ConvUtilTest ConvUtilTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(RangeTest RangeTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(ConvPlanTest ConvPlanTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(StdArithmeticTests StdArithmeticTests.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(GraphFunctionTest GraphFunctionTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(WinogradConvolution WinogradConv.cpp)
add_popnn_unit_test(NonLinearityTest NonLinearityTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(BigNLVertices BigNLVertices.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(GraphProgLocationTest GraphProgLocationTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(StdOperatorTest StdOperatorsTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(BatchNormTest BatchNormTest.cpp)
add_popnn_unit_test(ReductionTests ReductionTests.cpp)
add_popnn_unit_test(LargeSplitRegionsTest LargeSplitRegionsTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(DynamicSliceTest DynamicSliceTest.cpp)
add_popnn_unit_test(CircBufTests CircBufTests.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(BroadcastToMatchTest BroadcastToMatchTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(ParallelTest ParallelTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(LossTest LossTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")
add_popnn_unit_test(EncodingTest EncodingTest.cpp
                    VARIANTS "IpuModel;Sim:cpp;Sim")

# Codelets
add_subdirectory(codelets)

# Random single_conv_layer tests with fixed seed.
set(NUM_CONV_RANDOM_TESTS 50)
foreach(n RANGE 1 ${NUM_CONV_RANDOM_TESTS})
  add_multitarget_test(
           NAME conv_random_${n}
           COMMAND ${PYTHON_EXECUTABLE}
                   ${CMAKE_SOURCE_DIR}/tools/single_conv_layer_random.py
                   --n 1
                   --seed ${n}
                   --binary $<TARGET_FILE:single_conv_layer>)
endforeach()

# Random reduce_op tests with fixed seed.
set(NUM_REDUCE_RANDOM_TESTS 150)
foreach(n RANGE 1 ${NUM_REDUCE_RANDOM_TESTS})
  add_test(NAME reduce_random_${n}
           COMMAND $<TARGET_FILE:reduce_op> --seed ${n})
endforeach()

#  ORIG Layer tests
add_multitarget_test(NAME conv3x3_stride_2_128_out
         COMMAND single_conv_layer
                 --input-channels=64
                 --output-channels=128
                 --field={28,20}
                 --kernel-size=3
                 --data-type=half
                 --padding=2
                 --stride=2)

add_multitarget_test(
         NAME conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=128
             --output-channels=256
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=1)

add_multitarget_test(
         NAME conv1x1_no_bias
        COMMAND single_conv_layer
            --input-channels=64
            --output-channels=32
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1
            --bias=0)

add_multitarget_test(
         NAME conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=128
            --output-channels=257
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1)

add_multitarget_test(
         NAME conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=129
            --output-channels=256
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1)

add_multitarget_test(
         NAME conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=129
            --output-channels=257
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1
            --extra-text=1)

add_multitarget_test(
         NAME conv1x1_stride_3
         COMMAND single_conv_layer
             --input-channels=64
             --output-channels=128
             --field={14,14}
             --tiles-per-ipu=8
             --stride=3
             --kernel-size=1)

add_multitarget_test(
         NAME conv1x1_thinly_spread
         COMMAND single_conv_layer
             --field={1,128}
             --input-channels=16
             --output-channels=8)

add_multitarget_test(
         NAME conv3x3_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=3)


add_multitarget_test(
         NAME conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=2
             --kernel-size=3)

add_multitarget_test(
         NAME conv3x3_asymmetric_stride
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,10}
             --tiles-per-ipu=16
             --stride={2,3}
             --kernel-size=3)

add_multitarget_test(
         NAME conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --padding=1)

add_multitarget_test(
         NAME conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --padding=2)

add_multitarget_test(
         NAME conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=2
                 --stride=2
                 --kernel-size=3)

add_multitarget_test(
         NAME conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=1
                 --stride=3)

add_multitarget_test(
         NAME conv3x3_batch_2
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={7,7}
                  --tiles-per-ipu=2
                  --stride=2
                  --kernel-size=3
                  --batch-size=2)

add_multitarget_test(
         NAME conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={28,20}
        --kernel-size=3
        --data-type=half
        --padding={1,2}
        --stride=1)

add_multitarget_test(
         NAME conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={7,7}
        --tiles-per-ipu=2
        --stride=2
        --kernel-size=3
        --padding={1,2}
        --batch-size=2)

add_multitarget_test(
         NAME conv3x3_stride_1_fwd_asym_padding
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --field={28,20}
                --kernel-size=3
                --data-type=half
                --padding-lower=1
                --padding-upper=2
                --stride=1)
add_multitarget_test(
         NAME conv3x3_stride_1_fwd_asym_padding_2
         COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --field={28,20}
                --kernel-size=3
                --data-type=half
                --padding-lower=3
                --padding-upper=0
                --stride=1)

add_multitarget_test(
         NAME conv5x5_stride_2_fwd_asym_padding
         COMMAND single_conv_layer
                  --input-channels=64
                  --output-channels=64
                  --field={28,20}
                  --kernel-size=5
                  --data-type=half
                  --padding-lower={1,2}
                  --padding-upper={3,3}
                  --stride=2
                  --tiles-per-ipu=16)

add_multitarget_test(
         NAME small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --partials-type=float
                 --tiles-per-ipu=1216)

add_multitarget_test(
         NAME small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --kernel-size=3
                 --field={14,14}
                 --partials-type=float
                 --tiles-per-ipu=1216)

add_multitarget_test(
         NAME conv1x1_shallow_input
         COMMAND single_conv_layer
                 --input-channels=8
                 --output-channels=1024
                 --field={16,16}
                 --data-type=half)

add_multitarget_test(
         NAME conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=1
                 --data-type=half)

add_multitarget_test(
         NAME conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=2
                 --data-type=half)

add_multitarget_test(
         NAME conv_large_kernel
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --stride=2)

add_multitarget_test(
         NAME conv_large_kernel_bwd
         COMMAND single_conv_layer
                 --single-phase=bwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --stride=2)

add_multitarget_test(
         NAME conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                 --field={1,16}
                 --padding={1,6}
                 --kernel-size={1,7}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_flip_input_and_kernel_1
         COMMAND single_conv_layer
                 --field={6,2}
                 --padding={0,1}
                 --flip-input={0,1}
                 --kernel-size={1,3}
                 --kernel-padding={0,0}
                 --flip-kernel={0,1}
                 --input-channels=2
                 --output-channels=1
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_flip_input_and_kernel_2
         COMMAND single_conv_layer
                 --field={1,32}
                 --padding={0,16}
                 --flip-input={0,1}
                 --kernel-size={1,48}
                 --flip-kernel={0,1}
                 --input-channels=32
                 --output-channels=8
                 --tiles-per-ipu=4)

add_multitarget_test(
         NAME conv_flip_input_only
         COMMAND single_conv_layer
                 --field={24,24}
                 --padding=1
                 --flip-input=1
                 --kernel-size=3
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_flip_kernel_only
         COMMAND single_conv_layer
                 --field={24,24}
                 --padding=1
                 --kernel-size=3
                 --flip-kernel=1
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_batch_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=half
                 --batch-size=4)

add_multitarget_test(
         NAME conv_batch_bwd_float
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4)

add_multitarget_test(
         NAME conv_batch_group
         COMMAND single_conv_layer
                 --field={28,28}
                 --input-channels=512
                 --output-channels=128
                 --batch-size=2)

# conv_batch_bwd_float test split into the three component phases
add_multitarget_test(
         NAME conv_batch_bwd_float_fwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=fwd)

add_multitarget_test(
         NAME conv_batch_bwd_float_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=bwd)

add_multitarget_test(
         NAME conv_batch_bwd_float_wu
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=wu)

add_multitarget_test(
         NAME conv_negative_input_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_kernel_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,1}
                --kernel-size={1,2}
                --kernel-truncation-upper={0,1}
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_positive_and_negative_kernel_padding
         COMMAND single_conv_layer
         --input-channels=16
         --output-channels=16
         --field={5,5}
         --kernel-size=2
         --kernel-padding-lower=1
         --kernel-truncation-upper=1
         --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_input_padding_lower_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,2}
                 --kernel-size={1,1}
                 --truncation-upper={0,1}
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_kernel_padding_lower_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --kernel-truncation-lower={0,1}
                 --tiles-per-ipu=1)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_multitarget_test(
         NAME conv_large_output_tensor
         COMMAND single_conv_layer
                --field={16,16}
                --input-channels=16
                --output-channels=128
                --tiles-per-ipu=16
                --data-type=half)

add_multitarget_test(
         NAME conv_large_weights
         COMMAND single_conv_layer
                --field={14,16}
                --padding={2,0}
                --flip-input={1,0}
                --input-channels=64
                --output-channels=64
                --kernel-size={16,1}
                --flip-kernel={1,0}
                --tiles-per-ipu=16
                --data-type=half
                --single-phase=fwd)

add_multitarget_test(
         NAME conv_outer_product
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1024
                 --field={1,512}
                 --tiles-per-ipu=64)

add_multitarget_test(
         NAME conv3x3_stride_1_half_partials
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=16
                 --stride=1
                 --kernel-size=3
                 --partials-type=half)

add_multitarget_test(
         NAME conv1x1_kernel_with_padding_larger_than_field
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --field={1,128}
                 --tiles-per-ipu=2
                 --padding={1,64})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=64
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=3
                 --stride=3
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor2
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=64
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=4
                 --stride=2
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=64
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=2
                 --stride=4
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor4
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=64
                --field={14,14}
                --kernel-size={3,3}
                --in-dilation=4
                --stride=6
                --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor5
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=64
                --field={14,14}
                --kernel-size={3,3}
                --in-dilation=6
                --stride=4
                --tiles-per-ipu=16)

# Example where the amount of padding that must be added to start of the input
# window for the conv nx1 vertex exceeds the amount of lower padding for the
# layer.
add_multitarget_test(
         NAME conv_nx1_vertex_lower_padding
         COMMAND single_conv_layer
                 --field={16,32}
                 --input-channels=4
                 --padding-lower={1,0}
                 --output-channels=8
                 --kernel-size={2,1}
                 --kernel-dilation={2,1}
                 --stride={3,1}
                 --tiles-per-ipu=1)

# Test where, after padding and dilation, the last input index that would be
# multiplied by the a kernel element is in the middle of the lower input
# padding.
add_multitarget_test(
         NAME conv1x1_input_ignored_with_stride
         COMMAND single_conv_layer
                 --field={2,2}
                 --in-dilation=2
                 --padding-lower=2
                 --padding-upper=1
                 --stride=7
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=1)

# Test with large amounts of output padding
add_multitarget_test(
         NAME conv1x1_output_padding
         COMMAND single_conv_layer
                 --field={1,1}
                 --input-channels=16
                 --output-channels=16
                 --output-padding=10
                 --tiles-per-ipu=16)

#  Grouped convolution layer tests
add_multitarget_test(
         NAME grouped_conv3x3_stride_2_256_out
         COMMAND single_conv_layer
                 --input-channels=64
                 --output-channels=128
                 --field={28,20}
                 --kernel-size=3
                 --data-type=half
                 --padding=2
                 --conv-groups=2
                 --stride=2)

add_multitarget_test(
         NAME grouped_conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=4
             --kernel-size=1)

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=64
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --conv-groups=2
            --kernel-size=1)

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=128
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --conv-groups=2
            --kernel-size=1)

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --conv-groups=2
            --kernel-size=1)

add_multitarget_test(
         NAME grouped_conv1x1_stride_3_odd_conv_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={14,14}
             --tiles-per-ipu=8
             --stride=3
             --conv-groups=3
             --kernel-size=1)

add_multitarget_test(
         NAME grouped_conv1x1_thinly_spread
         COMMAND single_conv_layer
             --field={1,128}
             --input-channels=16
             --conv-groups=2
             --output-channels=8)

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_odd_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=3
             --kernel-size=3)


add_multitarget_test(
         NAME grouped_conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=2
             --conv-groups=2
             --kernel-size=3)

add_multitarget_test(
         NAME grouped_conv3x3_asymmetric_stride
         COMMAND single_conv_layer
             --input-channels=16
             --output-channels=8
             --field={7,10}
             --tiles-per-ipu=16
             --stride={2,3}
             --conv-groups=2
             --kernel-size=3)

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --conv-groups=2
              --padding=1)

add_multitarget_test(
         NAME grouped_conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=16
              --output-channels=8
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --conv-groups=2
              --padding=2)

add_multitarget_test(
         NAME grouped_conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=2
                 --stride=2
                 --conv-groups=2
                 --kernel-size=3)

add_multitarget_test(
         NAME grouped_conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=1
                 --conv-groups=2
                 --stride=3)

add_multitarget_test(
         NAME grouped_conv3x3_batch_2
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={7,7}
                  --tiles-per-ipu=2
                  --stride=2
                  --kernel-size=3
                  --conv-groups=2
                  --batch-size=2)

add_multitarget_test(
         NAME grouped_conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={28,20}
        --kernel-size=3
        --data-type=half
        --padding={1,2}
        --conv-groups=2
        --stride=1)

add_multitarget_test(
         NAME grouped_conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={7,7}
        --tiles-per-ipu=2
        --stride=2
        --kernel-size=3
        --padding={1,2}
        --conv-groups=2
        --batch-size=2)

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_fwd_asym_padding
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --field={28,20}
                --kernel-size=3
                --data-type=half
                --padding-lower=1
                --padding-upper=2
                --conv-groups=2
                --stride=1)
add_multitarget_test(
         NAME grouped_conv3x3_stride_1_fwd_asym_padding_2
         COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --field={28,20}
                --kernel-size=3
                --data-type=half
                --padding-lower=3
                --padding-upper=0
                --conv-groups=2
                --stride=1)

add_multitarget_test(
         NAME grouped_conv5x5_stride_2_fwd_asym_padding
         COMMAND single_conv_layer
                  --input-channels=32
                  --output-channels=64
                  --field={28,20}
                  --kernel-size=5
                  --data-type=half
                  --padding-lower={1,2}
                  --padding-upper={3,3}
                  --stride=2
                  --conv-groups=2
                  --tiles-per-ipu=16)

add_multitarget_test(
         NAME grouped_small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --partials-type=float
                 --conv-groups=2
                 --tiles-per-ipu=1216)

add_multitarget_test(
         NAME grouped_small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --kernel-size=3
                 --field={14,14}
                 --partials-type=float
                 --conv-groups=2
                 --tiles-per-ipu=1216)

add_multitarget_test(
         NAME grouped_conv1x1_shallow_input
        COMMAND single_conv_layer
                --input-channels=8
                --output-channels=1024
                --field={16,16}
                --conv-groups=2
                --data-type=half)

add_multitarget_test(
         NAME grouped_conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=1
                 --conv-groups=2
                 --data-type=half)

add_multitarget_test(
         NAME grouped_conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=2
                 --conv-groups=2
                 --data-type=half)

add_multitarget_test(
         NAME grouped_conv_large_kernel
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --conv-groups=2
                 --stride=2)

add_multitarget_test(
         NAME grouped_conv_large_kernel_bwd
         COMMAND single_conv_layer
                 --single-phase=bwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --conv-groups=2
                 --stride=2)

add_multitarget_test(
         NAME grouped_conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                --field={1,16}
                 --padding={6,1}
                 --kernel-size={7,1}
                 --input-channels=16
                 --output-channels=16
                 --conv-groups=2
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME grouped_conv_batch_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=half
                 --conv-groups=2
                 --batch-size=4)

add_multitarget_test(
         NAME grouped_conv_batch_bwd_float
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --conv-groups=2
                 --batch-size=4)

add_multitarget_test(
         NAME grouped_conv_batch_group
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=512
                 --output-channels=128
                 --tiles-per-ipu=304
                 --conv-groups=2
                 --batch-size=2)

# conv_batch_bwd_float test split into the three component phases
add_multitarget_test(
         NAME grouped_conv_batch_bwd_float_fwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=fwd)

add_multitarget_test(
         NAME grouped_conv_batch_bwd_float_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=bwd)

add_multitarget_test(
         NAME grouped_conv_batch_bwd_float_wu
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=wu)

add_multitarget_test(
         NAME grouped_conv_negative_input_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --conv-groups=2
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_kernel_padding_upper_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --kernel-truncation-lower={0,1}
                 --conv-groups=2
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_input_padding_lower_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --conv-groups=2
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_kernel_padding_lower_simple_${phase}
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --conv-groups=2
                 --kernel-truncation-lower={0,1}
                 --tiles-per-ipu=1)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_multitarget_test(
         NAME grouped_conv_large_output_tensor
         COMMAND single_conv_layer
                --field={16,16}
                --input-channels=16
                --output-channels=64
                --tiles-per-ipu=16
                --conv-groups=2
                --data-type=half)

add_multitarget_test(
         NAME grouped_conv_large_weights
         COMMAND single_conv_layer
                --field={14,16}
                --padding={2,0}
                --flip-input={1,0}
                --input-channels=32
                --output-channels=32
                --kernel-size={16,1}
                --flip-kernel={1,0}
                --tiles-per-ipu=16
                --data-type=half
                --conv-groups=2
                --single-phase=fwd)

add_multitarget_test(
         NAME grouped_conv_outer_product
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1024
                 --field={1,512}
                 --conv-groups=2
                 --tiles-per-ipu=64)

add_multitarget_test(
         NAME conv0d_simple
         COMMAND single_conv_layer
                 --field={}
                 --batch-size=4
                 --input-channels=256
                 --output-channels=128
                 --tiles-per-ipu=48)

add_multitarget_test(
         NAME conv1d_simple
         COMMAND single_conv_layer
                  --field={80}
                  --kernel-size=3
                  --padding=1
                  --input-channels=128
                  --output-channels=128
                  --tiles-per-ipu=48)

add_multitarget_test(
         NAME conv3d_simple
         COMMAND single_conv_layer
                 --field={5,6,7}
                 --kernel-size={4,3,2}
                 --padding={1,1,1}
                 --input-channels=32
                 --output-channels=64
                 --tiles-per-ipu=48)

add_multitarget_test(
         NAME conv3d_1_output_chan
         COMMAND single_conv_layer
                 --field={5,6,7}
                 --kernel-size={4,3,2}
                 --padding={1,1,1}
                 --input-channels=32
                 --output-channels=1
                 --tiles-per-ipu=48)

# 3D convolution with multiple batches, convolution groups, input dilation,
# kernel dilation and striding.
add_multitarget_test(
         NAME conv3d_complex
         COMMAND single_conv_layer
                 --conv-groups=2
                 --batch-size=2
                 --field={4,3,32}
                 --in-dilation={1,2,1}
                 --padding={1,0,1}
                 --kernel-size={3,2,3}
                 --kernel-dilation={1,1,2}
                 --input-channels=16
                 --output-channels=64
                 --stride={1,1,3}
                 --tiles-per-ipu=16)

add_test(NAME fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half)

add_test(NAME fully_connected_half_fwd
         COMMAND fully_connected_layer
                 --single-phase=fwd
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half)

add_test(NAME fully_connected_half_bwd
        COMMAND fully_connected_layer
                --single-phase=bwd
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half)

add_test(NAME fully_connected_small_field_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 4
                --output-size 4
                --tiles-per-ipu 16
                --data-type=half)

add_test(NAME fully_connected_large_field_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half)

add_test(NAME fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type=half)

add_test(NAME fully_connected_float
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float)

add_test(NAME fully_connected_float_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type=float)

add_test(NAME fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --batch-size=4)

add_test(NAME fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --ipus=2
                 --batch-size=4)

add_test(NAME fully_connected_half_batch_7
      COMMAND fully_connected_layer
              --input-size 256
              --output-size 256
              --tiles-per-ipu 16
              --data-type=half
              --batch-size=7)

add_test(NAME fully_connected_half_batch_7_bwd_as_wu
      COMMAND fully_connected_layer
              --input-size 256
              --output-size 256
              --tiles-per-ipu 16
              --data-type=half
              --batch-size=7)

add_test(NAME fully_connected_float_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float
                 --batch-size=4)

add_test(NAME fully_connected_float_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float
                 --ipus=2
                 --batch-size=4)

add_test(NAME fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 1
                 --data-type=half
                 --batch-size=4
                 --inference-only)

add_test(NAME fully_connected_input_size_1
         COMMAND fully_connected_layer
                 --input-size=1
                 --output-size=100
                 --tiles-per-ipu=4
                 --data-type=half
                 --batch-size=16)

add_test(NAME fully_connected_output_size_1
         COMMAND fully_connected_layer
                 --input-size=100
                 --output-size=1
                 --tiles-per-ipu=4
                 --data-type=half
                 --batch-size=16)

add_test(NAME grouped_fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --num-groups 2
                 --data-type=half)


add_test(NAME grouped_fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --num-groups 3
                 --data-type=half)


add_test(NAME grouped_fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --num-groups 2
                 --batch-size=4)

add_test(NAME grouped_fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --num-groups 3
                 --ipus=2
                 --batch-size=4)

add_test(NAME grouped_fully_connected_half_batch_7
      COMMAND fully_connected_layer
              --input-size 256
              --output-size 256
              --tiles-per-ipu 16
              --data-type=half
              --num-groups 3
              --batch-size=7)

add_test(NAME grouped_fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 1
                 --data-type=half
                 --batch-size=4
                 --num-groups 2
                 --inference-only)

# Fully connected layer where the RHS has fewer elements than the LHS
add_test(NAME fully_connected_small_rhs
         COMMAND fully_connected_layer
                 --batch-size=64
                 --input-size=512
                 --output-size=4
                 --data-type=half)

add_test(NAME max_pool_layer_half
         COMMAND pooling_layer
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME max_pool_layer_float
         COMMAND pooling_layer
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)
               

add_test(NAME max_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --width=100
                 --height=100
                 --kernel-size=3
                 --stride=2
                 --padding=1
                 --data-type=float)

add_test(NAME max_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME max_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_test(NAME max_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --width=56
                 --height=56
                 --kernel-size=3
                 --tiles-per-ipu 304
                 --stride=2
                 --ipus 2)

add_test(NAME max_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --width=32
                 --height=1
                 --kernel-height=1
                 --kernel-width=4
                 --stride-height=1
                 --stride-width=4)

add_test(NAME max_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=1
                --padding-width-upper=2
                --padding-height-lower=3
                --padding-height-upper=4)

add_test(NAME max_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=-1
                --padding-width-upper=2
                --padding-height-lower=-3
                --padding-height-upper=4)

add_test(NAME avg_pool_layer_half
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=avg
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME avg_pool_layer_float
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=avg
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_test(NAME avg_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --pooling-type=avg
                 --width=100
                 --height=100
                 --kernel-size=3
                 --stride=2
                 --padding=1
                 --data-type=float)

add_test(NAME avg_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=avg
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME avg_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=avg
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_test(NAME avg_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --pooling-type=avg
                 --bwd-chans-per-group=8
                 --width=56
                 --height=56
                 --kernel-size=3
                 --tiles-per-ipu=304
                 --stride=2
                 --ipus 2)

add_test(NAME avg_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=avg
                 --width=32
                 --height=1
                 --kernel-height=1
                 --kernel-width=4
                 --stride-height=1
                 --stride-width=4)

add_test(NAME avg_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --pooling-type=avg
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=1
                --padding-width-upper=2
                --padding-height-lower=3
                --padding-height-upper=4)

add_test(NAME avg_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --pooling-type=avg
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=-1
                --padding-width-upper=2
                --padding-height-lower=-3
                --padding-height-upper=4)

add_test(NAME sum_pool_layer_half
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=sum
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME sum_pool_layer_float
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=sum
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)
               

add_test(NAME sum_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --pooling-type=sum
                 --width=100
                 --height=100
                 --kernel-size=3
                 --stride=2
                 --padding=1
                 --data-type=float)

add_test(NAME sum_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=sum
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME sum_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=sum
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_test(NAME sum_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --pooling-type=sum
                 --bwd-chans-per-group=8
                 --width=56
                 --height=56
                 --kernel-size=3
                 --tiles-per-ipu=304
                 --stride=2
                 --ipus 2)

add_test(NAME sum_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=sum
                 --width=32
                 --height=1
                 --kernel-height=1
                 --kernel-width=4
                 --stride-height=1
                 --stride-width=4)

add_test(NAME sum_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --pooling-type=sum
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=1
                --padding-width-upper=2
                --padding-height-lower=3
                --padding-height-upper=4)

add_test(NAME sum_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --pooling-type=sum
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=-1
                --padding-width-upper=2
                --padding-height-lower=-3
                --padding-height-upper=4)

add_test(NAME gemm_5x0x4
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=normal
                  --right-matrix-op=normal)

add_test(NAME gemm_0x3x4
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=normal
                  --right-matrix-op=normal)

add_test(NAME gemm_5x3x0
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=normal
                  --right-matrix-op=normal)


add_test(NAME gemm_5x0x4_trA
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=transpose
                  --right-matrix-op=normal)

add_test(NAME gemm_0x3x4_trA
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=transpose
                  --right-matrix-op=normal)

add_test(NAME gemm_5x3x0_trA
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=transpose
                  --right-matrix-op=normal)

add_test(NAME gemm_5x0x4_trB
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=normal
                  --right-matrix-op=transpose)

add_test(NAME gemm_0x3x4_trB
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=normal
                  --right-matrix-op=transpose)

add_test(NAME gemm_5x3x0_trB
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=normal
                  --right-matrix-op=transpose)

add_test(NAME gemm_5x0x4_trA_trB
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=transpose
                  --right-matrix-op=transpose)

add_test(NAME gemm_0x3x4_trA_trB
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=transpose
                  --right-matrix-op=transpose)

add_test(NAME gemm_5x3x0_trA_trB
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --left-matrix-op=transpose
                  --right-matrix-op=transpose)

add_test(NAME gemm_1x3x1
          COMMAND general_matrix_multiply
                  --m 1
                  --k 3
                  --n 1
                  --alpha 1
                  --beta 1
                  --left-matrix-op=normal
                  --right-matrix-op=normal)

add_test(NAME gemm_1x1000x1
        COMMAND general_matrix_multiply
                --m 1
                --k 1000
                --n 1
                --alpha 2
                --beta 1
                --left-matrix-op=normal
                --right-matrix-op=normal)


add_test(NAME gemm_40x40x40
         COMMAND general_matrix_multiply
                 --m 40
                 --k 40
                 --n 40 
                 --alpha 2
                 --beta 1 
                 --left-matrix-op=normal
                 --right-matrix-op=normal)


add_test(NAME gemm_40x39x38
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38 
                 --alpha 2
                 --beta 1 
                 --left-matrix-op=normal
                 --right-matrix-op=normal)

add_test(NAME gemm_40x39x38_left_transpose
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38 
                 --alpha 2
                 --beta 1 
                 --left-matrix-op=transpose
                 --right-matrix-op=normal)


add_test(NAME gemm_40x39x38_right_transpose
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38 
                 --alpha 2
                 --beta 1 
                 --left-matrix-op=normal
                 --right-matrix-op=transpose)

add_test(NAME rnn_3x2x38_no_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --output-size 38
                 --batch-size=2
                 --nonlinearity-type sigmoid)


add_test(NAME rnn_3x32x2x38_with_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --input-size 32
                 --output-size 38
                 --batch-size=2
                 --phase all
                 --nonlinearity-type sigmoid
                 --apply-feedforward-weights) 
                 
add_test(NAME basic_lstm_40x4x38_seq_2_half_data
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --phase all
                 --sequence-size 2)

add_test(NAME basic_lstm_40x4x38_seq_2_half_data_preweight_inp
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --pre-weight-input=1
                 --phase all
                 --sequence-size 2)

add_test(NAME basic_lstm_40x4x38_seq_2_float_data
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --sequence-size 2
                 --phase all
                 --data-type=float)

add_multitarget_test(
         NAME conv1x1_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --tiles-per-ipu=1
                 --in-dilation=2)

add_multitarget_test(
         NAME conv1x1_stride_3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2)

add_multitarget_test(
         NAME conv3x3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation=2)

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2)

add_multitarget_test(
         NAME conv3x3_in_dilation_asymmetric
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation={2,3})

add_multitarget_test(
         NAME conv3x3_kernel_dilation_amp
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --weight-update-method=AMP
                --field={14,28}
                --kernel-size={3,5}
                --kernel-dilation={3,2}
                --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv3x3_kernel_padding_amp
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --weight-update-method=AMP
                 --field={14,14}
                 --kernel-size={3,3}
                 --kernel-padding-lower={4,2}
                 --kernel-padding-upper={4,3}
                 --tiles-per-ipu=16)
