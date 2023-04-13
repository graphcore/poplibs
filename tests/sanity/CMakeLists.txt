# Compile the public headers using C++11.
# When possible use more than one compiler to get better conformance coverage.
set(COMPILERS ${CMAKE_CXX_COMPILER})
find_program(CLANGPP clang++)
if(NOT CLANGPP STREQUAL CLANGPP-NOTFOUND AND
   NOT CLANGPP STREQUAL ${CMAKE_CXX_COMPILER})
  list(APPEND COMPILERS ${CLANGPP})
endif()
foreach(compiler ${COMPILERS})
  add_test(
    NAME ApiCppVersionTest-${compiler}
    COMMAND ${compiler} -std=c++11 -Wall -Wextra -pedantic -Wno-error=deprecated-declarations -Werror ${TEST_CXXFLAGS}
            ${CMAKE_CURRENT_SOURCE_DIR}/ApiCppVersionTest.cpp -lpoplar -lpopops
  )
endforeach()

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

function (add_ipu_hw_test version)
  add_test(NAME CheckIpuHw${version}Available
        COMMAND EnumerateDevices --;ipu${version}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
  set_tests_properties(CheckIpuHw${version}Available
        PROPERTIES FIXTURES_SETUP Ipu${version}IsAvailable)
endfunction()

add_unit_test(WrongSyncZone
              HangTest.cpp
              VARIANTS Hw
              LABELS multicard;CE_reproducers
              PROPERTIES
                RUN_SERIAL TRUE)

add_unit_test(ConsistentExecutableTest
              ConsistentExecutableTest.cpp
              VARIANTS NoTarget)
if(TARGET ConsistentExecutableTest)
  target_compile_definitions(
    ConsistentExecutableTest PRIVATE
    -DEXECUTABLE="${CMAKE_BINARY_DIR}/tools/consistent_executable_tool")
endif()

if (NOT "asan" IN_LIST SANITIZERS)
  # This test causes an internal error in address sanitizer.
  # See T13392.
  add_unit_test(ParallelTest ParallelTest.cpp)
endif()

# Stop transforms.py from rotting.
add_multitarget_test(
  NAME transformsTool
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tools/transforms.py
          --ci-test
          --remove-files
          --test-binary $<TARGET_FILE:single_conv_layer>
  VARIANTS "Hw"
  LABELS "python_tool")

# Stop stalls_highligther.py from rotting.
add_multitarget_test(
  NAME stallsHighligtherTool
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tools/stalls_highlighter.py
          --ci-test
          --test-binary $<TARGET_FILE:single_conv_layer>
  VARIANTS "Sim2"
  LABELS "python_tool")

# Test single_conv_layer_random options.
# POPLAR-320 IpuModel21 does not currently pass this test
add_multitarget_test(
  NAME singleConvLayerRandomTest
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tools/single_conv_layer_random.py
          --ci-test
          --binary $<TARGET_FILE:single_conv_layer>
          --partials-type half
          --constraints-file ${CMAKE_SOURCE_DIR}/tests/poplin/json/slic411.json
  VARIANTS IpuModel2
  LABELS "python_tool")

# Check that compilation is deterministic.
add_multitarget_test(
  NAME determinism_single_conv_layer
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tools/test_determinism.py
          $<TARGET_FILE:single_conv_layer>
          --input-channels=16
          --output-channels=16
          --field={1}
          --kernel-size=1
          --tiles-per-ipu=16
          --compile-only
  LABELS "python_tool"
  VARIANTS ${SIM_VARIANTS})

add_multitarget_test(
  NAME determinism_general_matrix_multiply
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tools/test_determinism.py
          $<TARGET_FILE:general_matrix_multiply>
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=transpose
          --right-matrix-op=transpose
          --compile-only
  LABELS "python_tool"
  VARIANTS ${SIM_VARIANTS})

add_multitarget_test(NAME determinism_pooling_layer
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tools/test_determinism.py
          $<TARGET_FILE:pooling_layer>
          --channels 16
          --field={9,14}
          --kernel-size=2
          --tiles-per-ipu=16
          --stride=2
          --data-type=half
          --use-introspection=1
          --compile-only
  LABELS "python_tool"
  VARIANTS ${SIM_VARIANTS})

add_multitarget_test(NAME determinism_rnn_layer
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tools/test_determinism.py
          $<TARGET_FILE:rnn_layer>
          --sequence-size 3
          --input-size 16
          --output-size 16
          --batch-size=2
          --phase all
          --tiles-per-ipu=16
          --nonlinearity-type sigmoid
          --apply-feedforward-weights
          --compile-only
  LABELS "python_tool"
  VARIANTS ${SIM_VARIANTS})

add_multitarget_test(NAME determinism_sparse_fc_layer
  COMMAND ${PYTHON_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tools/test_determinism.py
          $<TARGET_FILE:sparse_fc_layer>
          --data-type=float
          --partials-type=float
          --input-size=128
          --output-size=128
          --batch-size=4
          --block-size=8
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --matmul-options={\"partitioner.forceBucketSpills\":\"true\"}
          --single-phase=all
          --compile-only
  LABELS "python_tool"
  VARIANTS ${SIM_VARIANTS})

add_subdirectory(dependencies)
