# Compile the public headers using C++11.
add_test(
  NAME ApiCppVersionTest
  COMMAND ${CMAKE_CXX_COMPILER} -std=c++11 ${TEST_CXXFLAGS}
          ${CMAKE_CURRENT_SOURCE_DIR}/ApiCppVersionTest.cpp -lpoplar -Werror -lpopops
)

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
              LABELS multicard;CE_reproducers)

if (NOT "asan" IN_LIST SANITIZERS)
  # This test causes an internal error in address sanitizer.
  # See T13392.
  add_unit_test(ParallelTest ParallelTest.cpp)
endif()
