find_package(PythonInterp 3)
find_package(GCCS REQUIRED)

# Controls verbose debug output on creating test cases
set(DEBUG_TEST_GENERATION FALSE)

# Add vars used by add_executable function
set(COMMON_ADD_EXECUTABLE_INCLUDE_DIRECTORIES
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/lib
        $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:poplin,INCLUDE_DIRECTORIES>
)

set(COMMON_ADD_EXECUTABLE_LIBRARIES
        popfloat
        poprand
        popnn
        poputil
        pva
        poplibs_support
        poplibs_test
        poplar
        popsolver
        Boost::unit_test_framework
        Boost::timer
        Boost::system
        Boost::filesystem
        Boost::program_options
        ${CMAKE_THREAD_LIBS_INIT}
)

if(TARGET popsparse)
  list(APPEND COMMON_ADD_EXECUTABLE_LIBRARIES popsparse)
endif()

set(COMMON_ADD_EXECUTABLE_COMPILE_FLAGS "")

include(${GCCS_CMAKE_DIR}/AddUnitTest.cmake)
include(${GCCS_CMAKE_DIR}/AddSystemTest.cmake)

# Function to add a multi-target test executable
function(add_multi_target_test_executable name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
      PRIVATE ${COMMON_ADD_EXECUTABLE_INCLUDE_DIRECTORIES})
  target_link_libraries(${name} ${COMMON_ADD_EXECUTABLE_LIBRARIES})
endfunction()

# Make an variant list that can be used to denote tests that specifically
# should not be run on Sim variants because the test times out if run on the
# simulator:
set(TimesOutOnSim "${DEFAULT_TEST_VARIANTS}")
list(REMOVE_ITEM TimesOutOnSim "Sim2" "Sim21")

set(TimesOutOnSimCpp "${TimesOutOnSim}")
list(REMOVE_ITEM TimesOutOnSimCpp "Sim2:cpp" "Sim21:cpp")

# Tests
add_subdirectory(popfloat)
add_subdirectory(poplibs_support)
add_subdirectory(poplibs_test)
add_subdirectory(poplin)
add_subdirectory(popnn)
add_subdirectory(popops)
add_subdirectory(poprand)

if(TARGET popsparse)
  add_subdirectory(popsparse)
endif()
add_subdirectory(poputil)
add_subdirectory(sanity)
