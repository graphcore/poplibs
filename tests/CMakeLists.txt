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
        poplibs_support 
        poplibs_test 
        poplar 
        Boost::unit_test_framework
        Boost::timer 
        Boost::system 
        Boost::program_options
        ${CMAKE_THREAD_LIBS_INIT}
)

if(TARGET popsparse)
  list(APPEND COMMON_ADD_EXECUTABLE_LIBRARIES popsparse)
endif()

set(COMMON_ADD_EXECUTABLE_COMPILE_FLAGS "")

# Function to add a multi-target test executable
function(add_multi_target_test_executable name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/tests
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poplin,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name}
    popfloat poprand popnn poputil poplar poplibs_support poplibs_test pva
    Boost::program_options ${CMAKE_THREAD_LIBS_INIT})
  if(TARGET popsparse)
    target_link_libraries(${name} popsparse)
  endif()
endfunction()

include(${GCCS_CMAKE_DIR}/AddUnitTest.cmake)
include(${GCCS_CMAKE_DIR}/AddSystemTest.cmake)

# Make an variant list that can be used to denote tests that specifically
# should not be run on Sim variants because the test times out if run on the
# simulator:
set(TimesOutOnSim "${DEFAULT_TEST_VARIANTS}")
list(REMOVE_ITEM TimesOutOnSim "Sim1" "Sim2")

set(TimesOutOnSimCpp "${TimesOutOnSim}")
list(REMOVE_ITEM TimesOutOnSimCpp "Sim1:cpp" "Sim2:cpp")

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
