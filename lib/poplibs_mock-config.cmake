# Find dependencies
find_package(poplibs REQUIRED)
find_package(GTest REQUIRED CONFIG)

# Compute paths
get_filename_component(POPLIBS_MOCK_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Our library dependencies (contains definitions for IMPORTED targets).
# These should be in the same order as the add_subdirectory()'s in the
# CMakeLists.txt so dependencies are handled correctly
foreach(t
      poputil_mock popops_mock poplin_mock
    )
  if(NOT TARGET ${t} AND NOT ${t}_BINARY_DIR)
    include("${POPLIBS_MOCK_CMAKE_DIR}/../${t}/${t}-targets.cmake")
  endif()
endforeach()
