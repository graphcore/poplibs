# Find dependencies
find_package(poplar REQUIRED)

# Compute paths
get_filename_component(POPLIBS_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Our library dependencies (contains definitions for IMPORTED targets).
# These should be in the same order as the add_subdirectory()'s in the
# CMakeLists.txt so dependencies are handled correctly
foreach(t
      popsolver poputil popops poprand poplin popnn
    )
  if(NOT TARGET ${t} AND NOT ${t}_BINARY_DIR)
    include("${POPLIBS_CMAKE_DIR}/../${t}/${t}-targets.cmake")
  endif()
endforeach()
