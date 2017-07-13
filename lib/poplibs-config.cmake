# Find dependencies
find_package(poplar REQUIRED)

# Compute paths
get_filename_component(POPLIBS_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Our library dependencies (contains definitions for IMPORTED targets)
if(NOT TARGET popstd AND NOT popstd_BINARY_DIR)
  include("${POPLIBS_CMAKE_DIR}/../popstd/popstd-targets.cmake")
endif()

if(NOT TARGET popsolver AND NOT popsolver_BINARY_DIR)
  include("${POPLIBS_CMAKE_DIR}/../popsolver/popsolver-targets.cmake")
endif()

if(NOT TARGET popreduce AND NOT popreduce_BINARY_DIR)
  include("${POPLIBS_CMAKE_DIR}/../popreduce/popreduce-targets.cmake")
endif()

if(NOT TARGET poprand AND NOT poprand_BINARY_DIR)
  include("${POPLIBS_CMAKE_DIR}/../poprand/poprand-targets.cmake")
endif()

if(NOT TARGET popconv AND NOT popconv_BINARY_DIR)
  include("${POPLIBS_CMAKE_DIR}/../popconv/popconv-targets.cmake")
endif()

if(NOT TARGET poplin AND NOT poplin_BINARY_DIR)
  include("${POPLIBS_CMAKE_DIR}/../poplin/poplin-targets.cmake")
endif()

if(NOT TARGET popnn AND NOT popnn_BINARY_DIR)
  include("${POPLIBS_CMAKE_DIR}/../popnn/popnn-targets.cmake")
endif()
