# Locate sparsehash
# This modules defines the following imported target:
#  Sparsehash::sparsehash
# This modules defines the following variables:
#  SPARSEHASH_FOUND - System has sparsehash library
#  SPARSEHASH_INCLUDE_DIRS - The sparsehash include directories

find_path(SPARSEHASH_INCLUDE_DIRS sparsehash/sparse_hash_map)

if(SPARSEHASH_INCLUDE_DIRS)
  if(NOT TARGET Sparsehash::sparsehash)
    add_library(Sparsehash::sparsehash INTERFACE IMPORTED)
    set_target_properties(Sparsehash::sparsehash PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${SPARSEHASH_INCLUDE_DIRS}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
# Sets SPARSEHASH_FOUND
find_package_handle_standard_args(SPARSEHASH DEFAULT_MSG
                                  SPARSEHASH_INCLUDE_DIRS)

mark_as_advanced(SPARSEHASH_INCLUDE_DIRS)
