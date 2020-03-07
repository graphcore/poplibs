# Search for zoltan include directory and libraries
# This module defines the following variables:
#   - ZOLTAN_INCLUDE_DIR
#   - ZOLTAN_LIBRARY
#   - ZOLTAN_SIMPI_LIBRARY

find_path(ZOLTAN_INCLUDE_DIR zoltan.h
  HINTS ${ZOLTAN_ROOT}/include $ENV{ZOLTAN_ROOT}/include)

find_library(ZOLTAN_LIBRARY zoltan
  HINTS ${ZOLTAN_ROOT}/lib $ENV{ZOLTAN_ROOT}/lib)

find_library(ZOLTAN_SIMPI_LIBRARY simpi
  HINTS ${ZOLTAN_ROOT}/lib $ENV{ZOLTAN_ROOT}/lib)

set(ZOLTAN_LIBRARIES ${ZOLTAN_LIBRARY} ${ZOLTAN_SIMPI_LIBRARY})

if(ZOLTAN_INCLUDE_DIR AND ZOLTAN_LIBRARY AND ZOLTAN_SIMPI_LIBRARY)
  if(NOT TARGET ZOLTAN::zoltan)
    add_library(ZOLTAN::zoltan STATIC IMPORTED)
    set_target_properties(ZOLTAN::zoltan PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ZOLTAN_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${ZOLTAN_LIBRARY}")
    add_library(ZOLTAN::simpi STATIC IMPORTED)
    set_target_properties(ZOLTAN::simpi PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ZOLTAN_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${ZOLTAN_SIMPI_LIBRARY}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
# Sets zoltan_FOUND
find_package_handle_standard_args(zoltan DEFAULT_MSG
  ZOLTAN_INCLUDE_DIR ZOLTAN_LIBRARY ZOLTAN_SIMPI_LIBRARY)
