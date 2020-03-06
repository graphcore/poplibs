# Search for zoltan include directory and libraries
# This module defines the following variables:
#   - ZOLTAN_INCLUDE_DIR
#   - ZOLTAN_LIB_DIR

set(SEARCH_PATHS ${CMAKE_PREFIX_PATH})
list(APPEND SEARCH_PATHS $ENV{CMAKE_PREFIX_PATH})

foreach(path ${SEARCH_PATHS})
  if(EXISTS "${path}/include/zoltan.h")
    set(ZOLTAN_INCLUDE_DIR ${path}/include)
    set(zoltan_FOUND ON)
  endif()

  if(EXISTS "${path}/lib/libzoltan.a")
    set(ZOLTAN_LIB_DIR ${path}/lib)
    set(zoltan_FOUND ON)
  endif()
endforeach(path)
