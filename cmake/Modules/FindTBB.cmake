# Locate Intel Threading Building Blocks (TBB) library
# This modules defines the following imported target:
#  TBB::TBB - The TBB library, if found
#  TBB::malloc - The TBB memory allocator library, if found
# This module defines the following variables:
#  TBB_FOUND - System has TBB library
#  TBB_INCLUDE_DIRS - The TBB include directories
#  TBB_LIBRARIES - The libraries needed to use TBB

# You can supply a hint of where to find the TBB library by setting TBB_ROOT or
# $ENV{TBB_ROOT}

find_path(TBB_INCLUDE_DIRS tbb/tbb_stddef.h
          HINTS ${TBB_ROOT}/${CMAKE_INSTALL_LIBDIR}
                $ENV{TBB_ROOT}/include)
find_library(TBB_LIBRARY tbb
             HINTS ${TBB_ROOT}/${CMAKE_INSTALL_LIBDIR}
                   $ENV{TBB_ROOT}/${CMAKE_INSTALL_LIBDIR})
find_library(TBB_MALLOC_LIBRARY tbbmalloc
             HINTS ${TBB_ROOT}/${CMAKE_INSTALL_LIBDIR}
                   $ENV{TBB_ROOT}/${CMAKE_INSTALL_LIBDIR})
set(TBB_LIBRARIES ${TBB_LIBRARY} ${TBB_MALLOC_LIBRARY})

if(TBB_INCLUDE_DIRS AND TBB_LIBRARY AND TBB_MALLOC_LIBRARY)
  if(NOT TARGET TBB::TBB)
    add_library(TBB::malloc UNKNOWN IMPORTED)
    set_target_properties(TBB::malloc PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${TBB_MALLOC_LIBRARY}")
    add_library(TBB::TBB UNKNOWN IMPORTED)
    set_target_properties(TBB::TBB PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES TBB::malloc
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${TBB_LIBRARY}")

    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
      # See https://github.com/intel/tbb/issues/146
      set_target_properties(TBB::malloc PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "TBB_USE_GLIBCXX_VERSION=50102")
      set_target_properties(TBB::TBB PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "TBB_USE_GLIBCXX_VERSION=50102")
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
# Sets TBB_FOUND
find_package_handle_standard_args(TBB DEFAULT_MSG TBB_LIBRARY
                                                  TBB_MALLOC_LIBRARY
                                                  TBB_INCLUDE_DIRS)

mark_as_advanced(TBB_INCLUDE_DIRS TBB_LIBRARIES)
