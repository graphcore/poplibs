include(GNUInstallDirs)

# By default, poplar_packaging/SuperProjectConfig.cmake sets the
# CMAKE_INSTALL_RPATH of everything to $ORIGIN/../lib. When an executable (or
# library) is linked with a library with that rpath, the rpath is copied to
# the executable (normally the path to the executable is added as an rpath
# in the executable). So when someone links with poplin, $ORIGIN/../lib
# is added to their rpath, and libpoplin.so can be found at runtime.
#
# Using $ORIGIN/../lib means tools can find the library (they are installed in
# /bin, so /bin/../lib find the library). However poplin depends on TBB,
# which is not in $ORIGIN/../lib, until it is packaged. We add
# $ORIGIN/../../tbb/lib to the rpath so that executables that link with
# poplin can find TBB when run from the build or install trees.
#
if (APPLE)
  set(CMAKE_INSTALL_RPATH
      "@loader_path/../${CMAKE_INSTALL_LIBDIR};@loader_path/../../tbb/${CMAKE_INSTALL_LIBDIR}"
      CACHE STRING "Relative RPATH for OSX." FORCE)
elseif (UNIX)
  set(CMAKE_INSTALL_RPATH
      "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../../tbb/${CMAKE_INSTALL_LIBDIR}"
      CACHE STRING "Relative RPATH for Unix systems." FORCE)
endif()

add_subdirectory(poplibs_support)
add_subdirectory(popsolver)
add_subdirectory(poputil)
add_subdirectory(popops)
add_subdirectory(poprand)
add_subdirectory(poplin)
add_subdirectory(popnn)
add_subdirectory(poplibs_test)
add_subdirectory(popfloat)

add_subdirectory(poplin_mock)
add_subdirectory(poputil_mock)
add_subdirectory(popops_mock)

if(${zoltan_FOUND})
  add_subdirectory(popsparse)
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/poplibs-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poplibs
        COMPONENT poplibs)
