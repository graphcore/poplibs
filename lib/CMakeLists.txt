add_subdirectory(popsolver)
add_subdirectory(poplibs_support)
add_subdirectory(poputil)
add_subdirectory(popops)
add_subdirectory(poprand)
add_subdirectory(poplin)
add_subdirectory(popnn)
add_subdirectory(poplibs_test)
add_subdirectory(popfloat)

# If we are linking to Poplar in an SDK, rename all Poplibs libraries to
# ensure we use the local Poplibs libraries - not the ones from the SDK.
if (DEFINED ENV{POPLAR_SDK_ENABLED})
  set_target_properties(popsolver PROPERTIES OUTPUT_NAME popsolver_local)
  set_target_properties(poplibs_support PROPERTIES OUTPUT_NAME poplibs_support_local)
  set_target_properties(poputil PROPERTIES OUTPUT_NAME poputil_local)
  set_target_properties(popops PROPERTIES OUTPUT_NAME popops_local)
  set_target_properties(poprand PROPERTIES OUTPUT_NAME poprand_local)
  set_target_properties(poplin PROPERTIES OUTPUT_NAME poplin_local)
  set_target_properties(popnn PROPERTIES OUTPUT_NAME popnn_local)
  set_target_properties(poplibs_test PROPERTIES OUTPUT_NAME poplibs_test_local)
  set_target_properties(popfloat PROPERTIES OUTPUT_NAME popfloat_local)
endif()

# Libraries and executables that load Poplibs libraries should not need to
# pollute their link interface with Poplibs' private dependencies so all
# Poplibs libraries must have relative rpaths that find the correct libraries
# in all cases (that is build, install and package):
if (APPLE)
  set(CMAKE_INSTALL_RPATH
      "@loader_path/../${CMAKE_INSTALL_LIBDIR}"
      CACHE STRING "Relative RPATH for OSX." FORCE)
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH
      "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
      CACHE STRING "Relative RPATH for Unix systems." FORCE)
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/poplibs-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poplibs
        COMPONENT poplibs)
