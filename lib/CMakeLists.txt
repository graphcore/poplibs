add_subdirectory(popsolver)
add_subdirectory(poplibs_support)
add_subdirectory(poputil)
add_subdirectory(popops)
add_subdirectory(poprand)
add_subdirectory(poplin)
add_subdirectory(popnn)
add_subdirectory(poplibs_test)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/poplibs-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poplibs
        COMPONENT poplibs)
