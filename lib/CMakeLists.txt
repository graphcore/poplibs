add_subdirectory(popstd)
add_subdirectory(popreduce)
add_subdirectory(poprand)
add_subdirectory(popconv)
add_subdirectory(poplin)
add_subdirectory(popnn)
add_subdirectory(popsolver)
add_subdirectory(poplib_test)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/poplibs-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poplibs
        COMPONENT poplibs)
