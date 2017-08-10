include(GNUInstallDirs)

add_library(poplin SHARED
  codelets.cpp
  MatMul.cpp
  ${CMAKE_SOURCE_DIR}/include/poplin/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poplin/MatMul.hpp
)

target_link_libraries(poplin poplar popconv popreduce popstd)

target_include_directories(poplin
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

install(TARGETS poplin
        EXPORT poplin
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poplin
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT poplin
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poplin
        FILE poplin-targets.cmake
        COMPONENT poplin)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/poplin
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poplin)
