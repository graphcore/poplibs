include(GNUInstallDirs)

get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popsparse SHARED
  BlockSparseMatMul.cpp
  BSMatrix.cpp
  ${CMAKE_SOURCE_DIR}/include/popsparse/experimental/BlockSparseMatMul.hpp
)

target_link_libraries(popsparse
  PUBLIC
    poplar poputil popops poplin ${CMAKE_DL_LIBS}
  PRIVATE
    poplibs_support 
)

target_include_directories(popsparse
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

install(TARGETS popsparse
        EXPORT popsparse
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popsparse
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popsparse
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popsparse
        FILE popsparse-targets.cmake
        COMPONENT popsparse)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popsparse/experimental
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/popsparse
        COMPONENT popsparse)
