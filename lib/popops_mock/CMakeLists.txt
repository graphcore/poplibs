# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
add_library(popops_mock SHARED
  ElementWise.cpp
  ElementWiseUtil.cpp
  codelets.cpp
)

target_link_libraries(popops_mock
  PRIVATE
    gccs
    GTest::gtest
    GTest::gmock
    Boost::boost
)

target_include_directories(popops_mock
  PUBLIC
    $<TARGET_PROPERTY:popops,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS popops_mock
        COMPONENT popops_mock
        EXPORT popops_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT popops_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popops_mock
        FILE popops_mock-targets.cmake
        COMPONENT popops_mock)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popops_mock
        COMPONENT popops_mock
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
