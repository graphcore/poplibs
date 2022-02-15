# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
add_library(poplin_mock SHARED MatMul.cpp)

target_link_libraries(poplin_mock
  PRIVATE
    gccs
    GTest::gtest
    GTest::gmock
    Boost::boost
)

target_include_directories(poplin_mock
  PUBLIC
    $<TARGET_PROPERTY:poplin,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS poplin_mock
        COMPONENT poplin_mock
        EXPORT poplin_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT poplin_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poplin_mock
        FILE poplin_mock-targets.cmake
        COMPONENT poplin_mock)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/poplin_mock
        COMPONENT poplin_mock
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
