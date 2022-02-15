# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
add_library(popfloat_mock SHARED
  codelets.cpp
)

target_link_libraries(popfloat_mock
  PRIVATE
    gccs
    GTest::gtest
    GTest::gmock
    Boost::boost
)

target_include_directories(popfloat_mock
  PUBLIC
    $<TARGET_PROPERTY:popfloat,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS popfloat_mock
        COMPONENT popfloat_mock
        EXPORT popfloat_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT popfloat_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popfloat_mock
        FILE popfloat_mock-targets.cmake
        COMPONENT popfloat_mock)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popfloat_mock
        COMPONENT popfloat_mock
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
