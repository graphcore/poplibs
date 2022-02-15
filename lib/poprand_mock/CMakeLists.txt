# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
add_library(poprand_mock SHARED
  codelets.cpp
)

target_link_libraries(poprand_mock
  PRIVATE
    gccs
    GTest::gtest
    GTest::gmock
    Boost::boost
)

target_include_directories(poprand_mock
  PUBLIC
    $<TARGET_PROPERTY:poprand,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS poprand_mock
        COMPONENT poprand_mock
        EXPORT poprand_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT poprand_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poprand_mock
        FILE poprand_mock-targets.cmake
        COMPONENT poprand_mock)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/poprand_mock
        COMPONENT poprand_mock
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
