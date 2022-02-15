# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
add_library(popnn_mock SHARED
  codelets.cpp
)

target_link_libraries(popnn_mock
  PRIVATE
    gccs
    GTest::gtest
    GTest::gmock
    Boost::boost
)

target_include_directories(popnn_mock
  PUBLIC
    $<TARGET_PROPERTY:popnn,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS popnn_mock
        COMPONENT popnn_mock
        EXPORT popnn_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT popnn_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popnn_mock
        FILE popnn_mock-targets.cmake
        COMPONENT popnn_mock)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popnn_mock
        COMPONENT popnn_mock
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
