# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
add_library(poputil_mock SHARED TileMapping.cpp)

target_link_libraries(poputil_mock
  PRIVATE
    gccs
    GTest::gtest
    GTest::gmock
    Boost::boost
)

target_include_directories(poputil_mock
  PUBLIC
    $<TARGET_PROPERTY:poputil,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS poputil_mock
        COMPONENT poputil_mock
        EXPORT poputil_mock-export
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/poputil_mock
        COMPONENT poputil_mock
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
