include(GNUInstallDirs)

find_package(poplar REQUIRED)
find_package(Boost REQUIRED)

add_library(poplibs_support SHARED
  IclUtil.cpp
)

target_link_libraries(poplibs_support PRIVATE Boost::boost poplar)

target_include_directories(poplibs_support
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

install(TARGETS poplibs_support
        EXPORT poplibs_support
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poplibs_support
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT poplibs_support
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poplibs_support
        FILE poplibs_support-targets.cmake
        COMPONENT poplibs_support)
