include(GNUInstallDirs)

find_package(poplar REQUIRED)
find_package(Boost REQUIRED)

add_library(poplibs_support STATIC
  IclUtil.cpp
)

target_link_libraries(poplibs_support
  PRIVATE
    Boost::boost poplar
)

target_include_directories(poplibs_support
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)
