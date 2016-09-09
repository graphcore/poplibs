include(GNUInstallDirs)
find_package(Boost REQUIRED)

add_library(popnn_ref SHARED
  Convolution.cpp
  NonLinearity.cpp
  ${PROJECT_SOURCE_DIR}/include/popnn_ref/Convolution.hpp
  ${PROJECT_SOURCE_DIR}/include/popnn_ref/NonLinearity.hpp
)

target_include_directories(popnn_ref
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${Boost_INCLUDE_DIRS}
)
