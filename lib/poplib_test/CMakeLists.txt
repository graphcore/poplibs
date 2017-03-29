include(GNUInstallDirs)

# Util.cpp depends on poplar. TODO remove this dependency
find_package(poplar REQUIRED)

add_library(poplib_test SHARED
  Convolution.cpp
  FullyConnected.cpp
  GeneralMatrixMultiply.cpp
  MaxPooling.cpp
  NonLinearity.cpp
  Util.cpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/Convolution.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/FullyConnected.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/GeneralMatrixMultiply.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/MaxPooling.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/NonLinearity.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/Util.hpp
)

target_link_libraries(poplib_test poplar)

target_include_directories(poplib_test
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
