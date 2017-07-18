include(GNUInstallDirs)

# Util.cpp depends on poplar. TODO remove this dependency
find_package(poplar REQUIRED)

add_library(poplib_test SHARED
  Convolution.cpp
  FullyConnected.cpp
  GeneralMatrixAdd.cpp  
  GeneralMatrixMultiply.cpp
  Lstm.cpp
  Pooling.cpp
  NonLinearity.cpp
  Pass.cpp
  Rnn.cpp
  Util.cpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/Convolution.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/FullyConnected.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/GeneralMatrixAdd.hpp  
  ${PROJECT_SOURCE_DIR}/include/poplib_test/GeneralMatrixMultiply.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/Lstm.hpp  
  ${PROJECT_SOURCE_DIR}/include/poplib_test/Pooling.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/NonLinearity.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/Pass.hpp
  ${PROJECT_SOURCE_DIR}/include/poplib_test/Rnn.hpp  
  ${PROJECT_SOURCE_DIR}/include/poplib_test/Util.hpp
)

target_link_libraries(poplib_test poplar)

target_include_directories(poplib_test
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
