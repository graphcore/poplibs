include(GNUInstallDirs)

# Util.cpp depends on poplar. TODO remove this dependency
find_package(poplar REQUIRED)
find_package(Boost REQUIRED)

add_library(poplibs_test SHARED
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
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Convolution.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/FullyConnected.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/GeneralMatrixAdd.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/GeneralMatrixMultiply.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Lstm.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Pooling.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/NonLinearity.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Pass.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Rnn.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Util.hpp
)

target_link_libraries(poplibs_test
  PUBLIC
    poplar Boost::boost
)

target_include_directories(poplibs_test
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
