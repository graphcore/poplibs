include(GNUInstallDirs)

# Util.cpp depends on poplar. TODO remove this dependency
find_package(poplar REQUIRED)
find_package(Boost REQUIRED)

# Before cmake 3.6 imported targets are only created if you specify a list of
# components (or not at all before cmake 3.5). Workaround this by adding the
# imported target ourselves.
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
endif()

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

target_link_libraries(poplibs_test poplar Boost::boost)

target_include_directories(poplibs_test
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
