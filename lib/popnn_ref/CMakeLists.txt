include(GNUInstallDirs)

# Util.cpp depends on poplar. TODO remove this dependency
find_package(poplar REQUIRED)

add_library(popnn_ref SHARED
  Convolution.cpp
  FullyConnected.cpp
  MaxPooling.cpp
  NonLinearity.cpp
  Util.cpp
  ${PROJECT_SOURCE_DIR}/include/popnn_ref/Convolution.hpp
  ${PROJECT_SOURCE_DIR}/include/popnn_ref/FullyConnected.hpp
  ${PROJECT_SOURCE_DIR}/include/popnn_ref/MaxPooling.hpp
  ${PROJECT_SOURCE_DIR}/include/popnn_ref/NonLinearity.hpp
  ${PROJECT_SOURCE_DIR}/include/popnn_ref/Util.hpp
)

target_link_libraries(popnn_ref poplar)

target_include_directories(popnn_ref
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
