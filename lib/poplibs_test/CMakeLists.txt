include(GNUInstallDirs)

add_library(poplibs_test SHARED
  Convolution.cpp
  CTCLoss.cpp
  CTCInference.cpp
  CTCUtil.cpp
  Embedding.cpp
  FullyConnected.cpp
  GeneralMatrixAdd.cpp
  GeneralMatrixMultiply.cpp
  Gru.cpp
  Lstm.cpp
  Multirate.cpp
  NonLinearity.cpp
  Norms.cpp
  Pass.cpp
  Pooling.cpp
  Rnn.cpp
  Util.cpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Convolution.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/CTCLoss.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/CTCInference.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/CTCUtil.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Embedding.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/FullyConnected.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/GeneralMatrixAdd.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/GeneralMatrixMultiply.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Gru.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Lstm.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/MatrixTransforms.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Multirate.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/NonLinearity.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Norms.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Pass.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Pooling.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Rnn.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/SparseMatrix.hpp
  ${PROJECT_SOURCE_DIR}/include/poplibs_test/Util.hpp
)

target_link_libraries(poplibs_test
  PUBLIC
    poplar
    poplar_test
    poputil
    Boost::boost
    spdlog::spdlog_header_only
)

target_include_directories(poplibs_test
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS poplibs_test
        DESTINATION internal_tools
        COMPONENT internal_tools)
