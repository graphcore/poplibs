include(GNUInstallDirs)

find_package(spdlog 1.8.0 REQUIRED)
add_library(poplibs_support STATIC
  Algorithms.cpp
  codelets.cpp
  ContiguousRegionsByTile.cpp
  forceInterleavedEstimates.cpp
  IclUtil.cpp
  logging.cpp
  StridedRegions.cpp
  TestDevice.cpp
  TileHierarchy.cpp
  TraceChannels.cpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/Algorithms.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/Compiler.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/ContiguousRegionsByTile.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/CTCInferenceDefs.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/forceInterleavedEstimates.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/gcd.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/IclUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/LogArithmetic.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/logging.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/print.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/StridedRegions.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/StructHelper.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/TestDevice.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/TileHierarchy.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/Tracepoint.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/VectorUtils.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/vv_iterator.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/FlopEstimation.hpp
)

set_property(TARGET poplibs_support PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(poplibs_support PUBLIC
    spdlog::spdlog_header_only
    libpvti
  PRIVATE
    Boost::boost
    poplar
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
)

target_include_directories(poplibs_support
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)
