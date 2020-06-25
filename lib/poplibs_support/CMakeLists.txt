include(GNUInstallDirs)

# We actually want Spdlog 0.16.3 but the official release has incorrect version number 0.16.2.
find_package(spdlog 0.16.2 REQUIRED)

add_library(poplibs_support STATIC
  Algorithms.cpp
  codelets.cpp
  ContiguousRegionsByTile.cpp
  IclUtil.cpp
  logging.cpp
  PlanConstraints.cpp
  TileHierarchy.cpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/Algorithms.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/Compiler.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/ContiguousRegionsByTile.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/cyclesTables.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/gcd.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/IclUtil.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/logging.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/PlanConstraints.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/print.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/StructHelper.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/TileConstants.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/TileHierarchy.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/VectorUtils.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/vv_iterator.hpp
)

set_property(TARGET poplibs_support PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(poplibs_support
  PRIVATE
    Boost::boost poplar ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(poplibs_support
  PUBLIC
    ${SPDLOG_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)
