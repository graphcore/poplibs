include(GNUInstallDirs)
find_package(spdlog 1.8.0 REQUIRED)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(poputil SHARED
  DebugInfo.cpp
  Broadcast.cpp
  exceptions.cpp
  GraphFunction.cpp
  TensorUseTracker.cpp
  TensorMetaData.cpp
  TileMapping.cpp
  VarStructure.cpp
  Util.cpp
  ${CMAKE_SOURCE_DIR}/include/poputil/DebugInfo.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/Broadcast.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/exceptions.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/GraphFunction.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/Loop.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/OptionParsing.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/TensorMetaData.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/TileMapping.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/VarStructure.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/VertexTemplates.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/Util.hpp
)

target_link_libraries(poputil
  PUBLIC
    poplar
  PRIVATE
    poplibs_support
    Boost::boost
    spdlog::spdlog_header_only
)

target_include_directories(poputil
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

install(TARGETS poputil
        EXPORT poputil
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poputil
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT poputil
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poputil
        FILE poputil-targets.cmake
        COMPONENT poputil)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/poputil
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poputil)
