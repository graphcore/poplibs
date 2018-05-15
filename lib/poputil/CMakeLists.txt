include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

find_package(Boost REQUIRED)

add_library(poputil SHARED
  Broadcast.cpp
  GraphFunction.cpp
  TileMapping.cpp
  Util.cpp
  ${CMAKE_SOURCE_DIR}/include/poputil/Broadcast.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/exceptions.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/GraphFunction.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/TileMapping.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/Util.hpp
  ${CMAKE_SOURCE_DIR}/include/poputil/VertexTemplates.hpp
)

target_link_libraries(poputil
  PUBLIC
    poplar
  PRIVATE
    Boost::boost
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
