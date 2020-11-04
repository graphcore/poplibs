include(GNUInstallDirs)

add_library(popsolver STATIC
  Constraint.cpp
  Constraint.hpp
  Model.cpp
  Scheduler.cpp
  Scheduler.hpp
  ${CMAKE_SOURCE_DIR}/include/popsolver/Model.hpp
  ${CMAKE_SOURCE_DIR}/include/popsolver/Variable.hpp
)

set_property(TARGET popsolver PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(popsolver
  PRIVATE
    poplar
    poplibs_support
    poputil # Required because of T22741
    Boost::boost
    spdlog::spdlog_header_only
)

target_include_directories(popsolver
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)
