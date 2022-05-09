include(GNUInstallDirs)

add_library(popsolver STATIC
  Constraint.cpp
  Constraint.hpp
  GraphUtil.hpp
  Model.cpp
  Scheduler.cpp
  Scheduler.hpp
  ${CMAKE_SOURCE_DIR}/include/popsolver/Model.hpp
  ${CMAKE_SOURCE_DIR}/include/popsolver/PriorityGroupID.hpp
  ${CMAKE_SOURCE_DIR}/include/popsolver/Variable.hpp
)

# Enable auto vectorization and common microarchitecture extensions available
# in modern x86-64 processors
TargetAddMarchExtensions(popsolver)

set_property(TARGET popsolver PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(popsolver
  PUBLIC
    gccs
    Boost::boost
  PRIVATE
    poplar
    spdlog::spdlog_header_only
)

target_include_directories(popsolver
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)
