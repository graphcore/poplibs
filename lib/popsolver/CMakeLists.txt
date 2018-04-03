include(GNUInstallDirs)

find_package(Boost REQUIRED)

add_library(popsolver SHARED
  Constraint.cpp
  Constraint.hpp
  Model.cpp
  Scheduler.cpp
  Scheduler.hpp
  ${CMAKE_SOURCE_DIR}/include/popsolver/Model.hpp
  ${CMAKE_SOURCE_DIR}/include/popsolver/Variable.hpp
)

target_link_libraries(popsolver PRIVATE Boost::boost)

target_include_directories(popsolver
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

install(TARGETS popsolver
        EXPORT popsolver
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popsolver
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popsolver
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popsolver
        FILE popsolver-targets.cmake
        COMPONENT popsolver)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popsolver
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popsolver)
