include(GNUInstallDirs)

add_library(popnn SHARED
  codelets.cpp
  BatchNorm.cpp
  Loss.cpp
  Lstm.cpp
  Pooling.cpp
  popnnCycleEstimators.cpp
  NonLinearity.cpp
  PerformanceEstimation.hpp
  Recurrent.cpp
  ${CMAKE_SOURCE_DIR}/include/popnn/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/BatchNorm.hpp  
  ${CMAKE_SOURCE_DIR}/include/popnn/Loss.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Recurrent.hpp  
  ${CMAKE_SOURCE_DIR}/include/popnn/Lstm.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Pooling.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/NonLinearity.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/Compiler.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/gcd.hpp
  ${CMAKE_SOURCE_DIR}/include/poplibs_support/VectorUtils.hpp
)

target_link_libraries(popnn
  PUBLIC
    poplar popconv poputil poplin ${CMAKE_DL_LIBS}
  PRIVATE
    Boost::boost
)

target_include_directories(popnn
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

add_subdirectory(codelets)
add_dependencies(popnn popnn_codelets)

install(TARGETS popnn
        EXPORT popnn
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popnn
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popnn
        FILE popnn-targets.cmake
        COMPONENT popnn)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popnn
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popnn)
