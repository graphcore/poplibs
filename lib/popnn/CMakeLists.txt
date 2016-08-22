include(GNUInstallDirs)
find_package(Boost REQUIRED)

add_library(popnn SHARED
  ActivationMapping.cpp
  Convolution.cpp
  ConvPlan.cpp
  ConvReuse.cpp
  ConvUtil.cpp
  ConvUtil.hpp
  FullyConnected.cpp
  FullyConnectedPlan.cpp
  gcd.hpp
  MaxPool.cpp
  Net.cpp
  NonLinearity.cpp
  PerformanceEstimation.hpp
  VertexTemplates.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/ActivationMapping.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Convolution.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/ConvPlan.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/DeviceInfo.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/exceptions.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/FullyConnected.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/FullyConnectedPlan.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/internal/ConvReuse.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/MaxPool.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Net.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/NonLinearity.hpp
)

target_include_directories(popnn
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

add_custom_command(
  OUTPUT
    popnn.gp
  COMMAND
    popc -O3 -I ${CMAKE_CURRENT_SOURCE_DIR} -I ${CMAKE_SOURCE_DIR}/include -g ${CMAKE_CURRENT_SOURCE_DIR}/popnn.cpp -o popnn.gp
  DEPENDS
    popnn.cpp
    PerformanceEstimation.hpp
    ${CMAKE_SOURCE_DIR}/include/popnn/NetDef.hpp
    ${CMAKE_SOURCE_DIR}/include/popnn/ConvDef.hpp
    ${CMAKE_SOURCE_DIR}/include/popnn/NonLinearityDef.hpp
)

add_custom_target(graph_program DEPENDS popnn.gp)
add_dependencies(popnn graph_program)

install(TARGETS popnn
        EXPORT popnn
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popnn DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popnn
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popnn)      
