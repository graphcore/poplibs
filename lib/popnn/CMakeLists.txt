include(GNUInstallDirs)
find_package(Boost REQUIRED program_options)
find_package(poplar REQUIRED)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popnn SHARED
  ActivationMapping.cpp
  Convolution.cpp
  ConvPlan.cpp
  ConvReuse.cpp
  ConvUtil.cpp
  ConvUtil.hpp
  ConvValidation.hpp
  ConvValidation.cpp
  DimShuffle.hpp
  DimShuffle.cpp
  FullyConnected.cpp
  FullyConnectedPlan.cpp
  gcd.hpp
  MaxPool.cpp
  Net.cpp
  NonLinearity.cpp
  Winograd.cpp
  PerformanceEstimation.hpp
  VertexTemplates.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/ActivationMapping.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Convolution.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/ConvPlan.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/exceptions.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/FullyConnected.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/FullyConnectedPlan.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/internal/ConvReuse.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/MaxPool.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/Net.hpp
  ${CMAKE_SOURCE_DIR}/include/popnn/NonLinearity.hpp
)

target_link_libraries(popnn poplar ${Boost_LIBRARIES})

target_include_directories(popnn
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(POPC_FLAGS -I ${CMAKE_CURRENT_SOURCE_DIR} -I ${CMAKE_SOURCE_DIR}/include)
if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
  list(APPEND POPC_FLAGS -g)
elseif(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)
  list(APPEND POPC_FLAGS -O3 -g)
else()
  list(APPEND POPC_FLAGS -O3)
endif()

add_custom_command(
  OUTPUT
    popnn.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o popnn.gp
    ${CMAKE_CURRENT_SOURCE_DIR}/popnn.cpp
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
