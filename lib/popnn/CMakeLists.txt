include(GNUInstallDirs)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

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
    poplibs_support Boost::boost
)

target_include_directories(popnn
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popnnCodelets.cpp
)

set(codelet_asm_sources
  # Assembly to be added
)
set(codelet_headers
  PerformanceEstimation.hpp
)

add_custom_command(
  OUTPUT
    popnn.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -o popnn.gp
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_asm_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_command(
  OUTPUT
    popnn_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS} -DPOPLIBS_DISABLE_ASM_CODELETS
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -o popnn_c.gp
    ${codelet_cpp_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_target(popnn_codelets DEPENDS popnn.gp popnn_c.gp
                  SOURCES ${codelets_cpp_sources} ${codelet_asm_sources})
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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popnn.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popnn_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popnn)
