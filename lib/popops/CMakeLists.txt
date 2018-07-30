include(GNUInstallDirs)

find_package(Boost REQUIRED)

get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_library(popops SHARED
  ScaledAdd.cpp
  AllTrue.cpp
  Cast.cpp
  CircBuf.cpp
  codelets.cpp
  DynamicSlice.cpp
  ElementWise.cpp
  Encoding.cpp
  Expr.cpp
  ExprOpUtil.cpp
  ExprOpUtil.hpp
  Pad.cpp
  Padder.cpp
  popopsCycleEstimators.cpp
  Zero.cpp
  ${CMAKE_SOURCE_DIR}/include/popops/codelets.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ScaledAdd.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/AllTrue.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Cast.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/CircBuf.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/DynamicSlice.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ElementWise.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Encoding.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Expr.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/ExprOp.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Pad.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Reduce.hpp
  ${CMAKE_SOURCE_DIR}/include/popops/Zero.hpp

  reduction/CycleEstimationFunctions.cpp
  reduction/CycleEstimationFunctions.hpp
  reduction/IntermediatePartials.cpp
  reduction/IntermediatePartials.hpp
  reduction/IntermediatePartialsUtil.cpp
  reduction/IntermediatePartialsUtil.hpp
  reduction/Reduction.cpp
  reduction/Reduction.hpp
  reduction/ReductionConnection.cpp
  reduction/ReductionConnection.hpp
  reduction/ReductionDebug.hpp
  reduction/ReductionPlan.cpp
  reduction/ReductionPlan.hpp
  reduction/ReductionStages.cpp
  reduction/ReductionStages.hpp
  reduction/ReductionVertex.hpp
  reduction/RegionWrapping.cpp
  reduction/RegionWrapping.hpp
)

target_link_libraries(popops
  PUBLIC
    poplar poputil ${CMAKE_DL_LIBS}
  PRIVATE
    poplibs_support Boost::boost
)

target_include_directories(popops
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)



set(codelet_cpp_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/dynamicSliceCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/elementwiseCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/encodingCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/miscCodelets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/reduceCodelets.cpp
)

set(codelet_asm_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/Clamp.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/select_bool.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/select_half.S
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/select_int_float.S
)

set(codelet_headers
  # Add any header files used here
)

add_custom_command(
  OUTPUT
    popops.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
    -o popops.gp
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
    popops_c.gp
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS} -DPOPLIBS_DISABLE_ASM_CODELETS
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}/codelets/
    -o popops_c.gp
    ${codelet_cpp_sources}
  DEPENDS
    ${codelet_cpp_sources}
    ${codelet_headers}
    popc_bin
)

add_custom_target(popops_codelets DEPENDS popops.gp popops_c.gp
                  SOURCES ${codelet_cpp_sources} ${codelet_asm_sources})
add_dependencies(popops popops_codelets)

install(TARGETS popops
        EXPORT popops
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popops
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT popops
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/popops
        FILE popops-targets.cmake
        COMPONENT popops)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/popops
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT popops)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popops.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popops)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popops_c.gp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT popops)
