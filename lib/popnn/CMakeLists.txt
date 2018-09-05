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
    poplar poplin poputil poplin ${CMAKE_DL_LIBS}
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

# Search for arch_man path
foreach(path ${CMAKE_PREFIX_PATH})
    if(EXISTS "${path}/colossus/include/tileimplconsts_tommy.h")
      set(ARCH_MAN_PATH ${path})
    endif()
endforeach(path)

set(codelet_asm_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/LossTransform.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MaxPoolingGrad_float.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/MaxPoolingGrad_half.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMaxClassGather.S
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/ReduceMaxClassSparse.S
)

# Create the pooling vertex definitions from the shared template.
foreach(DATA_TYPE float half)
  foreach(POOL_TYPE max avg sum)
    if("${POOL_TYPE}" STREQUAL "max")
      set(VERTEX_NAME "__runCodelet_popnn__MaxPooling___${DATA_TYPE}")
      set(OP "max")
    else()
      string(TOUPPER "${POOL_TYPE}" PTYPE_UPPER)

      # split across two lines for character length reasons!
      set(VERTEX_NAME_PREFIX "__runCodelet_popnn__ScaledSumPooling___\
${DATA_TYPE}_popnn__PoolingType__${PTYPE_UPPER}")
      set(VERTEX_NAME "${VERTEX_NAME_PREFIX}${VERTEX_NAME_SUFFIX}")
      set(OP "add")
    endif()

    if("${POOL_TYPE}" STREQUAL "max")
      if("${DATA_TYPE}" STREQUAL "float")
        # -std::numeric_limits<float>::infinity()
        set(IDENTITY_VALUE 0xFF800000)
      else()
        # std::numeric_limits<half>::lowest()
        set(IDENTITY_VALUE 0xFBFF)
      endif()
    else()
      # identity value for avg and sum pooling is 0.
      set(IDENTITY_VALUE 0)
    endif()

    if("${POOL_TYPE}" STREQUAL "avg")
      set(AVERAGE_RESULT 1)
    else()
      set(AVERAGE_RESULT 0)
    endif()

    if("${DATA_TYPE}" STREQUAL "half")
      set(BIT_WIDTH 16)
    else()
      set(BIT_WIDTH 32)
    endif()

    # calc function will be something like "f16v4max"
    set(CALC_FUNC "f${BIT_WIDTH} ## n ## ${OP} dst, dst, src")

    set(INPUT_FILE
      ${CMAKE_CURRENT_SOURCE_DIR}/codelets/detail/Pooling_${DATA_TYPE}.S.in)
    set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/codelets/${VERTEX_NAME}.S)

    configure_file(${INPUT_FILE} ${OUTPUT_FILE})
    list(APPEND codelet_asm_sources ${OUTPUT_FILE})
  endforeach()
endforeach()

# Generate non-linearity assembly implementation variants
foreach(VERTEX_TYPE Supervisor 2D GradSupervisor Grad2D)
  set(VARIANT_TEMPLATE "codelets/NonLinearity${VERTEX_TYPE}.S.in")
  foreach(NL_TYPE sigmoid relu tanh)
    set(VARIANT_NAME "NonLinearity${VERTEX_TYPE}_${NL_TYPE}")
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/codelets/${VARIANT_NAME}.S")
    string(TOUPPER "${NL_TYPE}" NL_TYPE_UPPER)
    configure_file(${VARIANT_TEMPLATE} ${VARIANT_SRC})
    list(APPEND codelet_asm_sources ${VARIANT_SRC})
  endforeach()
endforeach()

set(codelet_cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/codelets/popnnCodelets.cpp
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
    -I ${ARCH_MAN_PATH}/colossus/include
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
