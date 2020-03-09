find_package(Boost 1.65.1 REQUIRED unit_test_framework timer system program_options)
find_package(PythonInterp 3)

find_package(Threads REQUIRED)

#Controls verbose debug output on creating test cases
set(DEBUG_TEST_GENERATION FALSE)

# ipu1 is our default arch, so convert ipu1->ipu
list(TRANSFORM ENABLED_IPU_ARCH_NAMES REPLACE "ipu1" "ipu" OUTPUT_VARIABLE ABBREV_TARGETS)
list(TRANSFORM ABBREV_TARGETS REPLACE "ipu" "Sim" OUTPUT_VARIABLE SIM_VARIANTS)
list(TRANSFORM SIM_VARIANTS APPEND ":cpp" OUTPUT_VARIABLE SIMCPP_VARIANTS)
list(TRANSFORM ABBREV_TARGETS REPLACE "ipu" "IpuModel" OUTPUT_VARIABLE IPUMODEL_VARIANTS)

set(DEFAULT_TEST_VARIANTS "Cpu;${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw" CACHE STRING
    "The device variants to run tests on when no variants are set explicitly")

if (DEFINED ENV{POPLAR_SDK_ENABLED})
set(ENABLED_TEST_VARIANTS "NoTarget;Cpu;${IPUMODEL_VARIANTS}" CACHE STRING
      "Generate tests for these devices and variants")
else()
set(ENABLED_TEST_VARIANTS "NoTarget;Cpu;${IPUMODEL_VARIANTS};${SIM_VARIANTS};${SIMCPP_VARIANTS}" CACHE STRING
      "Generate tests for these devices and variants")
endif()

add_subdirectory(popsolver)

# Compile the public headers using C++11.
add_test(
  NAME ApiCppVersionTest
  COMMAND ${CMAKE_CXX_COMPILER} -std=c++11 ${TEST_CXXFLAGS}
          ${CMAKE_CURRENT_SOURCE_DIR}/ApiCppVersionTest.cpp -lpoplar -Werror
)

# Function to add a multi-target test executable
function(add_multi_target_test_executable name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/tests
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poplin,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name}
    popfloat poprand popnn poputil poplar poplibs_test
    Boost::program_options ${CMAKE_THREAD_LIBS_INIT})
  if(TARGET popsparse)
    target_link_libraries(${name} popsparse)
  endif()
endfunction()

# Remove any requested variants not present in ${ENABLED_TEST_VARIANTS}
# If no variants were requested filtered ${DEFAULT_TEST_VARIANTS} are returned.
function(sanitise_variants variant_name)
  set(variants ${${variant_name}})
  set(old_variants "${variants}")
  if (NOT old_variants)
    set(old_variants ${DEFAULT_TEST_VARIANTS})
  endif()
  set(new_variants "")
  # Only allow each variant if it is enabled:
  foreach(variant ${old_variants})
    if (${variant} IN_LIST ENABLED_TEST_VARIANTS)
      list(APPEND new_variants ${variant})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES new_variants)
  set(${variant_name} "${new_variants}" PARENT_SCOPE)
endfunction()

function(add_test_executable name)
  add_executable(${name} ${ARGN})
  if (NOT TEST_TARGET STREQUAL "NoTarget")
    set_property(TARGET ${name}
                 APPEND_STRING PROPERTY
                 COMPILE_FLAGS "-DTEST_TARGET=DeviceType::${TEST_TARGET}")
  endif()
  target_compile_options(${name} PRIVATE -Wno-deprecated-declarations)
  target_include_directories(${name}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/tests
      ${CMAKE_SOURCE_DIR}/lib
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poplin,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} popfloat poprand popnn poputil
    poplar poplibs_test Boost::unit_test_framework
    Boost::timer Boost::system Boost::program_options
    ${CMAKE_THREAD_LIBS_INIT})
  if(TARGET popsparse)
    target_link_libraries(${name} popsparse)
  endif()
  if (NOT Boost_UNIT_TEST_FRAMEWORK_LIBRARY MATCHES "\\.a$")
    target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  endif()
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts the "target" part storing it in the second argument.
function(extract_target variant target)
  string(REPLACE ":" ";" vspec ${${variant}})
  list(GET vspec 0 HEAD)
  set(${target} ${HEAD} PARENT_SCOPE)
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts the "runconfig" part storing it in the second argument.
function(extract_runconfig variant run_config)
  string(REPLACE ":" ";" vspec ${${variant}})
  list(LENGTH vspec vlen)
  if (${vlen} EQUAL "1")
    set(${run_config} "default" PARENT_SCOPE)
  else()
    list(GET vspec 1 config)
    set(${run_config} ${config} PARENT_SCOPE)
  endif()
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts a list containing just the "target" parts. Where targets are listed
# with multiple runconfigs in the original list the duplicates are removed
# from the returned list.
function(extract_targets variants targets)
  set(extracted_targets "")
  foreach(variant ${${variants}})
    string(REPLACE ":" ";" vspec ${variant})
    list(GET vspec 0 HEAD)
    list(APPEND extracted_targets ${HEAD})
  endforeach()
  list(REMOVE_DUPLICATES extracted_targets)
  set(${targets} ${extracted_targets} PARENT_SCOPE)
endfunction()

# If running tests on Ipu hardware is enabled then add a test fixture that
# checks if a physical IPU device is present. Hw tests will list this test as a
# "required fixture" and they will therefore not be run if no Ipu device is
# present at run time (and the fixture will be listed as a failure):
extract_targets(ENABLED_TEST_VARIANTS ENABLED_TEST_TARGETS)
if ("Hw" IN_LIST ENABLED_TEST_TARGETS)
  add_test_executable(EnumerateDevices EnumerateDevices.cpp)
  add_test(NAME CheckIpuHwAvailable
           COMMAND EnumerateDevices
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
  set_tests_properties(CheckIpuHwAvailable
                       PROPERTIES FIXTURES_SETUP IpuIsAvailable)
endif()

# Add a test to run with the specified VARIANTS (or all enabled VARIANTS if
# none are specified). The first parameter is the test name, the following are
# source files, and finally everything after VARIANTS are variant
# specifications.
#
# A variation specification must be of the form "target:[config]"
#
# Currently supported targets are "Cpu;IpuModel;Sim;Hw"
# Note that the Cpu target does not support multi-tile tests
# Currently supported configs are "cpp;default" which specify the codelet types
function(add_unit_test name)
  cmake_parse_arguments(add_unit_test "" "FILES" "VARIANTS;SUITES;LABELS" "FILES;${ARGN}")
  sanitise_variants(add_unit_test_VARIANTS)
  set(FILES "${add_unit_test_FILES}")
  extract_targets(add_unit_test_VARIANTS TARGETS)

  if(DEBUG_TEST_GENERATION)
    message(STATUS
            "Adding test '${name}' with variants: ${add_unit_test_VARIANTS}")
  endif()

  foreach(TEST_TARGET ${TARGETS})
    set(executable_name "${TEST_TARGET}_${name}")
    add_test_executable(${executable_name} ${FILES})
  endforeach()

  foreach(VARIANT ${add_unit_test_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    macro(add_unit_test_impl)
      set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")
      set(executable_name "${TEST_TARGET}_${name}")

      set(test_suite "")
      if (SUITE)
        string(APPEND test_name "_${SUITE}")
        string(APPEND test_suite "--run_test=${SUITE}/*")
      endif()

      add_test(NAME "${test_name}"
        COMMAND ${executable_name} ${test_suite}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
      if (add_unit_test_LABELS)
        set_tests_properties(${test_name}
            PROPERTIES LABELS "${add_unit_test_LABELS}")
      endif()

      set(test_env ${TEST_ENVIRONMENT})
      if (${TEST_CONFIG} STREQUAL "cpp")
        list(APPEND test_env "POPLIBS_CODELET_SUFFIX=_c")
      endif()

      set_tests_properties(${test_name} PROPERTIES
                           ENVIRONMENT "${test_env}")

      if (${TEST_TARGET} STREQUAL "Hw")
        # Make sure tests that use physical IPUs only run if an appropriate
        # number were available according to the relevant test fixture:
        set_tests_properties(${test_name}
          PROPERTIES FIXTURES_REQUIRED IpuIsAvailable)
      endif()
    endmacro()

    if (add_unit_test_SUITES)
      foreach(SUITE ${add_unit_test_SUITES})
        add_unit_test_impl()
      endforeach()
    else()
      add_unit_test_impl()
    endif()
  endforeach()
endfunction()

# add a test to run on the specified VARIANTS
function(add_multitarget_test)
  cmake_parse_arguments(add_multitarget_test "" "NAME"
                        "VARIANTS;COMMAND;LABELS"
                        "${ARGN}")
  sanitise_variants(add_multitarget_test_VARIANTS)
  set(name ${add_multitarget_test_NAME})
  if (NOT add_multitarget_test_NAME)
    message(FATAL_ERROR "add_multitarget_test requires a NAME to be specified")
  endif()

  if(DEBUG_TEST_GENERATION)
    message(STATUS
      "Adding multi target test '${name}' with "
      "variants: ${add_multitarget_test_VARIANTS}")
  endif()

  foreach(VARIANT ${add_multitarget_test_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")
    add_test(NAME "${test_name}"
      COMMAND ${add_multitarget_test_COMMAND} "--device-type=${TEST_TARGET}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)

    set(test_env ${TEST_ENVIRONMENT})
    if (${TEST_CONFIG} STREQUAL "cpp")
      list(APPEND test_env "POPLIBS_CODELET_SUFFIX=_c")
    endif()

    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "${test_env}")

    if (${TEST_TARGET} STREQUAL "Hw")
      # Make sure tests that use physical IPUs only run if an appropriate
      # number were available according to the relevant test fixture:
      set_tests_properties(${test_name}
        PROPERTIES FIXTURES_REQUIRED IpuIsAvailable)
    endif()

    if (add_multitarget_test_LABELS)
      set_tests_properties(${test_name}
        PROPERTIES LABELS "${add_multitarget_test_LABELS}")
    endif()
  endforeach()
endfunction()

function(add_benchmark)
  cmake_parse_arguments(BENCHMARK "" "NAME;BINARY;PARALLEL_LEVEL" "PARAMS" "${ARGN}")

  if (NOT DEFINED BENCHMARK_PARALLEL_LEVEL)
    set(BENCHMARK_LABEL "benchmarks")
  elseif(BENCHMARK_PARALLEL_LEVEL EQUAL 2)
    set(BENCHMARK_LABEL "benchmarks_j2")
  elseif(BENCHMARK_PARALLEL_LEVEL EQUAL 4)
    set(BENCHMARK_LABEL "benchmarks_j4")
  elseif(BENCHMARK_PARALLEL_LEVEL EQUAL 8)
    set(BENCHMARK_LABEL "benchmarks_j8")
  else()
    message(FATAL_ERROR "Cannot set a parallel level that is not 2, 4 or 8 as "
                        "buildbot is currently only configured to split the "
                        "tasks at that granularity.")
  endif()

  add_multitarget_test(
    NAME "${BENCHMARK_NAME}_benchmark"
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tools/bench.py
            --name ${BENCHMARK_NAME}
            ${BENCHMARK_BINARY} ${BENCHMARK_PARAMS}
    LABELS ${BENCHMARK_LABEL}
    #TODO T17190: enable for "${IPUMODEL_VARIANTS}")
    VARIANTS "IpuModel")
endfunction()

function(add_conv_benchmark)
  cmake_parse_arguments(CONV_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}" )

  # common arguments for all conv benchmarks
  list(APPEND CONV_BENCHMARK_PARAMS --bias 0 --ignore-data)
  add_benchmark(
      NAME ${CONV_BENCHMARK_NAME}
      PARALLEL_LEVEL ${CONV_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:single_conv_layer>
      PARAMS ${CONV_BENCHMARK_PARAMS})
endfunction()

function(add_fc_benchmark)
  cmake_parse_arguments(FC_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}")

  # common arguments for all fc benchmarks
  list(APPEND FC_BENCHMARK_PARAMS --bias 0 --ignore-data)
  add_benchmark(
      NAME ${FC_BENCHMARK_NAME}
      PARALLEL_LEVEL ${FC_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:fully_connected_layer>
      PARAMS ${FC_BENCHMARK_PARAMS})
endfunction()

function(add_embedding_benchmark)
  cmake_parse_arguments(EMBED_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}")

  # common arguments for all embedding benchmarks
  list(APPEND EMBED_BENCHMARK_PARAMS --ignore-data 1)
  add_benchmark(
      NAME ${EMBED_BENCHMARK_NAME}
      PARALLEL_LEVEL ${EMBED_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:embedding_layer>
      PARAMS ${EMBED_BENCHMARK_PARAMS})
endfunction()

function(add_gemm_benchmark)
  cmake_parse_arguments(GEMM_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}")

  # common arguments for all gemm benchmarks
  list(APPEND GEMM_BENCHMARK_PARAMS --ignore-data)
  add_benchmark(
      NAME ${GEMM_BENCHMARK_NAME}
      PARALLEL_LEVEL ${GEMM_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:general_matrix_multiply>
      PARAMS ${GEMM_BENCHMARK_PARAMS})
endfunction()

function(add_reduction_benchmark)
  cmake_parse_arguments(REDUCTION_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}" )

  list(APPEND REDUCTION_BENCHMARK_PARAMS --ignore-data)
  add_benchmark(
      NAME ${REDUCTION_BENCHMARK_NAME}
      PARALLEL_LEVEL ${REDUCTION_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:reduce_op>
      PARAMS ${REDUCTION_BENCHMARK_PARAMS})
endfunction()

function(add_pooling_benchmark)
  cmake_parse_arguments(POOLING_BENCHMARK "" "NAME;PARALLEL_LEVEL" "PARAMS" "${ARGN}" )

  list(APPEND POOLING_BENCHMARK_PARAMS --ignore-data)
  add_benchmark(
      NAME ${POOLING_BENCHMARK_NAME}
      PARALLEL_LEVEL ${POOLING_BENCHMARK_PARALLEL_LEVEL}
      BINARY $<TARGET_FILE:pooling_layer>
      PARAMS ${POOLING_BENCHMARK_PARAMS})
endfunction()

# Make an alias for IpuModel that can be used to denote tests that specifically
# should not be run on Sim variants because the test times out if run on the
# simulator:
set(TimesOutOnSim "${IPUMODEL_VARIANTS};Hw")

# Make an alias for IpuModel that can be used to denote tests that specifically
# should not be run on the Sim:cpp variant due to test timeouts:
set(TimesOutOnSimCpp "${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw")

# Stochastic rounding and determinism test
# Sim::cpp not supported for random number generation
add_unit_test(StochasticRound StochasticRound.cpp
              VARIANTS ${IPUMODEL_VARIANTS} ${SIM_VARIANTS} Hw)
configure_file(StochasticRound.cpp StochasticRound.cpp COPYONLY)

# Unit tests
add_unit_test(AlgorithmTest AlgorithmTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(ConvUtilTest ConvUtilTest.cpp)
add_unit_test(CopyToIpu CopyToIpu.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(DuplicateTensor DuplicateTensor.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(RangeTest RangeTest.cpp)
add_unit_test(ConvOptionsTest ConvOptionsTest.cpp)
add_unit_test(ConvPlanTest ConvPlanTest.cpp)
add_unit_test(ConvTest ConvTest.cpp)
add_unit_test(StdArithmeticTests StdArithmeticTests.cpp)
# GraphFunctionTest is variant-independent
add_unit_test(GraphFunctionTest GraphFunctionTest.cpp
              VARIANTS ${IPUMODEL_VARIANTS} ${SIM_VARIANTS})
add_unit_test(CloneNTest CloneNTest.cpp
              VARIANTS ${IPUMODEL_VARIANTS} ${SIM_VARIANTS})
add_unit_test(WinogradConvolution WinogradConv.cpp
              VARIANTS ${TimesOutOnSim})
add_unit_test(NonLinearityTest NonLinearityTest.cpp)
add_unit_test(BigNLVertices BigNLVertices.cpp)
add_unit_test(GraphProgLocationTest GraphProgLocationTest.cpp)
add_unit_test(LargeSplitRegionsTest LargeSplitRegionsTest.cpp)
add_unit_test(MeshGridTest MeshGridTest.cpp)
add_unit_test(SpatialSoftmaxTest SpatialSoftmaxTest.cpp)
add_unit_test(DynamicSliceTest DynamicSliceTest.cpp
              SUITES SingleDim MultiDim LargeBuffer Update Misc MultiSlice
                     MultiUpdate MultiUpdateSingles MultiUpdateMultiples)
add_unit_test(DynamicSliceTestCpu DynamicSliceTest.cpp
              SUITES CpuChecks
              VARIANTS Cpu)
add_unit_test(DynamicSliceCreation.cpp DynamicSliceCreation.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(CircBufTests CircBufTests.cpp)
add_unit_test(BroadcastToMatchTest BroadcastToMatchTest.cpp)
add_unit_test(LossTest LossTest.cpp
              SUITES ArgMinMax TopK SUM_SQUARED_LOSS_suite
                     CROSS_ENTROPY_LOSS_suite Accuracy)
add_unit_test(EncodingTest EncodingTest.cpp)
add_unit_test(GatherTest GatherTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(GatherSimpleTest GatherSimpleTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(AllTrueTest AllTrueTest.cpp)
add_unit_test(ScatterTest ScatterTest.cpp)
add_unit_test(ScatterUpdateTest ScatterUpdateTest.cpp)
add_unit_test(ReduceEdgeCases ReduceEdgeCases.cpp)
add_unit_test(PaddingTest PaddingTest.cpp)
add_unit_test(PlanConstraintsTest PlanConstraintsTest.cpp
              VARIANTS NoTarget)
add_unit_test(ElementWiseUtilTest ElementWiseUtilTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(TileMappingTest TileMappingTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(VarStructureTest VarStructureTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(SortTest SortTest.cpp)
add_unit_test(GraphReplication GraphReplication.cpp)
add_unit_test(MultiArrayTest MultiArrayTest.cpp VARIANTS NoTarget)
add_unit_test(SelectScalarFromRows SelectScalarFromRowsTest.cpp)
add_unit_test(NaNTest NaNTest.cpp)
add_unit_test(UpdateScalarInRows UpdateScalarInRowsTest.cpp)
add_unit_test(DynamicSlicePlanningTest DynamicSlicePlanningTest.cpp)
if(TARGET popsparse)
  add_unit_test(BlockSparseTest BlockSparseTest.cpp VARIANTS IpuModel Hw)
endif()

if (NOT "asan" IN_LIST SANITIZERS)
  # This test causes an internal error in address sanitizer.
  # See T13392.
  add_unit_test(ParallelTest ParallelTest.cpp)
endif()

# MapFusionTests
macro(add_map_fusion_test test)
  add_multitarget_test(NAME MapFusion_${test} COMMAND MapFusionTest --test ${test})
endmacro()

add_multi_target_test_executable(MapFusionTest MapFusionTest.cpp)
add_map_fusion_test(Abs)
add_map_fusion_test(Add)
add_map_fusion_test(And)
add_map_fusion_test(Atan2)
add_map_fusion_test(BitwiseAnd)
add_map_fusion_test(BitwiseNot)
add_map_fusion_test(BitwiseOr)
add_map_fusion_test(BitwiseXnor)
add_map_fusion_test(BitwiseXor)
add_map_fusion_test(Ceil)
add_map_fusion_test(Clamp)
add_map_fusion_test(Cos)
add_map_fusion_test(Divide)
add_map_fusion_test(Equal)
add_map_fusion_test(Exp)
add_map_fusion_test(Expm1)
add_map_fusion_test(Floor)
add_map_fusion_test(Gt)
add_map_fusion_test(Gte)
add_map_fusion_test(Inv)
add_map_fusion_test(IsFinite)
add_map_fusion_test(Log)
add_map_fusion_test(Log1p)
add_map_fusion_test(Lt)
add_map_fusion_test(Lte)
add_map_fusion_test(Max)
add_map_fusion_test(Min)
add_map_fusion_test(Mul)
add_map_fusion_test(Neg)
add_map_fusion_test(Not)
add_map_fusion_test(NotEqual)
add_map_fusion_test(Or)
add_map_fusion_test(Pow)
add_map_fusion_test(Rem)
add_map_fusion_test(Round)
add_map_fusion_test(Rsqrt)
add_map_fusion_test(Select)
add_map_fusion_test(Shl)
add_map_fusion_test(Shr)
add_map_fusion_test(ShrSE)
add_map_fusion_test(Sigmoid)
add_map_fusion_test(Signum)
add_map_fusion_test(Sin)
add_map_fusion_test(Sqrt)
add_map_fusion_test(Square)
add_map_fusion_test(Sub)
add_map_fusion_test(Tanh)

add_map_fusion_test(Fusion)
add_map_fusion_test(MissingPlaceholder)

# StdOperatorsTests
macro(add_std_operators_test test)
  add_multitarget_test(
    NAME StdOperators_${test}
    COMMAND StdOperatorsTest --test ${test})
endmacro()

add_multi_target_test_executable(StdOperatorsTest StdOperatorsTest.cpp)
add_std_operators_test(AbsFloat)
add_std_operators_test(AbsInt)
add_std_operators_test(AddFloat)
add_std_operators_test(Asin)
add_std_operators_test(Atan2Float)
add_std_operators_test(AddInt)
add_std_operators_test(BitwiseAndInt)
add_std_operators_test(BitwiseOrInt)
add_std_operators_test(BitwiseNotInt)
add_std_operators_test(BitwiseXorInt)
add_std_operators_test(BitwiseXnorInt)
add_std_operators_test(Ceil)
add_std_operators_test(Cos)
add_std_operators_test(CountLeadingZeros)
add_std_operators_test(DivideInt)
add_std_operators_test(DivideHalf)
add_std_operators_test(DivideFloat)
add_std_operators_test(EqualFloat)
add_std_operators_test(GreaterThanBool)
add_std_operators_test(GreaterThanEqualBool)
add_std_operators_test(LessThanBool)
add_std_operators_test(LessThanEqualBool)
add_std_operators_test(Exponent)
add_std_operators_test(ExponentMinus1)
add_std_operators_test(Floor)
add_std_operators_test(GreaterThanFloat)
add_std_operators_test(GreaterThanInt)
add_std_operators_test(GreaterThanEqualFloat)
add_std_operators_test(LessThanFloat)
add_std_operators_test(LessThanEqualFloat)
add_std_operators_test(Logarithm)
add_std_operators_test(Logarithm1Plus)
add_std_operators_test(LogicalAnd)
add_std_operators_test(LogicalNot)
add_std_operators_test(LogicalOr)
add_std_operators_test(MaxFloat)
add_std_operators_test(MaxInt)
add_std_operators_test(MinFloat)
add_std_operators_test(MinInt)
add_std_operators_test(Multiply)
add_std_operators_test(NotEqualFloat)
add_std_operators_test(NotEqualBool)
add_std_operators_test(NegateFloat)
add_std_operators_test(NegateInt)
add_std_operators_test(Popcount)
add_std_operators_test(Power)
add_std_operators_test(RemainderFloat)
add_std_operators_test(RemainderInt)
add_std_operators_test(ShiftLeftInt)
add_std_operators_test(ShiftRightInt)
add_std_operators_test(ShiftRightSignExtendInt)
add_std_operators_test(SignumFloat)
add_std_operators_test(SignumInt)
add_std_operators_test(Sin)
add_std_operators_test(Tan)
add_std_operators_test(Tanh)
add_std_operators_test(Square)
add_std_operators_test(SquareRoot)
add_std_operators_test(SubtractFloat)
add_std_operators_test(SubtractHalf)
add_std_operators_test(SubtractInt)
add_std_operators_test(RoundFloat)
add_std_operators_test(SelectFloat)
add_std_operators_test(SelectFloatLHSConst)
add_std_operators_test(SelectFloatRHSConst)
add_std_operators_test(SelectFloatLHSAndRHSConst)
add_std_operators_test(SelectHalfLHSAndRHSConst)
add_std_operators_test(SelectInt)
add_std_operators_test(BroadcastSelectorSelectInt)
add_std_operators_test(BroadcastSelectorSelectFloat)
add_std_operators_test(BroadcastSelectorSelectInPlaceInt)
add_std_operators_test(BroadcastSelectorSelectInPlaceFloat)
add_std_operators_test(ClampFloat)
add_std_operators_test(ClampFloatMinConst)
add_std_operators_test(ClampFloatMaxConst)
add_std_operators_test(ClampInt)
add_std_operators_test(ClampInPlaceFloat)
add_std_operators_test(BroadcastClampInt)
add_std_operators_test(BroadcastClampInPlaceInt)
add_std_operators_test(BroadcastClampFloat)
add_std_operators_test(BroadcastClampInPlaceFloat)
add_std_operators_test(BroadcastClampSingleElementSrcFloat)
add_std_operators_test(BinaryOutputMapChoice)
add_std_operators_test(TrinaryOutputMapChoice)
add_std_operators_test(AllTrueBad)
add_std_operators_test(AllTrue)
add_std_operators_test(IsFinite)
add_std_operators_test(Map)
add_std_operators_test(MapCast)
add_std_operators_test(MapCastInPlace)
add_std_operators_test(MapCastIntToFloat)
add_std_operators_test(MapMultiTensor)
add_std_operators_test(MapInPlace)
add_std_operators_test(MapInPlaceBroadcast)
add_std_operators_test(MapInferType)
add_std_operators_test(MapInferTypeCast)
add_std_operators_test(MapInferTypeEqual)
add_std_operators_test(MapInferTypeNot)
add_std_operators_test(AddInPlace)
add_std_operators_test(MapAllScalar)
add_std_operators_test(MapSomeScalar)
add_std_operators_test(BinaryConcat)
add_std_operators_test(UnaryConcat)
add_std_operators_test(MultiplyFloatInPlaceConstScalarTest)
add_std_operators_test(AddHalfConstScalarTest)

# Binary Operator tests - currently not used.
# add_multi_target_test_executable(BinaryOpTest BinaryOpTest.cpp)

# NonLinearity Sweep tests
add_multi_target_test_executable(NonLinearitySweepTest NonLinearitySweepTest.cpp)

add_multitarget_test(NAME NonLinearitySweepTest
  COMMAND NonLinearitySweepTest
  --nl-type gelu)

# NonLinearity Grad Sweep tests
add_multi_target_test_executable(NonLinearityGradSweepTest NonLinearityGradSweepTest.cpp)

add_multitarget_test(NAME NonLinearityGradSweepTest
  COMMAND NonLinearityGradSweepTest
  --nl-type gelu)

# Broadcast Optimise tests
add_multi_target_test_executable(BroadcastOptimiseTest BroadcastOptimiseTest.cpp)

add_multitarget_test(NAME BroadcastOptimiseTest_multiply_inplace
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,16,16}
  --operation MULTIPLY
  --dim=2
  --dim-shuffle={0,1,2}
  --tiles 16)

add_multitarget_test(NAME BroadcastOptimiseTest_multiply_inplace_2
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,16,16}
  --operation MULTIPLY
  --dim=0
  --dim-shuffle={0,1,2}
  --tiles 16)

add_multitarget_test(NAME BroadcastOptimiseTest_multiply_inplace_3
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,2,64}
  --operation MULTIPLY
  --dim=2
  --dim-shuffle={0,1,2}
  --tiles 4)

add_multitarget_test(NAME BroadcastOptimiseTest_subtract_inplace
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,2,16,16}
  --operation SUBTRACT
  --dim=3
  --dim-shuffle={3,2,1,0}
  --tiles 5)

add_multitarget_test(NAME BroadcastOptimiseTest_add_inplace
  COMMAND BroadcastOptimiseTest
  --data-type float
  --dims={2,16,16}
  --operation ADD
  --dim=1
  --dim-shuffle={0,1,2}
  --tiles 1)

add_multitarget_test(NAME BroadcastOptimiseTest_add_inplace_2
  COMMAND BroadcastOptimiseTest
  --data-type float
  --dims={2,8,64}
  --operation ADD
  --dim=1
  --dim-shuffle={0,1,2}
  --tiles 5)

add_multitarget_test(NAME BroadcastOptimiseTest_add_inplace_3
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={4,16,128}
  --operation ADD
  --dim=2
  --dim-shuffle={0,1,2}
  --tiles 3)

add_multitarget_test(NAME BroadcastOptimiseTest_multiply
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,8,4,16}
  --operation MULTIPLY
  --dim=0
  --in-place 0
  --dim-shuffle={0,1,2,3}
  --tiles 16)

add_multitarget_test(NAME BroadcastOptimiseTest_subtract
  COMMAND BroadcastOptimiseTest
  --data-type float
  --dims={2,16,16}
  --operation SUBTRACT
  --dim=2
  --in-place 0
  --dim-shuffle={2,1,0}
  --tiles 16)


add_multitarget_test(NAME BroadcastOptimiseTest_subtract_2
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,16,20}
  --operation SUBTRACT
  --dim=1
  --dim-shuffle={0,1,2}
  --tiles 1)

add_multitarget_test(NAME BroadcastOptimiseTest_add
  COMMAND BroadcastOptimiseTest
  --data-type float
  --dims={2,16,16}
  --operation ADD
  --dim=2
  --in-place 0
  --dim-shuffle={2,0,1}
  --tiles 16)

# Broadcast Patterns Generator
add_multi_target_test_executable(BroadcastGeneratePatterns BroadcastGeneratePatterns.cpp)

add_multitarget_test(NAME BroadcastGeneratePatterns_multiply_inplace
  COMMAND BroadcastGeneratePatterns
  --data-type half
  --pattern={3,2,5}
  --tiles 5
  --regions-per-tile 3
  --operation MULTIPLY
  --in-place 1)

add_multitarget_test(NAME BatchNormConv_Batch2_Dim28x28_Ch32_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=false
    --norm-type=BN)


add_multitarget_test(NAME BatchNormConv_Batch2_Dim28x28_Ch32_SmallEps_stable
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=true
    --norm-type=BN)

add_multitarget_test(NAME GroupNormConv_Batch2_Dim28x28_Ch32_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --num-groups=2
    --stable-algo-for-stats=false
    --norm-type=GN)

add_multitarget_test(NAME GroupNormConv_Batch2_Dim28x28_Ch32_SmallEps_stable
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --num-groups=2
    --stable-algo-for-stats=true
    --norm-type=GN)

add_multitarget_test(NAME GroupNormConv_Batch3_Dim4x4_Ch6_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={3,6,4,4}
    --num-groups=3
    --stable-algo-for-stats=true
    --norm-type=GN)

add_multitarget_test(NAME InstanceNormConv_Batch2_Dim28x28_Ch32_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=false
    --norm-type=IN)


add_multitarget_test(NAME InstanceNormConv_Batch2_Dim28x28_Ch32_SmallEps_stable
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=true
    --norm-type=IN)

add_multitarget_test(NAME LayerNormConv_Batch2_Dim28x28_Ch32_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=false
    --norm-type=LN)

add_multitarget_test(NAME LayerNormConv_Batch2_Dim28x28_Ch32_SmallEps_stable
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=true
    --norm-type=LN)

add_multitarget_test(NAME BatchNormConv_Batch4_Dim20x20_Ch32_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=24
    --dims={4,32,20,20}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormConv_Batch4_Dim20x20_Ch32_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=24
    --dims={4,32,20,20}
    --num-groups=16
    --norm-type=GN)

add_multitarget_test(NAME InstanceNormConv_Batch4_Dim20x20_Ch32_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=24
    --dims={4,32,20,20}
    --norm-type=IN)

add_multitarget_test(NAME LayerNormConv_Batch4_Dim20x20_Ch32_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=24
    --dims={4,32,20,20}
    --norm-type=LN)

add_multitarget_test(NAME BatchNormConv_Batch16_Dim7x7_Ch8
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,8,7,7}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormConv_Batch16_Dim7x7_Ch8
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,8,7,7}
    --num-groups=4
    --norm-type=GN)

add_multitarget_test(NAME InstanceNormConv_Batch16_Dim7x7_Ch8
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,8,7,7}
    --norm-type=IN)

add_multitarget_test(NAME LayerNormConv_Batch16_Dim7x7_Ch8
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,8,7,7}
    --norm-type=LN)

add_multitarget_test(NAME BatchNormConv_Batch1_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56,56}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormConv_Batch1_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56,56}
    --num-groups=2
    --norm-type=GN)

add_multitarget_test(NAME InstanceNormConv_Batch1_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56,56}
    --norm-type=IN)

add_multitarget_test(NAME LayerNormConv_Batch1_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56,56}
    --norm-type=LN)

add_multitarget_test(NAME BatchNormConv_Batch1_DataFhalf_field_dims_1
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56}
    --norm-type=BN)

add_multitarget_test(NAME BatchNormConv_Batch1_DataFhalf_field_dims_3
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,14,2,7}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormFc_Batch4_Acts2048
  COMMAND norm_layer
    --eps=0.001
    --unbiased-var-est=0
    --learning-rate=0.01
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={4,2048}
    --num-groups=64
    --stable-algo-for-stats=false
    --norm-type=GN)

add_multitarget_test(NAME GroupNormFc_Batch4_Acts2048_stable
  COMMAND norm_layer
    --eps=0.001
    --unbiased-var-est=0
    --learning-rate=0.01
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={4,2048}
    --num-groups=64
    --stable-algo-for-stats=true
    --norm-type=GN)

add_multitarget_test(NAME LayerNormFc_Batch4_Acts2048
  COMMAND norm_layer
    --eps=0.001
    --unbiased-var-est=0
    --learning-rate=0.01
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={4,2048}
    --norm-type=LN)


add_multitarget_test(NAME BatchNormFc_Batch16_Acts256_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,256}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormFc_Batch16_Acts256_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,256}
    --num-groups=16
    --norm-type=GN)

add_multitarget_test(NAME LayerNormFc_Batch16_Acts256_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,256}
    --norm-type=LN)

add_multitarget_test(NAME BatchNormFc_Batch8_Acts512_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormFc_Batch8_Acts512_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --num-groups=16
    --norm-type=GN)

add_multitarget_test(NAME LayerNormFc_Batch8_Acts512_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --num-groups=16
    --norm-type=LN)

add_multitarget_test(NAME BatchNormFc_Batch8_Acts512_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormFc_Batch8_Acts512_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --num-groups=8
    --norm-type=GN)

add_multitarget_test(NAME LayerNormFc_Batch8_Acts512_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --norm-type=LN)

add_multitarget_test(NAME InstanceNormCOnv_Batch2_field_dims_4
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,4,6,7,8,9}
    --norm-type=IN)


# Reduction tests
foreach(type half float)
  foreach(update true false)
    foreach(scale 1.0 1.2)
      foreach(op ADD SQUARE_ADD)
        add_multitarget_test(
          NAME reduce_op_no_reduce_${op}_${type}_${update}_${scale}
          COMMAND $<TARGET_FILE:reduce_op>
            "--shape=32,4,1"
            "--dims=2"
            --type=${type}
            --scale ${scale}
            --update ${update}
            --operation ${op}
            --tiles-per-ipu=1)
      endforeach()
    endforeach()
  endforeach()
endforeach()


foreach(type half float)
  foreach(update true false)
    foreach(scale 1.0 1.2)
      add_multitarget_test(
        NAME reduce_op_${partialsType}_${type}_${update}_${scale}
        COMMAND $<TARGET_FILE:reduce_op>
          "--shape=32,4,4"
          "--dims=0"
          --type=${type}
          --scale ${scale}
          --update ${update}
          --operation ADD
          --tiles-per-ipu=1)
    endforeach()
  endforeach()
endforeach()

foreach(operation ADD SQUARE_ADD)
  add_multitarget_test(
    NAME reduce_op_empty_dims_${operation}
    COMMAND $<TARGET_FILE:reduce_op>
      "--shape=32,4,4"
      --type=half
      --scale=1.2
      --update false
      --operation ${operation}
      --tiles-per-ipu=1)
endforeach()

foreach(type half float)
  foreach(update true false)
    foreach(scale 1.0 1.2)
      foreach(tiles 1 2)
        add_multitarget_test(
          NAME reduce_op_shuffle_${partialsType}_${type}_${update}_${scale}_${tiles}
          COMMAND $<TARGET_FILE:reduce_op>
            "--shape=128,8,2"
            "--initial-shape=256,4,2"
            "--shuffle=0,2,1"
            "--dims=0"
            --type=${type}
            --scale ${scale}
            --update ${update}
            --operation ADD
            --tiles-per-ipu=${tiles})
      endforeach()
    endforeach()
  endforeach()
endforeach()

add_multi_target_test_executable(ReductionTests ReductionTests.cpp)
add_multitarget_test(NAME Reduce_100x100_float_float_noupdate
  COMMAND ReductionTests
    --dims={100,100}
    --partials-type=float
    --out-type=float
    --k=1.0
    --update=false
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_10x200_half_half
  COMMAND ReductionTests
    --dims={10,200}
    --partials-type=half
    --out-type=half
    --k=2.0
    --update=false
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_scale_half_half
  COMMAND ReductionTests
    --dims={31,201}
    --partials-type=half
    --out-type=half
    --k=3.0
    --update=false
    --scale=true
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_scale_float_half
  COMMAND ReductionTests
    --dims={31,201}
    --partials-type=float
    --out-type=half
    --k=-1.5
    --update=false
    --scale=true
    --test=Add)

add_multitarget_test(NAME Reduce_1x201_scale_float_half
  COMMAND ReductionTests
    --dims={1,201}
    --partials-type=float
    --out-type=half
    --k=-1.5
    --update=false
    --scale=true
    --test=Add)

add_multitarget_test(NAME Reduce_1x201_scale_half_half
  COMMAND ReductionTests
    --dims={1,201}
    --partials-type=half
    --out-type=half
    --k=-1.5
    --update=false
    --scale=true
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_update_float_float
  COMMAND ReductionTests
    --dims={31,101}
    --partials-type=float
    --out-type=float
    --k=-1.5
    --update=true
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_update_half_half
  COMMAND ReductionTests
    --dims={31,101}
    --partials-type=half
    --out-type=half
    --k=2.0
    --update=true
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_update_float_half
  COMMAND ReductionTests
    --dims={31,101}
    --partials-type=float
    --out-type=half
    --k=-1.5
    --update=true
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_Add_float
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=float
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_Add_half
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=half
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_Add_int
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=int
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_SquareAdd_float
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=float
    --operation=SQUARE_ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_SquareAdd_half
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=half
    --operation=SQUARE_ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_SquareAdd_int
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=int
    --operation=SQUARE_ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_Mul_float
  COMMAND ReductionTests
    --dims={33,22,11}
    --red-vect={0}
    --out-type=float
    --operation=MUL
    --test=Ops)

add_multitarget_test(NAME Reduce_Mul_half
  COMMAND ReductionTests
    --dims={33,22,11}
    --red-vect={0}
    --out-type=half
    --operation=MUL
    --test=Ops)

add_multitarget_test(NAME Reduce_Mul_int
  COMMAND ReductionTests
    --dims={33,22,11}
    --red-vect={0}
    --out-type=int
    --operation=MUL
    --test=Ops)

add_multitarget_test(NAME Reduce_Max_float
  COMMAND ReductionTests
    --dims={20,30,40}
    --red-vect={0,1}
    --out-type=float
    --operation=MAX
    --test=Ops)

add_multitarget_test(NAME Reduce_Max_half
  COMMAND ReductionTests
    --dims={20,30,40}
    --red-vect={0,1}
    --out-type=half
    --operation=MAX
    --test=Ops)

add_multitarget_test(NAME Reduce_Max_int
  COMMAND ReductionTests
    --dims={20,30,40}
    --red-vect={0,1}
    --out-type=int
    --operation=MAX
    --test=Ops)

add_multitarget_test(NAME Reduce_Min_float
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=float
    --operation=MIN
    --test=Ops)

add_multitarget_test(NAME Reduce_Min_half
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=half
    --operation=MIN
    --test=Ops)

add_multitarget_test(NAME Reduce_Min_int
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=int
    --operation=MIN
    --test=Ops)

add_multitarget_test(NAME Reduce_And_bool
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=bool
    --operation=LOGICAL_AND
    --test=Ops)

add_multitarget_test(NAME Reduce_Or_bool
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=bool
    --operation=LOGICAL_OR
    --test=Ops)

add_multitarget_test(NAME Reduce_All_ADD_float
  COMMAND ReductionTests
    --dims={20,30,11}
    --red-vect={1,0,2}
    --out-type=float
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_None_ADD_float
  COMMAND ReductionTests
    --dims={20,30,11}
    --red-vect={}
    --out-type=float
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_Skip_ADD_float
  COMMAND ReductionTests
    --dims={1,1,11}
    --red-vect={0,1}
    --out-type=float
    --operation=ADD
    --test=Ops)

add_unit_test(ReplicatedAllToAll
              ReplicatedAllToAll.cpp
              VARIANTS Hw
              LABELS multicard;Collectives)

add_unit_test(ReplicatedReduceScatter
              ReplicatedReduceScatter.cpp
              VARIANTS Hw
              LABELS multicard;Collectives)

# Codelets
add_subdirectory(codelets)

if (PYTHONINTERP_FOUND AND PYTHON_VERSION_MAJOR EQUAL 3)
  # Random single_conv_layer tests with fixed seed.
  set(NUM_CONV_RANDOM_TESTS 50)
  foreach(n RANGE 1 ${NUM_CONV_RANDOM_TESTS})
  add_multitarget_test(
          NAME conv_random_${n}
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_SOURCE_DIR}/tools/single_conv_layer_random.py
                  --n 1
                  --seed ${n}
                  --binary $<TARGET_FILE:single_conv_layer>)
  endforeach()

  # Random single_conv_layer tests with fixed seed on multi ipus
  set(NUM_CONV_RANDOM_TESTS 25)
  foreach(n RANGE 1 ${NUM_CONV_RANDOM_TESTS})
    add_multitarget_test(
             NAME multi_ipu_conv_random_${n}
             COMMAND ${PYTHON_EXECUTABLE}
                     ${CMAKE_SOURCE_DIR}/tools/single_conv_layer_random.py
                     --n 1
                     --seed ${n}
                     --ipus 2
                     --binary $<TARGET_FILE:single_conv_layer>
             VARIANTS "Hw;${IPUMODEL_VARIANTS}")
    if ("Hw" IN_LIST ENABLED_TEST_TARGETS)
      set_tests_properties(Hw_default_multi_ipu_conv_random_${n} PROPERTIES RUN_SERIAL TRUE)
    endif()
  endforeach()

  # Some of the following benchmarks use quite a lot of host memory, too much
  # in fact for a buildslave (which currently have 32GB RAM and 16 cores) to
  # parallelise across the number of cores for all of these benchmarks. In this
  # case an optional PARALLEL_LEVEL argument is added that can specify that this
  # test must only be run with a -j flag of either 2, 4 or 8. If this option
  # does not exist then the buildbot will run the benchmark at a parallel
  # level of `nproc`.

  # These numbers must match those of bench.py
  set(GEMM_BENCHMARK_MS 200 600 1000)
  set(GEMM_BENCHMARK_KS 64 256 512)
  set(GEMM_BENCHMARK_NS 10000 20000 30000)

  # Matrix multiplication benchmarks with regression testing.
  foreach(m ${GEMM_BENCHMARK_MS})
    foreach(k ${GEMM_BENCHMARK_KS})
      foreach(n ${GEMM_BENCHMARK_NS})
        set(NAME "gemm_${m}x${k}x${n}")

        unset(GEMM_EXTRA_ARGS)
        if(NAME STREQUAL "gemm_600x256x30000" OR
           NAME STREQUAL "gemm_600x512x20000" OR
           NAME STREQUAL "gemm_600x512x30000" OR
           NAME STREQUAL "gemm_1000x256x20000" OR
           NAME STREQUAL "gemm_1000x256x30000" OR
           NAME STREQUAL "gemm_1000x512x20000" OR
           NAME STREQUAL "gemm_1000x512x30000")
          set(GEMM_EXTRA_ARGS PARALLEL_LEVEL 8)
        endif()

        add_gemm_benchmark(NAME ${NAME} PARAMS --m ${m} --k ${k} --n ${n} ${GEMM_EXTRA_ARGS})
      endforeach()
    endforeach()
  endforeach()

  # resnet50 conv layer benchmarks (batch size 4). each name is the enigma name
  # the first time that layer is seen.
  add_conv_benchmark(NAME resnet50_tr_bs4_cnv
          PARAMS --field {224,224}
                 --kernel-size 7
                 --stride 2
                 --padding 3
                 --input-channels 4
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm64L0_projection
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 256
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm64L0
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm64L0A1
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 64
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm64L1A0
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L0_projection
          PARAMS --field {56,56}
                 --kernel-size 1
                 --stride 2
                 --input-channels 256
                 --output-channels 512
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L0A0
          PARAMS --field {56,56}
                 --kernel-size 1
                 --stride 2
                 --input-channels 256
                 --output-channels 128
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L0A1
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L0A2
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 128
                 --output-channels 512
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm128L1A0
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 128
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L0_projection
          PARAMS --field {28,28}
                 --kernel-size 1
                 --stride 2
                 --input-channels 512
                 --output-channels 1024
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L0A0
          PARAMS --field {28,28}
                 --kernel-size 1
                 --stride 2
                 --input-channels 512
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L0A1
          PARAMS --field {14,14}
                 --kernel-size 3
                 --padding 1
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L0A2
          PARAMS --field {14,14}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 1024
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm256L1A0
          PARAMS --field {14,14}
                 --kernel-size 1
                 --input-channels 1024
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L0_projection
          PARAMS --field {14,14}
                 --kernel-size 1
                 --stride 2
                 --input-channels 1024
                 --output-channels 2048
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L0A0
          PARAMS --field {14,14}
                 --kernel-size 1
                 --stride 2
                 --input-channels 1024
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L0A1
          PARAMS --field {7,7}
                 --kernel-size 3
                 --padding 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L0A2
          PARAMS --field {7,7}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 2048
                 --batch-size 4)
  add_conv_benchmark(NAME resnet50_tr_bs4_bm512L1A0
          PARAMS --field {7,7}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 512
                 --batch-size 4)

  # inception conv layer benchmarks (batch size 4). each name is the enigma name
  # the first time that layer is seen.
  add_conv_benchmark(NAME inception_tr_bs4_layer1
          PARAMS --field {299,299}
                 --kernel-size 3
                 --stride 2
                 --input-channels 4
                 --output-channels 32
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME inception_tr_bs4_layer3
          PARAMS --field {149,149}
                 --kernel-size 3
                 --input-channels 32
                 --output-channels 32
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME inception_tr_bs4_layer5
          PARAMS --field {147,147}
                 --kernel-size 3
                 --padding 1
                 --input-channels 32
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME inception_tr_bs4_layer8
          PARAMS --field {147,147}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 80
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME inception_tr_bs4_layer10
          PARAMS --field {73,73}
                 --kernel-size 3
                 --input-channels 80
                 --output-channels 192
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_a1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 192
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_b1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 192
                 --output-channels 48
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_b5x5
          PARAMS --field {35,35}
                 --kernel-size 5
                 --padding 2
                 --input-channels 48
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME inception_tr_bs4_i1_db1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 192
                 --output-channels 32
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_c3x3a
          PARAMS --field {35,35}
                 --kernel-size 3
                 --padding 1
                 --input-channels 64
                 --output-channels 96
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i1_c3x3b
          PARAMS --field {35,35}
                 --kernel-size 3
                 --padding 1
                 --input-channels 96
                 --output-channels 96
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i2_a1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i2_b1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 48
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i3_a1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 288
                 --output-channels 64
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i3_b1x1
          PARAMS --field {35,35}
                 --kernel-size 1
                 --input-channels 288
                 --output-channels 48
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i4_a3x3
          PARAMS --field {35,35}
                 --kernel-size 3
                 --stride 2
                 --input-channels 288
                 --output-channels 384
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME inception_tr_bs4_i4_b3x3b
          PARAMS --field {35,35}
                 --kernel-size 3
                 --stride 2
                 --input-channels 96
                 --output-channels 95
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_a1x1
          PARAMS --field {17,17}
                 --kernel-size 1
                 --input-channels 768
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_b1x1
          PARAMS --field {17,17}
                 --kernel-size 1
                 --input-channels 768
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_b1x7
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_c7x1a
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_b7x1
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 128
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i5_c1x7b
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 128
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_b1x1
          PARAMS --field {17,17}
                 --kernel-size 1
                 --input-channels 768
                 --output-channels 160
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_b1x7
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 160
                 --output-channels 160
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_c7x1a
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 128
                 --output-channels 160
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_b7x1
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 160
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_c7x1b
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 160
                 --output-channels 160
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i6_c1x7b
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 160
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i7_b1x7
          PARAMS --field {17,17}
                 --kernel-size {1,7}
                 --padding {0,3}
                 --input-channels 192
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i7_b7x1
          PARAMS --field {17,17}
                 --kernel-size {7,1}
                 --padding {3,0}
                 --input-channels 192
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i8_a3x3
          PARAMS --field {17,17}
                 --kernel-size 3
                 --stride 2
                 --input-channels 192
                 --output-channels 320
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i8_b3x3
          PARAMS --field {17,17}
                 --kernel-size 3
                 --stride 2
                 --input-channels 192
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_a1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 1280
                 --output-channels 320
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_c1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 1280
                 --output-channels 448
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_ba1x3
          PARAMS --field {8,8}
                 --kernel-size {1,3}
                 --padding {0,1}
                 --input-channels 384
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_bb3x1
          PARAMS --field {8,8}
                 --kernel-size {3,1}
                 --padding {1,0}
                 --input-channels 384
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_d1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 1280
                 --output-channels 192
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i9_c3x3
          PARAMS --field {8,8}
                 --kernel-size 3
                 --padding 1
                 --input-channels 448
                 --output-channels 384
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME inception_tr_bs4_i10_a1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 320
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i10_b1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i10_c1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 448
                 --batch-size 4)
  add_conv_benchmark(NAME inception_tr_bs4_i10_d1x1
          PARAMS --field {8,8}
                 --kernel-size 1
                 --input-channels 2048
                 --output-channels 192
                 --batch-size 4)

  # vgg16 conv layer benchmarks (batch size 4). each name is the enigma name
  # the first time that layer is seen.
  add_conv_benchmark(NAME vgg16_tr_bs4_v1L0
          PARAMS --field {224,224}
                 --kernel-size 3
                 --padding 1
                 --input-channels 4
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME vgg16_tr_bs4_v1L1
          PARAMS --field {224,224}
                 --kernel-size 3
                 --padding 1
                 --input-channels 64
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 2)
  add_conv_benchmark(NAME vgg16_tr_bs4_v2L0
          PARAMS --field {112,112}
                 --kernel-size 3
                 --padding 1
                 --input-channels 64
                 --output-channels 128
                 --batch-size 4
          PARALLEL_LEVEL 2)
  add_conv_benchmark(NAME vgg16_tr_bs4_v2L1
          PARAMS --field {112,112}
                 --kernel-size 3
                 --padding 1
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4
          PARALLEL_LEVEL 2)
  add_conv_benchmark(NAME vgg16_tr_bs4_v3L0
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 128
                 --output-channels 256
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME vgg16_tr_bs4_v3L1
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME vgg16_tr_bs4_v4L0
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 256
                 --output-channels 512
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME vgg16_tr_bs4_v4L1
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME vgg16_tr_bs4_v5L0
          PARAMS --field {14,14}
                 --kernel-size 3
                 --padding 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4
          PARALLEL_LEVEL 8)

  # alexnet conv layer benchmarks (batch size 4). each name is the enigma name
  # the first time that layer is seen.
  add_conv_benchmark(NAME alexnet_tr_bs4_layer1
          PARAMS --field {224,224}
                 --kernel-size 11
                 --padding 3
                 --stride 4
                 --input-channels 4
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME alexnet_tr_bs4_layer4
          PARAMS --field {27,27}
                 --kernel-size 5
                 --padding 2
                 --input-channels 64
                 --output-channels 192
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME alexnet_tr_bs4_layer7
          PARAMS --field {13,13}
                 --kernel-size 3
                 --input-channels 192
                 --output-channels 384
                 --batch-size 4)
  add_conv_benchmark(NAME alexnet_tr_bs4_layer9
          PARAMS --field {13,13}
                 --kernel-size 3
                 --input-channels 384
                 --output-channels 256
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME alexnet_tr_bs4_layer11
          PARAMS --field {13,13}
                 --kernel-size 3
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4)

  # mobilenet conv layer benchmarks (batch size 4). each name is derived from
  # the first time that layer is seen in the public_examples tensorflow model.
  add_conv_benchmark(NAME mobilenet_conv1_1
          PARAMS --field {224,224}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 4
                 --output-channels 32
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_depthwise
          PARAMS --field {112,112}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 32
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_1_1
          PARAMS --field {112,112}
                 --kernel-size 1
                 --input-channels 32
                 --output-channels 64
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_depthwise_1
          PARAMS --field {112,112}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 64
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_2_1
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 64
                 --output-channels 128
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_depthwise_2
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 128
                 --batch-size 4
          PARALLEL_LEVEL 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_3_1
          PARAMS --field {56,56}
                 --kernel-size 1
                 --input-channels 128
                 --output-channels 128
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_depthwise_3
          PARAMS --field {56,56}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 128
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_conv_pw_4_1
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 128
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_4
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 256
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_conv_pw_5_1
          PARAMS --field {28,28}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 256
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_5
          PARAMS --field {28,28}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 256
                 --batch-size 4
          PARALLEL_LEVEL 8)
  add_conv_benchmark(NAME mobilenet_conv_pw_6_1
          PARAMS --field {14,14}
                 --kernel-size 1
                 --input-channels 256
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_6
          PARAMS --field {14,14}
                 --kernel-size 1
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 512
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_7_1
          PARAMS --field {14,14}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 512
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_11
          PARAMS --field {14,14}
                 --kernel-size 3
                 --padding-lower 1
                 --stride 2
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 512
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_12_1
          PARAMS --field {7,7}
                 --kernel-size 1
                 --input-channels 512
                 --output-channels 1024
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_depthwise_12
          PARAMS --field {7,7}
                 --kernel-size 3
                 --padding 1
                 --input-channels 1
                 --output-channels 1
                 --conv-groups 1024
                 --batch-size 4)
  add_conv_benchmark(NAME mobilenet_conv_pw_13_1
          PARAMS --field {7,7}
                 --kernel-size 1
                 --input-channels 1024
                 --output-channels 1024
                 --batch-size 4)
  add_conv_benchmark(NAME conv_5x200_1_in_100_out_bs1440
          PARAMS --field {15,200}
                 --kernel-size {5,200}
                 --input-channels 1
                 --output-channels 100
                 --batch-size 1440
          PARALLEL_LEVEL 4)

  # fc and embedding layer benchmarks (batch size 16). each name is derived from
  # the layer naming
  add_fc_benchmark(NAME fc_layer_1440x100x200
          PARAMS --batch-size 1440
                 --input-size 100
                 --output-size 200)
  add_fc_benchmark(NAME fc_layer_1440x200x400
          PARAMS --batch-size 1440
                 --input-size 200
                 --output-size 400)
  add_fc_benchmark(NAME fc_layer_16x1324x100
          PARAMS --batch-size 16
                 --input-size 1324
                 --output-size 100)
  add_fc_benchmark(NAME fc_layer_80_1324_100
          PARAMS --batch-size 80
                 --input-size 1324
                 --output-size 100)
  add_embedding_benchmark(NAME embedding_small
          PARAMS --shape {1000,200}
                 --num-indices 21600
          PARALLEL_LEVEL 8)
  add_embedding_benchmark(NAME embedding_vlarge
          PARAMS --shape {100000,200}
                 --num-indices 1440
          PARALLEL_LEVEL 8)

  # Small fully connected layers at low batch sizes (half and float).
  foreach(type half float)
    foreach(bs 1 4)
      add_fc_benchmark(NAME fc_layer_${bs}_1000_1000_${type}
                      PARAMS --batch-size ${bs}
                             --input-size 1000
                             --output-size 1000
                             --data-type=${type})
      add_fc_benchmark(NAME fc_layer_${bs}_1000_5_${type}
                       PARAMS --batch-size ${bs}
                              --input-size 1000
                              --output-size 5
                              --data-type=${type})
    endforeach()
  endforeach()

  # bert fc layer benchmarks (batch size 1). based off of the following the from
  # the model in enigma:
  #   KQV concat FC (S , H) * (H, 3*H)
  #   Grouped (first dim is group)  (B * K, S, 64) * (B * K, 64, S)
  #   Attention projection FC  (S , H) * (H, H)
  #   FFN1 FC (S , H) * (H, FX)
  #   FFN2 FC (S , FX) * (FX, H)
  #   where
  #     H is the set (768, 1024)
  #     K is (12, 16) depending on H
  #     FX is (3072, 4096) depending on H
  #     S is an independent dimension and can be (128, 384, 512)
  set(BERT_H 768 1024)
  set(BERT_K 12 16)
  set(BERT_FX 3072 4096)
  set(BERT_S 128 384 512)

  foreach(idx 0 1)
    list(GET BERT_H ${idx} h)
    list(GET BERT_K ${idx} k)
    list(GET BERT_FX ${idx} fx)
    math(EXPR h3 "3 * ${h}")
    foreach(s ${BERT_S})
      add_fc_benchmark(NAME bert_kqv_${s}x${h}x${h3}
              PARAMS --input-size ${h}
                     --output-size ${h3}
                     --batch-size ${s}
              PARALLEL_LEVEL 8)
      add_fc_benchmark(NAME bert_grouped_${k}x${s}x64x${s}
              PARAMS --input-size 64
                     --output-size ${s}
                     --batch-size ${s}
                     --num-groups ${k})
      add_fc_benchmark(NAME bert_proj_${s}x${h}x${h}
              PARAMS --input-size ${h}
                     --output-size ${s}
                     --batch-size ${h}
              PARALLEL_LEVEL 8)
      add_fc_benchmark(NAME bert_ffn1_${s}x${h}x${fx}
              PARAMS --input-size ${h}
                     --output-size ${fx}
                     --batch-size ${s}
              PARALLEL_LEVEL 8)
      add_fc_benchmark(NAME bert_ffn2_${s}x${fx}x${h}
              PARAMS --input-size ${fx}
                     --output-size ${h}
                     --batch-size ${s}
              PARALLEL_LEVEL 8)
    endforeach()
  endforeach()

  # inception layer pooling benchmarks. each name is the enigma name
  # the first time that layer is seen.
  add_pooling_benchmark(NAME inception_tr_bs1_pool1
          PARAMS --field {147,147}
                 --kernel {3,3}
                 --channels 64
                 --tiles-per-ipu 1216
                 --stride 2
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8)
  add_pooling_benchmark(NAME inception_tr_bs1_pool2
          PARAMS --field {71,71}
                 --kernel {3,3}
                 --channels 192
                 --tiles-per-ipu 1216
                 --stride 2
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8)
  add_pooling_benchmark(NAME inception_tr_bs1_i1_dmaxpool
          PARAMS --field {35,35}
                 --kernel {3,3}
                 --channels 192
                 --tiles-per-ipu 1216
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i2_dmaxpool
          PARAMS --field {35,35}
                 --kernel {3,3}
                 --channels 256
                 --tiles-per-ipu 1216
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i3_dmaxpool
          PARAMS --field {35,35}
                 --kernel {3,3}
                 --channels 288
                 --tiles-per-ipu 1216
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i4_cmaxpool
          PARAMS --field {35,35}
                 --kernel {3,3}
                 --channels 288
                 --tiles-per-ipu 1216
                 --stride 2
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8)
  add_pooling_benchmark(NAME inception_tr_bs1_i5_dmaxpool
          PARAMS --field {17,17}
                 --kernel {3,3}
                 --channels 768
                 --tiles-per-ipu 1216
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i6_dmaxpool
          PARAMS --field {17,17}
                 --kernel {3,3}
                 --channels 768
                 --tiles-per-ipu 1216
                 --stride 2
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8)
  add_pooling_benchmark(NAME inception_tr_bs1_i9_dmax_pool
          PARAMS --field {8,8}
                 --kernel {3,3}
                 --channels 1280
                 --tiles-per-ipu 1216
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 1
                 --padding-upper 1
                 --padding-lower 1)
  add_pooling_benchmark(NAME inception_tr_bs1_i10_dmax_pool
          PARAMS --field {8,8}
                 --kernel {3,3}
                 --channels 2048
                 --tiles-per-ipu 1216
                 --stride 1
                 --data-type half
                 --batch-size 1
                 --fwd-chans-per-group 8
                 --padding-upper 1
                 --padding-lower 1)

else()
  message(WARNING "Python 3 could not be found so several tests and benchmarks will not be run.")
endif()

# Based on Layer0/Attention/Z/Softmax/125/nonLinearityOutplace/Nonlinearity/SoftMax from BERT
add_reduction_benchmark(NAME bert_reduce_0
                        PARAMS --shape=16,4,16,32,8
                               --dims=1,3
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=MAX
                               --tiles-per-ipu=1024)

add_reduction_benchmark(NAME bert_reduce_1
                        PARAMS --shape=16,4,16,32,8
                               --dims=1,3
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1024)

# Based on Embedding/GroupNormalization/106/groupNormStatistics/Norm/statistics/mean/ConstScale from BERT
add_reduction_benchmark(NAME bert_reduce_2
                        PARAMS --shape=16,256,8,4
                               --dims=1,3
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1024)

# Based on Embedding/GroupNormalization/106/groupNormStatistics/Norm/statistics/power/ConstScale from BERT
add_reduction_benchmark(NAME bert_reduce_3
                        PARAMS --shape=16,256,8,4
                               --dims=1,3
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=SQUARE_ADD
                               --tiles-per-ipu=1024)

# Based on Layer0/FF/GroupNormalization/210/operandGrad/Norm/deltas/JointGammaDelta from BERT
add_reduction_benchmark(NAME bert_reduce_4
                        PARAMS --shape=128,16,8,16
                               --dims=0,2
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1216)

# Based on cnv-gn/Fwd/Norm/statistics/power/ConstScale from Resnet50
add_reduction_benchmark(NAME resnet50_tr_bs1_cnv_reduce
                        PARAMS --shape=4,25088,8
                               --dims=1
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=SQUARE_ADD
                               --tiles-per-ipu=1216)

# Based on bm64L0A0-gn/Fwd/Norm/statistics/mean/ConstScale from Resnet50
add_reduction_benchmark(NAME resnet50_tr_bs1_bm64L0A0_reduce
                        PARAMS --shape=4,6272,8
                               --dims=1
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1200)

# Based on bm128L0A0-gn/Fwd/Norm/statistics/mean/ConstScale from Resnet50
add_reduction_benchmark(NAME resnet50_tr_bs1_bm128L0A0_reduce
                        PARAMS --shape=4,3136,8
                               --dims=1
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1216)

# Based on bm128L0-gn/Fwd/Norm/statistics/mean/ConstScale from Resnet50
add_reduction_benchmark(NAME resnet50_tr_bs1_bm128L0_reduce
                        PARAMS --shape=4,12544,8
                               --dims=1
                               --type=half
                               --scale=1.0
                               --update=false
                               --operation=ADD
                               --tiles-per-ipu=1216)

# a few multi ipu tests that don't take too long that we can run on the Sim.
add_multitarget_test(NAME multi_ipu_conv_smoke_1
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={6,3}
                 --input-channels=4
                 --in-dilation={3,1}
                 --padding-upper={1,1}
                 --padding-lower={1,1}
                 --output-channels=16
                 --kernel-size={1,2}
                 --kernel-dilation={1,3}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={0,0}
                 --stride={2,2}
                 --tiles-per-ipu=28
                 --num-io-tiles=4
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\",\"startTileMultiplier\":\"0\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_2
         COMMAND single_conv_layer
                 --data-type=float
                 --conv-groups=1
                 --batch-size=2
                 --field={3,2}
                 --input-channels=12
                 --in-dilation={1,1}
                 --padding-upper={2,2}
                 --padding-lower={1,2}
                 --output-channels=19
                 --kernel-size={1,1}
                 --kernel-dilation={1,2}
                 --kernel-padding-upper={0,1}
                 --kernel-padding-lower={0,2}
                 --stride={4,2}
                 --tiles-per-ipu=26
                 --num-io-tiles=2
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\",\"startTileMultiplier\":\"34\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_3
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={12,3}
                 --input-channels=3
                 --in-dilation={1,1}
                 --padding-upper={1,1}
                 --padding-lower={2,1}
                 --output-channels=16
                 --kernel-size={5,3}
                 --kernel-dilation={1,1}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={2,2}
                 --stride={2,2}
                 --tiles-per-ipu=24
                 --num-io-tiles=0
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\",\"startTileMultiplier\":\"72\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_4
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={4,2}
                 --input-channels=16
                 --in-dilation={1,3}
                 --padding-upper={1,1}
                 --padding-lower={1,1}
                 --output-channels=3
                 --kernel-size={3,2}
                 --kernel-dilation={1,4}
                 --kernel-padding-upper={0,1}
                 --kernel-padding-lower={0,0}
                 --stride={1,1}
                 --tiles-per-ipu=17
                 --num-io-tiles=1
                 --enable-shared-structures
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\",\"startTileMultiplier\":\"8\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_5
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=3
                 --field={9,4}
                 --input-channels=4
                 --in-dilation={1,1}
                 --padding-upper={2,1}
                 --padding-lower={1,3}
                 --output-channels=1
                 --kernel-size={2,1}
                 --kernel-dilation={4,1}
                 --kernel-padding-upper={0,1}
                 --kernel-padding-lower={0,1}
                 --stride={2,1}
                 --tiles-per-ipu=26
                 --num-io-tiles=2
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\",\"startTileMultiplier\":\"58\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_6
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={9,1}
                 --input-channels=16
                 --in-dilation={2,1}
                 --padding-upper={1,2}
                 --padding-lower={1,2}
                 --output-channels=4
                 --kernel-size={1,2}
                 --kernel-dilation={1,1}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={1,0}
                 --stride={1,3}
                 --tiles-per-ipu=18
                 --num-io-tiles=2
                 --enable-shared-structures
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\",\"startTileMultiplier\":\"44\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_7
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={8,1}
                 --input-channels=46
                 --in-dilation={1,2}
                 --padding-upper={3,2}
                 --padding-lower={1,1}
                 --output-channels=4
                 --kernel-size={1,1}
                 --kernel-dilation={1,2}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={2,1}
                 --stride={1,1}
                 --tiles-per-ipu=20
                 --num-io-tiles=4
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\",\"startTileMultiplier\":\"44\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_8
         COMMAND single_conv_layer
                 --data-type=float
                 --conv-groups=1
                 --batch-size=1
                 --field={2,1}
                 --input-channels=4
                 --in-dilation={2,2}
                 --padding-upper={1,4}
                 --padding-lower={1,3}
                 --output-channels=20
                 --kernel-size={2,1}
                 --kernel-dilation={4,2}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={0,0}
                 --stride={1,2}
                 --tiles-per-ipu=1
                 --num-io-tiles=0
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\",\"startTileMultiplier\":\"32\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_9
         COMMAND single_conv_layer
                 --data-type=float
                 --conv-groups=1
                 --batch-size=1
                 --field={4,11}
                 --input-channels=4
                 --in-dilation={1,3}
                 --padding-upper={1,1}
                 --padding-lower={2,1}
                 --output-channels=4
                 --kernel-size={1,2}
                 --kernel-dilation={3,3}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={1,0}
                 --stride={1,2}
                 --tiles-per-ipu=17
                 --num-io-tiles=1
                 --enable-shared-structures
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\",\"startTileMultiplier\":\"32\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_10
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=2
                 --field={1,3}
                 --input-channels=16
                 --in-dilation={3,1}
                 --padding-upper={3,1}
                 --padding-lower={1,2}
                 --output-channels=16
                 --kernel-size={1,2}
                 --kernel-dilation={2,1}
                 --kernel-padding-upper={0,1}
                 --kernel-padding-lower={0,0}
                 --stride={1,2}
                 --tiles-per-ipu=28
                 --num-io-tiles=4
                 --enable-shared-structures
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\",\"startTileMultiplier\":\"8\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS}")

#Check reduction patterns.
add_test_executable (ReductionPatternsTest ReductionPatternsTest.cpp)
add_test(NAME ReductionPatternsTest
         COMMAND ReductionPatternsTest)

# Random reduce_op tests with fixed seed.
set(NUM_REDUCE_RANDOM_TESTS 150)
foreach(n RANGE 1 ${NUM_REDUCE_RANDOM_TESTS})
  add_multitarget_test(NAME reduce_random_${n}
           COMMAND $<TARGET_FILE:reduce_op> --seed ${n})
endforeach()

# Random reduce_op tests with fixed seed.
set(NUM_REDUCE_RANDOM_TESTS 15)
foreach(n RANGE 1 ${NUM_REDUCE_RANDOM_TESTS})
  add_multitarget_test(NAME multi_ipu_reduce_random_${n}
           COMMAND $<TARGET_FILE:reduce_op> --seed ${n} --ipus=2
           VARIANTS "Hw;${IPUMODEL_VARIANTS}")
  if ("Hw" IN_LIST ENABLED_TEST_TARGETS)
    set_tests_properties(Hw_default_multi_ipu_reduce_random_${n} PROPERTIES RUN_SERIAL TRUE)
  endif()
endforeach()

add_unit_test(collective-control-code
              collective-control-code.cpp
              VARIANTS Hw
              LABELS multicard;CE_reproducers;Collectives)

# Collective operations
foreach(collective reduce_scatter
                   all_gather
                   all_reduce)
  foreach(method bidirectional_ring_pair
                 meet_in_middle_ring
                 clockwise_ring
                 anticlockwise_ring)
    foreach(num_ipus 2 4 8 16)
      add_multitarget_test(
        NAME collective_${method}_${collective}_${num_ipus}ipus
        COMMAND collectives
                --reduction-operator=ADD
                --collective=${collective}
                --ipus=${num_ipus}
                --tiles-per-ipu=64
                --elements=1024
                --method=${method}
                --shuffle-mapping=true
        LABELS Collectives
        VARIANTS ${IPUMODEL_VARIANTS})
    endforeach()
  endforeach()
endforeach()

# Collectives where each rank consists of multiple IPUs.
foreach(collective all_reduce)
  foreach(ipus_per_rank 2 4 8)
    add_multitarget_test(
      NAME collective_${collective}_8_ipus_${ipus_per_rank}_ipus_per_rank
      COMMAND collectives
              --reduction-operator=ADD
              --collective=${collective}
              --ipus-per-rank=${ipus_per_rank}
              --ipus=8
              --tiles-per-ipu=64
              --elements=1024
              --shuffle-mapping=true
      LABELS Collectives
      VARIANTS ${IPUMODEL_VARIANTS})
  endforeach()
endforeach()

# Replicated collectives
foreach(collective all_reduce)
  foreach(method bidirectional_ring_pair
                 meet_in_middle_ring
                 clockwise_ring
                 anticlockwise_ring)
    foreach(num_ipus 2 4 8 16)
      add_multitarget_test(
        NAME replicated_collective_${method}_${collective}_${num_ipus}ipus
        COMMAND replicated_collectives
                --use-replicated-implementation
                --reduction-operator=ADD
                --collective=${collective}
                --ipus=${num_ipus}
                --tiles-per-ipu=64
                --elements=1024
                --method=${method}
                --shuffle-mapping=true
        LABELS Collectives
        VARIANTS ${IPUMODEL_VARIANTS})
    endforeach()
  endforeach()

  foreach(ipus_per_rank 2 4 8)
    add_multitarget_test(
      NAME replicated_collective_${collective}_8_ipus_${ipus_per_rank}_ipus_per_rank
      COMMAND replicated_collectives
              --use-replicated-implementation
              --reduction-operator=ADD
              --collective=${collective}
              --ipus-per-rank=${ipus_per_rank}
              --ipus=8
              --tiles-per-ipu=64
              --elements=1291
              --shuffle-mapping=true
      LABELS Collectives
      VARIANTS ${IPUMODEL_VARIANTS})
  endforeach()

  foreach(ipus_per_rank 1 2 4)
    foreach(method clockwise_ring
                   anticlockwise_ring
                   bidirectional_ring_pair
                   meet_in_middle_ring)
        add_multitarget_test(
        NAME replicated_collective_${collective}_8_ipus_${ipus_per_rank}_ipus_per_rank_${method}
        COMMAND replicated_collectives
                --use-replicated-implementation
                --replicate-top-level-graph=true
                --reduction-operator=ADD
                --collective=${collective}
                --method=${method}
                --ipus-per-rank=${ipus_per_rank}
                --ipus=8
                --tiles-per-ipu=64
                --elements=1399
                --shuffle-mapping=true
        LABELS Collectives multicard
        VARIANTS Hw)
    endforeach()
  endforeach()
endforeach()

add_multitarget_test(NAME replicated_collectives_few_elements
                     COMMAND replicated_collectives
                             --use-replicated-implementation
                             --reduction-operator=ADD
                             --collective=all_reduce
                             --ipus-per-rank=1
                             --ipus=2
                             --tiles-per-ipu=4
                             --elements=1
                             --shuffle-mapping=false
                      LABELS Collectives;CE_reproducers
                      VARIANTS ${IPUMODEL_VARIANTS})

add_multitarget_test(NAME replicated_collectives_force_mapping
                      COMMAND replicated_collectives
                              --use-replicated-implementation
                              --reduction-operator=ADD
                              --collective=all_reduce
                              --ipus-per-rank=2
                              --ipus=4
                              --tiles-per-ipu=4
                              --elements=1
                              --shuffle-mapping=false
                              --force-mapping=0
                       LABELS Collectives
                       VARIANTS ${IPUMODEL_VARIANTS})


foreach(operator ADD MUL MIN MAX)
  add_multitarget_test(
    NAME collective_${operator}_reduce_scatter
    COMMAND collectives
            --reduction-operator=${operator}
            --collective=reduce_scatter
            --ipus=4
            --tiles-per-ipu=64
            --elements=1024
            --method=auto
            --shuffle-mapping=true
    LABELS Collectives
    VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

# Test each method with multi ipu rank
foreach(collective all_reduce)
  foreach(method bidirectional_ring_pair
                 meet_in_middle_ring
                 clockwise_ring
                 anticlockwise_ring)
    add_multitarget_test(
      NAME collective_${collective}_${method}_2_ipus_per_rank
      COMMAND collectives
                --reduction-operator=ADD
                --collective=${collective}
                --ipus=8
                --ipus-per-rank=2
                --tiles-per-ipu=64
                --elements=1024
                --method=${method}
                --shuffle-mapping=true
        LABELS Collectives
        VARIANTS ${IPUMODEL_VARIANTS})

  endforeach()
endforeach()

# Test it collectives handles tensors with no elements on some ipus
add_multitarget_test(
      NAME collective_empty_ipus
      COMMAND collectives
                --reduction-operator=ADD
                --collective=all_reduce
                --ipus=4
                --ipus-per-rank=2
                --tiles-per-ipu=4
                --elements=1
                --method=meet_in_middle_ring
        LABELS Collectives
        VARIANTS ${IPUMODEL_VARIANTS})

# Basic convolution to test profiling option
add_multitarget_test(NAME conv_profile
         COMMAND single_conv_layer
         --input-channels=16
         --output-channels=16
         --field={1}
         --kernel-size=1
         --tiles-per-ipu=16
         --profile)

add_multitarget_test(NAME conv1x1_float_input_odd
         COMMAND single_conv_layer
         --single-phase=fwd
         --input-channels=37
         --output-channels=1
         --field={5,6,15}
         --kernel-size=1
         --data-type=float
         --tiles-per-ipu=12
         --padding=0
         --stride=1)

#  ORIG Layer tests
add_multitarget_test(NAME conv3x3_stride_2_128_out
         COMMAND single_conv_layer
                 --input-channels=48
                 --output-channels=128
                 --field={20,12}
                 --kernel-size=3
                 --data-type=half
                 --tiles-per-ipu=12
                 --padding=2
                 --stride=2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=128
             --output-channels=256
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_no_bias
         COMMAND single_conv_layer
            --input-channels=64
            --output-channels=32
            --field={7,7}
            --tiles-per-ipu=8
            --stride=1
            --kernel-size=1
            --bias=0
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
  NAME conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=128
            --output-channels=257
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=129
            --output-channels=256
            --field={7,7}
            --tiles-per-ipu=12
            --stride=1
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_3
         COMMAND single_conv_layer
             --input-channels=64
             --output-channels=128
             --field={14,14}
             --tiles-per-ipu=8
             --stride=3
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_thinly_spread
         COMMAND single_conv_layer
             --field={1,56}
             --input-channels=16
             --output-channels=8
             --tiles-per-ipu=24)

add_multitarget_test(
         NAME conv3x3_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=3
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=2
             --kernel-size=3)

add_multitarget_test(
         NAME conv3x3_asymmetric_stride
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,10}
             --tiles-per-ipu=16
             --stride={2,3}
             --kernel-size=3)

add_multitarget_test(
         NAME conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --padding=1
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --padding=2
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=2
                 --stride=2
                 --kernel-size=3)

add_multitarget_test(
         NAME conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=1
                 --stride=3
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_batch_2
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={7,7}
                  --tiles-per-ipu=2
                  --stride=2
                  --kernel-size=3
                  --batch-size=2)

add_multitarget_test(
         NAME conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={28,20}
        --kernel-size=3
        --data-type=half
        --padding={1,2}
        --tiles-per-ipu=16
        --stride=1
        VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={7,7}
        --tiles-per-ipu=2
        --stride=2
        --kernel-size=3
        --padding={1,2}
        --batch-size=2
        VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_1_asym_padding
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=24
                --field={20,12}
                --kernel-size=3
                --data-type=half
                --padding-lower=1
                --padding-upper=2
                --tiles-per-ipu=12
                --stride=1
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_1_asym_padding_2
         COMMAND single_conv_layer
                --input-channels=32
                --output-channels=40
                --field={20,12}
                --kernel-size=3
                --data-type=half
                --padding-lower=3
                --padding-upper=0
                --tiles-per-ipu=16
                --stride=1
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv5x5_stride_2_asym_padding
         COMMAND single_conv_layer
                  --input-channels=32
                  --output-channels=32
                  --field={20,12}
                  --kernel-size=5
                  --data-type=half
                  --padding-lower={1,2}
                  --padding-upper={3,3}
                  --stride=2
                  --tiles-per-ipu=12
                  VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --field={5,5}
                 --tiles-per-ipu=24
                 --convolution-options={\"partialsType\":\"float\"})

add_multitarget_test(
         NAME small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --kernel-size=3
                 --field={7,7}
                 --tiles-per-ipu=24
                 --convolution-options={\"partialsType\":\"float\"})

add_multitarget_test(
         NAME conv1x1_shallow_input
         COMMAND single_conv_layer
                 --input-channels=8
                 --output-channels=512
                 --field={8,8}
                 --data-type=half
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=1
                 --data-type=half
                 --tiles-per-ipu=2)

add_multitarget_test(
         NAME conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=2
                 --data-type=half
                 --tiles-per-ipu=2)

add_multitarget_test(
         NAME conv_large_kernel
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --stride=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_large_kernel_bwd
         COMMAND single_conv_layer
                 --single-phase=bwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --stride=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                 --field={1,16}
                 --padding={1,6}
                 --kernel-size={1,7}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_flip_input_and_kernel_1
         COMMAND single_conv_layer
                 --field={6,2}
                 --padding={0,1}
                 --flip-input={0,1}
                 --kernel-size={1,3}
                 --kernel-padding={0,0}
                 --flip-kernel={0,1}
                 --input-channels=2
                 --output-channels=1
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_flip_input_and_kernel_2
         COMMAND single_conv_layer
                 --field={1,32}
                 --padding={0,16}
                 --flip-input={0,1}
                 --kernel-size={1,48}
                 --flip-kernel={0,1}
                 --input-channels=32
                 --output-channels=8
                 --tiles-per-ipu=4
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_flip_input_only
         COMMAND single_conv_layer
                 --field={20,20}
                 --padding=1
                 --flip-input=1
                 --kernel-size=3
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=8
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_flip_kernel_only
         COMMAND single_conv_layer
                 --field={20,20}
                 --padding=1
                 --kernel-size=3
                 --flip-kernel=1
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=8
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=12
                 --data-type=half
                 --batch-size=4
                 --single-phase=bwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_float
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_group
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=256
                 --output-channels=64
                 --batch-size=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

# conv_batch_float test split into the three component phases
add_multitarget_test(
         NAME conv_batch_float_fwd
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=12
                 --data-type=float
                 --batch-size=4
                 --single-phase=fwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_float_bwd
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=12
                 --data-type=float
                 --batch-size=4
                 --single-phase=bwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_float_wu
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=12
                 --data-type=float
                 --batch-size=4
                 --single-phase=wu)

add_multitarget_test(
         NAME conv_negative_input_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_kernel_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,1}
                --kernel-size={1,2}
                --kernel-truncation-upper={0,1}
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_positive_and_negative_kernel_padding
         COMMAND single_conv_layer
         --input-channels=16
         --output-channels=16
         --field={5,5}
         --kernel-size=2
         --kernel-padding-lower=1
         --kernel-truncation-upper=1
         --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_input_padding_lower_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,2}
                 --kernel-size={1,1}
                 --truncation-upper={0,1}
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_kernel_padding_lower_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --kernel-truncation-lower={0,1}
                 --tiles-per-ipu=1)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_multitarget_test(
         NAME conv_large_output_tensor
         COMMAND single_conv_layer
                --field={16,16}
                --input-channels=16
                --output-channels=128
                --tiles-per-ipu=12
                --data-type=half
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_large_weights
         COMMAND single_conv_layer
                --field={14,16}
                --padding={2,0}
                --flip-input={1,0}
                --input-channels=64
                --output-channels=64
                --kernel-size={16,1}
                --flip-kernel={1,0}
                --tiles-per-ipu=16
                --data-type=half
                --single-phase=fwd
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_outer_product
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=384
                 --field={1,128}
                 --tiles-per-ipu=12
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_1_half_partials
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=16
                 --stride=1
                 --kernel-size=3
                 --convolution-options={\"partialsType\":\"half\"}
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_kernel_with_padding_larger_than_field
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --field={1,128}
                 --tiles-per-ipu=4
                 --padding={1,64})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=64
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=3
                 --stride=3
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor2
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=4
                 --stride=2
                 --tiles-per-ipu=12
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=2
                 --stride=4
                 --tiles-per-ipu=8
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor4
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field={14,14}
                --kernel-size={3,3}
                --in-dilation=4
                --stride=6
                --tiles-per-ipu=8
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor5
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field={14,14}
                --kernel-size={3,3}
                --in-dilation=6
                --stride=4
                --tiles-per-ipu=8
                VARIANTS ${TimesOutOnSim})

# Example where the amount of padding that must be added to start of the input
# window for the conv nx1 vertex exceeds the amount of lower padding for the
# layer.
add_multitarget_test(
         NAME conv_nx1_vertex_lower_padding
         COMMAND single_conv_layer
                 --field={16,32}
                 --input-channels=4
                 --padding-lower={1,0}
                 --output-channels=8
                 --kernel-size={2,1}
                 --kernel-dilation={2,1}
                 --stride={3,1}
                 --tiles-per-ipu=1
                 VARIANTS ${TimesOutOnSim})

# Test where, after padding and dilation, the last input index that would be
# multiplied by the a kernel element is in the middle of the lower input
# padding.
add_multitarget_test(
         NAME conv1x1_input_ignored_with_stride
         COMMAND single_conv_layer
                 --field={2,2}
                 --in-dilation=2
                 --padding-lower=2
                 --padding-upper=1
                 --stride=7
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=1)

# Test with large amounts of output padding
add_multitarget_test(
         NAME conv1x1_output_padding
         COMMAND single_conv_layer
                 --field={1,1}
                 --input-channels=16
                 --output-channels=16
                 --output-padding=10
                 --tiles-per-ipu=1)

# Convolution with output padding and non-view-only expandDims transform
add_multitarget_test(
         NAME conv1x20_expanddims_with_output_padding
         COMMAND single_conv_layer
                 --field={1,20}
                 --input-channels=4
                 --output-channels=4
                 --kernel-size={1,10}
                 --padding-lower={1,0}
                 --inference-only
                 --tiles-per-ipu=16
                 --fwd-plan-constraints={\"0\":{\"transform\":{\"expandDims\":[1,0]}}})

#  Grouped convolution layer tests
add_multitarget_test(
         NAME grouped_conv147x147_stride_1_8_out
         COMMAND single_conv_layer
                 --input-channels=4
                 --output-channels=8
                 --field={147,147}
                 --kernel-size={49,49}
                 --data-type=float
                 --padding=0
                 --conv-groups=2
                 --stride=1
                 --kernel-dilation={3,3}
                 --inference-only
                 VARIANTS Cpu)

add_multitarget_test(
         NAME grouped_conv3x3_stride_2_32_out
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={20,12}
                 --kernel-size=3
                 --data-type=half
                 --padding=2
                 --conv-groups=2
                 --stride=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=4
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME zero_conv_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=0
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=64
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=8
            --stride=1
            --conv-groups=2
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=128
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --conv-groups=2
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=12
            --stride=1
            --conv-groups=2
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_3_odd_conv_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={14,14}
             --tiles-per-ipu=16
             --stride=3
             --conv-groups=3
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_thinly_spread
         COMMAND single_conv_layer
             --field={1,128}
             --input-channels=16
             --conv-groups=2
             --output-channels=8
             --tiles-per-ipu=24)

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_odd_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=3
             --kernel-size=3
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=2
             --conv-groups=2
             --kernel-size=3
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_asymmetric_stride
         COMMAND single_conv_layer
             --input-channels=16
             --output-channels=8
             --field={7,10}
             --tiles-per-ipu=16
             --stride={2,3}
             --conv-groups=2
             --kernel-size=3)

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --conv-groups=2
              --padding=1
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=16
              --output-channels=8
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --conv-groups=2
              --padding=2
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=2
                 --stride=2
                 --conv-groups=2
                 --kernel-size=3
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=4
                 --conv-groups=2
                 --stride=3)

add_multitarget_test(
         NAME grouped_conv3x3_batch_2
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={7,7}
                  --tiles-per-ipu=4
                  --stride=2
                  --kernel-size=3
                  --conv-groups=2
                  --batch-size=2
                  VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={28,20}
        --kernel-size=3
        --data-type=half
        --padding={1,2}
        --conv-groups=2
        --stride=1
        --tiles-per-ipu=12
        VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={7,7}
        --tiles-per-ipu=4
        --stride=2
        --kernel-size=3
        --padding={1,2}
        --conv-groups=2
        --batch-size=2
        VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_asym_padding
        COMMAND single_conv_layer
                --input-channels=16
                --output-channels=24
                --field={20,12}
                --kernel-size=3
                --data-type=half
                --padding-lower=1
                --padding-upper=2
                --conv-groups=2
                --tiles-per-ipu=12
                --stride=1
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_asym_padding_2
         COMMAND single_conv_layer
                --input-channels=16
                --output-channels=24
                --field={20,12}
                --kernel-size=3
                --data-type=half
                --padding-lower=3
                --padding-upper=0
                --conv-groups=2
                --tiles-per-ipu=12
                --stride=1
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv5x5_stride_2_asym_padding
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={20,12}
                  --kernel-size=5
                  --data-type=half
                  --padding-lower={1,2}
                  --padding-upper={3,3}
                  --stride=2
                  --conv-groups=2
                  --tiles-per-ipu=12
                  VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --conv-groups=2
                 --tiles-per-ipu=12
                 --convolution-options={\"partialsType\":\"float\"}
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --kernel-size=3
                 --field={7,7}
                 --conv-groups=2
                 --tiles-per-ipu=20
                 --convolution-options={\"partialsType\":\"float\"}
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_shallow_input
        COMMAND single_conv_layer
                --input-channels=8
                --output-channels=384
                --field={8,8}
                --conv-groups=2
                --data-type=half
                --tiles-per-ipu=16
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=1
                 --conv-groups=2
                 --data-type=half
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME grouped_conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=2
                 --conv-groups=2
                 --data-type=half
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME grouped_conv_large_kernel_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=8
                 --padding=1
                 --conv-groups=2
                 --stride=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_large_kernel_bwd
         COMMAND single_conv_layer
                 --single-phase=bwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=8
                 --padding=1
                 --conv-groups=2
                 --stride=2
                 --tiles-per-ipu=20
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                --field={1,16}
                 --padding={6,1}
                 --kernel-size={7,1}
                 --input-channels=16
                 --output-channels=16
                 --conv-groups=2
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME grouped_conv_batch_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=half
                 --conv-groups=2
                 --batch-size=4
                 --single-phase=fwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_batch_float
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --conv-groups=2
                 --batch-size=2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_batch_group
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=256
                 --output-channels=64
                 --tiles-per-ipu=32
                 --conv-groups=2
                 --batch-size=2
                 VARIANTS ${TimesOutOnSim})

# conv_batch_bwd_float test split into the three component phases
add_multitarget_test(
         NAME grouped_conv_batch_float_fwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=fwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_batch_float_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=bwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_batch_float_wu
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=wu
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_negative_input_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --conv-groups=2
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_kernel_padding_upper_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --kernel-truncation-lower={0,1}
                 --conv-groups=2
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_input_padding_lower_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --conv-groups=2
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_kernel_padding_lower_simple_${phase}
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --conv-groups=2
                 --kernel-truncation-lower={0,1}
                 --tiles-per-ipu=1)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_multitarget_test(
         NAME grouped_conv_large_output_tensor
         COMMAND single_conv_layer
                --field={16,16}
                --input-channels=16
                --output-channels=64
                --tiles-per-ipu=12
                --conv-groups=2
                --data-type=half
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_large_weights
         COMMAND single_conv_layer
                --field={14,16}
                --padding={2,0}
                --flip-input={1,0}
                --input-channels=32
                --output-channels=16
                --kernel-size={16,1}
                --flip-kernel={1,0}
                --tiles-per-ipu=12
                --data-type=half
                --conv-groups=2
                --single-phase=fwd
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_outer_product
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=256
                 --field={1,128}
                 --conv-groups=2
                 --tiles-per-ipu=20
                 VARIANTS ${TimesOutOnSim})

# depthwise convolution with the combineConvGroups transformation applied tests.
set(DEPTHWISE_CONV_PLAN ${CMAKE_SOURCE_DIR}/tests/simple_depthwise_conv.json)
foreach(TYPE half float)
  foreach(GROUP 3 4 8)
    foreach(BS 1 2)
      set(CIs 1)
      if (TYPE STREQUAL half)
        list(APPEND CIs 2)
      endif()

      foreach(CI ${CIs})
        foreach(CO 1 2)
          foreach(IPUS 1 2)
            foreach(SIZE 2 3)
              if (SIZE EQUAL 2)
                set(FIELD "{16,16}")
              else()
                set(FIELD "{8,8,8}")
              endif()

              # a kernel size of more than 1 prevents the batch dimension from
              # being flattened.
              # TODO: T12985 Allow us to constrain flattenDims directly.
              add_multitarget_test(
                       NAME "depthwise_conv${SIZE}d_ipus${IPUS}_g${GROUP}_bs${BS}_ci${CI}_co${CO}_${TYPE}"
                       COMMAND single_conv_layer
                               --ipus=${IPUS}
                               --kernel-size=2
                               --data-type=${TYPE}
                               --field=${FIELD}
                               --batch-size=${BS}
                               --conv-groups=${GROUP}
                               --input-channels=${CI}
                               --output-channels=${CO}
                               --tiles-per-ipu=20
                               --single-phase=fwd
                               --fwd-plan-constraints-file=${DEPTHWISE_CONV_PLAN})
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

add_multitarget_test(
         NAME conv0d_simple
         COMMAND single_conv_layer
                 --field={}
                 --batch-size=4
                 --input-channels=128
                 --output-channels=64
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv1d_simple
         COMMAND single_conv_layer
                  --field={40}
                  --kernel-size=3
                  --padding=1
                  --input-channels=128
                  --output-channels=128
                  --tiles-per-ipu=20
                  VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3d_simple
         COMMAND single_conv_layer
                 --field={5,6,7}
                 --kernel-size={4,3,2}
                 --padding={1,1,1}
                 --input-channels=16
                 --output-channels=32
                 --tiles-per-ipu=20
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3d_1_output_chan
         COMMAND single_conv_layer
                 --field={5,6,7}
                 --kernel-size={4,3,2}
                 --padding={1,1,1}
                 --input-channels=32
                 --output-channels=1
                 --tiles-per-ipu=8
                 VARIANTS ${TimesOutOnSim})

# 3D convolution with multiple batches, convolution groups, input dilation,
# kernel dilation and striding.
add_multitarget_test(
         NAME conv3d_complex
         COMMAND single_conv_layer
                 --conv-groups=2
                 --batch-size=2
                 --field={4,3,16}
                 --in-dilation={1,2,1}
                 --padding={1,0,1}
                 --kernel-size={3,2,3}
                 --kernel-dilation={1,1,2}
                 --input-channels=16
                 --output-channels=32
                 --stride={1,1,3}
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

# Replicate across tiles
add_multitarget_test(
  NAME conv2d_replicate_over_tiles
  COMMAND single_conv_layer
          --tiles-per-ipu=8
          --single-phase=fwd
          --input-channels=32
          --output-channels=64
          --field={14,14}
          --kernel-size=3
          --padding=1
          --replication-factor=2
  VARIANTS ${TimesOutOnSimCpp})

# Replicate across IPUs
foreach(rf 1 2 4 8 16)
  math(EXPR bs "16 / ${rf}")
  add_multitarget_test(
    NAME conv2d_replicate_over_ipus_rf_${rf}
    COMMAND single_conv_layer
            --tiles-per-ipu=8
            --ipus=16
            --single-phase=fwd
            --input-channels=32
            --output-channels=64
            --field={14,14}
            --kernel-size=3
            --padding=1
            --batch-size=${bs}
            --replication-factor=${rf}
    VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

add_multitarget_test(
         NAME fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half)

add_multitarget_test(
         NAME fully_connected_half_no_bias
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --bias 0
                 --data-type=half)

add_multitarget_test(
         NAME fully_connected_half_fwd
         COMMAND fully_connected_layer
                 --single-phase=fwd
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half)

add_multitarget_test(
        NAME fully_connected_half_bwd
        COMMAND fully_connected_layer
                --single-phase=bwd
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half)

add_multitarget_test(
        NAME fully_connected_small_field_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 4
                --output-size 4
                --tiles-per-ipu 16
                --data-type=half)

add_multitarget_test(
        NAME fully_connected_large_field_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half)

add_multitarget_test(
        NAME fully_connected_half_fwd_128_bit_load
        COMMAND fully_connected_layer
                --single-phase=fwd
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half
                --matmul-options={\"use128BitConvUnitLoad\":\"true\"})

add_test(
         NAME fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type=half)

add_multitarget_test(
         NAME fully_connected_float
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float)

add_test(
         NAME fully_connected_float_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type=float)

add_multitarget_test(
         NAME fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim})

add_test(
         NAME fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --ipus=2
                 --batch-size=4)

add_multitarget_test(
      NAME fully_connected_half_batch_7
      COMMAND fully_connected_layer
              --input-size 96
              --output-size 96
              --tiles-per-ipu 8
              --data-type=half
              --batch-size=7
              VARIANTS ${TimesOutOnSim})


add_multitarget_test(
         NAME fully_connected_float_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim})

add_test(
         NAME fully_connected_float_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float
                 --ipus=2
                 --batch-size=4)

add_multitarget_test(
         NAME fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 220
                 --output-size 84
                 --tiles-per-ipu 1
                 --data-type=half
                 --batch-size=4
                 --inference-only)

add_multitarget_test(
         NAME fully_connected_input_size_1
         COMMAND fully_connected_layer
                 --input-size=1
                 --output-size=100
                 --tiles-per-ipu=4
                 --data-type=half
                 --batch-size=16)

add_multitarget_test(
         NAME fully_connected_output_size_1
         COMMAND fully_connected_layer
                 --input-size=100
                 --output-size=1
                 --tiles-per-ipu=4
                 --data-type=half
                 --batch-size=16)

add_multitarget_test(
         NAME grouped_fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 160
                 --output-size 100
                 --tiles-per-ipu 12
                 --num-groups 2
                 --data-type=half)


add_test(
         NAME grouped_fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --num-groups 3
                 --data-type=half)


add_multitarget_test(
         NAME grouped_fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 160
                 --output-size 100
                 --tiles-per-ipu 12
                 --data-type=half
                 --num-groups 2
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim})

add_test(
         NAME grouped_fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --num-groups 3
                 --ipus=2
                 --batch-size=4)

add_multitarget_test(
      NAME grouped_fully_connected_half_batch_7
      COMMAND fully_connected_layer
              --input-size 112
              --output-size 112
              --tiles-per-ipu 8
              --data-type=half
              --num-groups 3
              --batch-size=7
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 108
                 --output-size 84
                 --tiles-per-ipu 1
                 --data-type=half
                 --batch-size=2
                 --num-groups 2
                 --inference-only)

add_multitarget_test(
         NAME grouped_fully_connected_half_batch_4_halffloat
         COMMAND fully_connected_layer
                 --input-size 160
                 --output-size 100
                 --tiles-per-ipu 12
                 --num-groups 2
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim}
                 --input-type=half
                 --output-type=float)

add_test(
         NAME grouped_fully_connected_half_batch_4_two_ipus_floathalf
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --num-groups 3
                 --ipus=2
                 --batch-size=4
                 --input-type=float
                 --output-type=half)

add_multitarget_test(
      NAME grouped_fully_connected_half_batch_7_halffloat
      COMMAND fully_connected_layer
              --input-size 112
              --output-size 112
              --tiles-per-ipu 8
              --num-groups 3
              --batch-size=7
              VARIANTS ${TimesOutOnSim}
              --input-type=half
              --output-type=float)

add_multitarget_test(
         NAME grouped_fully_connected_batch_fwd_one_tile_floathalf
         COMMAND fully_connected_layer
                 --input-size 108
                 --output-size 84
                 --tiles-per-ipu 1
                 --batch-size=2
                 --num-groups 2
                 --inference-only
                 --input-type=float
                 --output-type=half)

# Fully connected layer where the RHS has fewer elements than the LHS
add_multitarget_test(
         NAME fully_connected_small_rhs
         COMMAND fully_connected_layer
                 --batch-size=64
                 --input-size=512
                 --tiles-per-ipu=16
                 --output-size=4
                 --data-type=half
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_half_with_introspection
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --stride=2
                 --data-type=half
                 --use-introspection=1)

add_multitarget_test(NAME max_pool_layer_half_with_introspection_and_scale_grad
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --stride=2
                 --data-type=half
                 --use-introspection=1)

add_multitarget_test(NAME max_pool_layer_3d_half_with_introspection_and_scale_grad
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14,7}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --stride={2,2,1}
                 --data-type=half
                 --use-introspection=1)

add_multitarget_test(NAME max_pool_layer_half_without_introspection
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --stride=2
                 --data-type=half
                 --use-introspection=0)

add_multitarget_test(NAME max_pool_layer_float
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --stride=2
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --field={50,50}
                 --kernel-size=3
                 --tiles-per-ipu=16
                 --stride=2
                 --padding-lower=1
                 --padding-upper=1
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_1chan_overlapping_kernel_scaled_grad
         COMMAND pooling_layer
                 --channels 1
                 --field={50,50}
                 --kernel-size=3
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --stride=2
                 --padding-lower=1
                 --padding-upper=1
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_3d_1chan_overlapping_kernel_scaled_grad
         COMMAND pooling_layer
                 --channels 1
                 --field={50,50,7}
                 --kernel-size={3,3,1}
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --stride=2
                 --padding-lower=1
                 --padding-upper=1
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --channels 16
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_multitarget_test(NAME max_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=2
                 --channels 16
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --field={56,56}
                 --kernel-size=3
                 --tiles-per-ipu 16
                 --stride=2
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_3d_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --field={14,14,4}
                 --kernel-size=3
                 --tiles-per-ipu 16
                 --stride={2,4,1}
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_2ipu_scaled_grad
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --field={56,56}
                 --kernel-size=3
                 --tiles-per-ipu 304
                 --stride=2
                 --use-scaled-grad=1
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_3d_2ipu_scaled_grad
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --field={14,14,4}
                 --kernel-size=3
                 --tiles-per-ipu 16
                 --stride={2,4,1}
                 --use-scaled-grad=1
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --field={1,32}
                 --tiles-per-ipu=16
                 --kernel-size={1,4}
                 --stride={1,4})

add_multitarget_test(NAME max_pool_layer_1x4_scaled_grad
         COMMAND pooling_layer
                 --channels 32
                 --field={1,32}
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --kernel-size={1,4}
                 --stride={1,4})

add_multitarget_test(NAME max_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --field={16,32}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={3,1}
                --padding-upper={4,2})

add_multitarget_test(NAME max_pool_layer_mixed_padding_scaled_grad
        COMMAND pooling_layer
                --channels 32
                --field={16,32}
                --tiles-per-ipu=16
                --use-scaled-grad=1
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={3,1}
                --padding-upper={4,2})

add_multitarget_test(NAME max_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --field={16,32}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={-3,-1}
                --padding-upper={4,2})

add_multitarget_test(NAME max_pool_layer_negative_mixed_padding_scaled_grad
        COMMAND pooling_layer
                --channels 32
                --field={16,32}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --use-scaled-grad=1
                --padding-lower={-3,-1}
                --padding-upper={4,2})

add_multitarget_test(NAME max_pool_layer_folded_spatial_dims
        COMMAND pooling_layer
                 --channels 1
                 --field={3,3}
                 --kernel-size={1,1}
                 --tiles-per-ipu=16
                 --stride={2,2}
                 --data-type=half
                 --use-introspection=1)

add_multitarget_test(NAME avg_pool_layer_half
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=avg
                 --field={28,28}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_multitarget_test(NAME avg_pool_layer_float
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=avg
                 --field={28,28}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_multitarget_test(NAME avg_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --pooling-type=avg
                 --field={100,100}
                 --tiles-per-ipu=16
                 --kernel-size=3
                 --stride=2
                 --padding-lower=1
                 --padding-upper=1
                 --data-type=float)

add_multitarget_test(NAME avg_pool_layer_same_field_and_kernel_size
         COMMAND pooling_layer
                 --channels 16
                 --pooling-type=avg
                 --field={7,7}
                 --tiles-per-ipu=16
                 --kernel-size=7
                 --stride=7
                 --data-type=half)

add_multitarget_test(NAME sum_pool_layer_same_field_and_kernel_size
         COMMAND pooling_layer
                 --channels 16
                 --pooling-type=sum
                 --field={7,7}
                 --tiles-per-ipu=16
                 --kernel-size=7
                 --stride=7
                 --data-type=half)

add_multitarget_test(NAME sum_pool_layer_3d_same_field_and_kernel_size
         COMMAND pooling_layer
                 --channels 16
                 --pooling-type=sum
                 --field={4,4,4}
                 --tiles-per-ipu=16
                 --kernel-size=4
                 --stride=4
                 --data-type=half)

add_multitarget_test(NAME max_pool_layer_same_field_and_kernel_size
         COMMAND pooling_layer
                 --channels 16
                 --pooling-type=max
                 --field={7,7}
                 --tiles-per-ipu=16
                 --kernel-size=7
                 --stride=7
                 --data-type=half)

add_multitarget_test(NAME avg_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=avg
                 --channels 16
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_multitarget_test(NAME avg_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=avg
                 --channels 32
                 --field={14,28}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=float
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME avg_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --pooling-type=avg
                 --bwd-chans-per-group=8
                 --field={56,56}
                 --kernel-size=3
                 --tiles-per-ipu=304
                 --stride=2
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME avg_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 16
                 --pooling-type=avg
                 --field={1,32}
                 --tiles-per-ipu=8
                 --kernel-size={1,4}
                 --stride={1,4})

add_multitarget_test(NAME avg_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 16
                --pooling-type=avg
                --field={16,24}
                --tiles-per-ipu=16
                --kernel={5,3}
                --stride={1,4}
                --padding-lower={3,1}
                --padding-upper={4,2})

add_multitarget_test(NAME avg_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 16
                --pooling-type=avg
                --field={16,24}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={-3,-1}
                --padding-upper={4,2})

add_multitarget_test(NAME avg_pool_layer_3d_negative_mixed_padding
        COMMAND pooling_layer
                --channels 16
                --pooling-type=avg
                --field={16,24,4}
                --tiles-per-ipu=16
                --kernel-size={5,3,1}
                --stride={1,4,1}
                --padding-lower={-3,-1,-1}
                --padding-upper={4,2,-1})

add_multitarget_test(NAME sum_pool_layer_half
         COMMAND pooling_layer
                 --channels 16
                 --pooling-type=sum
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_multitarget_test(NAME sum_pool_layer_float
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=sum
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_multitarget_test(NAME sum_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --pooling-type=sum
                 --field={100,100}
                 --tiles-per-ipu=16
                 --kernel-size=3
                 --stride=2
                 --padding-lower=1
                 --padding-upper=1
                 --data-type=float)

add_multitarget_test(NAME sum_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=sum
                 --channels 16
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_multitarget_test(NAME sum_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=sum
                 --channels 16
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_multitarget_test(NAME sum_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --pooling-type=sum
                 --bwd-chans-per-group=8
                 --field={56,56}
                 --kernel-size=3
                 --tiles-per-ipu=16
                 --stride=2
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME sum_pool_layer_3d_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --pooling-type=sum
                 --bwd-chans-per-group=8
                 --field={14,14,4}
                 --kernel-size={3,3,1}
                 --tiles-per-ipu=16
                 --stride=2
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME sum_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=sum
                 --field={1,32}
                 --tiles-per-ipu=16
                 --kernel-size={1,4}
                 --stride={1,4})

add_multitarget_test(NAME sum_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 16
                --pooling-type=sum
                --field={16,24}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={3,1}
                --padding-upper={4,2})

add_multitarget_test(NAME sum_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 16
                --pooling-type=sum
                --field={16,24}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={-3,-1}
                --padding-upper={4,2})

add_multitarget_test(NAME avg_pool_half_single_channel
        COMMAND pooling_layer
                --channels=1
                --field={10,10}
                --kernel-size={5,5}
                --padding-lower={0,0}
                --padding-upper={0,0}
                --batch-size=1
                --pooling-type=avg
                --data-type=half
                VARIANTS ${IPUMODEL_VARIANTS};Cpu)

# the ignore data option disables validation, this test basically just checks
# that the option or any of the profiling options do not cause a crash.
add_multitarget_test(
          NAME gemm_5x0x4_ignore_data_profile
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --tiles-per-ipu=1
                  --ignore-data
                  --profile
                  --report-plan
                  --show-execution-steps
                  --show-var-storage)

add_multitarget_test(
          NAME gemm_5x0x4_groups3_ignore_data_profile
          COMMAND general_matrix_multiply
                  --g 3
                  --m 5
                  --k 0
                  --n 4
                  --tiles-per-ipu=1
                  --ignore-data
                  --profile
                  --report-plan
                  --show-execution-steps
                  --show-var-storage)

add_multitarget_test(
          NAME gemm_5x0x4
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=normal
                  --right-matrix-op=normal)

add_multitarget_test(
          NAME gemm_0x3x4
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=normal
                  --right-matrix-op=normal)

add_multitarget_test(
          NAME gemm_5x3x0
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=normal
                  --right-matrix-op=normal)


add_multitarget_test(
          NAME gemm_5x0x4_trA
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=transpose
                  --right-matrix-op=normal)

add_multitarget_test(
          NAME gemm_0x3x4_trA
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=transpose
                  --right-matrix-op=normal)

add_multitarget_test(
          NAME gemm_5x3x0_trA
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=transpose
                  --right-matrix-op=normal)

add_multitarget_test(
          NAME gemm_5x0x4_trB
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=normal
                  --right-matrix-op=transpose)

add_multitarget_test(
          NAME gemm_0x3x4_trB
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=normal
                  --right-matrix-op=transpose)

add_multitarget_test(
          NAME gemm_5x3x0_trB
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=normal
                  --right-matrix-op=transpose)

add_multitarget_test(
          NAME gemm_5x0x4_trA_trB
          COMMAND general_matrix_multiply
                  --m 5
                  --k 0
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=transpose
                  --right-matrix-op=transpose)

add_multitarget_test(
          NAME gemm_0x3x4_trA_trB
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=transpose
                  --right-matrix-op=transpose)

add_multitarget_test(
          NAME gemm_5x3x0_trA_trB
          COMMAND general_matrix_multiply
                  --m 0
                  --k 3
                  --n 4
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=1
                  --left-matrix-op=transpose
                  --right-matrix-op=transpose)

add_multitarget_test(
          NAME gemm_1x3x1
          COMMAND general_matrix_multiply
                  --m 1
                  --k 3
                  --n 1
                  --alpha 1
                  --beta 1
                  --tiles-per-ipu=16
                  --left-matrix-op=normal
                  --right-matrix-op=normal)

add_multitarget_test(
        NAME gemm_1x1000x1
        COMMAND general_matrix_multiply
                --m 1
                --k 1000
                --n 1
                --alpha 2
                --beta 1
                --tiles-per-ipu=16
                --left-matrix-op=normal
                --right-matrix-op=normal)


add_multitarget_test(
         NAME gemm_40x40x40
         COMMAND general_matrix_multiply
                 --m 40
                 --k 40
                 --n 40
                 --alpha 2
                 --beta 1
                 --tiles-per-ipu=16
                 --left-matrix-op=normal
                 --right-matrix-op=normal)


add_multitarget_test(
         NAME gemm_10x20x30_groups5
         COMMAND general_matrix_multiply
                 --g 5
                 --m 10
                 --k 20
                 --n 30
                 --alpha 2
                 --beta 3
                 --tiles-per-ipu=16
                 --left-matrix-op=normal
                 --right-matrix-op=normal)

add_multitarget_test(
         NAME gemm_40x39x38
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38
                 --alpha 2
                 --beta 1
                 --tiles-per-ipu=16
                 --left-matrix-op=normal
                 --right-matrix-op=normal)

add_multitarget_test(
         NAME gemm_40x39x38_left_transpose
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38
                 --alpha 2
                 --beta 1
                 --tiles-per-ipu=16
                 --left-matrix-op=transpose
                 --right-matrix-op=normal)


add_multitarget_test(
         NAME gemm_40x39x38_right_transpose
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38
                 --alpha 2
                 --beta 1
                 --tiles-per-ipu=16
                 --left-matrix-op=normal
                 --right-matrix-op=transpose)

add_multitarget_test(
         NAME gemm_40x40x40_halffloat
         COMMAND general_matrix_multiply
                 --m 40
                 --k 40
                 --n 40
                 --alpha 2
                 --beta 1
                 --tiles-per-ipu=16
                 --left-matrix-op=normal
                 --right-matrix-op=normal
                 --input-type=half
                 --output-type=float)

add_multitarget_test(
         NAME gemm_40x39x38_floathalf
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38
                 --alpha 2
                 --beta 1
                 --tiles-per-ipu=16
                 --left-matrix-op=normal
                 --right-matrix-op=normal
                 --input-type=float
                 --output-type=half)

add_multitarget_test(
         NAME gemm_40x39x38_left_transpose_halffloat
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38
                 --alpha 2
                 --beta 1
                 --tiles-per-ipu=16
                 --left-matrix-op=transpose
                 --right-matrix-op=normal
                 --input-type=half
                 --output-type=float)

add_multitarget_test(
         NAME gemm_40x39x38_right_transpose_floathalf
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38
                 --alpha 2
                 --beta 1
                 --tiles-per-ipu=16
                 --left-matrix-op=normal
                 --right-matrix-op=transpose
                 --input-type=float
                 --output-type=half)

add_multitarget_test(
         NAME gemm_100x100x100_serial_split_nx2
         COMMAND general_matrix_multiply
                 --m 100
                 --k 100
                 --n 100
                 --tiles-per-ipu=16
                 --plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/pc_serial_split_ocx2.json
                 )

add_multitarget_test(
         NAME gemm_200x100x50_serial_split_nx2
         COMMAND general_matrix_multiply
                 --m 200
                 --k 100
                 --n 50
                 --tiles-per-ipu=16
                 --plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/pc_serial_split_ocx2.json
                 )

add_multitarget_test(
         NAME gemm_40x80x140_serial_split_nx4
         COMMAND general_matrix_multiply
                 --m 40
                 --k 80
                 --n 140
                 --tiles-per-ipu=16
                 --plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/pc_serial_split_ocx4.json
                 )

add_multitarget_test(
         NAME rnn_3x2x38_no_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --output-size 38
                 --tiles-per-ipu=16
                 --batch-size=2
                 --nonlinearity-type sigmoid)


add_multitarget_test(
         NAME rnn_3x32x2x38_with_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --input-size 24
                 --output-size 32
                 --batch-size=2
                 --phase all
                 --tiles-per-ipu=16
                 --nonlinearity-type sigmoid
                 --apply-feedforward-weights)


foreach(PARTIALS_TYPE half float)
        add_multitarget_test(
                NAME basic_lstm_40x4x38_seq_2_half_data_${PARTIALS_TYPE}_partials
                COMMAND lstm_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --partials-type=${PARTIALS_TYPE}
                        --accumulators-type=${PARTIALS_TYPE}
                        VARIANTS ${TimesOutOnSim}
                        LABELS lstm)
endforeach()

foreach(PARTIALS_TYPE half float)
  add_multitarget_test(
         NAME basic_lstm_40x4x38_seq_2_half_data_preweight_inp_${PARTIALS_TYPE}_partials
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --pre-weight-input=1
                 --tiles-per-ipu=16
                 --phase all
                 --sequence-size 2
                 --partials-type=${PARTIALS_TYPE}
                 --accumulators-type=${PARTIALS_TYPE}
                 VARIANTS ${TimesOutOnSim}
                 LABELS lstm)
endforeach()

add_multitarget_test(
         NAME basic_lstm_40x4x38_seq_2_float_data
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --sequence-size 2
                 --tiles-per-ipu=16
                 --phase all
                 --data-type=float
                 VARIANTS ${TimesOutOnSim}
                 LABELS lstm)

add_multitarget_test(
         NAME basic_lstm_40x4x38_seq_2_float_data_fwd_only
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --sequence-size 2
                 --tiles-per-ipu=16
                 --phase fwd
                 --data-type=float
                 VARIANTS ${TimesOutOnSim}
                 LABELS lstm)

foreach(PARTIALS_TYPE half float)
  add_multitarget_test(
         NAME basic_lstm_40x4x38_seq_2_half_data_celltanhrecomp_${PARTIALS_TYPE}_partials
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --tiles-per-ipu=16
                 --phase all
                 --sequence-size 2
                 --recomputation-mode=cellAndTanh
                 --partials-type=${PARTIALS_TYPE}
                 --accumulators-type=${PARTIALS_TYPE}
                 VARIANTS ${TimesOutOnSim}
                 LABELS lstm)
endforeach()

foreach(PARTIALS_TYPE half float)
  add_multitarget_test(
         NAME basic_lstm_40x4x38_seq_2_half_data_runs_2_${PARTIALS_TYPE}_partials
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --tiles-per-ipu=16
                 --phase all
                 --sequence-size 2
                 --runs 2
                 --partials-type=${PARTIALS_TYPE}
                 --accumulators-type=${PARTIALS_TYPE}
                 VARIANTS ${TimesOutOnSim}
                 LABELS lstm)
endforeach()

add_multitarget_test(
         NAME basic_gru_40x4x38_seq_2_half_data
         COMMAND gru_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --tiles-per-ipu=16
                 --phase all
                 --sequence-size 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME basic_gru_40x4x38_seq_2_float_data
         COMMAND gru_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --sequence-size 2
                 --tiles-per-ipu=16
                 --phase all
                 --data-type=float
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME basic_gru_40x4x38_seq_2_float_data_fwd_only
         COMMAND gru_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --sequence-size 2
                 --tiles-per-ipu=16
                 --phase fwd
                 --data-type=float
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME basic_gru_40x4x38_seq_2_half_data_runs_2
         COMMAND gru_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --tiles-per-ipu=16
                 --phase all
                 --sequence-size 2
                 --runs 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --tiles-per-ipu=1
                 --in-dilation=2)

add_multitarget_test(
         NAME conv1x1_stride_3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2)

add_multitarget_test(
         NAME conv3x3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation=2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2)

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd_minMem
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2
                 --convolution-options={\"availableMemoryProportion\":\"0\"})

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd_minCycles
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2
                 --convolution-options={\"availableMemoryProportion\":\"1\"})

add_multitarget_test(
          NAME conv3x3_stride_3_in_dilation_2_fwd_128bit_load
          COMMAND single_conv_layer
                  --single-phase=fwd
                  --input-channels=16
                  --output-channels=16
                  --field={14,14}
                  --kernel-size=3
                  --tiles-per-ipu=1
                  --stride=3
                  --in-dilation=2
                  --convolution-options={\"use128BitConvUnitLoad\":\"true\"})

add_multitarget_test(
         NAME conv3x3_in_dilation_asymmetric
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation={2,3}
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_kernel_dilation_amp
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --wu-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/method_amp.json
                --field={14,28}
                --kernel-size={3,5}
                --kernel-dilation={3,2}
                --tiles-per-ipu=12
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_flattening
         COMMAND single_conv_layer
                --input-channels=46
                --output-channels=4
                --fwd-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/zero_conv_flattening.json
                --field={8,1}
                --in-dilation={1,2}
                --padding-upper={3,2}
                --padding-lower={1,1}
                --kernel-size={1,1}
                --kernel-dilation={1,2}
                --kernel-padding-lower={2,1}
                --stride={1,1}
                --ipus=2
                --tiles-per-ipu=20)

add_multitarget_test(
         NAME conv3x3_kernel_padding_amp
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --wu-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/method_amp.json
                 --field={14,14}
                 --kernel-size={3,3}
                 --kernel-padding-lower={4,2}
                 --kernel-padding-upper={4,3}
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd_minCycles_halffloat
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2
                 --input-type=half
                 --output-type=float
                 --convolution-options={\"availableMemoryProportion\":\"1\"})

add_multitarget_test(
         NAME conv3x3_in_dilation_asymmetric_floathalf
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation={2,3}
                 VARIANTS ${TimesOutOnSim}
                 --input-type=float
                 --output-type=half)

add_multitarget_test(
         NAME conv3x3_kernel_dilation_amp_halffloat
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --wu-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/method_amp.json
                --field={14,28}
                --kernel-size={3,5}
                --kernel-dilation={3,2}
                --tiles-per-ipu=12
                VARIANTS ${TimesOutOnSim}
                --input-type=half
                --output-type=float)

add_multitarget_test(
         NAME conv3x3_kernel_padding_amp_floathalf
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --wu-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/method_amp.json
                 --field={14,14}
                 --kernel-size={3,3}
                 --kernel-padding-lower={4,2}
                 --kernel-padding-upper={4,3}
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim}
                 --input-type=float
                 --output-type=half)

add_multitarget_test(
         NAME conv1x1_stride_1_fwd_serial_split_ocx2
         COMMAND single_conv_layer
             --input-channels=128
             --output-channels=256
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=1
             --single-phase=fwd
             --fwd-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/pc_serial_split_ocx2.json
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
        NAME conv3x3_stride_2_128_out_fwd_serial_split_ocx4
        COMMAND single_conv_layer
            --input-channels=48
            --output-channels=128
            --field={20,12}
            --kernel-size=3
            --data-type=half
            --tiles-per-ipu=12
            --padding=2
            --stride=2
            --fwd-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/pc_serial_split_ocx4.json
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME T8298_conv
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={32,7,1}
                 --input-channels=2
                 --in-dilation={1,1,1}
                 --padding-upper={2,1,2}
                 --padding-lower={1,1,1}
                 --output-channels=1
                 --kernel-size={1,1,3}
                 --kernel-dilation={2,2,1}
                 --kernel-padding-upper={0,1,1}
                 --kernel-padding-lower={1,1,0}
                 --stride={1,2,1}
                 --tiles-per-ipu=2
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\",\"startTileMultiplier\":\"0\"})

# Convolution seen in resnet 50 which exposed a bug when generating WU pass
add_multitarget_test(
         NAME conv7x7_stride_2_1024_in_512_out_serial_fail_case
         COMMAND single_conv_layer
            --input-channels=1024
            --output-channels=512
            --field={14,14}
            --stride={2,2}
            --batch-size=4
            --single-phase=wu
            --tiles-per-ipu=1216
            --ipus=1
            --wu-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/pc_conv7x7_stride_2_1024_in_512_out_serial_fail_case.json
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME random_gen_uniform_half
         COMMAND random_generator
                 --rand-test=Uniform
                 --min-val=-2.0
                 --max-val=0.0
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --half-data-type=true
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_int
         COMMAND random_generator
                 --rand-test=UniformInt
                 --min-val=-20.0
                 --max-val=2.0
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --in-size=40000
                 --tiles-per-ipu=2
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_int_max_range
         COMMAND random_generator
                 --rand-test=UniformInt
                 --min-val=-2147483648
                 --max-val=2147483647
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_float
         COMMAND random_generator
                 --rand-test=Uniform
                 --min-val=-1.0
                 --max-val=1.0
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_float_1_0
         COMMAND random_generator
                 --rand-test=Uniform
                 --min-val=1e-9
                 --max-val=1.0
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_half
         COMMAND random_generator
                 --rand-test=Bernoulli
                 --half-data-type=true
                 --prob=0.75
                 --percent-error=5.0
                 --seed=6513234
                 --seed-modifier=24543
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_float
         COMMAND random_generator
                 --rand-test=Bernoulli
                 --prob=0.25
                 --percent-error=5.0
                 --seed=986427
                 --seed-modifier=1234
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_int
         COMMAND random_generator
                 --rand-test=BernoulliInt
                 --prob=0.5
                 --percent-error=5.0
                 --seed=9885343
                 --seed-modifier=8643232
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_float_prob0
         COMMAND random_generator
                 --rand-test=Bernoulli
                 --prob=0.0
                 --percent-error=5.0
                 --seed=9786553
                 --seed-modifier=6765354
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_int_prob1
         COMMAND random_generator
                 --rand-test=BernoulliInt
                 --prob=1.0
                 --percent-error=5.0
                 --seed=876854
                 --seed-modifier=89766465
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_normal_half
         COMMAND random_generator
                 --rand-test=Normal
                 --half-data-type=true
                 --mean=2.5
                 --std-dev=2.5
                 --percent-error=5.0
                 --seed=827437
                 --seed-modifier=7665654
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_normal_float
         COMMAND random_generator
                 --rand-test=Normal
                 --mean=-0.5
                 --std-dev=2.5
                 --percent-error=5.0
                 --seed=1452764
                 --seed-modifier=7861245
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_truncated_normal_half
         COMMAND random_generator
                 --rand-test=TruncatedNormal
                 --half-data-type=true
                 --mean=1.0
                 --std-dev=1.0
                 --alpha=2.0
                 --percent-error=5.0
                 --seed=1387532
                 --seed-modifier=985436
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_truncated_normal_float
         COMMAND random_generator
                 --rand-test=TruncatedNormal
                 --mean=-1.0
                 --std-dev=1.0
                 --alpha=2.0
                 --percent-error=5.0
                 --seed=8956342
                 --seed-modifier=249065
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_dropout_float
         COMMAND random_generator
                 --rand-test=Dropout
                 --prob=0.3
                 --percent-error=2.0
                 --seed=9887532
                 --seed-modifier=575329
                 --in-size=40000
                 --tiles-per-ipu=2
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_dropout_half
         COMMAND random_generator
                 --rand-test=Dropout
                 --half-data-type=true
                 --prob=0.25
                 --percent-error=2.0
                 --seed=9077511
                 --seed-modifier=709815
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)


add_multitarget_test(
         NAME random_gen_uniform_float_no_seed
         COMMAND random_generator
                 --rand-test=Uniform
                 --min-val=-1.0
                 --max-val=1.0
                 --percent-error=5.0
                 --in-size=40000
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_set_seeds
         COMMAND random_generator
                 --rand-test=SetSeeds
                 --tiles-per-ipu=16
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_set_hw_seeds
         COMMAND random_generator
                 --rand-test=SetHwSeeds
                 --tiles-per-ipu=16
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME conv_T10392
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field=1
                 --kernel-size=3
                 --kernel-truncation-lower=1
                 --kernel-truncation-upper=1
                 --single-phase=fwd
                 --fwd-plan-constraints-file=${CMAKE_SOURCE_DIR}/tests/pc_T10392.json
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_serialization_and_post_conv_dilation
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --field={3,3}
		 --kernel-size={1,1}
		 --in-dilation={2,2}
		 --padding-upper={1,1}
		 --batch-size=1
                 --single-phase=fwd
		 --tiles-per-ipu=16
		 --convolution-options={\"availableMemoryProportion\":\"0.1\"}
		 --fwd-plan-constraints={\"0\":{\"partition\":{\"outChanSplit\":{\"serial\":2}}}})

add_multitarget_test(NAME conv5x200_1_in_100_out
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=100
                 --field={15,200}
                 --kernel-size={5,200}
                 --batch-size=16
                 --data-type=half
                 --tiles-per-ipu=16
                 --fwd-plan-constraints={\"0\":{\"transform\":{\"swapOperands\":false,\"expandDims\":[1]}}}
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(NAME conv5x200_1_in_100_out_swap_operands
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=100
                 --field={15,200}
                 --kernel-size={5,200}
                 --batch-size=16
                 --data-type=half
                 --tiles-per-ipu=16
                 --fwd-plan-constraints={\"0\":{\"transform\":{\"swapOperands\":true,\"expandDims\":[1]}}}
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_10_RZ
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --tiles-per-ipu=16
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_10_bias30_no_inf_RA
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --round-mode=RA
                 --tiles-per-ipu=32
                 --bias=30
                 --enable-infs-and-nans=0
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_4_11_bias13_no_denorm_RN_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=4
                 --man=11
                 --bias=13
                 --enable-denorms=0
                 --round-mode=RN
                 --tiles-per-ipu=16
                 --storage-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_4_11_bias13_no_denorm_RN_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=4
                 --man=11
                 --bias=13
                 --enable-denorms=0
                 --round-mode=RN
                 --tiles-per-ipu=32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_3_4_bias3_no_inf_RD_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=3
                 --man=4
                 --bias=3
                 --enable-infs-and-nans=0
                 --round-mode=RD
                 --tiles-per-ipu=16
                 --storage-type=FP16
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_3_4_bias3_no_inf_RD_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=3
                 --man=4
                 --bias=3
                 --enable-infs-and-nans=0
                 --round-mode=RD
                 --tiles-per-ipu=16
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_3_4_bias3_no_inf_RD_quantise_fp32_calc_type
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=3
                 --man=4
                 --bias=3
                 --enable-infs-and-nans=0
                 --round-mode=RD
                 --tiles-per-ipu=16
                 --storage-type=FP16
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_3_4_bias3_no_inf_RD_gfloat_as_int_fp32_calc_type
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=3
                 --man=4
                 --bias=3
                 --enable-infs-and-nans=0
                 --round-mode=RD
                 --tiles-per-ipu=32
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_6_9_bias62_no_inf_no_denorm_RA_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=6
                 --man=9
                 --bias=62
                 --enable-infs-and-nans=0
                 --enable-denorms=0
                 --round-mode=RA
                 --tiles-per-ipu=16
                 --storage-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_6_9_bias62_no_inf_no_denorm_RA_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=6
                 --man=9
                 --bias=62
                 --enable-infs-and-nans=0
                 --enable-denorms=0
                 --round-mode=RA
                 --tiles-per-ipu=32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_2_bias29_no_inf_RU_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=5
                 --man=2
                 --bias=29
                 --enable-infs-and-nans=0
                 --round-mode=RD
                 --storage-type=FP16
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_2_bias29_no_inf_RU_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=5
                 --man=2
                 --bias=29
                 --enable-infs-and-nans=0
                 --round-mode=RD
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_2_bias29_no_inf_RU_quantise_fp32_calc_type
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=5
                 --man=2
                 --bias=29
                 --enable-infs-and-nans=0
                 --round-mode=RD
                 --storage-type=FP16
                 --calc-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_2_bias29_no_inf_RU_gfloat_as_int_fp32_calc_type
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=5
                 --man=2
                 --bias=29
                 --enable-infs-and-nans=0
                 --round-mode=RD
                 --calc-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_2_bias29_nanoo_RD_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=5
                 --man=2
                 --bias=29
                 --enable-nanoo-mode=1
                 --round-mode=RD
                 --tiles-per-ipu=100
                 --storage-type=FP16
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_2_bias29_nanoo_RD_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=5
                 --man=2
                 --bias=29
                 --enable-nanoo-mode=1
                 --round-mode=RD
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_2_bias29_nanoo_RD_quantise_fp32_calc_type
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=5
                 --man=2
                 --bias=29
                 --enable-nanoo-mode=1
                 --round-mode=RD
                 --tiles-per-ipu=100
                 --storage-type=FP16
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_5_2_bias29_nanoo_RD_gfloat_as_int_fp32_calc_type
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=5
                 --man=2
                 --bias=29
                 --enable-nanoo-mode=1
                 --round-mode=RD
                 --calc-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_6_9_bias62_no_inf_RZ_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=6
                 --man=9
                 --bias=62
                 --enable-infs-and-nans=0
                 --round-mode=RZ
                 --storage-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_6_9_bias62_no_inf_RZ_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=6
                 --man=9
                 --bias=62
                 --enable-infs-and-nans=0
                 --round-mode=RZ
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_1_14_bias0_noInf_RU_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=1
                 --man=14
                 --bias=0
                 --enable-infs-and-nans=0
                 --round-mode=RU
                 --storage-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_1_14_bias0_noInf_RU_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=1
                 --man=14
                 --bias=0
                 --enable-infs-and-nans=0
                 --round-mode=RU
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_1_6_bias0_noInf_RA_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=1
                 --man=6
                 --bias=0
                 --enable-infs-and-nans=0
                 --round-mode=RA
                 --storage-type=FP16
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_1_6_bias0_noInf_RA_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=1
                 --man=6
                 --bias=0
                 --enable-infs-and-nans=0
                 --round-mode=RA
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_1_6_bias0_noInf_RA_quantise_fp32_calc_type
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=1
                 --man=6
                 --bias=0
                 --enable-infs-and-nans=0
                 --round-mode=RA
                 --storage-type=FP16
                 --calc-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_1_6_bias0_noInf_RA_gfloat_as_int_fp32_calc_type
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=1
                 --man=6
                 --bias=0
                 --enable-infs-and-nans=0
                 --round-mode=RA
                 --calc-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_0_15_bias-1_RA_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=0
                 --man=15
                 --bias=-1
                 --enable-infs-and-nans=0
                 --round-mode=RA
                 --storage-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_0_15_bias-1_RA_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=0
                 --man=15
                 --bias=-1
                 --enable-infs-and-nans=0
                 --round-mode=RA
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_6_1_bias31_RU_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=6
                 --man=1
                 --bias=31
                 --round-mode=RU
                 --storage-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_6_1_bias31_RU_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=6
                 --man=1
                 --bias=31
                 --round-mode=RU
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_7_0_bias63_RN_quantise
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=7
                 --man=0
                 --bias=63
                 --round-mode=RN
                 --storage-type=FP32
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_1_7_0_bias63_RN_gfloat_as_int
         COMMAND cast_to_gfloat
                 --input-size=40000
                 --exp=7
                 --man=0
                 --bias=63
                 --round-mode=RN
                 --tiles-per-ipu=100
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_Uniform
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=6
                 --man=9
                 --bias=31
                 --sr-noise-density=Uniform
                 --sr-noise-min=0.65
                 --sr-noise-max=0.85
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Uniform
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=4
                 --man=3
                 --bias=7
                 --sr-noise-density=Uniform
                 --sr-noise-min=0.05
                 --sr-noise-max=0.34
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Uniform_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=4
                 --man=3
                 --bias=7
                 --sr-noise-density=Uniform
                 --sr-noise-min=0.05
                 --sr-noise-max=0.34
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_Normal
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=6
                 --man=2
                 --bias=31
                 --sr-noise-density=Normal
                 --sr-noise-min=0.0
                 --sr-noise-max=0.42
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Normal
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=3
                 --man=6
                 --bias=1
                 --sr-noise-density=Normal
                 --sr-noise-min=0.0 --sr-noise-max=0.22
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Normal_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=3
                 --man=6
                 --bias=1
                 --sr-noise-density=Normal
                 --sr-noise-min=0.0
                 --sr-noise-max=0.22
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_TruncatedNormal
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=5
                 --man=10
                 --bias=30
                 --sr-noise-density=TruncatedNormal
                 --sr-noise-min=-0.15
                 --sr-noise-max=0.25
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_TruncatedNormal
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=5
                 --man=2
                 --bias=15
                 --sr-noise-density=TruncatedNormal
                 --sr-noise-min=-0.1
                 --sr-noise-max=0.5
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_TruncatedNormal_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=5
                 --man=2
                 --bias=15
                 --sr-noise-density=TruncatedNormal
                 --sr-noise-min=-0.1
                 --sr-noise-max=0.5
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_TruncatedLaplace
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=7
                 --man=1
                 --bias=15
                 --sr-noise-density=TruncatedLaplace
                 --sr-noise-min=-0.2
                 --sr-noise-max=0.5
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_TruncatedLaplace
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=5
                 --man=2
                 --bias=15
                 --sr-noise-density=TruncatedLaplace
                 --sr-noise-min=-0.16
                 --sr-noise-max=0.43
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_TruncatedLaplace_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=5
                 --man=2
                 --bias=15
                 --sr-noise-density=TruncatedLaplace
                 --sr-noise-min=-0.16
                 --sr-noise-max=0.43
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_Laplace
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=4
                 --man=13
                 --bias=10
                 --sr-noise-density=Laplace
                 --sr-noise-min=0.0
                 --sr-noise-max=0.5
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Laplace
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=2
                 --man=4
                 --bias=6
                 --sr-noise-density=Laplace
                 --sr-noise-min=0.0
                 --sr-noise-max=0.3
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Laplace_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=2
                 --man=4
                 --bias=6
                 --sr-noise-density=Laplace
                 --sr-noise-min=-0.41
                 --sr-noise-max=0.3
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_TruncatedLogistic
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=4
                 --man=11
                 --bias=7
                 --sr-noise-density=TruncatedLogistic
                 --sr-noise-min=-0.35
                 --sr-noise-max=-0.15
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_Logistic
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=6
                 --man=7
                 --bias=17
                 --sr-noise-density=Logistic
                 --sr-noise-min=-0.35
                 --sr-noise-max=0.15
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Logistic
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=0
                 --man=7
                 --bias=4
                 --sr-noise-density=Logistic
                 --sr-noise-min=-0.5
                 --sr-noise-max=0.15
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Logistic_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=2
                 --man=9
                 --bias=4
                 --sr-noise-density=Logistic
                 --sr-noise-min=-0.5
                 --sr-noise-max=0.15
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_TruncatedLogitNormal
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=3
                 --man=16
                 --bias=7
                 --sr-noise-density=TruncatedLogitNormal
                 --sr-noise-min=0.15
                 --sr-noise-max=0.85
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_TruncatedLogitNormal
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=4
                 --man=7
                 --bias=4
                 --sr-noise-density=TruncatedLogitNormal
                 --sr-noise-min=0.0
                 --sr-noise-max=0.9
                 --input-value=2047
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_TruncatedLogitNormal_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=4
                 --man=7
                 --bias=4
                 --sr-noise-density=TruncatedLogitNormal
                 --sr-noise-min=0.0
                 --sr-noise-max=0.9
                 --input-value=2047
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_LogitNormal
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=2
                 --man=5
                 --bias=4
                 --sr-noise-density=LogitNormal
                 --sr-noise-min=0.0
                 --sr-noise-max=0.5
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_LogitNormal
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=6
                 --man=7
                 --bias=31
                 --sr-noise-density=LogitNormal
                 --sr-noise-min=0.05
                 --sr-noise-max=0.45
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_LogitNormal_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=2
                 --man=5
                 --bias=4
                 --sr-noise-density=LogitNormal
                 --sr-noise-min=0.0
                 --sr-noise-max=0.5
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_float_Bernoulli
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=8
                 --man=7
                 --bias=127
                 --sr-noise-density=Bernoulli
                 --sr-prob-truncate=0.45
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Bernoulli
         COMMAND cast_to_gfloat_sr
                --input-size=40000
                --tiles-per-ipu=32
                --exp=0
                --man=3
                --bias=4
                --sr-noise-density=Bernoulli
                --sr-prob-truncate=0.26
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME cast_to_gfloat_sr_half_Bernoulli_fp32_calc
         COMMAND cast_to_gfloat_sr
                 --input-size=40000
                 --tiles-per-ipu=32
                 --exp=4
                 --man=6
                 --bias=4
                 --sr-noise-density=Bernoulli
                 --sr-prob-truncate=0.76
                 --calc-type=FP32
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME embedding_layer_float_10x20
         COMMAND embedding_layer
                 --data-type=float
                 --shape={10,20}
                 --num-indices=50
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME embedding_layer_half_10x20
         COMMAND embedding_layer
                 --data-type=half
                 --shape={10,20}
                 --num-indices=50
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME embedding_layer_float_10x20_2_index_sets
         COMMAND embedding_layer
                 --data-type=float
                 --shape={10,20}
                 --num-indices={100,10}
                 --tiles-per-ipu=16)

add_multitarget_test(
	 NAME embedding_layer_half_10x20_2_index_sets
         COMMAND embedding_layer
                 --data-type=half
                 --shape={10,20}
                 --num-indices={5,20}
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME embedding_layer_half_pad_grain1
         COMMAND embedding_layer
                 --shape {5,4}
                 --num-indices 3
                 --grain-size 1
		 --use-embedding-plan=0
                 --data-type half
                 --tiles-per-ipu 4)

add_multitarget_test(
         NAME embedding_layer_half_pad_grain3
         COMMAND embedding_layer
                 --shape {5,4}
                 --num-indices 3
                 --grain-size 3
		 --use-embedding-plan=0
                 --data-type half
                 --tiles-per-ipu 4)

# These tests are quite slow to execute on the simulator
add_multitarget_test(
         NAME embedding_layer_big_embedding
         COMMAND embedding_layer
                 --data-type=half
                 --shape={100000,200}
                 --num-indices=1440
                 --use-embedding-plan=1
                 --ignore-data=1
         VARIANTS ${IPUMODEL_VARIANTS};Hw)

add_multitarget_test(
         NAME embedding_layer_many_lookups
         COMMAND embedding_layer
                 --data-type=half
                 --shape={1000,200}
                 --num-indices=18000
                 --use-embedding-plan=1
                 --ignore-data=1
         VARIANTS ${IPUMODEL_VARIANTS};Hw)

# This test is too slow and fails on MACS
#add_multitarget_test(
         #NAME embedding_layer_vmany_lookups
         #COMMAND embedding_layer
                 #--data-type=half
                 #--shape={1000,200}
                 #--num-indices=40000
                 #--use-embedding-plan=1
                 #--ignore-data=1
         #VARIANTS ${IPUMODEL_VARIANTS};Hw)

# This will also split the unsliced dim
add_multitarget_test(
         NAME embedding_layer_quick_split_sliced_dim
         COMMAND embedding_layer
                 --data-type=half
                 --shape={5000,100}
                 --num-indices=1440
                 --use-embedding-plan=1
                 --ignore-data=1
                 --tiles-per-ipu=30
                 --plan-constraints={\"slicedDimSplit\":\"6\"})

# This will also split the unsliced dim
add_multitarget_test(
         NAME embedding_layer_quick_split_lookup
         COMMAND embedding_layer
                 --data-type=half
                 --shape={100,200}
                 --num-indices=450
                 --use-embedding-plan=1
                 --ignore-data=1
                 --tiles-per-ipu=30
                 --plan-constraints={\"lookupSplit\":\"3\"})

add_multitarget_test(
         NAME embedding_layer_quick_unsliced_split_lookup
         COMMAND embedding_layer
                 --data-type=half
                 --shape={100,200}
                 --num-indices=450
                 --use-embedding-plan=1
                 --ignore-data=1
                 --tiles-per-ipu=30
                 --plan-constraints={\"unslicedDimSplit\":\"10\"})
add_unit_test(WrongSyncZone
              HangTest.cpp
              VARIANTS Hw
              LABELS multicard;CE_reproducers)
