#-----------------------------------------------------------------------------
# Test executables for all UnaryOp, BinaryOp, BroadcastScalar,
# BroadcastVectorInner and BroadcastVectorOuter vertices.
add_multi_target_test_executable(UnaryCodeletsTest UnaryCodeletsTest.cpp)
add_multi_target_test_executable(BinaryCodeletsTest BinaryCodeletsTest.cpp)

# Each of the tests for elementwise operators below runs many sub-tests in
# parallel in the same graph & compute set, to speed regression test execution.
# If/when a test fails, this makes it hard to find which specific vertex is
# responsible for the failure
# In this case, you can run exactly the same command, but with the added
# '--group-tests 1' option. This will run each vertex individually, which is
# very much slower but make easier finding the failing vertex.
# You can then run the command again, specifying only the single
# vertex/datatype/operand size that is causing the failure, to speedily debug
# and fix the problem.

# Basic tests to verify that all expected vertices are instantiated, using the
# smallest possible data sizes.
# '--disable-fp-exceptions' is set because three unary (IS_INF, IS_NAN,
# IS_FINITE) and one binary operations (INV_STD_DEV_TO_VARIANCE) can generate
# FP exceptions (when they are enabled; this currently happens in Debug builds).
add_multitarget_test(NAME "UnaryOpCodeletsTests_all"
                     COMMAND UnaryCodeletsTest
                             --size 1
                             --disable-fp-exceptions true
                     LABELS codelet
                     VARIANTS ${IPUMODEL_VARIANTS};Hw)
add_multitarget_test(NAME "BinaryOpCodeletsTests_all"
                     COMMAND BinaryCodeletsTest
                             --size 1
                             --disable-fp-exceptions true
                     LABELS codelet
                     VARIANTS ${IPUMODEL_VARIANTS};Hw)

# Run some tests with big sizes, only on IpuModels and Hw, for speed
add_multitarget_test(NAME "UnaryOpCodeletsTests_NoSim"
                     COMMAND UnaryCodeletsTest
                             --size 480 168 [4095,666]
                     LABELS codelet
                     VARIANTS ${IPUMODEL_VARIANTS};Hw)
add_multitarget_test(NAME "BinaryOpCodeletsTests_NoSim"
                     COMMAND BinaryCodeletsTest
                             --size 480 168 8176 8192
                             --size2  8   8 2044 2048
                     LABELS codelet
                     VARIANTS ${IPUMODEL_VARIANTS};Hw)

# The operations below (trigonometric, power, remainder and integer division),
# implemented with a software library, are extremely slow on the Sim devices, so
# we test them separately, on IpuModel and Hw only.
# Also, because the vertices themselves are implemented in C++, they should need
# less testing on various different sizes for the inputs
add_multitarget_test(NAME "UnaryOpCodeletsTests_slow"
                     COMMAND UnaryCodeletsTest
                             --operation ASIN COS SIN TAN
                             --size 1 2 3 4 7 8 9 [5,2,25]
                     LABELS codelet
                     VARIANTS ${IPUMODEL_VARIANTS};Hw)
add_multitarget_test(NAME "BinaryOpCodeletsTests_slow_floating_point"
                     COMMAND BinaryCodeletsTest
                             --operation ATAN2 POWER REMAINDER
                             --size 1 2 3 4 7 8 9 [5,2,25]
                     LABELS codelet
                     VARIANTS ${IPUMODEL_VARIANTS};Hw)
add_multitarget_test(NAME "BinaryOpCodeletsTests_slow_integer"
                     COMMAND BinaryCodeletsTest
                             --operation DIVIDE
                             --data-type int uint
                             --size [5,2,25]
                     LABELS codelet
                     VARIANTS ${IPUMODEL_VARIANTS};Hw)

# Main function to test the majority of vertices. Called specifying "1D"
# or "2D" and different sizes for the operands
#   TEST_TAG  Just a tag to distinguish various invocation of this function
#   TYPE      Will be "1D" or "2D"
#   SIZE      Specify the operand sizes, passed to the test executables.
#             Look at their help to understand the details of the specification
#             of sizes for the various vertices.
#
# Note that where --data-type option is NOT specified in a command, the test is
# run for all valid data types for the specified operator(s).
function(test_elementwise TEST_TAG TYPE SIZE)
  add_multitarget_test(NAME "UnaryOpCodeletsTests_${TEST_TAG}"
                       COMMAND UnaryCodeletsTest
                               --vertexRE ${TYPE}
                               --operation ABSOLUTE BITWISE_NOT CEIL CBRT
                                           COUNT_LEADING_ZEROS ERF EXPONENT
                                           EXPONENT_MINUS_ONE FLOOR GELU_ERF
                                           INVERSE LOGARITHM LOGARITHM_ONE_PLUS
                                           LOGICAL_NOT NEGATE POPCOUNT RELU SQUARE
                                           SIGNUM TANH ROUND RSQRT SQRT SIGMOID
                               --size ${SIZE}
                       LABELS codelet
                       VARIANTS ${SIM_VARIANTS})

  add_multitarget_test(NAME "BinaryOpCodeletsTests_${TEST_TAG}"
                       COMMAND BinaryCodeletsTest
                               --vertexRE ${TYPE}
                               --operation ADD BITWISE_AND BITWISE_OR
                                           BITWISE_XOR BITWISE_XNOR EQUAL
                                           GREATER_THAN_EQUAL GREATER_THAN
                                           LESS_THAN_EQUAL LOGICAL_AND LOGICAL_OR
                                           LESS_THAN MAXIMUM MINIMUM MULTIPLY
                                           NOT_EQUAL SHIFT_LEFT SHIFT_RIGHT
                                           SHIFT_RIGHT_SIGN_EXTEND SUBTRACT
                                           VARIANCE_TO_INV_STD_DEV
                               --size ${SIZE}
                       LABELS codelet
                       VARIANTS ${SIM_VARIANTS})
endfunction()

# Run ALL the 1D elementwise test with various sizes of the operands
# We call test_elementwise multiple times to avoid taking too long and
# timing out.
# Note that the strings with semicolons are part of CMake syntax for function
# parameters, so that inside the function they are space-separated lists, while
# the comma-separated, square-bracket arrays are passed directly to the test
# executables
test_elementwise(S1  1D   "2;3;4;8")
test_elementwise(S2  1D   "12;16;24")
test_elementwise(S3  1D   "25;26;27")
test_elementwise(S4  1D   "28;44;50")
# Run ALL the 2D elementwise test once with a 2D vectors containing many
# different sizes.
test_elementwise(2D_1 2D "[1,2,3,4,8,9,12,22,24]"  1)
test_elementwise(2D_2 2D "[25,26,27,28]"  1)

# These operations will generate FP exceptions when they are enabled (this
# currently happens in Debug builds).
# This should be fixed and the operations moved back in 'test_elementwise' above
#   For IS_FINITE, IS_INF, IS_NAN      : see T33934
#   For SQUARE, INV_STD_DEV_TO_VARIANCE: see T34981, T36789
add_multitarget_test(NAME "UnaryOpCodeletsTests_no_fp_except"
                      COMMAND UnaryCodeletsTest
                              --operation IS_FINITE IS_INF IS_NAN
                              --disable-fp-exceptions true
                              --size "2;3;4;8;12;16;24;25;26;27;28;44;50;[1,2,23,24]"
                      LABELS codelet
                      VARIANTS ${SIM_VARIANTS})
add_multitarget_test(NAME "BinaryOpCodeletsTests_no_fp_except"
                      COMMAND BinaryCodeletsTest
                              --operation INV_STD_DEV_TO_VARIANCE
                              --disable-fp-exceptions true
                              --size "2;3;4;8;12;16;24;25;26;27;28;44;50;[1,2,23,24]"
                      LABELS codelet
                      VARIANTS ${SIM_VARIANTS})

# DIVIDE is handled separately here for half and float because for int, uint
# it is handled in the "BinaryOpCodeletsTests_slow_integer" test outside of
# this function.
add_multitarget_test(NAME "BinaryOpCodeletsTests_div_float"
COMMAND BinaryCodeletsTest
        --operation DIVIDE
        --data-type half float
        --size 2 3 4 5 6 8 9 24 25 26 27 28
        --group-tests
LABELS codelet
VARIANTS ${SIM_VARIANTS})

# Vector Inner vertices are exercised with these sizes as well
add_multitarget_test(NAME "BinaryOpCodeletsTests_VectorInner"
COMMAND BinaryCodeletsTest
        --vertexRE VectorInner
        --size   8 16 32 15  [15,12,32,15]
        --size2  1  4  8  5      [1,4,8,5]
        --group-tests
LABELS codelet
VARIANTS ${SIM_VARIANTS})
add_multitarget_test(NAME "BinaryOpCodeletsTests_VectorInner_long"
COMMAND BinaryCodeletsTest
        --vertexRE VectorInner1
        --data-type half
        --operation add mul
        --size   4080 4088 4096
        --size2  2040 2044 2048
        --group-tests
LABELS codelet
VARIANTS ${SIM_VARIANTS})

add_multitarget_test(NAME "BinaryOpCodeletsTests_VectorInner_b_slices_1"
COMMAND BinaryCodeletsTest
        --vertexRE VectorInner1
        --data-type half
        --operation add div subtract mul
        --size   48 128 256 784 1024
        --size2  8 16
        --group-tests
LABELS codelet
VARIANTS ${SIM_VARIANTS})

add_multitarget_test(NAME "BinaryOpCodeletsTests_VectorInner_b_slices_2"
COMMAND BinaryCodeletsTest
        --vertexRE VectorInner1
        --data-type half
        --operation add div subtract mul
        --size   128 256 1024
        --size2  8 16 32
        --group-tests
        --b-slices=2
LABELS codelet
VARIANTS ${SIM_VARIANTS})

add_multitarget_test(NAME "BinaryOpCodeletsTests_VectorInner_b_slices_4"
COMMAND BinaryCodeletsTest
        --vertexRE VectorInner1
        --data-type half
        --operation add div subtract mul
        --size   128 256 1024
        --size2  4 8 16
        --group-tests
        --b-slices=4
LABELS codelet
VARIANTS ${SIM_VARIANTS})



# Vector Outer vertices are exercised with these sizes as well.
# When 'allowMisaligned' is false, the rows must be a multiple of atom size
add_multitarget_test(NAME "BinaryOpCodeletsTests_VectorOuter_misalign"
COMMAND BinaryCodeletsTest
        --vertexRE VectorOuter
        --allow-misaligned true
        --size  1x4 1x5 1x20 1x28 3x4 3x5 3x20 3x28 8x4 8x5 8x20 8x28 3x9 4x8 80x8 80x11 80x32 80x33 80x8 80x11 80x32 80x33
        --size2   7   7    7   7    7   7    7    7   7   7    7    7   3   4    7     7     7     7    2     2     2     2
        --group-test
        --num-tiles=8
LABELS codelet
VARIANTS ${SIM_VARIANTS})

add_multitarget_test(NAME "BinaryOpCodeletsTests_VectorOuter_align"
COMMAND BinaryCodeletsTest
        --vertexRE VectorOuter
        --allow-misaligned false
        --size  1x4 1x20 1x28 3x4 3x20 3x28 8x4 8x20 8x28 4x8 80x8 80x11 80x32 80x8 80x32
        --size2   7    7   7    7    7    7   7    7    7   4    7     7     7    2     2
        --group-test
        --num-tiles=8
LABELS codelet
VARIANTS ${SIM_VARIANTS})


# Some Binary/Broadcast codelets use different code paths depending if the two
# operands are located close to each other or 32 Kbytes (and a bit) away in
# memory.
function(runFastSlowPathBinaryTests TEST_TAG OPS DATA_TYPES)
  foreach(OFFS 0 32824)
    add_multitarget_test(NAME "BinaryOpCodeletsTests_${TEST_TAG}_offset_${OFFS}"
                         COMMAND BinaryCodeletsTest
                                 --vertexRE BinaryOp|BroadcastScalar
                                 --operation ${OPS}
                                 --data-type ${DATA_TYPES}
                                 --offset-operands ${OFFS}
                                 --size [511,511]
                                 --group-tests
                                 --num-tiles=16
                         LABELS codelet
                         VARIANTS ${SIM_VARIANTS})
  endforeach()
endfunction()
runFastSlowPathBinaryTests(float "ADD;SUBTRACT;MULTIPLY" "float;half")
runFastSlowPathBinaryTests(short "BITWISE_AND;BITWISE_OR" "short;ushort")
#-----------------------------------------------------------------------------
include(GNUInstallDirs)
set(GP_OBJECTS "${CMAKE_INSTALL_FULL_LIBDIR}")

if(enable_vv_tests)
  # BinaryOps are C++ vertices, and the compiler generates RPT loops
  # for them. This test verifies if this is a case by looking into
  # trace output from vertex verifer.
  set(BinaryOpRptLoopOps "MINIMUM;MAXIMUM")
  set(BinaryOpRptLoopTypes "half;float")
  foreach(OP ${BinaryOpRptLoopOps})
    foreach(TYPE ${BinaryOpRptLoopTypes})
      foreach(SIM_VARIANT ${SIM_VARIANTS})
        string(REPLACE "Sim" "ipu" ARCH "${SIM_VARIANT}")
        get_target_property(VV_EXECUTABLE "vertex_verifier_${ARCH}" LOCATION)
        extract_runconfig(SIM_VARIANT TEST_CONFIG)
        string(JOIN "_" TEST_NAME ${SIM_VARIANT} ${TEST_CONFIG} "BinaryOpRptLoopTest" ${OP} ${TYPE})
        add_test(NAME ${TEST_NAME}
          COMMAND ${VV_EXECUTABLE}
            --input-file=${GP_OBJECTS}/popops.gp
            --vertex=popops::BinaryOp1D<popops::expr::BinaryOpType::${OP},${TYPE}>
            --target=${ARCH}
            --constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/BinaryOpRptLoopTest.vv
            --trace-mask=0b0000010 # Set to trace only worker 1
            --trace
        )
        set_tests_properties(${TEST_NAME} PROPERTIES
            PASS_REGULAR_EXPRESSION ".*:[ ]+rpt[ ]+\\$m")
      endforeach()
    endforeach()
  endforeach()
endif()

# This run tests on all the vertices of the 'ScaledAdd' family, checking for
# overruns at the end of the data
add_multi_target_test_executable(ScaledAddCodeletsTest ScaledAddCodeletsTest.cpp)
add_multitarget_test(NAME "ScaledAddCodeletsTests"
                     COMMAND ScaledAddCodeletsTest
                             --tolerance 0.00001 0.001
                             --size 1 2 3 4 8 9 10 11 12 13 14 15 16 23 24 25 [47,48,49]
                     VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS})



add_multi_target_test_executable(FillCodeletsTest FillCodeletsTest.cpp)

add_multitarget_test(NAME "FillCodelets"
                     COMMAND FillCodeletsTest
                           --size 1 2 3 4 5 6 7 8 10 [1,1] 2x4 2x7 2x8
                     VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS})

# Fill vertices for 8 bit types need to be tested with a lot of different data
# lengths and start offsets (from 0 to 7), because the assembly code is quite
# specialised for the various combinations.
# We only test for char, as the code is the same for all the 8 bit types
add_multitarget_test(NAME "FillCodelets_8bit_align"
                     COMMAND FillCodeletsTest --group-test
                             --size 1 2 3 4 5 6 7 8 9 10 11 12 43
                                    [44,11,13] [45,12,14] 46 47 48
                             --data-type char
                             --align-start 0 1 2 3 4 5 6 7
                     LABELS codelet
                     TEST_IPUMODEL
                     VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS})



add_unit_test(DynamicSliceCodeletTest
              DynamicSliceCodeletTest.cpp
              LABELS codelet
              VARIANTS ${SIM_VARIANTS})
add_unit_test(DynamicSlice1DCodeletTest
              DynamicSlice1DCodeletTest.cpp
              LABELS codelet
              VARIANTS ${SIM_VARIANTS})
add_unit_test(MultiSliceCodeletTest
              MultiSliceCodeletTest.cpp
              LABELS codelet
              VARIANTS ${SIM_VARIANTS})

#Histogram
add_multi_target_test_executable(histogramCodeletTest histogramCodeletTest.cpp)
foreach(DATA_SIZE  1 2 3 4 99)
  foreach(DATA_TYPE half float)
    foreach(DO_ABS true false)
      foreach(LIMITS_SIZE 1 5 13)
        add_multitarget_test(NAME histogramSupervisor_by_limit_d${DATA_SIZE}_t${DATA_TYPE}_l${LIMITS_SIZE}_abs${DO_ABS}
                             COMMAND histogramCodeletTest
                             --data-size=${DATA_SIZE}
                             --limits-size=${LIMITS_SIZE}
                             --type=${DATA_TYPE}
                             --absolute=${DO_ABS}
                             --supervisor=true
                             --supervisor-by-limit=true
                             LABELS codelet
                             VARIANTS ${SIM_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(DATA_SIZE  1 2 3 4 5 6 7 8 99)
  foreach(DATA_TYPE half float)
    foreach(DO_ABS true false)
      foreach(LIMITS_SIZE 1 7)
        add_multitarget_test(NAME histogramSupervisor_by_data_d${DATA_SIZE}_t${DATA_TYPE}_l${LIMITS_SIZE}_abs${DO_ABS}
                             COMMAND histogramCodeletTest
                             --data-size=${DATA_SIZE}
                             --limits-size=${LIMITS_SIZE}
                             --type=${DATA_TYPE}
                             --absolute=${DO_ABS}
                             --supervisor=true
                             --supervisor-by-limit=false
                             LABELS codelet
                             VARIANTS ${SIM_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(DATA_SIZE  3 4 99)
  foreach(DATA_TYPE half float)
    foreach(DO_ABS true false)
      foreach(LIMITS_SIZE 1 10)
        foreach(ROWS 1 2)
          add_multitarget_test(NAME histogram2D_d${DATA_SIZE}_t${DATA_TYPE}_l${LIMITS_SIZE}_r${ROWS}_abs${DO_ABS}
                               COMMAND histogramCodeletTest
                               --data-size=${DATA_SIZE}
                               --limits-size=${LIMITS_SIZE}
                               --type=${DATA_TYPE}
                               --absolute=${DO_ABS}
                               --supervisor=false
                               --data-rows=${ROWS}
                               LABELS codelet
                               VARIANTS ${SIM_VARIANTS})
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

#Clamp<InType>
foreach(DATA_TYPE half float int)
  foreach(FUNC_TYPE Clamp BroadcastClamp ClampInPlace BroadcastClampInPlace)
    set(VARIANT_NAME "${FUNC_TYPE}Test_${DATA_TYPE}")
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
    string(TOUPPER "${FUNC_TYPE}" FUNC_TYPE_UPPER)
    string(TOUPPER "${DATA_TYPE}" DATA_TYPE_UPPER)
    configure_file(ClampTest.cpp.in ${VARIANT_SRC})
    add_unit_test(${VARIANT_NAME} ${VARIANT_SRC} LABELS codelet
                  VARIANTS ${SIM_VARIANTS})
  endforeach()
endforeach()

#ReduceAdd
add_multi_target_test_executable(ReduceAdd ReduceAdd.cpp)
foreach(OUTER_DIM RANGE 1 15)
  foreach(INNER_DIM RANGE 1 3)
    foreach(PARTIALS_TYPE half float)
      foreach(OUT_TYPE half float)
        set(VARIANT_NAME
            "ReduceAdd_${OUTER_DIM}_${INNER_DIM}_${PARTIALS_TYPE}_${OUT_TYPE}")
        add_multitarget_test(NAME ${VARIANT_NAME}
                             COMMAND ReduceAdd
                               --partials-type=${PARTIALS_TYPE}
                               --out-type=${OUT_TYPE}
                               --outer-dim=${OUTER_DIM}
                               --inner-dim=${INNER_DIM}
                               LABELS codelet
                               VARIANTS ${SIM_VARIANTS})
        endforeach()
      endforeach()
  endforeach()
endforeach()

foreach(OUTER_DIM 48 49 50 51 52 53 54 55 56 500)
  foreach(INNER_DIM RANGE 1 3)
    foreach(PARTIALS_TYPE half float)
      foreach(OUT_TYPE half float)
        set(VARIANT_NAME
            "ReduceAdd_${OUTER_DIM}_${INNER_DIM}_${PARTIALS_TYPE}_${OUT_TYPE}")
        add_multitarget_test(NAME ${VARIANT_NAME}
                             COMMAND ReduceAdd
                               --partials-type=${PARTIALS_TYPE}
                               --out-type=${OUT_TYPE}
                               --outer-dim=${OUTER_DIM}
                               --inner-dim=${INNER_DIM}
                               LABELS codelet
                               VARIANTS ${SIM_VARIANTS})
        endforeach()
      endforeach()
  endforeach()
endforeach()

foreach(OUTER_DIM 4 8 20 24 28 48 52)
  foreach(INNER_DIM RANGE 1 3)
    foreach(OUT_TYPE float half)
      set(VARIANT_NAME
          "ReduceAddSingleIn_${OUTER_DIM}_${INNER_DIM}_float_${OUT_TYPE}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ReduceAdd
                             --partials-type=float
                             --out-type=${OUT_TYPE}
                             --outer-dim=${OUTER_DIM}
                             --inner-dim=${INNER_DIM}
                             --single-input=true
                             --constrain-partials=true
                             LABELS codelet
                             VARIANTS ${SIM_VARIANTS})

      endforeach()
  endforeach()
endforeach()

foreach(OUTER_DIM 8 16 40 48 56 96 104)
  foreach(INNER_DIM RANGE 1 3)
    foreach(OUT_TYPE float half)
      set(VARIANT_NAME
          "ReduceAddSingleIn_${OUTER_DIM}_${INNER_DIM}_half_${OUT_TYPE}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ReduceAdd
                             --partials-type=half
                             --out-type=${OUT_TYPE}
                             --outer-dim=${OUTER_DIM}
                             --inner-dim=${INNER_DIM}
                             --single-input=true
                             --constrain-partials=true
                             LABELS codelet
                             VARIANTS ${SIM_VARIANTS})
      endforeach()
  endforeach()
endforeach()

foreach(PARTIALS_TYPE float half)
  foreach(OUT_TYPE float half)
    set(VARIANT_NAME
        "ReduceAddSingleInNoConstrain_${PARTIALS_TYPE}_${OUT_TYPE}")
    add_multitarget_test(NAME ${VARIANT_NAME}
                         COMMAND ReduceAdd
                           --partials-type=${PARTIALS_TYPE}
                           --out-type=${OUT_TYPE}
                           --outer-dim=96
                           --inner-dim=4
                           --single-input=true
                           --constrain-partials=false
                           LABELS codelet
                           VARIANTS ${SIM_VARIANTS})
    endforeach()
  endforeach()


add_multi_target_test_executable(Reduce Reduce.cpp)
foreach(OUTER_DIM RANGE 1 16)
  foreach(IN_TYPE half float)
    foreach(OUT_TYPE half float)
      set(VARIANT_NAME "Reduce_${IN_TYPE}_${OUT_TYPE}_${OUTER_DIM}")
      set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND Reduce
                            --in-type=${IN_TYPE}
                            --out-type=${OUT_TYPE}
                            --outer-dim=${OUTER_DIM}
                            LABELS codelet
                            VARIANTS ${SIM_VARIANTS})
    endforeach()
  endforeach()
endforeach()

add_multi_target_test_executable(ContinuousReduce ContinuousReduce.cpp)
foreach(OUTER_DIM RANGE 1 4)
  foreach(INNER_DIM RANGE 1 8)
    foreach(IN_TYPE half float)
      foreach(OP MIN MAX)
      set(VARIANT_NAME "ContinuousReduce_${IN_TYPE}_${OUTER_DIM}_${INNER_DIM}_${OP}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                          COMMAND ContinuousReduce
                          --partials-type=${IN_TYPE}
                          --out-type=${IN_TYPE}
                          --outer-dim=${OUTER_DIM}
                          --inner-dim=${INNER_DIM}
                          --operation=${OP}
                          LABELS codelet
                          VARIANTS ${SIM_VARIANTS})
      endforeach()

      foreach(OUT_TYPE half float)
        set(VARIANT_NAME "ContinuousReduce_${IN_TYPE}_${OUT_TYPE}_${OUTER_DIM}_${INNER_DIM}")
        add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ContinuousReduce
                            --partials-type=${IN_TYPE}
                            --out-type=${OUT_TYPE}
                            --outer-dim=${OUTER_DIM}
                            --inner-dim=${INNER_DIM}
                            --operation=SQUARE_ADD
                            LABELS codelet
                            VARIANTS ${SIM_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(OPERATION ADD SQUARE_ADD)
  set(VARIANT_NAME "ContinuousReduce_${OPERATION}")
  set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
  add_multitarget_test(NAME ${VARIANT_NAME}
                     COMMAND ContinuousReduce
                      --partials-type=half
                      --out-type=half
                      --outer-dim=4
                      --inner-dim=8
                      --operation=${OPERATION}
                      LABELS codelet
                      VARIANTS ${SIM_VARIANTS})
endforeach()

foreach(IN_TYPE int)
  set(VARIANT_NAME "ContinuousReduce_${IN_TYPE}_${IN_TYPE}_32_32")
  set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
  add_multitarget_test(NAME ${VARIANT_NAME}
                          COMMAND ContinuousReduce
                          --partials-type=${IN_TYPE}
                          --out-type=${IN_TYPE}
                          --outer-dim=32
                          --inner-dim=32
                          --operation=ADD
                          LABELS codelet
                          VARIANTS ${SIM_VARIANTS})
endforeach()

add_multi_target_test_executable(ReduceSpecial23 ReduceSpecial23.cpp)
foreach(TYPE half float)
  foreach(OPERATION MAX MIN)
    foreach(UPDATE true false)
      foreach(OUTPUT_DIM 4 8 16)
        set(VARIANT_NAME "ReduceSpecial3_${TYPE}_${OPERATION}_${UPDATE}_${OUTPUT_DIM}")
        set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
        add_multitarget_test(NAME ${VARIANT_NAME}
                        COMMAND ReduceSpecial23
                          --partials-type=${TYPE}
                          --out-type=${TYPE}
                          --outer-dim=5
                          --inner-dim=16
                          --output-dim=${OUTPUT_DIM}
                          --operation=${OPERATION}
                          --update=${UPDATE}
                          --specialisation=3
                          --num-outer-strides=1
                          --outer-stride=0
                          --num-partials=5
                          LABELS codelet
                          VARIANTS ${SIM_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(PARTIALS_TYPE half float)
  foreach(OUT_TYPE half float)
    foreach(OPERATION ADD SQUARE_ADD LOG_ADD)
      foreach(UPDATE true false)
        foreach(SCALE 1.0 2.0)
                set(VARIANT_NAME "ReduceSpecial3_${PARTIALS_TYPE}_${OUT_TYPE}_${OPERATION}_${UPDATE}_${SCALE}")
                set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
                add_multitarget_test(NAME ${VARIANT_NAME}
                                COMMAND ReduceSpecial23
                                  --partials-type=${PARTIALS_TYPE}
                                  --out-type=${OUT_TYPE}
                                  --outer-dim=13
                                  --inner-dim=16
                                  --output-dim=16
                                  --operation=${OPERATION}
                                  --update=${UPDATE}
                                  --specialisation=3
                                  --num-outer-strides=3
                                  --outer-stride=2
                                  --num-partials=3
                                  --scale=${SCALE}
                                  LABELS codelet
                                  VARIANTS ${SIM_VARIANTS})
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(TYPE half float)
  foreach(OPERATION ADD SQUARE_ADD)
    foreach(INNER_DIM 1 2 4 8 16 17 18)
      set(VARIANT_NAME "ReduceSpecial2_${TYPE}_${OPERATION}_${INNER_DIM}")
      set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
      add_multitarget_test(NAME ${VARIANT_NAME}
                      COMMAND ReduceSpecial23
                        --partials-type=float
                        --out-type=${TYPE}
                        --outer-dim=1
                        --inner-dim=${INNER_DIM}
                        --output-dim=1
                        --operation=${OPERATION}
                        --update=false
                        --specialisation=2
                        --scale=1.0
                        LABELS codelet
                        VARIANTS ${SIM_VARIANTS})
    endforeach()
  endforeach()
endforeach()

# Select
foreach(DATA_TYPE int unsigned_int float half bool)
  set(VARIANT_NAME "Select_${DATA_TYPE}")
  set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
  string(TOUPPER "${DATA_TYPE}" TYPE)
  set(DATA_HEADER "select/${DATA_TYPE}.hpp")
  configure_file(Select.cpp.in ${VARIANT_SRC})
  configure_file(${DATA_HEADER} ${CMAKE_CURRENT_BINARY_DIR}/${DATA_HEADER} COPYONLY)
  add_unit_test(${VARIANT_NAME} ${VARIANT_SRC} LABELS codelet
                VARIANTS ${SIM_VARIANTS})
endforeach()

# BroadcastSelect
foreach(DATA_TYPE int unsigned_int float half bool)
  set(VARIANT_NAME "BroadcastSelect_${DATA_TYPE}")
  set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
  string(TOUPPER "${DATA_TYPE}" TYPE)
  set(DATA_HEADER "select/${DATA_TYPE}.hpp")
  configure_file(BroadcastSelect.cpp.in ${VARIANT_SRC})
  configure_file(${DATA_HEADER} ${CMAKE_CURRENT_BINARY_DIR}/${DATA_HEADER} COPYONLY)
  add_unit_test(${VARIANT_NAME} ${VARIANT_SRC} LABELS codelet
                VARIANTS ${SIM_VARIANTS})
endforeach()

add_unit_test(FloatPointBehaviour
              FloatPointBehaviour.cpp
              VARIANTS ${SIM_VARIANTS}
              LABELS codelet)

add_multi_target_test_executable(ReduceMaxClassGather ReduceMaxClassGather.cpp)
foreach(ACTIVATION_TYPE half float int unsigned)
  foreach(SIZE RANGE 1 24)
    set(VARIANT_NAME "ReduceMaxClassGather_${ACTIVATION_TYPE}_${SIZE}/4")
    add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ReduceMaxClassGather
                           --activation-type=${ACTIVATION_TYPE}
                           --divisor=4
                           --size=${SIZE}
                           LABELS codelet
                           VARIANTS ${SIM_VARIANTS})
  endforeach()
endforeach()

add_multi_target_test_executable(ReduceMaxClassSparse ReduceMaxClassSparse.cpp)
foreach(ACTIVATION_TYPE float int unsigned)
  foreach(SIZE RANGE 1 24)
    foreach(LABEL_TYPE int unsigned)
      set(VARIANT_NAME "ReduceMaxClassSparse_${ACTIVATION_TYPE}_${SIZE}_${LABEL_TYPE}/4")
      add_multitarget_test(NAME ${VARIANT_NAME}
                             COMMAND ReduceMaxClassSparse
                             --activation-type=${ACTIVATION_TYPE}
                             --label-type=${LABEL_TYPE}
                             --size=${SIZE}
                             LABELS codelet
                             VARIANTS ${SIM_VARIANTS})
    endforeach()
  endforeach()
endforeach()


add_multi_target_test_executable(ReduceNMaxClassGather ReduceNMaxClassGather.cpp)
foreach(ACTIVATION_TYPE float int)
  foreach(K RANGE 1 4)
    foreach(SIZE RANGE ${K} 12)
        set(VARIANT_NAME "ReduceNMaxClassGather_top${K}_${ACTIVATION_TYPE}_${SIZE}")
        add_multitarget_test(NAME ${VARIANT_NAME}
                              COMMAND ReduceNMaxClassGather
                              --activation-type=${ACTIVATION_TYPE}
                              --divisor=4
                              --size=${SIZE}
                              --k=${K}
                              LABELS codelet
                              VARIANTS ${SIM_VARIANTS})
    endforeach()
  endforeach()

  # Add some larger tests.
  set(VARIANT_NAME "ReduceNMaxClassGather_top8_${ACTIVATION_TYPE}_100")
  add_multitarget_test(NAME ${VARIANT_NAME}
                        COMMAND ReduceNMaxClassGather
                        --activation-type=${ACTIVATION_TYPE}
                        --divisor=128
                        --size=100
                        --k=8
                        LABELS codelet
                        VARIANTS ${SIM_VARIANTS})


  # Add some larger tests.
  set(VARIANT_NAME "ReduceNMaxClassGather_top162_${ACTIVATION_TYPE}_200")
  add_multitarget_test(NAME ${VARIANT_NAME}
                        COMMAND ReduceNMaxClassGather
                        --activation-type=${ACTIVATION_TYPE}
                        --divisor=64
                        --size=200
                        --k=162
                        LABELS codelet
                        VARIANTS ${SIM_VARIANTS})

endforeach()

add_multi_target_test_executable(ReduceNMaxClassSparse ReduceNMaxClassSparse.cpp)
foreach(ACTIVATION_TYPE int float)
  foreach(K RANGE 1 12)
    foreach(SIZE RANGE ${K} 13)
        set(VARIANT_NAME "ReduceNMaxClassSparse_top${K}_${ACTIVATION_TYPE}_${SIZE}")
        add_multitarget_test(NAME ${VARIANT_NAME}
                              COMMAND ReduceNMaxClassSparse
                              --size=${SIZE}
                              --k=${K}
                              --activation-type=${ACTIVATION_TYPE}
                              LABELS codelet
                              VARIANTS ${SIM_VARIANTS})
    endforeach()
  endforeach()
endforeach()

set(VARIANT_NAME "ReduceNMaxClassSparse_top150_float_200")
add_multitarget_test(NAME ${VARIANT_NAME}
                      COMMAND ReduceNMaxClassSparse
                      --size=200
                      --k=150
                      --activation-type=float
                      LABELS codelet
                      VARIANTS ${SIM_VARIANTS})


add_multi_target_test_executable(CastTest CastTest.cpp)

# Cast half to quarter
foreach(COLUMNS 1 2 3 4 8 24 48 141)
  foreach(IN_TYPE half quarter char schar)
    add_multitarget_test(NAME CastTest_quarter_1_${IN_TYPE}_${COLUMNS}_Supervisor
      COMMAND CastTest
        --in-type ${IN_TYPE}
        --out-type quarter
        --rows=1
        --columns=${COLUMNS}
        --supervisor=1
        --out-offset=0
        --fp8-format=F152
        --fp8-scale=1
        --fp8-format-out=F152
        --fp8-scale-out=1
        LABELS codelet
        VARIANTS ;${SIM_VARIANTS};Sim21)
  endforeach()
  add_multitarget_test(NAME CastTest_quarter_1_float_${COLUMNS}_Supervisor
    COMMAND CastTest
      --in-type float
      --out-type quarter
      --rows=1
      --columns=${COLUMNS}
      --supervisor=1
      --out-offset=0
      --fp8-format=F152
      --fp8-scale=1
      --fp8-format-out=F152
      --fp8-scale-out=1
      LABELS codelet
      VARIANTS Sim21)
endforeach()

foreach(ROWS 1 2)
  foreach(COLUMNS 1 2 3 4 8 15)
    foreach(IN_TYPE half quarter char uchar schar)
      add_multitarget_test(NAME CastTest_quarter_${IN_TYPE}_${ROWS}_${COLUMNS}_Worker
        COMMAND CastTest
          --in-type ${IN_TYPE}
          --out-type quarter
          --rows=${ROWS}
          --columns=${COLUMNS}
          --supervisor=0
          --out-offset=0
          --fp8-format=F143
          --fp8-scale=2
          --fp8-format-out=F143
          --fp8-scale-out=2
          LABELS codelet
          VARIANTS ${SIM_VARIANTS};Sim21)
    endforeach()
    add_multitarget_test(NAME CastTest_quarter_float_${ROWS}_${COLUMNS}_Worker
      COMMAND CastTest
        --in-type float
        --out-type quarter
        --rows=${ROWS}
        --columns=${COLUMNS}
        --supervisor=0
        --out-offset=0
        --fp8-format=F143
        --fp8-scale=2
        --fp8-format-out=F143
        --fp8-scale-out=2
        LABELS codelet
        VARIANTS Sim21)
  endforeach()
endforeach()

foreach(ROWS 1 2)
  foreach(IN_TYPE half quarter char uchar schar)
    add_multitarget_test(NAME CastTest_quarter_${IN_TYPE}_quarter_${ROWS}_100_Worker
        COMMAND CastTest
          --in-type ${IN_TYPE}
          --out-type quarter
          --rows=${ROWS}
          --columns=100
          --supervisor=0
          --out-offset=0
          --fp8-format=F143
          --fp8-scale=2
          --fp8-format-out=F143
          --fp8-scale-out=2
          LABELS codelet
          VARIANTS ${SIM_VARIANTS};Sim21)
  endforeach()
  add_multitarget_test(NAME CastTest_quarter_float_quarter_${ROWS}_100_Worker
      COMMAND CastTest
        --in-type float
        --out-type quarter
        --rows=${ROWS}
        --columns=100
        --supervisor=0
        --out-offset=0
        --fp8-format=F143
        --fp8-scale=2
        --fp8-format-out=F143
        --fp8-scale-out=2
        LABELS codelet
        VARIANTS Sim21)
endforeach()

foreach(ROWS 1 2)
  foreach(OUT_TYPE half quarter char uchar schar)
    add_multitarget_test(NAME CastTest_quarter_${OUT_TYPE}_${ROWS}_100_Worker
        COMMAND CastTest
          --in-type quarter
          --out-type ${OUT_TYPE}
          --rows=${ROWS}
          --columns=100
          --supervisor=0
          --out-offset=0
          --fp8-format=F143
          --fp8-scale=2
          --fp8-format-out=F143
          --fp8-scale-out=2
          LABELS codelet
          VARIANTS ${SIM_VARIANTS};Sim21)
  endforeach()
  add_multitarget_test(NAME CastTest_quarter_float_${ROWS}_100_Worker
      COMMAND CastTest
        --in-type quarter
        --out-type float
        --rows=${ROWS}
        --columns=100
        --supervisor=0
        --out-offset=0
        --fp8-format=F143
        --fp8-scale=2
        --fp8-format-out=F143
        --fp8-scale-out=2
        LABELS codelet
        VARIANTS Sim21)
endforeach()

foreach (COLUMNS 1 2 3 256)
  foreach(SUPERVISOR 0 1)
    foreach(QUARTER_FORMAT F143 F152)
      foreach(SCALE 0 -1)
        add_multitarget_test(NAME CastTest_quarter_half_binary_${SUPERVISOR}_${QUARTER_FORMAT}_${SCALE}_${COLUMNS}
          COMMAND CastTest
            --in-type quarter
            --out-type half
            --rows=1
            --columns=${COLUMNS}
            --supervisor=${SUPERVISOR}
            --out-offset=0
            --fp8-format=${QUARTER_FORMAT}
            --fp8-scale=${SCALE}
            --binary-mode
            LABELS codelet
            VARIANTS ${SIM_VARIANTS};Sim21)
      endforeach()
      foreach(SCALE 0 1)
        add_multitarget_test(NAME CastTest_half_quarter_binary_${SUPERVISOR}_${QUARTER_FORMAT}_${SCALE}_${COLUMNS}
          COMMAND CastTest
            --in-type half
            --out-type quarter
            --rows=1
            --columns=${COLUMNS}
            --supervisor=${SUPERVISOR}
            --out-offset=0
            --fp8-format-out=${QUARTER_FORMAT}
            --fp8-scale-out=${SCALE}
            --binary-mode
            LABELS codelet
            VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()
#Cast<InType, OutType>


add_multi_target_test_executable(CastCodeletsTest CastCodeletsTest.cpp)

# Basic tests to verify that all expected vertices are instantiated, using the
# smallest possible data sizes.
add_multitarget_test(NAME CastCodeletsTest_all_exist
              COMMAND CastCodeletsTest
                      --size 1
              LABELS codelet
              VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS})

# Group some cast tests together to run them with various sizes
function(test_cast TEST_TAG SIZES VARIANTS_TO_USE)
  # For 8 bit types we check with all the possible --align-start
  add_multitarget_test(NAME CastCodeletsTest_from_8bit_types_${TEST_TAG}
                COMMAND CastCodeletsTest
                        --data-type char schar uchar bool
                        --size ${SIZES}
                        --align-start 0 1 2 3 4 5 6 7
                LABELS codelet
                VARIANTS ${VARIANTS_TO_USE})

  add_multitarget_test(NAME CastCodeletsTest_from_16bit_types_${TEST_TAG}
                COMMAND CastCodeletsTest
                        --data-type half short ushort
                        --size ${SIZES}
                        --align-start 0 2
                LABELS codelet
                VARIANTS ${VARIANTS_TO_USE})

  foreach (FP8_FORMAT F143) # TODO T66923: Enable F152 when fixed
    add_multitarget_test(NAME CastCodeletsTest_from_half_to_quarter_${FP8_FORMAT}_${TEST_TAG}
                  COMMAND CastCodeletsTest
                          --data-type half short ushort
                          --cast quarter
                          --size ${SIZES}
                          --fp8-format ${FP8_FORMAT}
                          --align-start 0 2
                  LABELS codelet
                  VARIANTS ${VARIANTS_TO_USE})
  endforeach()

  add_multitarget_test(NAME CastCodeletsTest_from_32_64bit_types_${TEST_TAG}
                COMMAND CastCodeletsTest
                        --data-type float int uint ulonglong longlong
                        --size ${SIZES}
                LABELS codelet
                VARIANTS ${VARIANTS_TO_USE})
endfunction()

# Note that the strings with semicolons are part of CMake syntax for function
# parameters; inside the function CMake will expand them to space-separated lists.
test_cast("small_sizes" "1;2;3;4;5" "${IPUMODEL_VARIANTS};${SIM_VARIANTS}")
test_cast("more_sizes" "6;7;[8,3,6,120];[9,8];22;23;24;25;26;4300" "${IPUMODEL_VARIANTS};Hw")



# Tests all the ScalarMultiply codelets
add_multi_target_test_executable(ScalarMultiplyCodeletsTest
                                 ScalarMultiplyCodeletsTest.cpp)
add_multitarget_test(
      ScalarMultiplyCodeletsTest
      COMMAND ScalarMultiplyCodeletsTest --group
                                         --size 1 2 3 4 5 6 [7,7,7,7] 8 9 10 11
                                                12 13 14 15 16 17 18 19 20 21 22
                                                23 24 25 26 27 48 49 50 51
                                         --tolerance 0 1
      LABELS codelet
      VARIANTS ${SIM_VARIANTS})
