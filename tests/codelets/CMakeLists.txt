get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

# NonLinearity{Supervisor,2D}<dataType, nlType>
foreach(NL_VERTEX Supervisor 2D)
  set(TEST_EXECUTABLE "NonLinearity${NL_VERTEX}")
  set(TEST_SRC "${TEST_EXECUTABLE}.cpp")
  add_multi_target_test_executable(${TEST_EXECUTABLE} ${TEST_SRC})
  foreach(DATA_TYPE half float)
    foreach(NL_TYPE sigmoid relu gelu tanh)
      set(VARIANT_NAME "NonLinearity${NL_VERTEX}_${DATA_TYPE}_${NL_TYPE}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ${TEST_EXECUTABLE}
                             --data-type=${DATA_TYPE}
                             --nl-type=${NL_TYPE}
                            LABELS codelet)
    endforeach()
  endforeach()
endforeach()

# Loss(LT_TYPE)Transform<dataType>
foreach(DATA_TYPE half float)
  foreach(LT_TYPE SumSquared Softmax)
    foreach(OUTPUT_SCALE 1 32768 65504)
      set(VARIANT_NAME "Loss${LT_TYPE}Transform_${DATA_TYPE}_Scale_${OUTPUT_SCALE}")
      set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
      string(TOUPPER "${DATA_TYPE}" DATA_TYPE_UPPER)
      string(TOUPPER "${LT_TYPE}_LOSS" LT_TYPE_UPPER)
      string(TOUPPER "${OUTPUT_SCALE}" OUTPUT_SCALE_UPPER)
      configure_file(LossTransform.cpp.in ${VARIANT_SRC})
      add_unit_test(${VARIANT_NAME} ${VARIANT_SRC}
                      LABELS codelet)
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
    add_unit_test(${VARIANT_NAME} ${VARIANT_SRC} LABELS codelet)
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
                               LABELS codelet)
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
                               LABELS codelet)
        endforeach()
      endforeach()
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
                            LABELS codelet)
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
                          LABELS codelet)
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
                            LABELS codelet)
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
                      LABELS codelet)
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
                          LABELS codelet)
endforeach()

add_multi_target_test_executable(PartialsEqualSizeReduce PartialsEqualSizeReduce.cpp)
foreach(IN_TYPE half float)
  foreach(OUT_TYPE half float)
    foreach(OPERATION ADD SQUARE_ADD)
      set(VARIANT_NAME "PartialsEqualSizeReduce_${IN_TYPE}_${OUT_TYPE}_${OPERATION}")
      set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
      add_multitarget_test(NAME ${VARIANT_NAME}
                      COMMAND PartialsEqualSizeReduce
                        --partials-type=${IN_TYPE}
                        --out-type=${OUT_TYPE}
                        --outer-dim=3
                        --inner-dim=16
                        --output-dim=8
                        --operation=${OPERATION}
                        LABELS codelet)
    endforeach()
  endforeach()
endforeach()

foreach(TYPE half float)
  foreach(OPERATION MAX MIN)
    foreach(UPDATE true false)
      set(VARIANT_NAME "PartialsEqualSizeReduce_${TYPE}_${OPERATION}_${UPDATE}")
      set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
      add_multitarget_test(NAME ${VARIANT_NAME}
                      COMMAND PartialsEqualSizeReduce
                        --partials-type=${TYPE}
                        --out-type=${TYPE}
                        --outer-dim=3
                        --inner-dim=16
                        --output-dim=8
                        --operation=${OPERATION}
                        --update=${UPDATE}
                        LABELS codelet)
    endforeach()
  endforeach()
endforeach()

add_multi_target_test_executable(ReduceSpecial23 ReduceSpecial23.cpp)
foreach(TYPE half float)
  foreach(OPERATION MAX MIN)
    foreach(UPDATE true false)
      set(VARIANT_NAME "ReduceSpecial3_${TYPE}_${OPERATION}_${UPDATE}")
      set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
      add_multitarget_test(NAME ${VARIANT_NAME}
                      COMMAND ReduceSpecial23
                        --partials-type=${TYPE}
                        --out-type=${TYPE}
                        --outer-dim=5
                        --inner-dim=8
                        --output-dim=8
                        --operation=${OPERATION}
                        --update=${UPDATE}
                        --specialisation=3
                        LABELS codelet)
    endforeach()
  endforeach()
endforeach()

foreach(PARTIALS_TYPE half float)
  foreach(OUT_TYPE half float)
    foreach(OPERATION ADD SQUARE_ADD)
      foreach(UPDATE true false)
        set(VARIANT_NAME "ReduceSpecial3_${PARTIALS_TYPE}_${OUT_TYPE}_${OPERATION}_${UPDATE}")
        set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
        add_multitarget_test(NAME ${VARIANT_NAME}
                        COMMAND ReduceSpecial23
                          --partials-type=${PARTIALS_TYPE}
                          --out-type=${OUT_TYPE}
                          --outer-dim=3
                          --inner-dim=16
                          --output-dim=16
                          --operation=${OPERATION}
                          --update=${UPDATE}
                          --specialisation=3
                          LABELS codelet)
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
                        LABELS codelet)
    endforeach()
  endforeach()
endforeach()

# Select
foreach(DATA_TYPE int float half bool)
  set(VARIANT_NAME "Select_${DATA_TYPE}")
  set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
  string(TOUPPER "${DATA_TYPE}" TYPE)
  set(DATA_HEADER "select/${DATA_TYPE}.hpp")
  configure_file(Select.cpp.in ${VARIANT_SRC})
  configure_file(${DATA_HEADER} ${CMAKE_CURRENT_BINARY_DIR}/${DATA_HEADER} COPYONLY)
  add_unit_test(${VARIANT_NAME} ${VARIANT_SRC} LABELS codelet)
endforeach()

# BroadcastSelect
foreach(DATA_TYPE int float half bool)
  set(VARIANT_NAME "BroadcastSelect_${DATA_TYPE}")
  set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
  string(TOUPPER "${DATA_TYPE}" TYPE)
  set(DATA_HEADER "select/${DATA_TYPE}.hpp")
  configure_file(BroadcastSelect.cpp.in ${VARIANT_SRC})
  configure_file(${DATA_HEADER} ${CMAKE_CURRENT_BINARY_DIR}/${DATA_HEADER} COPYONLY)
  add_unit_test(${VARIANT_NAME} ${VARIANT_SRC} LABELS codelet)
endforeach()

add_unit_test(FloatPointBehaviour
              FloatPointBehaviour.cpp
              VARIANTS Sim
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
                           LABELS codelet)
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
                             LABELS codelet)
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
                              LABELS codelet)
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
                        LABELS codelet)


  # Add some larger tests.
  set(VARIANT_NAME "ReduceNMaxClassGather_top162_${ACTIVATION_TYPE}_200")
  add_multitarget_test(NAME ${VARIANT_NAME}
                        COMMAND ReduceNMaxClassGather
                        --activation-type=${ACTIVATION_TYPE}
                        --divisor=64
                        --size=200
                        --k=162
                        LABELS codelet)

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
                              LABELS codelet)
    endforeach()
  endforeach()
endforeach()

set(VARIANT_NAME "ReduceNMaxClassSparse_top150_float_200")
add_multitarget_test(NAME ${VARIANT_NAME}
                      COMMAND ReduceNMaxClassSparse
                      --size=200
                      --k=150
                      --activation-type=float
                      LABELS codelet)

#Binary / Unary Operations
add_multi_target_test_executable(StdOpTest StdOpTest.cpp)
  foreach(ROWS 1 2 )
    foreach(COLUMNS 1 2 3 4 8 12 24 25 26 27 28)
      foreach(TYPE half float int unsigned)
        foreach(IN_PLACE 0 1)
          if("${TYPE}" STREQUAL "int")
            set(OPERATIONS NEGATE ADD SIGNUM GREATER_THAN)
          elseif("${TYPE}" STREQUAL "unsigned")
            set(OPERATIONS GREATER_THAN MULTIPLY SQUARE)
          else()
            set(OPERATIONS NEGATE ADD SIGNUM GREATER_THAN TAN IS_FINITE IS_INF IS_NAN INVERSE)
          endif()
          foreach(OPERATION ${OPERATIONS})
            set(TEST_OK 1)
            if(IN_PLACE STREQUAL 1 )
              if(OPERATION STREQUAL GREATER_THAN)
                set(TEST_OK 0)
              endif()
              if(OPERATION STREQUAL IS_FINITE OR OPERATION STREQUAL IS_INF OR OPERATION STREQUAL IS_NAN)
                set(TEST_OK 0)
              endif()
            endif()
            if(TEST_OK STREQUAL 1)
              set(VARIANT_NAME "StdOpTest_${TYPE}_${ROWS}_${COLUMNS}_${IN_PLACE}_${OPERATION}")
              add_multitarget_test(NAME ${VARIANT_NAME}
                              COMMAND StdOpTest
                                  --data-type=${TYPE}
                                  --rows=${ROWS}
                                  --columns=${COLUMNS}
                                  --operation=${OPERATION}
                                  --in-place=${IN_PLACE}
                                  LABELS codelet)
            endif()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

#Verify operation of the binary ops fast path
foreach(ROWS 1 2)
  foreach(TYPE half float)
    foreach(IN_PLACE 0 1)
      foreach(IN1_OFFSET 0 16416)
        set(VARIANT_NAME "StdOpTestBinaryFastPath_${ROWS}_${TYPE}_${IN_PLACE}_${IN1_OFFSET}")
          add_multitarget_test(NAME ${VARIANT_NAME}
                              COMMAND StdOpTest
                                --data-type=${TYPE}
                                --rows=${ROWS}
                                --columns=511
                                --operation=SUBTRACT
                                --in1-offset=${IN1_OFFSET}
                                --in-place=${IN_PLACE}
                                LABELS codelet)
        endforeach()
      endforeach()
    endforeach()
  endforeach()

#Broadcasting Operations
add_multi_target_test_executable(BroadcastOpTest BroadcastOpTest.cpp)
foreach(ROWS 1 2)
  foreach(COLUMNS 1 2 3 4 8 12 24 25 26 27 28)
    foreach(TYPE half float)
      foreach(OPERATION INV_STD_DEV_TO_VARIANCE VARIANCE_TO_INV_STD_DEV)
        set(VARIANT_NAME "BroadcastOpTest_inplace_${TYPE}_${ROWS}_${COLUMNS}_${OPERATION}")
        add_multitarget_test(NAME ${VARIANT_NAME}
            COMMAND BroadcastOpTest
              --data-type=${TYPE}
              --rows=${ROWS}
              --columns=${COLUMNS}
              --operation=${OPERATION}
              --in-place=true
              LABELS codelet)
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(SUPERVISOR true false)
  foreach(INPLACE true false)
    foreach(TYPE half float)
      foreach(OPERATION INV_STD_DEV_TO_VARIANCE VARIANCE_TO_INV_STD_DEV)
        set(VARIANT_NAME "BroadcastOpTest_checkAllExist${TYPE}_${SUPERVISOR}_${INPLACE}_${OPERATION}")
        add_multitarget_test(NAME ${VARIANT_NAME}
            COMMAND BroadcastOpTest
              --data-type=${TYPE}
              --rows=2
              --columns=8
              --operation=${OPERATION}
              --in-place=${INPLACE}
              --supervisor=${SUPERVISOR}
              LABELS codelet)
      endforeach()
    endforeach()
  endforeach()
endforeach()

#Verify operation of the broadcast ops fast path
foreach(ROWS 1 2)
  foreach(TYPE half float)
    foreach(IN_PLACE 0 1)
      foreach(IN1_OFFSET 0 16416)
        set(VARIANT_NAME "BroadcastTestFastPath_${ROWS}_${TYPE}_${IN_PLACE}_${IN1_OFFSET}")
        add_multitarget_test(NAME ${VARIANT_NAME}
          COMMAND BroadcastOpTest
            --data-type=${TYPE}
            --rows=${ROWS}
            --columns=511
            --operation=SUBTRACT
            --in1-offset=${IN1_OFFSET}
            --in-place=${IN_PLACE}
            LABELS codelet)
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(SUPER 0 1)
  foreach(TYPE half float)
    foreach(IN1_OFFSET 0 16416)
      set(VARIANT_NAME "BroadcastTestFastPathShort_${SUPER}_${TYPE}_${IN1_OFFSET}")
      add_multitarget_test(NAME ${VARIANT_NAME}
          COMMAND BroadcastOpTest
            --data-type=${TYPE}
            --rows=1
            --columns=8
            --operation=MULTIPLY
            --in1-offset=${IN1_OFFSET}
            --in-place=0
            --supervisor=${SUPER}
            LABELS codelet)
    endforeach()
  endforeach()
endforeach()

#Broadcasting Operations
foreach(INPLACE true false)
  foreach(ROWS 1 3 8)
    foreach(COLUMNS  4 5 20 28)
      foreach(TYPE float half)
        foreach(BY_ROW true false)
          set(VARIANT_NAME "BroadcastOpTest_${TYPE}_${ROWS}_${COLUMNS}_${BY_ROW}_${INPLACE}")
          add_multitarget_test(NAME ${VARIANT_NAME}
              COMMAND BroadcastOpTest
                --b-length=7
                --data-type=${TYPE}
                --rows=${ROWS}
                --columns=${COLUMNS}
                --supervisor=1
                --operation=ADD
                --divide-by-row=${BY_ROW}
                --in-place=${INPLACE}
                LABELS codelet)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()


foreach(COLUMNS 1 2 3 4 8 12 24 25 26 27 28)
  foreach(TYPE half float)
    foreach(OPERATION VARIANCE_TO_INV_STD_DEV INV_STD_DEV_TO_VARIANCE)
      set(VARIANT_NAME "BroadcastOpTest_scalar_lengths_${TYPE}_${COLUMNS}_${OPERATION}")
      add_multitarget_test(NAME ${VARIANT_NAME}
          COMMAND BroadcastOpTest
            --data-type=${TYPE}
            --rows=1
            --b-length=1
            --columns=${COLUMNS}
            --operation=${OPERATION}
            --supervisor=false
            --in-place=false
            LABELS codelet)
    endforeach()
  endforeach()
endforeach()

foreach(SUPERVISOR false true)
  foreach(B_LENGTH 1 3)
    set(VARIANT_NAME "BroadcastOpTest_${SUPERVISOR}_${B_LENGTH}")
    add_multitarget_test(NAME ${VARIANT_NAME}
        COMMAND BroadcastOpTest
          --data-type=float
          --rows=3
          --columns=9
          --operation=ADD
          --supervisor=${SUPERVISOR}
          --b-length=${B_LENGTH}
          --in-place=false
          LABELS codelet)
  endforeach()
endforeach()

add_multitarget_test(NAME  "BroadcastOpTest_multiply_vector_operand"
  COMMAND BroadcastOpTest
    --data-type half
    --rows=4
    --columns=8
    --operation=MULTIPLY
    --supervisor=false
    --b-length=4
    --in-place=true
    LABELS codelet)

add_multitarget_test(NAME  "BroadcastOpTest_add_supervisor_vector_operand"
  COMMAND BroadcastOpTest
    --data-type half
    --rows=4
    --columns=8
    --operation=ADD
    --supervisor=true
    --b-length=4
    --in-place=true
    LABELS codelet)

foreach(B_LENGTH 7 2)
  foreach(TYPE half float)
    foreach(COLUMNS 8 11 32 33)
      add_multitarget_test(NAME  "BroadcastOpTest_add_supervisor_${B_LENGTH}_${TYPE}_${COLUMNS}"
        COMMAND BroadcastOpTest
          --data-type=${TYPE}
          --rows=80
          --columns=${COLUMNS}
          --operation=ADD
          --supervisor=true
          --b-length=${B_LENGTH}
          --in-place=false
          LABELS codelet)
    endforeach()
  endforeach()
endforeach()

foreach(SUPER true false)
  foreach(COLUMNS 1 2 3 4 22 25 )
    foreach(ROWS 1 2)
      add_multitarget_test(NAME  "InvStdDevToVarianceTestCasting_${SUPER}_${COLUMNS}_${ROWS}"
        COMMAND BroadcastOpTest
          --data-type=half
          --rows=${ROWS}
          --columns=${COLUMNS}
          --operation=INV_STD_DEV_TO_VARIANCE
          --supervisor=${SUPER}
          --b-length=1
          --in-place=false
          --cast-out=true
          LABELS codelet)
    endforeach()
  endforeach()
endforeach()

foreach(SUPER true false)
  foreach(COLUMNS 1 2 3 4 22 25 )
    foreach(ROWS 1 2)
      add_multitarget_test(NAME  "VarianceToInvStdDevTestCasting_${SUPER}_${COLUMNS}_${ROWS}"
        COMMAND BroadcastOpTest
          --data-type=float
          --rows=${ROWS}
          --columns=${COLUMNS}
          --operation=VARIANCE_TO_INV_STD_DEV
          --supervisor=${SUPER}
          --b-length=1
          --in-place=false
          --cast-out=true
          LABELS codelet)
    endforeach()
  endforeach()
endforeach()

add_multitarget_test(NAME  "BroadcastOpTest_add_supervisor_vector_b_length_small"
  COMMAND BroadcastOpTest
    --data-type half
    --rows=80
    --columns=8
    --operation=ADD
    --supervisor=true
    --b-length=2
    --in-place=true
    LABELS codelet)

#Cast<InType, OutType>
add_multi_target_test_executable(CastTest CastTest.cpp)

foreach(IN_TYPE half float)
  if(IN_TYPE STREQUAL half)
    set(OUT_TYPE float)
  else()
    set(OUT_TYPE half)
  endif()
  foreach(ROWS 1 2 )
    foreach(COLUMNS RANGE 1 8)
      foreach(OFFSET 0 2)
        set(VARIANT_NAME "CastTest_${IN_TYPE}_${OUT_TYPE}_${ROWS}_${COLUMNS}_${OFFSET}")
        add_multitarget_test(NAME ${VARIANT_NAME}
                      COMMAND CastTest
                          --in-type=${IN_TYPE}
                          --out-type=${OUT_TYPE}
                          --rows=${ROWS}
                          --columns=${COLUMNS}
                          --out-offset=${OFFSET}
                          LABELS codelet)
      endforeach()
    endforeach()
  endforeach()

  set(ROWS 1)
  set(COLUMNS 4300)
  set(OFFSET 0)
  foreach(SUPERVISOR 0 1)
    set(VARIANT_NAME "CastTest_${IN_TYPE}_${OUT_TYPE}_${ROWS}_${COLUMNS}_${OFFSET}_${SUPERVISOR}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                    COMMAND CastTest
                        --in-type=${IN_TYPE}
                        --out-type=${OUT_TYPE}
                        --rows=${ROWS}
                        --columns=${COLUMNS}
                        --out-offset=${OFFSET}
                        --supervisor=${SUPERVISOR}
                        LABELS codelet)
  endforeach()
endforeach()

add_unit_test(AddToChannel
              AddToChannel.cpp
              LABELS codelet)
add_unit_test(AddToChannel2D
              AddToChannel2D.cpp
              LABELS codelet)
add_unit_test(ChannelMul
              ChannelMul.cpp
              LABELS codelet)
add_unit_test(ChannelMul2D
              ChannelMul2D.cpp
              LABELS codelet)
add_unit_test(OuterProductTest
              OuterProductTest.cpp
              LABELS codelet)
add_unit_test(ScaledAdd2D_fp
              ScaledAdd2D_fp.cpp
              SUITES ScaledAdd2DHalfConst ScaledAdd2DHalfTensor ScaledSubtract2DHalfTensor
                     ScaledAdd2DFloatConst ScaledAdd2DFloatTensor ScaledAdd2DHalfHalfFloatConst
                     ScaledAdd2DFloatHalfHalf ScaledAdd2DFloatHalfFloat
                     ScaledAdd2DHalfHalfFloatTensorHighTol ScaledAdd2DHalfHalfFloatTensorLowTol
                     ScaledSubtract2DFloatTensor aXPlusbYHalfConst aXPlusbYHalfTensor
                     XMinusaXPlusbYHalfConst XMinusaXPlusbYHalfTensor aXMinusbYHalfTensor
              LABELS codelet)
add_unit_test(ScaledAdd2D_integral
              ScaledAdd2D_integral.cpp
              SUITES ScaledAdd2DIntConst ScaledAdd2DUnsignedIntConst ScaledAdd2DIntTensor
                     ScaledAdd2DUnsignedIntTensor ScaledSubtract2DIntTensor ScaledSubtract2DUnsignedIntTensor
              LABELS codelet)
add_unit_test(ScaledAddSupervisor_fp
              ScaledAddSupervisor_fp.cpp
              SUITES ScaledAddSupervisorHalfConst ScaledAddSupervisorFloatConst
                     ScaledAddSupervisorFloatHalfHalf ScaledAddSupervisorFloatHalfFloat
                     ScaledAddSupervisorHalfFloatHalfConst ScaledAddSupervisorHalfHalfFloatConst
                     ScaledAddSupervisorHalfHalfFloatTensorHighTol ScaledAddSupervisorHalfHalfFloatTensorLowTol
                     ScaledAddSupervisorHalfTensor ScaledAddSupervisorFloatTensor
                     ScaledAddSupervisorFloatHalfTensor ScaledSubtractSupervisorHalfTensor
                     ScaledSubtractSupervisorFloatTensor ScaledSubtractSupervisorFloatHalfTensor
                     aXPlusbYSupervisorHalfConst aXPlusbYSupervisorHalfTensor
                     XMinusaXPlusbYSupervisorHalfConst XMinusaXPlusbYSupervisorHalfTensor
                     aXMinusbYSupervisorHalfTensor
              LABELS codelet)
add_unit_test(ScaledAddSupervisor_integral
              ScaledAddSupervisor_integral.cpp
              SUITES ScaledAddSupervisorIntConstant ScaledAddSupervisorUnsignedIntConstant
                     ScaledAddSupervisorIntTensor ScaledAddSupervisorUnsignedIntTensor
                     ScaledSubtractSupervisorIntTensor ScaledSubtractSupervisorUnsignedIntTensor
              LABELS codelet)
add_unit_test(Transpose2dTest
              Transpose2dTest.cpp
              LABELS codelet)
add_unit_test(ZeroTest
              ZeroTest.cpp
              LABELS codelet)
add_unit_test(DynamicSliceCodeletTest
              DynamicSliceCodeletTest.cpp
              LABELS codelet)
add_unit_test(DynamicSliceSupervisorCodeletTest
              DynamicSliceSupervisorCodeletTest.cpp
              LABELS codelet)
add_unit_test(MultiSliceCodeletTest
              MultiSliceCodeletTest.cpp
              LABELS codelet)
