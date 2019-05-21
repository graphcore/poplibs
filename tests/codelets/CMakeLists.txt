get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

# NonLinearity{Supervisor,2D}<dataType, nlType>
foreach(NL_VERTEX Supervisor 2D)
  set(TEST_EXECUTABLE "NonLinearity${NL_VERTEX}")
  set(TEST_SRC "${TEST_EXECUTABLE}.cpp")
  add_multi_target_test_executable(${TEST_EXECUTABLE} ${TEST_SRC})
  foreach(DATA_TYPE half float)
    foreach(NL_TYPE sigmoid relu tanh)
      set(VARIANT_NAME "NonLinearity${NL_VERTEX}_${DATA_TYPE}_${NL_TYPE}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ${TEST_EXECUTABLE}
                             --data-type=${DATA_TYPE}
                             --nl-type=${NL_TYPE})
    endforeach()
  endforeach()
endforeach()

# Loss(LT_TYPE)Transform<dataType>
foreach(DATA_TYPE half float)
  foreach(LT_TYPE SumSquared Softmax)
    set(VARIANT_NAME "Loss${LT_TYPE}Transform_${DATA_TYPE}")
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
    string(TOUPPER "${DATA_TYPE}" DATA_TYPE_UPPER)
    string(TOUPPER "${LT_TYPE}_LOSS" LT_TYPE_UPPER)
    configure_file(LossTransform.cpp.in ${VARIANT_SRC})
    add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC})
  endforeach()
endforeach()

#Clamp<InType>
foreach(DATA_TYPE half float int)
   set(VARIANT_NAME "ClampTest_${DATA_TYPE}")
   set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
   string(TOUPPER "${DATA_TYPE}" DATA_TYPE_UPPER)
   configure_file(ClampTest.cpp.in ${VARIANT_SRC})
   add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC} )
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
                               --inner-dim=${INNER_DIM})
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
                               --inner-dim=${INNER_DIM})
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
                            --outer-dim=${OUTER_DIM})
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
  add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC})
endforeach()

include_directories(${ARCH_MAN_PATH}/include)

# Only run this on the simulator as the Cpu doesn't have the cycle count vertex
# and the hardware will give a different result as ciss isn't cycle accurate.
add_popnn_unit_test(Popsys
                    FILES Popsys.cpp Delay1000.gp VARIANTS Sim)

add_custom_target(
  delay1000
  COMMAND
    ${POPC_EXECUTABLE} ${POPC_FLAGS}
    -o Delay1000.gp
    ${CMAKE_CURRENT_SOURCE_DIR}/Delay1000.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Delay1000.S
  DEPENDS
    Delay1000.cpp
    Delay1000.S
    popc_bin
)

if(TARGET Sim_Popsys)
  add_dependencies(Sim_Popsys delay1000)
endif()

add_multi_target_test_executable(ReduceMaxClassGather ReduceMaxClassGather.cpp)
foreach(ACTIVATION_TYPE half float)
  foreach(SIZE RANGE 1 24)
    set(VARIANT_NAME "ReduceMaxClassGather_${ACTIVATION_TYPE}_${SIZE}/4")
    add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ReduceMaxClassGather
                           --activation-type=${ACTIVATION_TYPE}
                           --divisor=4
                           --size=${SIZE})
  endforeach()
endforeach()

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
            set(OPERATIONS NEGATE ADD SIGNUM GREATER_THAN TANH IS_FINITE INVERSE)
          endif()
          foreach(OPERATION ${OPERATIONS})
            set(TEST_OK 1)
            if(IN_PLACE STREQUAL 1 )
              if(OPERATION STREQUAL GREATER_THAN)
                set(TEST_OK 0)
              endif()
              if(OPERATION STREQUAL IS_FINITE)
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
                                  --in-place=${IN_PLACE})
            endif()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

#Broadcasting Operations
add_multi_target_test_executable(BroadcastOpTest BroadcastOpTest.cpp)
foreach(ROWS 1 2)
  foreach(COLUMNS 1 2 3 4 8 12 24 25 26 27 28)
    foreach(TYPE half float)
      set(VARIANT_NAME "BroadcastOpTest_inplace_${TYPE}_${ROWS}_${COLUMNS}")
      add_multitarget_test(NAME ${VARIANT_NAME}
          COMMAND BroadcastOpTest
            --data-type=${TYPE}
            --rows=${ROWS}
            --columns=${COLUMNS}
            --operation=INV_STD_DEV_TO_VARIANCE
            --in-place=true)
    endforeach()
  endforeach()
endforeach()


#Broadcasting Operations
foreach(INPLACE true false)
  foreach(ROWS 1 2 3 4 8)
    foreach(COLUMNS 5 17)
      foreach(TYPE float half)
        set(VARIANT_NAME "BroadcastOpTest_${TYPE}_${ROWS}_${COLUMNS}_${INPLACE}")
        add_multitarget_test(NAME ${VARIANT_NAME}
            COMMAND BroadcastOpTest
              --b-length=7
              --data-type=${TYPE}
              --rows=${ROWS}
              --columns=${COLUMNS}
              --supervisor=1
              --operation=ADD
              --in-place=${INPLACE})
      endforeach()
    endforeach()
  endforeach()
endforeach()


foreach(COLUMNS 1 2 3 4 8 12 24 25 26 27 28)
  foreach(TYPE half float)
    set(VARIANT_NAME "BroadcastOpTest_scalar_lengths_${TYPE}_${COLUMNS}")
    add_multitarget_test(NAME ${VARIANT_NAME}
        COMMAND BroadcastOpTest
          --data-type=${TYPE}
          --rows=2
          --b-length=1
          --columns=${COLUMNS}
          --operation=SUBTRACT
          --in-place=true)
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
          --in-place=false)
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
    --in-place=true)

add_multitarget_test(NAME  "BroadcastOpTest_add_supervisor_vector_operand"
  COMMAND BroadcastOpTest
    --data-type half
    --rows=4
    --columns=8
    --operation=ADD
    --supervisor=true
    --b-length=4
    --in-place=true)

add_multitarget_test(NAME  "BroadcastOpTest_add_supervisor_vector_b_length_large"
  COMMAND BroadcastOpTest
    --data-type half
    --rows=80
    --columns=8
    --operation=ADD
    --supervisor=true
    --b-length=7
    --in-place=true)

add_multitarget_test(NAME  "BroadcastOpTest_add_supervisor_vector_b_length_small"
  COMMAND BroadcastOpTest
    --data-type half
    --rows=80
    --columns=8
    --operation=ADD
    --supervisor=true
    --b-length=2
    --in-place=true)

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
                            --out-offset=${OFFSET})
      endforeach()
    endforeach()
  endforeach()

  set(ROWS 1)
  set(COLUMNS 4300)
  set(OFFSET 0)
  set(VARIANT_NAME "Casttest_${IN_TYPE}_${OUT_TYPE}_${ROWS}_${COLUMNS}_${OFFSET}")
  add_multitarget_test(NAME ${VARIANT_NAME}
                COMMAND CastTest
                    --in-type=${IN_TYPE}
                    --out-type=${OUT_TYPE}
                    --rows=${ROWS}
                    --columns=${COLUMNS}
                    --out-offset=${OFFSET})
endforeach()

add_popnn_unit_test(AddToChannel
                    AddToChannel.cpp)
add_popnn_unit_test(AddToChannel2D
                    AddToChannel2D.cpp)
add_popnn_unit_test(ChannelMul
                    ChannelMul.cpp)
add_popnn_unit_test(ChannelMul2D
                    ChannelMul2D.cpp)
add_popnn_unit_test(OuterProductTest
                    OuterProductTest.cpp)
add_popnn_unit_test(ScaledAdd2D_fp
                    ScaledAdd2D_fp.cpp)
add_popnn_unit_test(ScaledAdd2D_integral
                    ScaledAdd2D_integral.cpp)
add_popnn_unit_test(ScaledAddSupervisor_fp
                    ScaledAddSupervisor_fp.cpp)
add_popnn_unit_test(ScaledAddSupervisor_integral
                    ScaledAddSupervisor_integral.cpp)
add_popnn_unit_test(Transpose2dTest
                    Transpose2dTest.cpp)
add_popnn_unit_test(ZeroTest
                    ZeroTest.cpp)
add_popnn_unit_test(DynamicSliceCodeletTest
                    DynamicSliceCodeletTest.cpp)
add_popnn_unit_test(DynamicSliceSupervisorCodeletTest
                    DynamicSliceSupervisorCodeletTest.cpp)
