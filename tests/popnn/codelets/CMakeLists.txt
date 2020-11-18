# NonLinearity{Supervisor,2D}<dataType, nlType>
foreach(NL_VERTEX Supervisor 2D)
  set(TEST_EXECUTABLE "NonLinearity${NL_VERTEX}")
  set(TEST_SRC "${TEST_EXECUTABLE}.cpp")
  add_multi_target_test_executable(${TEST_EXECUTABLE} ${TEST_SRC})
  foreach(DATA_TYPE half float)
    set(VARIANT_NAME "NonLinearity${NL_VERTEX}_${DATA_TYPE}_gelu")
    add_multitarget_test(NAME ${VARIANT_NAME}
                         COMMAND ${TEST_EXECUTABLE}
                           --data-type=${DATA_TYPE}
                           --nl-type=gelu
                          LABELS codelet)
  endforeach()
endforeach()

# For tanh, relu and sigmoid the fwd non-linearity is implemented as a unary op
# So only test the bwd (grad) variant here
foreach(NL_VERTEX Supervisor 2D)
  set(TEST_EXECUTABLE "NonLinearity${NL_VERTEX}")
  foreach(DATA_TYPE half float)
    foreach(NL_TYPE tanh relu sigmoid)
      set(VARIANT_NAME "NonLinearity${NL_VERTEX}_${DATA_TYPE}_${NL_TYPE}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ${TEST_EXECUTABLE}
                             --data-type=${DATA_TYPE}
                             --nl-type=${NL_TYPE}
                             --test-fwd=false
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

add_multi_target_test_executable(PoolingCodeletTest PoolingCodeletTest.cpp)
foreach (FIELD 8 9 10 11 12)
  foreach (KERNEL 2 3 4)
    foreach (CHANNELS 8 16)
      foreach (CHANNEL_GROUPS 1 2)
        foreach (DATA_TYPE half float)
          foreach(POOLING_TYPE max avg sum)
            add_multitarget_test(NAME "PoolingCodelet_f${FIELD}_k${KERNEL}_c${CHANNELS}_g${CHANNEL_GROUPS}_${DATA_TYPE}_${POOLING_TYPE}"
            COMMAND PoolingCodeletTest
              --field=${FIELD}
              --kernel=${KERNEL}
              --channels=${CHANNELS}
              --pooling-type=${POOLING_TYPE}
              --data-type=${DATA_TYPE}
              --channel-groups=${CHANNEL_GROUPS}
            LABELS codelet)
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
