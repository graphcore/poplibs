# NonLinearity{1D,2D}<dataType, nlType>
foreach(NL_VERTEX 1D 2D)
  set(TEST_EXECUTABLE "NonLinearity${NL_VERTEX}")
  set(TEST_SRC "${TEST_EXECUTABLE}.cpp")
  add_multi_target_test_executable(${TEST_EXECUTABLE} ${TEST_SRC})
  foreach(DATA_TYPE half float)
    foreach(NL_TYPE gelu swish)
      set(VARIANT_NAME "NonLinearity${NL_VERTEX}_${DATA_TYPE}_${NL_TYPE}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ${TEST_EXECUTABLE}
                             --data-type=${DATA_TYPE}
                             --nl-type=${NL_TYPE}
                             --fwd-inplace=true
                            LABELS codelet
                            VARIANTS Cpu;${SIM_VARIANTS})
      if(NL_TYPE STREQUAL "swish")
            add_multitarget_test(NAME ${VARIANT_NAME}_NonInPlace
                           COMMAND ${TEST_EXECUTABLE}
                             --data-type=${DATA_TYPE}
                             --nl-type=${NL_TYPE}
                             --fwd-inplace=false
                            LABELS codelet
                            VARIANTS Cpu;${SIM_VARIANTS})
      endif()
    endforeach()
  endforeach()
endforeach()

# For tanh, relu, gelu_erf and sigmoid the fwd non-linearity is implemented as a unary op
# So only test the bwd (grad) variant here
foreach(NL_VERTEX 1D 2D)
  set(TEST_EXECUTABLE "NonLinearity${NL_VERTEX}")
  foreach(DATA_TYPE half float)
    foreach(NL_TYPE tanh relu gelu_erf sigmoid)
      set(VARIANT_NAME "NonLinearity${NL_VERTEX}_${DATA_TYPE}_${NL_TYPE}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ${TEST_EXECUTABLE}
                             --data-type=${DATA_TYPE}
                             --nl-type=${NL_TYPE}
                             --test-fwd=false
                            LABELS codelet
                            VARIANTS Cpu;${SIM_VARIANTS})
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
                      LABELS codelet
                      VARIANTS Cpu;${SIM_VARIANTS})
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
            LABELS codelet
            VARIANTS Cpu;${SIM_VARIANTS})
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Placeholder tests for prototype vertices
add_multi_target_test_executable(CTCLossCodeletTest CTCLossCodeletTest.cpp)

foreach (VERTEX alpha beta gradGivenAlpha gradGivenBeta)
  if(VERTEX STREQUAL "alpha")
    set(TIME_STEP "3")
  endif()
  if(VERTEX STREQUAL "gradGivenAlpha")
    set(TIME_STEP "8")
  endif()
  if(VERTEX STREQUAL "beta")
    set(TIME_STEP "5")
  endif()
  if(VERTEX STREQUAL "gradGivenBeta")
    set(TIME_STEP "0")
  endif()
  add_multitarget_test(NAME "CTCLossCodeletTest_${VERTEX}"
    COMMAND CTCLossCodeletTest
      --time=8
      --time-step=${TIME_STEP}
      --num-classes=4
      --test=${VERTEX}
    LABELS codelet
    VARIANTS Cpu;${SIM_VARIANTS})
endforeach()

add_multi_target_test_executable(CTCInferenceCodeletTest
  CTCInferenceCodeletTest.cpp
  CTCInferenceCodeletTestConnection.cpp
  CTCInferenceCodeletTestConnection.hpp
  CTCInferenceGenerateCandidates.cpp
  CTCInferenceMergeCandidates.cpp
  CTCInferenceRankAndReduceCandidates.cpp
  CTCInferenceUpdate.cpp
  CTCInferenceGenerateOutput.cpp
)

foreach (SEED RANGE 1 5)
  foreach (TIMESTEP 0 1 4 14)
    foreach (VERTEX generate_copy generate_extend merge update rank output)
      add_multitarget_test(NAME "CTCInferenceCodeletTest_seed_${SEED}_timestep_${TIMESTEP}_${VERTEX}"
          COMMAND CTCInferenceCodeletTest
            --seed=${SEED}
            --beamwidth=5
            --num-classes=25
            --max-time=15
            --timestep=${TIMESTEP}
            --vertex-type=${VERTEX}
          LABELS codelet
          VARIANTS Cpu;${SIM_VARIANTS})
    endforeach()
  endforeach()
endforeach()

# More specific tests for codelets with assembler
foreach (SEED RANGE 1 5)
  foreach (TIMESTEP 0 10)
    foreach (BEAMWIDTH 4 5)
      foreach (CLASSES RANGE 5 8)
        add_multitarget_test(NAME "CTCInferenceCodeletTest_seed_${SEED}_${TIMESTEP}_${CLASSES}_${BEAMWIDTH}_rank"
            COMMAND CTCInferenceCodeletTest
              --seed=${SEED}
              --beamwidth=${BEAMWIDTH}
              --num-classes=${CLASSES}
              --max-time=15
              --timestep=${TIMESTEP}
              --vertex-type=rank
            LABELS codelet
            VARIANTS Cpu;${SIM_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach (SEED RANGE 1 5)
  foreach (PARTITIONS 1 2 3 4 5)
    add_multitarget_test(NAME "CTCInferenceCodeletTest_seed_${SEED}_partitions_${PARTITIONS}_reduce"
        COMMAND CTCInferenceCodeletTest
          --seed=${SEED}
          --beamwidth=5
          --num-classes=25
          --max-time=15
          --timestep=4
          --vertex-type=reduce
          --reduce-partitions=${PARTITIONS}
        LABELS codelet
        VARIANTS Cpu;${SIM_VARIANTS})
  endforeach()
endforeach()

foreach (BEAMWIDTH 4 5 8)
  foreach (INVALID 1 2 3)
    add_multitarget_test(NAME "CTCInferenceCodeletTest_seed_${BEAMWIDTH}_invalid_${INVALID}_update"
        COMMAND CTCInferenceCodeletTest
          --seed=4
          --beamwidth=${BEAMWIDTH}
          --num-classes=25
          --max-time=8
          --timestep=4
          --vertex-type=update
          --update-invalid-candidate=${INVALID}
        LABELS codelet
        VARIANTS Cpu;${SIM_VARIANTS})
  endforeach()
endforeach()
