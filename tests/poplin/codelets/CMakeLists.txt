add_unit_test(ConvPartial1x1Out
              ConvPartial1x1Out.cpp
              LABELS codelet)
add_unit_test(OuterProductTest
              OuterProductTest.cpp
              LABELS codelet)

add_multi_target_test_executable(ConvPartial1xNSLIC ConvPartial1xNSLIC.cpp)
foreach(MODE 0 1 2)
  if (${MODE} EQUAL 0)
    set(CONV_GROUPS_PER_GROUP 4)
    set(CHANS_PER_GROUP 1)
  elseif(${MODE} EQUAL 1)
    set(CONV_GROUPS_PER_GROUP 2)
    set(CHANS_PER_GROUP 2)
  else()
    set(CONV_GROUPS_PER_GROUP 1)
    set(CHANS_PER_GROUP 4)
  endif()
  foreach(OUT_STRIDE 1 2)
    foreach(PARTIALS_TYPE half float)

      if(${PARTIALS_TYPE} STREQUAL "half")
        set(CONV_CHAIN_VARIANTS "2;4")
        set(SLIC_TEST_VARIANTS "Cpu;Sim2")
      else()
        set(CONV_CHAIN_VARIANTS "2")
        set(SLIC_TEST_VARIANTS "Cpu")
      endif()

      foreach(CONV_CHAINS ${CONV_CHAIN_VARIANTS})
        string(CONCAT TEST_NAME_PREFIX
               "ConvPartial1x4SLIC" "_"
               "cgpg${CONV_GROUPS_PER_GROUP}" "_"
               "cpg${CHANS_PER_GROUP}" "_"
               "st${OUT_STRIDE}" "_"
               "par${PARTIALS_TYPE}" "_"
               "conv${CONV_CHAINS}")

        add_multitarget_test(NAME "${TEST_NAME_PREFIX}_x4_y4_kx4_ky1"
          COMMAND ConvPartial1xNSLIC
            --input-type=half
            --partials-type=${PARTIALS_TYPE}
            --conv-chains=${CONV_CHAINS}
            --conv-groups-per-group=${CONV_GROUPS_PER_GROUP}
            --chans-per-group=${CHANS_PER_GROUP}
            --in-field-size={4,4}
            --kernel-size={1,4}
            --output-stride={1,${OUT_STRIDE}}
            LABELS codelet
          VARIANTS ${SLIC_TEST_VARIANTS})

        add_multitarget_test(NAME "${TEST_NAME_PREFIX}_cgg4_x20_y4_kx4_ky4"
          COMMAND ConvPartial1xNSLIC
            --input-type=half
            --partials-type=${PARTIALS_TYPE}
            --conv-chains=${CONV_CHAINS}
            --conv-groups-per-group=${CONV_GROUPS_PER_GROUP}
            --chans-per-group=${CHANS_PER_GROUP}
            --conv-group-groups=4
            --in-field-size={4,20}
            --kernel-size={4,4}
            --output-stride={1,${OUT_STRIDE}}
            LABELS codelet
          VARIANTS ${SLIC_TEST_VARIANTS})

        # Intentionally even no. of 1x4 sub-kernels
        add_multitarget_test(NAME "${TEST_NAME_PREFIX}_bs4_cgg4_x24_y24_kx8_ky8"
          COMMAND ConvPartial1xNSLIC
            --input-type=half
            --partials-type=${PARTIALS_TYPE}
            --conv-chains=${CONV_CHAINS}
            --conv-groups-per-group=${CONV_GROUPS_PER_GROUP}
            --chans-per-group=${CHANS_PER_GROUP}
            --conv-group-groups=4
            --batch-size=4
            --in-field-size={24,24}
            --kernel-size={8,8}
            --output-stride={1,${OUT_STRIDE}}
          LABELS codelet
        VARIANTS ${SLIC_TEST_VARIANTS})

        # Intentionally odd no. of 1x4 sub-kernels
        add_multitarget_test(NAME "${TEST_NAME_PREFIX}_bs4_cgg4_x24_y24_kx12_ky3"
          COMMAND ConvPartial1xNSLIC
            --input-type=half
            --partials-type=${PARTIALS_TYPE}
            --conv-chains=${CONV_CHAINS}
            --conv-groups-per-group=${CONV_GROUPS_PER_GROUP}
            --chans-per-group=${CHANS_PER_GROUP}
            --conv-group-groups=4
            --batch-size=4
            --in-field-size={24,24}
            --kernel-size={3,12}
            --output-stride={1,${OUT_STRIDE}}
          LABELS codelet
          VARIANTS ${SLIC_TEST_VARIANTS})

        # Output padding
        add_multitarget_test(NAME "${TEST_NAME_PREFIX}_output_padding"
          COMMAND ConvPartial1xNSLIC
            --input-type=half
            --partials-type=${PARTIALS_TYPE}
            --conv-chains=${CONV_CHAINS}
            --conv-groups-per-group=${CONV_GROUPS_PER_GROUP}
            --chans-per-group=${CHANS_PER_GROUP}
            --conv-group-groups=1
            --batch-size=1
            --in-field-size={3,2,5}
            --kernel-size={2,1,4}
            --output-padding-lower={1,1,1}
            --output-padding-upper={1,1,2}
            --output-stride={1,1,${OUT_STRIDE}}
          LABELS codelet
          VARIANTS ${SLIC_TEST_VARIANTS})

        # T18510 - Try and catch overreads with NaNs
        add_multitarget_test(NAME "${TEST_NAME_PREFIX}_bs1_cgg1_x70_kx4"
          COMMAND ConvPartial1xNSLIC
            --input-type=half
            --partials-type=${PARTIALS_TYPE}
            --conv-chains=${CONV_CHAINS}
            --conv-groups-per-group=${CONV_GROUPS_PER_GROUP}
            --chans-per-group=${CHANS_PER_GROUP}
            --conv-group-groups=1
            --batch-size=1
            --in-field-size={3,2,5}
            --kernel-size={2,1,4}
            --output-padding-lower={1,1,1}
            --output-padding-upper={1,1,2}
            --output-stride={1,1,${OUT_STRIDE}}
          LABELS codelet
          VARIANTS ${SLIC_TEST_VARIANTS})
        endforeach() # CONV_UNIT_VARIANTS
    endforeach() # PARTIALS_TYPE
  endforeach() # OUT_STRIDE
endforeach() # MODE


add_multitarget_test(NAME "ConvPartial1xNSLIC_disable_sr"
COMMAND ConvPartial1xNSLIC
  --input-type=half
  --partials-type=half
  --conv-chains=2
  --conv-groups-per-group=4
  --chans-per-group=1
  --in-field-size={4,4}
  --kernel-size={1,4}
  --output-stride={1,2}
  --disable-sr-template-arg=true
  LABELS codelet
VARIANTS "Sim2;IpuModel2")


add_multitarget_test(NAME "ConvPartial1xNSLIC_disable_sr_cgg4_x20_y4_kx4_ky4"
COMMAND ConvPartial1xNSLIC
  --input-type=half
  --partials-type=float
  --conv-chains=2
  --conv-groups-per-group=4
  --chans-per-group=1
  --conv-group-groups=4
  --in-field-size={4,20}
  --kernel-size={4,4}
  --output-stride={1,1}
  --disable-sr-template-arg=true
  LABELS codelet
  VARIANTS "Sim2;IpuModel2")

foreach(OUT_STRIDE 2 3 4)
  add_multitarget_test(NAME "ConvPartial1x4SLIC_stride_1_${OUT_STRIDE}_1"
    COMMAND ConvPartial1xNSLIC
      --input-type=half
      --partials-type=float
      --conv-groups-per-group=4
      --chans-per-group=1
      --conv-group-groups=1
      --batch-size=1
      --in-field-size={10,8,11}
      --kernel-size={1,1,4}
      --output-stride={1,${OUT_STRIDE},1}
    LABELS codelet
    VARIANTS "Cpu;${DEFAULT_TEST_VARIANTS}")

  add_multitarget_test(NAME "ConvPartial1x4SLIC_stride_${OUT_STRIDE}_${OUT_STRIDE}_2"
    COMMAND ConvPartial1xNSLIC
      --input-type=half
      --partials-type=float
      --conv-groups-per-group=4
      --chans-per-group=1
      --conv-group-groups=1
      --batch-size=1
      --in-field-size={10,8,11}
      --kernel-size={1,1,4}
      --output-stride={${OUT_STRIDE},${OUT_STRIDE},2}
    LABELS codelet
    VARIANTS "Cpu;${DEFAULT_TEST_VARIANTS}")
endforeach() # OUT_STRIDE

foreach(LOWER_TRUNCATION 0 1 2)
  foreach(UPPER_TRUNCATION 0 1 3)
    add_multitarget_test(NAME "ConvPartial1x4SLIC_truncation1_upper${UPPER_TRUNCATION}_lower${LOWER_TRUNCATION}"
      COMMAND ConvPartial1xNSLIC
        --input-type=half
        --partials-type=float
        --conv-groups-per-group=4
        --chans-per-group=1
        --conv-group-groups=1
        --batch-size=1
        --in-field-size={10,8,11}
        --kernel-size={1,1,4}
        --output-truncation-lower={0,0,${LOWER_TRUNCATION}}
        --output-truncation-upper={0,0,${UPPER_TRUNCATION}}
      LABELS codelet
      VARIANTS "Cpu;${DEFAULT_TEST_VARIANTS}")

    add_multitarget_test(NAME "ConvPartial1x4SLIC_truncation2_upper${UPPER_TRUNCATION}_lower${LOWER_TRUNCATION}"
      COMMAND ConvPartial1xNSLIC
        --input-type=half
        --partials-type=float
        --conv-groups-per-group=4
        --chans-per-group=1
        --conv-group-groups=1
        --batch-size=1
        --in-field-size={10,8,11}
        --kernel-size={1,1,4}
        --output-truncation-upper={0,1,${UPPER_TRUNCATION}}
        --output-truncation-lower={1,${LOWER_TRUNCATION},0}
      LABELS codelet
      VARIANTS "Cpu;${DEFAULT_TEST_VARIANTS}")
  endforeach() # UPPER_TRUNCATION
endforeach() # LOWER_TRUNCATION

foreach (STRIDE 1 2)
  foreach(KERNEL_SIZE 4 8 12)
    foreach(FIELD_SIZE 12 13 14 15 16 32 256)
      add_multitarget_test(NAME "ConvPartial1x4SLIC_quarter_${KERNEL_SIZE}_${FIELD_SIZE}_${STRIDE}"
        COMMAND ConvPartial1xNSLIC
          --input-type=quarter
          --partials-type=half
          --conv-groups-per-group=1
          --chans-per-group=8
          --conv-group-groups=3
          --batch-size=2
          --in-field-size={4,${FIELD_SIZE}}
          --kernel-size={1,${KERNEL_SIZE}}
          --output-stride={1,${STRIDE}}
          --conv-chains=4
        LABELS codelet
        VARIANTS "Sim21")
    endforeach()
  endforeach()
endforeach()

foreach(STRIDE 1 2)
  foreach(FIELD_SIZE 4 5 6 7 8 9 10)
    add_multitarget_test(NAME "ConvPartial1x4SLIC_quarter_kernel_4_${FIELD_SIZE}_${STRIDE}"
      COMMAND ConvPartial1xNSLIC
        --input-type=quarter
        --partials-type=half
        --conv-groups-per-group=1
        --chans-per-group=8
        --conv-group-groups=1
        --batch-size=1
        --in-field-size={1,${FIELD_SIZE}}
        --kernel-size={1,4}
        --output-stride={1,${STRIDE}}
        --conv-chains=4
      LABELS codelet
      VARIANTS "Sim21")
  endforeach()
endforeach()
