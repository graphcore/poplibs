add_subdirectory(codelets)

add_unit_test(SparseFormatsTest SparseFormatsTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
# TODO: T22622: Re-enable/refine these tests. Disabled due to planner timeout but not essential.
#add_unit_test(PopsparseFullyConnectedPlan PopsparseFullyConnectedPlan.cpp VARIANTS ${IPUMODEL_VARIANTS})

add_unit_test(BlockSparseTest BlockSparseTest.cpp VARIANTS "${IPUMODEL_VARIANTS};Hw")
add_unit_test(BlockSparseOpsTest BlockSparseOpsTest.cpp VARIANTS "${IPUMODEL_VARIANTS};Hw")


add_unit_test(SparseFormatsValidateTest SparseFormatsValidateTest.cpp VARIANTS ${IPUMODEL_VARIANTS})

add_test_executable(ShardedSparseMatMul ShardedSparseMatMul.cpp)

add_multitarget_test(
    NAME ShardedSparseMatMulTest}
    COMMAND ShardedSparseMatMul
    VARIANTS "Hw;${IPUMODEL_VARIANTS}")


set(SPARSITY_MATRIX ${CMAKE_SOURCE_DIR}/tests/popsparse/bs-m8x8_0.8_nr.txt)
foreach(PART_METHOD "block" "block-naive" "strip" "stripv0" "block-group2")
  add_multitarget_test(
    NAME BlockSparseMtTest_dsd_fp32_${PART_METHOD}
    COMMAND bs_matmul_test
      --tiles-per-ipu=16
      --data-type=float
      --sparsity-matrix=${SPARSITY_MATRIX}
      --scenario=dsd
      --lhs-rows=64
      --lhs-block-row=8
      --lhs-block-col=8
      --rhs-block=8
      --partition-method=${PART_METHOD}
      --check-result)

  add_multitarget_test(
    NAME BlockSparseMtTest_dds_fp32_${PART_METHOD}
    COMMAND bs_matmul_test
      --tiles-per-ipu=16
      --data-type=float
      --sparsity-matrix=${SPARSITY_MATRIX}
      --scenario=dds
      --lhs-block-cols=8
      --lhs-block-row=8
      --lhs-block-col=8
      --rhs-block=8
      --partition-method=${PART_METHOD}
      --check-result)

  add_multitarget_test(
    NAME BlockSparseMtTest_dsd_fp16_${PART_METHOD}
    COMMAND bs_matmul_test
      --tiles-per-ipu=16
      --data-type=half
      --sparsity-matrix=${SPARSITY_MATRIX}
      --scenario=dsd
      --lhs-rows=64
      --lhs-block-row=16
      --lhs-block-col=16
      --rhs-block=16
      --check-result)

  add_multitarget_test(
    NAME BlockSparseMtTest_dsd_fp16_need_transpose_${PART_METHOD}
    COMMAND bs_matmul_test
      --tiles-per-ipu=16
      --data-type=half
      --sparsity-matrix=${SPARSITY_MATRIX}
      --scenario=dsd
      --lhs-rows=64
      --lhs-block-row=16
      --lhs-block-col=16
      --rhs-block=16
      --rhs-need-transpose=1
      --check-result)

  add_multitarget_test(
    NAME BlockSparseMtTest_dsd_fp32_groups_${PART_METHOD}
    COMMAND bs_matmul_test
      --tiles-per-ipu=16
      --data-type=float
      --sparsity-matrix=${SPARSITY_MATRIX}
      --scenario=dsd
      --lhs-rows=16
      --lhs-block-row=8
      --lhs-block-col=8
      --rhs-block=8
      --rhs-need-transpose=1
      --number-of-groups=2
      --check-result)
endforeach()

foreach(DATA_TYPE half float)
  # Tests specifically covering block-sparsity
  set(PARTIALS_TYPE_CANDIDATES float)
  if (${DATA_TYPE} STREQUAL half)
    list(APPEND PARTIALS_TYPE_CANDIDATES half)
  endif()
  foreach(PARTIALS_TYPE ${PARTIALS_TYPE_CANDIDATES})
    foreach(BLOCK_SIZE 4 8 16)
      add_multitarget_test(
        NAME sparse_fc_layer_${DATA_TYPE}_${PARTIALS_TYPE}_1024in_1024out_4b_block${BLOCK_SIZE}_0.1sl_eb0
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --partials-type=${PARTIALS_TYPE}
          --input-size=1024
          --output-size=1024
          --batch-size=4
          --block-size=${BLOCK_SIZE}
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --matmul-options={\"partitioner.forceBucketSpills\":\"true\"}
          --plan-constraints={\"exchange\":{\"gradWExchangeBuckets\":false}})

      add_multitarget_test(
        NAME sparse_fc_layer_${DATA_TYPE}_${PARTIALS_TYPE}_1024in_1024out_4b_block${BLOCK_SIZE}_0.1sl_eb1
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --partials-type=${PARTIALS_TYPE}
          --input-size=1024
          --output-size=1024
          --batch-size=4
          --block-size=${BLOCK_SIZE}
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --matmul-options={\"partitioner.forceBucketSpills\":\"true\"}
          --plan-constraints={\"exchange\":{\"gradWExchangeBuckets\":true}})

      add_multitarget_test(
        NAME sparse_fc_layer_${DATA_TYPE}_${PARTIALS_TYPE}_1024in_1024out_8b_block${BLOCK_SIZE}_0.1sl
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --partials-type=${PARTIALS_TYPE}
          --input-size=1024
          --output-size=1024
          --batch-size=8
          --block-size=${BLOCK_SIZE}
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --matmul-options={\"partitioner.forceBucketSpills\":\"true\"}
        VARIANTS ${TimesOutOnSim})

      add_multitarget_test(
        NAME sparse_fc_layer_${DATA_TYPE}_${PARTIALS_TYPE}_256in_256out_64b_block${BLOCK_SIZE}_0.1sl
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --partials-type=${PARTIALS_TYPE}
          --input-size=256
          --output-size=256
          --batch-size=64
          --block-size=${BLOCK_SIZE}
          --sparsity-factor=0.01
          --tiles-per-ipu=24
          --matmul-options={\"partitioner.forceBucketSpills\":\"true\"})
    endforeach()
  endforeach()

  # Tests specifically covering element-wise sparsity
  foreach(SHARED_BUCKETS true false)
    foreach(PASS_TYPE fwd bwd wu)
      if (${SHARED_BUCKETS} STREQUAL "true" AND (${PASS_TYPE} STREQUAL "fwd" OR ${PASS_TYPE} STREQUAL "wu"))
        continue()
      endif()

      add_multitarget_test(
        NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_float_512in_512out_4b_0.1sl_eb0_sb_${SHARED_BUCKETS}
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --input-size=512
          --output-size=512
          --batch-size=4
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\",\"partitioner.forceBucketSpills\":\"true\",\"availableMemoryProportion\":\"0.6\"}
          --plan-constraints={\"exchange\":{\"gradWExchangeBuckets\":false}}
          --single-phase=${PASS_TYPE})

      if (${PASS_TYPE} STREQUAL "wu" OR ${PASS_TYPE} STREQUAL "all")
        add_multitarget_test(
          NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_float_1024in_1024out_4b_0.1sl_eb1_sb_${SHARED_BUCKETS}
          COMMAND sparse_fc_layer
            --data-type=${DATA_TYPE}
            --input-size=512
            --output-size=512
            --batch-size=4
            --sparsity-factor=0.1
            --tiles-per-ipu=24
            --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\",\"partitioner.forceBucketSpills\":\"true\",\"availableMemoryProportion\":\"0.6\"}
            --plan-constraints={\"exchange\":{\"gradWExchangeBuckets\":true}}
            --single-phase=${PASS_TYPE})
      endif()

      if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        # TODO: T31789 - Re-enable these tests in debug once speed issues are resolved
        add_multitarget_test(
          NAME sparse_fc_layer_T27490_${PASS_TYPE}_${DATA_TYPE}_sb_${SHARED_BUCKETS}
          COMMAND sparse_fc_layer
            --data-type=${DATA_TYPE}
            --input-size=784
            --output-size=300
            --batch-size=16
            --sparsity-factor=0.1
            --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\",\"metaInfoBucketOversizeProportion\":\"0.5\"}
            --single-phase=${PASS_TYPE}
            VARIANTS ${TimesOutOnSim})
      endif()

      if (${SHARED_BUCKETS} STREQUAL "true")
        add_multitarget_test(
          NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_float_1024in_1024out_8b_0.1sl_imbalanced_sb_${SHARED_BUCKETS}
          COMMAND sparse_fc_layer
            --data-type=${DATA_TYPE}
            --input-size=1024
            --output-size=1024
            --batch-size=8
            --sparsity-factor=0.1
            --weighted-area-begin={384,128}
            --weighted-area-end={400,786}
            --weighted-area-weight=8
            --tiles-per-ipu=24
            --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\",\"availableMemoryProportion\":\"0.6\"}
            --single-phase=${PASS_TYPE}
          VARIANTS ${TimesOutOnSim})
      endif()
    endforeach()
  endforeach()
endforeach()

# This test exists primarily to demonstrate and exercise plan constraints.
add_multitarget_test(
  NAME sparse_fc_layer_fwd_half_float_plan_constraints
  COMMAND sparse_fc_layer
    --data-type=half
    --input-size=512
    --output-size=512
    --batch-size=64
    --sparsity-factor=0.01
    --tiles-per-ipu=24
    --matmul-options={\"planConstraints\":\"{\\\"partition\\\":{\\\"x\\\":6,\\\"y\\\":4,\\\"z\\\":1}}\"}
    --single-phase=fwd)

# Test reporting of gradw serial splits
add_multitarget_test(
        NAME sparse_fc_layer_report_serial_splits_256in_256out_64b_block4_0.1sl
        COMMAND sparse_fc_layer
          --input-size=256
          --output-size=256
          --batch-size=64
          --block-size=4
          --sparsity-factor=0.01
          --tiles-per-ipu=24
          --matmul-options={\"partitioner.forceBucketSpills\":\"true\"}
          --report-dense-gradw-serial-splits=1)

# Dense version tests
foreach(INPUT_SIZE 32 47 512)
  foreach(OUTPUT_SIZE 23 128)
    set(BLOCK_SIZE 1)
    add_multitarget_test(
      NAME sparse_matmul_force_dense_${PASS}_${INPUT_SIZE}_${OUTPUT_SIZE}
      COMMAND sparse_fc_layer
        --data-type=float
        --partials-type=float
        --input-size=${INPUT_SIZE}
        --output-size=${OUTPUT_SIZE}
        --batch-size=4
        --block-size=${BLOCK_SIZE}
        --sparsity-factor=0.3
        --tiles-per-ipu=24
        --variable-seed
        --plan-constraints={\"useDense\":\"true\"}
        --single-phase=fwd
    )
  endforeach()
endforeach()


# Test when all dimensions overflow
add_multitarget_test(
  NAME sparse_fc_layer_all_half_1024in_1080out_4b_0.1sl_sb_true_exc_0.005
        COMMAND sparse_fc_layer
          --data-type=half --input-size=1024
          --output-size=1080
          --batch-size=4
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --matmul-options={\"sharedBuckets\":\"true\",\"metaInfoBucketOversizeProportion\":\".005\"}
          --single-phase=all)

# Test to disable structure rearrangement
add_multitarget_test(
  NAME sparse_fc_layer_all_half_1024in_1088out_4b_0.1sl_sb_true_exc_0.2_wo_struct_rearr
        COMMAND sparse_fc_layer
          --data-type=half --input-size=512
          --output-size=544
          --batch-size=16
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --block-size=16
          --matmul-options={\"sharedBuckets\":\"true\",\"metaInfoBucketOversizeProportion\":\".2\",\"enableStructuredRearrangements\":\"false\"}
          --single-phase=all)

add_test_executable(SparsePartitionerTest SparsePartitionerTests.cpp)
foreach(BLOCK_XY 1 4)
  foreach(XSPLIT 2 7)
    foreach(YSPLIT 2 7)
      foreach(ZSPLIT 2 4)
        foreach(SPARSITY 0.1 .05)
          foreach(EXCESS .1 .001)
            add_test(
              NAME SparsePartitionerTest_rows100_cols100_B8_block${BLOCK_XY}x${BLOCK_XY}_xs${XSPLIT}_ys${YSPLIT}_zs${ZSPLIT}_sp${SPARSITY}_ex${EXCESS}_small
                COMMAND SparsePartitionerTest
                  --matmul-shape={100,100,8}
                  --split-shape={${XSPLIT},${YSPLIT},${ZSPLIT}}
                  --block-shape={${BLOCK_XY},${BLOCK_XY}}
                  --sparsity-level=${SPARSITY}
                  --excess=${EXCESS})

            # Run larger tests without generating actual sparsity data impl as
            # we start to hit the limit for the data type currently used
            add_test(
              NAME SparsePartitionerTest_rows1000_cols1000_B16_block${BLOCK_XY}x${BLOCK_XY}_xs${XSPLIT}_ys${YSPLIT}_zs${ZSPLIT}_sp${SPARSITY}_ex${EXCESS}_large
                COMMAND SparsePartitionerTest
                  --matmul-shape={1000,1000,16}
                  --split-shape={${XSPLIT},${YSPLIT},${ZSPLIT}}
                  --block-shape={${BLOCK_XY},${BLOCK_XY}}
                  --sparsity-level=${SPARSITY}
                  --excess=${EXCESS}
                  --disable-sparsity-data-impl-checks)
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(BLOCK_SIZE 1 4 8)
  set(BATCH 8)
  foreach(Y_SIZE 24 64)
    foreach(X_SIZE 40 80)
      add_test(NAME SparsePartitionerTest_${BLOCK_SIZE}_${BATCH}_${Y_SIZE}_${X_SIZE}_densepartition
                COMMAND SparsePartitionerTest
                  --matmul-shape={${X_SIZE},${Y_SIZE},${BATCH}}
                  --split-shape={2,2,8}
                  --block-shape={${BLOCK_SIZE},${BLOCK_SIZE}}
                  --sparsity-level=0.1
                  --use-dense=1
                )
    endforeach()
  endforeach()
endforeach()


add_test_executable(StaticSparsePartitionerTest StaticSparsePartitionerTest.cpp)
foreach(BLOCK_LENGTH 1 4 8 16)
  foreach(BATCH_SIZE 1 8)
    foreach(TYPE half float)
      foreach(REP csr coo csc)
        add_test(NAME StaticSparsePartitionerTest_${BLOCK_LENGTH}_128x112x${BATCH_SIZE}_${REP}_${TYPE}
                  COMMAND StaticSparsePartitionerTest
                    --matmul-shape {128,112,${BATCH}}
                    --tiles-per-ipu 4
                    --block-length=${BLOCK_LENGTH}
                    --sparsity-level=0.5
                    --sparse-representation=${REP}
                    --data-type=${TYPE}
                )
        add_test(NAME StaticSparsePartitionerTest_${BLOCK_LENGTH}_32x512x${BATCH_SIZE}_${REP}_${TYPE}
                COMMAND StaticSparsePartitionerTest
                  --matmul-shape {32,512,${BATCH}}
                  --tiles-per-ipu 4
                  --block-length=${BLOCK_LENGTH}
                  --sparsity-level=0.1
                  --sparse-representation=${REP}
                  --data-type=${TYPE}
              )
      endforeach()
    endforeach()
  endforeach()
endforeach()


# Sparse embedding (elementwise sparsity)
foreach(DATA_TYPE float half)
  foreach(ROWS 4 67 126)
    foreach(COLUMNS 8 65)
      set(BATCH_SIZE 4)
      if(${COLUMNS} STREQUAL 8)
        set(WEIGHTED_COLUMN_END 6)
      endif()
      if(${COLUMNS} STREQUAL 67)
        set(WEIGHTED_COLUMN_END 32)
      endif()
      if(${COLUMNS} STREQUAL 127)
        set(WEIGHTED_COLUMN_END 100)
      endif()

      foreach(INDICES 1 2 4 5 16 17)
        add_multitarget_test(
          NAME sparse_embedding_tied_matmul_${DATA_TYPE}_R${ROWS}_C${COLUMNS}_Ba${BATCH_SIZE}_I${INDICES}_W${WEIGHTED_COLUMN_END}
          COMMAND sparse_embedding_tied_matmul
            --data-type=${DATA_TYPE}
            --num-entries=${ROWS}
            --embedding-size=${COLUMNS}
            --batch-size=${BATCH_SIZE}
            --tiles-per-ipu=16
            --num-indices=${INDICES}
            --block-size=1
            --sparsity-factor=0.2
            --matmul-options={\"metaInfoBucketOversizeProportion\":\"0.5\"}
            --weighted-area-end={2,${WEIGHTED_COLUMN_END}}
            --weighted-area-weighting=1.0)
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Sparse embedding (block sparsity) - dims need to be a multiple of the block
# size
foreach(DATA_TYPE float half)
  foreach(ROWS 24 120)
    foreach(COLUMNS 8 24)
      set(BATCH_SIZE 4)
      foreach(BLOCK_ROWS 4 8)
        if(${COLUMNS} STREQUAL 8)
          set(WEIGHTED_COLUMN_END 4)
        endif()
        if(${COLUMNS} STREQUAL 24)
          set(WEIGHTED_COLUMN_END 8)
        endif()
        if(${COLUMNS} STREQUAL 56)
          set(WEIGHTED_COLUMN_END 16)
        endif()
        if(${BLOCK_ROWS} STREQUAL 4)
          set(BLOCK_COLUMNS 2)
        endif()
        if(${BLOCK_ROWS} STREQUAL 8)
          set(BLOCK_COLUMNS 4)
        endif()

        foreach(INDICES 1 2 17)
          add_multitarget_test(
            NAME sparse_embedding_tied_matmul_${DATA_TYPE}_R${ROWS}_C${COLUMNS}_Ba${BATCH_SIZE}_I${INDICES}_W${WEIGHTED_COLUMN_END}_Br${BLOCK_ROWS}_Bc${BLOCK_COLUMNS}
            COMMAND sparse_embedding_tied_matmul
              --data-type=${DATA_TYPE}
              --num-entries=${ROWS}
              --embedding-size=${COLUMNS}
              --batch-size=${BATCH_SIZE}
              --tiles-per-ipu=32
              --num-indices=${INDICES}
              --block-size={${BLOCK_ROWS},${BLOCK_COLUMNS}}
              --test-matmul=false
              --sparsity-factor=0.1
              --matmul-options={\"metaInfoBucketOversizeProportion\":\"0.6\"}
              --weighted-area-end={2,${WEIGHTED_COLUMN_END}}
              --weighted-area-weighting=1.0)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()


foreach(DATA_TYPE half float)
  # Tests specifically covering block-sparsity
  foreach(BLOCK_SIZE 1 4 8 16)
    add_multitarget_test(
      NAME static_sparse_matmul_${DATA_TYPE}_${BLOCK_SIZE}x${BLOCK_SIZE}_${BLOCK_SIZE}_sparsity1_batch1
      COMMAND static_sparse_matmul
          --data-type=${DATA_TYPE}
          --m ${BLOCK_SIZE}
          --k ${BLOCK_SIZE}
          --n 1
          --block-length ${BLOCK_SIZE}
          --sparsity-factor=1
          --tiles-per-ipu=4
        VARIANTS "${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw")

    add_multitarget_test(
        NAME static_sparse_matmul_${DATA_TYPE}_${BLOCK_SIZE}x${BLOCK_SIZE}__${BLOCK_SIZE}_sparsity1_batch4
        COMMAND static_sparse_matmul
          --data-type=${DATA_TYPE}
          --m 64
          --k 64
          --n 32
          --block-length=${BLOCK_SIZE}
          --sparsity-factor=1
          --tiles-per-ipu=4
        VARIANTS "${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw")

    add_multitarget_test(
        NAME static_sparse_matmul_${DATA_TYPE}_${BLOCK_SIZE}x${BLOCK_SIZE}__${BLOCK_SIZE}_sparsity0p01
        COMMAND static_sparse_matmul
          --data-type=${DATA_TYPE}
          --m 512
          --k 64
          --n 8
          --block-length ${BLOCK_SIZE}
          --sparsity-factor=0.01
          --tiles-per-ipu=4
        VARIANTS "${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw")


    add_multitarget_test(
        NAME static_sparse_matmul_${DATA_TYPE}_${BLOCK_SIZE}x${BLOCK_SIZE}__${BLOCK_SIZE}_sparsity0p1_batch13
        COMMAND static_sparse_matmul
          --data-type=${DATA_TYPE}
          --m 512
          --k 64
          --n 13
          --block-length ${BLOCK_SIZE}
          --sparsity-factor=0.1
          --tiles-per-ipu=4
        VARIANTS "${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw")
  endforeach()
endforeach()


add_multitarget_test(
  NAME static_sparse_matmul_half_half_512x8__7_sparsity0p1_options
    COMMAND static_sparse_matmul
      --data-type=half
      --m 512
      --k 8
      --n 7
      --block-length 4
      --sparsity-factor=0.1
      --verbose
      --available-memory-proportion=0.45
      --tiles-per-ipu=4
    VARIANTS "${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw")


add_multitarget_test(
  NAME static_sparse_matmul_half_half_512x8__8_sparsity_allzeros
    COMMAND static_sparse_matmul
      --data-type=half
      --m 512
      --k 8
      --n 7
      --block-length 4
      --sparsity-factor=0.0001
      --verbose
      --tiles-per-ipu=4
    VARIANTS "${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw")

# Note: This test reproduced the issue T71209
add_multitarget_test(
  NAME static_sparse_matmul_half_half_52x129x1__1_nonuniform
  COMMAND static_sparse_matmul
    --data-type float
    --m 52
    --k 129
    --n 1
    --do-dense-sparse 1
    --sparsity-factor 0.2
    --tiles-per-ipu=1
    --sparsity-matrix-file=${CMAKE_CURRENT_SOURCE_DIR}/static_sparse_T71209_mask.txt)
