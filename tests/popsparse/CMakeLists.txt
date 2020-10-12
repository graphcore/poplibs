add_subdirectory(codelets)

add_unit_test(SparseFormatsTest SparseFormatsTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
# TODO: T22622: Re-enable/refine these tests. Disabled due to planner timeout but not essential.
#add_unit_test(PopsparseFullyConnectedPlan PopsparseFullyConnectedPlan.cpp VARIANTS ${IPUMODEL_VARIANTS})

add_unit_test(BlockSparseTest BlockSparseTest.cpp VARIANTS "${IPUMODEL_VARIANTS};Hw")
add_unit_test(BlockSparseOpsTest BlockSparseOpsTest.cpp VARIANTS "${IPUMODEL_VARIANTS};Hw")

add_unit_test(SparseFormatsValidateTest SparseFormatsValidateTest.cpp VARIANTS ${IPUMODEL_VARIANTS})

set(SPARSITY_MATRIX ${CMAKE_SOURCE_DIR}/tests/popsparse/bs-m8x8_0.8_nr.txt)
foreach(PART_METHOD "block" "block-naive" "strip" "stripv0")
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
    foreach(PASS_TYPE fwd bwd wu all)

      foreach(BLOCK_SIZE 4 8 16)
        add_multitarget_test(
          NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_${PARTIALS_TYPE}_1024in_1024out_4b_block${BLOCK_SIZE}_0.1sl
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
            --single-phase=${PASS_TYPE})

        add_multitarget_test(
          NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_${PARTIALS_TYPE}_1024in_1024out_8b_block${BLOCK_SIZE}_0.1sl
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
            --single-phase=${PASS_TYPE}
          VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
          NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_${PARTIALS_TYPE}_256in_256out_64b_block${BLOCK_SIZE}_0.1sl
          COMMAND sparse_fc_layer
            --data-type=${DATA_TYPE}
            --partials-type=${PARTIALS_TYPE}
            --input-size=256
            --output-size=256
            --batch-size=64
            --block-size=${BLOCK_SIZE}
            --sparsity-factor=0.01
            --tiles-per-ipu=24
            --matmul-options={\"partitioner.forceBucketSpills\":\"true\"}
            --single-phase=${PASS_TYPE})
      endforeach()
    endforeach()
  endforeach()

  # Tests specifically covering element-wise sparsity
  foreach(SHARED_BUCKETS true false)
    foreach(PASS_TYPE fwd bwd wu all)

      if (${SHARED_BUCKETS} STREQUAL "true" AND (${PASS_TYPE} STREQUAL "fwd" OR ${PASS_TYPE} STREQUAL "wu"))
        continue()
      endif()

      add_multitarget_test(
        NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_float_1024in_1024out_4b_0.1sl_sb_${SHARED_BUCKETS}
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --input-size=1024
          --output-size=1024
          --batch-size=4
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\",\"partitioner.forceBucketSpills\":\"true\",\"availableMemoryProportion\":\"0.6\"}
          --single-phase=${PASS_TYPE})

      add_multitarget_test(
        NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_float_512in_512out_8b_0.1sl_sb_${SHARED_BUCKETS}
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --input-size=512
          --output-size=512
          --batch-size=8
          --sparsity-factor=0.1
          --tiles-per-ipu=24
          --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\",\"partitioner.forceBucketSpills\":\"true\",\"availableMemoryProportion\":\"0.6\"}
          --single-phase=${PASS_TYPE}
        VARIANTS ${TimesOutOnSim})

      add_multitarget_test(
        NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_float_256in_256out_64b_0.1sl_sb_${SHARED_BUCKETS}
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --input-size=256
          --output-size=256
          --batch-size=64
          --sparsity-factor=0.01
          --tiles-per-ipu=24
          --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\",\"partitioner.forceBucketSpills\":\"true\"}
          --single-phase=${PASS_TYPE})

      # This test exists primarily to demonstrate and exercise plan constraints.
      add_multitarget_test(
        NAME sparse_fc_layer_${PASS_TYPE}_${DATA_TYPE}_float_plan_constraints_${SHARED_BUCKETS}
        COMMAND sparse_fc_layer
          --data-type=${DATA_TYPE}
          --input-size=512
          --output-size=512
          --batch-size=64
          --sparsity-factor=0.01
          --tiles-per-ipu=24
          --matmul-options={\"planConstraints\":\"{\\\"partition\\\":{\\\"x\\\":6,\\\"y\\\":4,\\\"z\\\":1}}\"}
          --single-phase=${PASS_TYPE})

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

    # Just a couple of tests to hit combinations of transpose. These are sparse fully connected
    # layers in implementation which should hit majority of test coverage required.
    foreach(TRANSPOSE_LHS 0 1)
      foreach(TRANSPOSE_RHS 0 1)
        add_multitarget_test(
          NAME sparse_matmul_${DATA_TYPE}_float_lhsT${TRANSPOSE_LHS}_rhsT${TRANSPOSE_RHS}_1024m_1024k_4n_0.1sl_sb_${SHARED_BUCKETS}
          COMMAND sparse_matmul
            --data-type=${DATA_TYPE}
            --m=1024
            --k=1024
            --n=4
            --tiles-per-ipu=24
            --sparsity-factor=0.1
            --transpose-lhs=${TRANSPOSE_LHS}
            --transpose-rhs=${TRANSPOSE_RHS}
            --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\"})
        add_multitarget_test(
          NAME sparse_matmul_${DATA_TYPE}_float_lhsT${TRANSPOSE_LHS}_rhsT${TRANSPOSE_RHS}_1024m_256k_16n_0.1sl_sb_${SHARED_BUCKETS}
          COMMAND sparse_matmul
            --data-type=${DATA_TYPE}
            --m=1024
            --k=256
            --n=16
            --tiles-per-ipu=24
            --sparsity-factor=0.1
            --transpose-lhs=${TRANSPOSE_LHS}
            --transpose-rhs=${TRANSPOSE_RHS}
            --matmul-options={\"sharedBuckets\":\"${SHARED_BUCKETS}\"})
      endforeach()
    endforeach()
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
  foreach(XSPLIT 2 4 7)
    foreach(YSPLIT 2 4 7)
      foreach(ZSPLIT 2 4)
        foreach(SPARSITY 0.1 .05)
          foreach(EXCESS .1 .01 .001)
            foreach(ROWS 100)
              foreach(COLS 100)
                foreach(BATCH 8 16)
                  add_test(
                    NAME SparsePartitionerTest_rows${ROWS}_cols${COLS}_B${BATCH}_block${BLOCK_XY}x${BLOCK_XY}_xs${XSPLIT}_ys${YSPLIT}_zs${ZSPLIT}_sp${SPARSITY}_ex${EXCESS}_small
                      COMMAND SparsePartitionerTest
                        --matmul-shape={${ROWS},${COLS},${BATCH}}
                        --split-shape={${XSPLIT},${YSPLIT},${ZSPLIT}}
                        --block-shape={${BLOCK_XY},${BLOCK_XY}}
                        --sparsity-level=${SPARSITY}
                        --excess=${EXCESS}
                    )
                endforeach()
              endforeach()
            endforeach()
            # Run larger tests without generating actual sparsity data impl as
            # we start to hit the limit for the data type currently used
            foreach(ROWS 1000 2400)
              foreach(COLS 1000 2400)
                foreach(BATCH 16 64)
                  add_test(
                    NAME SparsePartitionerTest_rows${ROWS}_cols${COLS}_B${BATCH}_block${BLOCK_XY}x${BLOCK_XY}_xs${XSPLIT}_ys${YSPLIT}_zs${ZSPLIT}_sp${SPARSITY}_ex${EXCESS}_large
                      COMMAND SparsePartitionerTest
                        --matmul-shape={${ROWS},${COLS},${BATCH}}
                        --split-shape={${XSPLIT},${YSPLIT},${ZSPLIT}}
                        --block-shape={${BLOCK_XY},${BLOCK_XY}}
                        --sparsity-level=${SPARSITY}
                        --excess=${EXCESS}
                        --disable-sparsity-data-impl-checks
                    )
                endforeach()
              endforeach()
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Sparse embedding
foreach(DATA_TYPE float half)
  foreach(ROWS 4 67 126)
    foreach(COLUMNS 8 65 127)
      foreach(BATCH_SIZE 4 128)
        if(${COLUMNS} STREQUAL 8)
          set(WEIGHTED_COLUMN_END 6)
        endif()
        if(${COLUMNS} STREQUAL 67)
          set(WEIGHTED_COLUMN_END 32)
        endif()
        if(${COLUMNS} STREQUAL 127)
          set(WEIGHTED_COLUMN_END 100)
        endif()
        foreach(INDICES 1 2 3 4 5 15 16 17 40)
          add_multitarget_test(
            NAME sparse_embedding_tied_matmul_${DATA_TYPE}_R${ROWS}_C${COLUMNS}_B${BATCH_SIZE}_I${INDICES}_W${WEIGHTED_COLUMN_END}
            COMMAND sparse_embedding_tied_matmul
              --data-type=${DATA_TYPE}
              --num-entries=${ROWS}
              --embedding-size=${COLUMNS}
              --batch-size=${BATCH_SIZE}
              --tiles-per-ipu=16
              --num-indices=${INDICES}
              --sparsity-factor=0.2
              --weighted-area-end={2,${WEIGHTED_COLUMN_END}}
              --weighted-area-weighting=1.0)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
