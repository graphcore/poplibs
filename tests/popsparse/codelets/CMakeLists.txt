add_multi_target_test_executable(SparseDenseMatMulElementWise
                      SparseDenseMatMulElementWise.cpp
                      SparseDensePartitionElementWise.cpp
                      SparseDensePartitionElementWise.hpp
                      SparseDenseUtils.cpp SparseDenseUtils.hpp)

set(SPARSE_DENSE_VERTEX_TYPES Forward GradA GradW Transposed)
set(SPARSE_DENSE_BLOCK_VERTEX_TYPES Forward GradA GradW GradWAmp)

foreach(VERTEX_TYPE ${SPARSE_DENSE_VERTEX_TYPES})
  foreach(NUM_BUCKETS 1 2)
    foreach(NUM_SPLITS 1 2)
      if (VERTEX_TYPE STREQUAL "GradW" AND NUM_BUCKETS GREATER 1)
        continue()
      endif()

      foreach(X 2 20)
        set(Y 20)

        # Test different batch size paths for input data types
        foreach(Z 2 4 12)
          add_multitarget_test(NAME SparseDenseMatMulElementWise_m${Z}_k${Y}_n${X}_sl0.5_${NUM_BUCKETS}bckts_${NUM_SPLITS}splits_float_in_${VERTEX_TYPE}${ZERO_PARTIALS}
            COMMAND SparseDenseMatMulElementWise
              --input-type=float
              --partials-type=float
              --a-shape={${X},${Y}}
              --b-shape={${Y},${Z}}
              --sparsity-level=0.5
              --num-buckets=${NUM_BUCKETS}
              --num-splits-per-bucket=${NUM_SPLITS}
              --vertex-type=${VERTEX_TYPE})
        endforeach()

        foreach(Z 4 8 28)
          add_multitarget_test(NAME SparseDenseMatMulElementWise_m${Z}_k${Y}_n${X}_sl0.5_${NUM_BUCKETS}bckts_${NUM_SPLITS}splits_half_in_${VERTEX_TYPE}${ZERO_PARTIALS}
            COMMAND SparseDenseMatMulElementWise
              --input-type=half
              --partials-type=float
              --a-shape={${X},${Y}}
              --b-shape={${Y},${Z}}
              --sparsity-level=0.5
              --num-buckets=${NUM_BUCKETS}
              --num-splits=${NUM_SPLITS}
              --vertex-type=${VERTEX_TYPE})
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  add_multitarget_test(NAME SparseDenseMatMulElementWise_${VERTEX_TYPE}_ZeroPartials
    COMMAND SparseDenseMatMulElementWise
      --zero-partials
      --input-type=float
      --partials-type=float
      --a-shape={2,20}
      --b-shape={20,12}
      --sparsity-level=0.5
      --num-buckets=1
      --num-splits-per-bucket=2
      --vertex-type=${VERTEX_TYPE})
endforeach()


add_multi_target_test_executable(SparseGatherTest
                                  SparseGatherTest.cpp)
foreach(M 1 2 3 4 5 6 7 8 10 13 15 16 20 24 25 40 100)
  foreach(TYPE half float)
    add_multitarget_test(NAME SparseGather_Elemwise_${M}_in_${TYPE}
    COMMAND SparseGatherTest
            --data-type=${TYPE}
            --num-indices=${M})
  endforeach()
endforeach()

add_multi_target_test_executable(SparseDenseMatMulBlock
                                 SparseDenseMatMulBlock.cpp
                                 SparseDensePartitionBlock.cpp
                                 SparseDensePartitionBlock.hpp
                                 SparseDenseUtils.cpp SparseDenseUtils.hpp)

foreach(VERTEX_TYPE ${SPARSE_DENSE_BLOCK_VERTEX_TYPES})
  foreach(NUM_BUCKETS 1 2)
    if ((VERTEX_TYPE STREQUAL "GradW" OR VERTEX_TYPE STREQUAL "GradWAmp") AND NUM_BUCKETS GREATER 1)
      continue()
    endif()

    set(NUM_SPLITS 1)
    foreach(X 2 3)
      set(Y 2)
      foreach(BLOCKSIZE 4 8 16)
        # Test different batch size paths for input data types
        math(EXPR NUM_ROWS "${X} * ${BLOCKSIZE}")
        math(EXPR NUM_COLS "${Y} * ${BLOCKSIZE}")
        if (VERTEX_TYPE STREQUAL "GradW" AND BLOCKSIZE EQUAL 16)
          set(TimesOutOnSimLocal ${TimesOutOnSim})
        endif()

        foreach(Z 2 12 16)
          math(EXPR Z_MODULO_8 "${Z} % 8")
          if (VERTEX_TYPE STREQUAL "GradWAmp" AND Z_MODULO_8 EQUAL 0)
            add_multitarget_test(NAME SparseDenseMatMulBlock_m${Z}_k${Y}_block${BLOCKSIZE}_n${X}_sl0.5_${NUM_BUCKETS}bckts_${NUM_SPLITS}splits_float_in_float_out_${VERTEX_TYPE}${ZERO_PARTIALS}
            COMMAND SparseDenseMatMulBlock
              --input-type=float
              --partials-type=float
              --a-shape={${NUM_ROWS},${NUM_COLS}}
              --b-shape={${NUM_COLS},${Z}}
              --block-size={${BLOCKSIZE},${BLOCKSIZE}}
              --sparsity-level=0.5
              --num-buckets=${NUM_BUCKETS}
              --num-splits-per-bucket=${NUM_SPLITS}
              --vertex-type=${VERTEX_TYPE}
            VARIANTS ${TimesOutOnSimLocal})
          endif()
        endforeach()

        foreach(Z 4 16 28)
          math(EXPR Z_MODULO_16 "${Z} % 16")
          if (VERTEX_TYPE STREQUAL "GradWAmp" AND Z_MODULO_16 EQUAL 0)
            foreach(PARTIALS_TYPE half float)
              add_multitarget_test(NAME SparseDenseMatMulBlock_m${Z}_k${Y}_n${X}_block${BLOCKSIZE}_sl0.5_${NUM_BUCKETS}bckts_${NUM_SPLITS}splits_half_in_${PARTIALS_TYPE}_out_${VERTEX_TYPE}${ZERO_PARTIALS}
                COMMAND SparseDenseMatMulBlock
                  --input-type=half
                  --partials-type=${PARTIALS_TYPE}
                  --a-shape={${NUM_ROWS},${NUM_COLS}}
                  --b-shape={${NUM_COLS},${Z}}
                  --block-size={${BLOCKSIZE},${BLOCKSIZE}}
                  --sparsity-level=0.5
                  --num-buckets=${NUM_BUCKETS}
                  --num-splits=${NUM_SPLITS}
                  --vertex-type=${VERTEX_TYPE})
            endforeach()
          endif()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  add_multitarget_test(NAME SparseDenseMatMulBlock_${VERTEX_TYPE}_ZeroPartials
    COMMAND SparseDenseMatMulBlock
      --zero-partials
      --input-type=float
      --partials-type=float
      --a-shape={8,24}
      --b-shape={24,16}
      --block-size={4,4}
      --sparsity-level=0.5
      --num-buckets=1
      --num-splits-per-bucket=2
      --vertex-type=${VERTEX_TYPE})
endforeach()


add_multi_target_test_executable(BlockTransposeGradWTest BlockTransposeGradWTest.cpp)

foreach(BLOCKSIZE 4 8 16)
  foreach(XYBLOCK 1 2 4)
    foreach(ZBLOCK 1 2)
      foreach(TYPE half float)
        math(EXPR XYSIZE "${XYBLOCK} * ${BLOCKSIZE}")
        if (TYPE STREQUAL "half")
          math(EXPR ZSIZE "${ZBLOCK} * 16")
        else()
          math(EXPR ZSIZE "${ZBLOCK} * 8")
        endif()

        add_multitarget_test(NAME BlockTransposeGradWTest_BLOCK${BLOCKSIZE}_xy${XYSIZE}_z${ZSIZE}_${TYPE}
        COMMAND BlockTransposeGradWTest
          --data-type=${TYPE}
          --x-y-size=${XYSIZE}
          --z-size=${ZSIZE}
          --block-size=${BLOCKSIZE})
      endforeach()
    endforeach()
  endforeach()
endforeach()

add_multi_target_test_executable(SparseDenseMultiSlice
                      SparseDenseMultiSlice.cpp
                      SparseDensePartitionElementWise.cpp
                      SparseDensePartitionElementWise.hpp
                      SparseDensePartitionBlock.cpp
                      SparseDensePartitionBlock.hpp
                      SparseDenseUtils.cpp SparseDenseUtils.hpp)
foreach(TYPE half float)
  foreach(UPDATE true false)
    foreach(BUCKETS 1 2)
      foreach(ZSIZE 4 36 108)
        foreach(OFFSETS 1 2 4 5 12 13)
          add_multitarget_test(NAME SparseDenseMultiSlice_${TYPE}_${UPDATE}_${BUCKETS}_${ZSIZE}_${OFFSETS}
            COMMAND SparseDenseMultiSlice
              --baseT-shape={41,32}
              --offsets=${OFFSETS}
              --sparsity-level=0.5
              --num-other-sub-groups=3
              --input-type=${TYPE}
              --num-buckets=${BUCKETS}
              --row-offset=2
              --initialise-subT=true
              --update-add=${UPDATE}
              --z-size=${ZSIZE}
              LABELS codelet)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Tests specific to update add, where it helps to vary the size of baseT
# due to that controlling the work division
foreach(TYPE half float)
  foreach(ROWS 7 8 45 67)
    foreach(COLUMNS 1 9 78 100)
      foreach(OFFSETS 10 16)
        add_multitarget_test(NAME SparseDenseMultiSlice_${TYPE}_${ROWS}_${COLUMNS}_${OFFSETS}
            COMMAND SparseDenseMultiSlice
              --baseT-shape={${ROWS},${COLUMNS}}
              --offsets=${OFFSETS}
              --sparsity-level=0.5
              --num-other-sub-groups=3
              --input-type=${TYPE}
              --num-buckets=2
              --row-offset=0
              --initialise-subT=true
              --update-add=true
              --z-size=4
              LABELS codelet)
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Tests for the block-sparse multislice codelets
foreach(TYPE half float)
  foreach(UPDATE true false)
    foreach(BLOCK_ROWS 2 8)
      foreach(BLOCK_COLUMNS 1 4 5)
        foreach(OFFSETS 10 16)
          add_multitarget_test(NAME SparseDenseMultiSliceBlock_${TYPE}_${OFFSETS}_${UPDATE}_${BLOCK_ROWS}_${BLOCK_COLUMNS}
              COMMAND SparseDenseMultiSlice
                --baseT-shape={64,100}
                --offsets=${OFFSETS}
                --sparsity-level=0.5
                --num-other-sub-groups=3
                --input-type=${TYPE}
                --num-buckets=2
                --row-offset=2
                --initialise-subT=true
                --update-add=${UPDATE}
                --block-size={${BLOCK_ROWS},${BLOCK_COLUMNS}}
                LABELS codelet)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
