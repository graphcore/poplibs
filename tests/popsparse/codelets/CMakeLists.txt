add_multi_target_test_executable(SparseDenseMatMulElementWise
                      SparseDenseMatMulElementWise.cpp
                      SparseDensePartition.cpp SparseDensePartition.hpp
                      SparseDenseUtils.cpp SparseDenseUtils.hpp)

set(SPARSE_DENSE_VERTEX_TYPES Forward GradA GradW Transposed)
foreach(ZERO_PARTIALS "" "ZeroPartials")
  set(ZERO_PARTIALS_OPT)
  if (ZERO_PARTIALS STREQUAL "ZeroPartials")
    set(ZERO_PARTIALS_OPT "--zero-partials")
  endif()
  foreach(VERTEX_TYPE ${SPARSE_DENSE_VERTEX_TYPES})
    foreach(NUM_BUCKETS 1 2)
      foreach(NUM_SPLITS 1 2)
        if (NOT (VERTEX_TYPE STREQUAL "GradW" AND NUM_BUCKETS GREATER 1))
          foreach(X 2 20)
            foreach(Y 20)
              # Test different batch size paths for input data types
              foreach(Z 2 4 12)
                add_multitarget_test(NAME SparseDenseMatMulElementWise_m${Z}_k${Y}_n${X}_sl0.5_${NUM_BUCKETS}bckts_${NUM_SPLITS}splits_float_in_${VERTEX_TYPE}${ZERO_PARTIALS}
                  COMMAND SparseDenseMatMulElementWise
                    ${ZERO_PARTIALS_OPT}
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
                    ${ZERO_PARTIALS_OPT}
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
        endif()
      endforeach()
    endforeach()
  endforeach()
endforeach()

add_multi_target_test_executable(SparseGatherTest
                                  SparseGatherTest.cpp)
foreach(M 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 40 100)
  foreach(TYPE half float)
    add_multitarget_test(NAME SparseGather_Elemwise_${M}_in_${TYPE}
    COMMAND SparseGatherTest
            --data-type=${TYPE}
            --num-indices=${M})
  endforeach()
endforeach()

add_multi_target_test_executable(SparseDenseMatMulBlock
                                 SparseDenseMatMulBlock.cpp SparseDenseUtils.cpp SparseDenseUtils.hpp)

set(SPARSE_DENSE_BLOCK_VERTEX_TYPES Forward GradA GradW GradWAmp)
foreach(ZERO_PARTIALS "" "ZeroPartials")
  set(ZERO_PARTIALS_OPT)
  if (ZERO_PARTIALS STREQUAL "ZeroPartials")
    set(ZERO_PARTIALS_OPT "--zero-partials")
  endif()
  foreach(VERTEX_TYPE ${SPARSE_DENSE_BLOCK_VERTEX_TYPES})
    foreach(NUM_BUCKETS 1 2)
      if (NOT ((VERTEX_TYPE STREQUAL "GradW" OR VERTEX_TYPE STREQUAL "GradWAmp") AND NUM_BUCKETS GREATER 1))
        foreach(NUM_SPLITS 1)
          foreach(X 2 3)
            foreach(Y 2 3)
              foreach(BLOCKSIZE 4 8 16)
                # Test different batch size paths for input data types
                math(EXPR NUM_ROWS "${X} * ${BLOCKSIZE}")
                math(EXPR NUM_COLS "${Y} * ${BLOCKSIZE}")
                if (VERTEX_TYPE STREQUAL "GradW" AND BLOCKSIZE EQUAL 16)
                  set(TimesOutOnSimLocal ${TimesOutOnSim})
                endif()
                foreach(Z 2 4 12 16 32)
                  math(EXPR Z_MODULO_8 "${Z} % 8")
                  if (VERTEX_TYPE STREQUAL "GradWAmp" AND Z_MODULO_8 EQUAL 0)
                    add_multitarget_test(NAME SparseDenseMatMulBlock_m${Z}_k${Y}_block${BLOCKSIZE}_n${X}_sl0.5_${NUM_BUCKETS}bckts_${NUM_SPLITS}splits_float_in_float_out_${VERTEX_TYPE}${ZERO_PARTIALS}
                    COMMAND SparseDenseMatMulBlock
                      ${ZERO_PARTIALS_OPT}
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
                foreach(Z 4 8 16 28 32)
                  math(EXPR Z_MODULO_16 "${Z} % 16")
                  if (VERTEX_TYPE STREQUAL "GradWAmp" AND Z_MODULO_16 EQUAL 0)
                    foreach(PARTIALS_TYPE half float)
                      add_multitarget_test(NAME SparseDenseMatMulBlock_m${Z}_k${Y}_n${X}_block${BLOCKSIZE}_sl0.5_${NUM_BUCKETS}bckts_${NUM_SPLITS}splits_half_in_${PARTIALS_TYPE}_out_${VERTEX_TYPE}${ZERO_PARTIALS}
                        COMMAND SparseDenseMatMulBlock
                          ${ZERO_PARTIALS_OPT}
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
        endforeach()
      endif()
    endforeach()
  endforeach()
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

add_multi_target_test_executable(SparseDenseMultiSliceElementWise
                      SparseDenseMultiSliceElementWise.cpp
                      SparseDensePartition.cpp SparseDensePartition.hpp
                      SparseDenseUtils.cpp SparseDenseUtils.hpp)
foreach(TYPE half float)
  foreach(UPDATE true false)
    foreach(BUCKETS 1 2)
      foreach(ZSIZE 4 8 33 128 1002)
        add_multitarget_test(NAME SparseMultiSlice_${TYPE}_${UPDATE}_${BUCKETS}_${ZSIZE}
          COMMAND SparseDenseMultiSliceElementWise
            --baseT-shape={10,16}
            --offsets=6
            --sparsity-level=0.5
            --num-other-sub-groups=20
            --input-type=${TYPE}
            --num-buckets=${BUCKETS}
            --row-offset=2
            --initialise-subT=true
            --update-add=${UPDATE}
            --z-size=${ZSIZE})
      endforeach()
    endforeach()
  endforeach()
endforeach()
