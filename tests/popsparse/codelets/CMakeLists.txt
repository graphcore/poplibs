add_multi_target_test_executable(SparseDenseMatMulElementWise
                                 SparseDenseMatMulElementWise.cpp SparseDenseUtils.cpp SparseDenseUtils.hpp)

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

set(SPARSE_DENSE_BLOCK_VERTEX_TYPES Forward GradA)
foreach(ZERO_PARTIALS "" "ZeroPartials")
  set(ZERO_PARTIALS_OPT)
  if (ZERO_PARTIALS STREQUAL "ZeroPartials")
    set(ZERO_PARTIALS_OPT "--zero-partials")
  endif()
  foreach(VERTEX_TYPE ${SPARSE_DENSE_BLOCK_VERTEX_TYPES})
    foreach(NUM_BUCKETS 1 2)
      foreach(NUM_SPLITS 1)
        foreach(X 2 3)
          foreach(Y 2 3)
            foreach(BLOCKSIZE 4 8 16)
              # Test different batch size paths for input data types
              math(EXPR NUM_ROWS "${X} * ${BLOCKSIZE}")
              math(EXPR NUM_COLS "${Y} * ${BLOCKSIZE}")

              foreach(Z 2 4 12 16 32)
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
                  --vertex-type=${VERTEX_TYPE})
              endforeach()
              foreach(Z 4 8 16 28)
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
              endforeach()
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
