add_subdirectory(codelets)

add_unit_test(CholeskyTest CholeskyTest.cpp)
add_unit_test(ConvOptionsTest ConvOptionsTest.cpp)
add_unit_test(ConvPlanTest ConvPlanTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(ConvTest ConvTest.cpp)
add_unit_test(ConvUtilTest ConvUtilTest.cpp)
add_unit_test(MeshGridTest MeshGridTest.cpp)
add_unit_test(MultiConvolutionPlanTest MultiConvolutionPlanTest.cpp)
add_unit_test(MultiConvolutionTest MultiConvolutionTest.cpp)
add_unit_test(RangeTest RangeTest.cpp)
add_unit_test(WinogradConvolution WinogradConv.cpp VARIANTS ${TimesOutOnSim})

# Triangular solver tests
add_multi_target_test_executable(TriangularSolveTest TriangularSolveTest.cpp)
foreach(DATA_TYPE half float)
  foreach(LEFT_SIDE false true)
    foreach(LOWER_TRIANGLE false true)
      foreach(UNIT_DIAGONAL false true)
        foreach(BLOCK_SOLVER false true)
          foreach(K 1 2 3 8)
            add_multitarget_test(NAME TriangularSolveTest_${DATA_TYPE}_${LEFT_SIDE}_${LOWER_TRIANGLE}_${UNIT_DIAGONAL}_${BLOCK_SOLVER}_${K}
              COMMAND TriangularSolveTest
                --type=${DATA_TYPE}
                --left-side=${LEFT_SIDE}
                --lower=${LOWER_TRIANGLE}
                --unit-diagonal=${UNIT_DIAGONAL}
                --block-solver=${BLOCK_SOLVER}
                --k=${K}
            )
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Norm layer tests.

add_multitarget_test(NAME BatchNormConv_Batch2_Dim28x28_Ch32_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=false
    --norm-type=BN)

add_multitarget_test(NAME BatchNormConv_Batch2_Dim28x28_Ch32_SmallEps_stable
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=true
    --norm-type=BN)

foreach (STRIDED_GROUPING true false)
  add_multitarget_test(NAME GroupNormConv_Batch2_Dim28x28_Ch32_SmallEps_${STRIDED_GROUPING}
    COMMAND norm_layer
      --eps=0.00001
      --learning-rate=0.01
      --data-type=half
      --unbiased-var-est=1
      --partials-type=float
      --tiles-per-ipu=16
      --dims={2,32,28,28}
      --num-groups=2
      --stable-algo-for-stats=false
      --norm-type=GN
      --strided-channel-grouping=${STRIDED_GROUPING})

  add_multitarget_test(NAME GroupNormConv_Batch2_Dim28x28_Ch32_SmallEps_stable_${STRIDED_GROUPING}
    COMMAND norm_layer
      --eps=0.00001
      --learning-rate=0.01
      --data-type=half
      --unbiased-var-est=1
      --partials-type=float
      --tiles-per-ipu=16
      --dims={2,32,28,28}
      --num-groups=2
      --stable-algo-for-stats=true
      --norm-type=GN
      --strided-channel-grouping=${STRIDED_GROUPING})
endforeach()

add_multitarget_test(NAME GroupNormConv_Batch3_Dim4x4_Ch6_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={3,6,4,4}
    --num-groups=3
    --stable-algo-for-stats=true
    --norm-type=GN)

add_multitarget_test(NAME InstanceNormConv_Batch2_Dim28x28_Ch32_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=false
    --norm-type=IN)


add_multitarget_test(NAME InstanceNormConv_Batch2_Dim28x28_Ch32_SmallEps_stable
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=true
    --norm-type=IN)

add_multitarget_test(NAME LayerNormConv_Batch2_Dim28x28_Ch32_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=false
    --norm-type=LN)

add_multitarget_test(NAME LayerNormConv_Batch2_Dim28x28_Ch32_SmallEps_stable
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --data-type=half
    --unbiased-var-est=1
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,32,28,28}
    --stable-algo-for-stats=true
    --norm-type=LN)

add_multitarget_test(NAME BatchNormConv_Batch4_Dim20x20_Ch32_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=24
    --dims={4,32,20,20}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormConv_Batch4_Dim20x20_Ch32_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=24
    --dims={4,32,20,20}
    --num-groups=16
    --norm-type=GN)

add_multitarget_test(NAME InstanceNormConv_Batch4_Dim20x20_Ch32_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=24
    --dims={4,32,20,20}
    --norm-type=IN)

add_multitarget_test(NAME LayerNormConv_Batch4_Dim20x20_Ch32_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=24
    --dims={4,32,20,20}
    --norm-type=LN)

add_multitarget_test(NAME BatchNormConv_Batch16_Dim7x7_Ch8
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,8,7,7}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormConv_Batch16_Dim7x7_Ch8
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,8,7,7}
    --num-groups=4
    --norm-type=GN)

add_multitarget_test(NAME InstanceNormConv_Batch16_Dim7x7_Ch8
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,8,7,7}
    --norm-type=IN)

add_multitarget_test(NAME LayerNormConv_Batch16_Dim7x7_Ch8
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,8,7,7}
    --norm-type=LN)

add_multitarget_test(NAME BatchNormConv_Batch1_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56,56}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormConv_Batch1_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56,56}
    --num-groups=2
    --norm-type=GN)

add_multitarget_test(NAME InstanceNormConv_Batch1_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56,56}
    --norm-type=IN)

add_multitarget_test(NAME LayerNormConv_Batch1_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56,56}
    --norm-type=LN)

add_multitarget_test(NAME BatchNormConv_Batch1_DataFhalf_field_dims_1
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,56}
    --norm-type=BN)

add_multitarget_test(NAME BatchNormConv_Batch1_DataFhalf_field_dims_3
  COMMAND norm_layer
    --eps=0.0001
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={1,8,14,2,7}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormFc_Batch4_Acts2048
  COMMAND norm_layer
    --eps=0.001
    --unbiased-var-est=0
    --learning-rate=0.01
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={4,2048}
    --num-groups=64
    --stable-algo-for-stats=false
    --norm-type=GN)

add_multitarget_test(NAME GroupNormFc_Batch4_Acts2048_stable
  COMMAND norm_layer
    --eps=0.001
    --unbiased-var-est=0
    --learning-rate=0.01
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={4,2048}
    --num-groups=64
    --stable-algo-for-stats=true
    --norm-type=GN)

add_multitarget_test(NAME LayerNormFc_Batch4_Acts2048
  COMMAND norm_layer
    --eps=0.001
    --unbiased-var-est=0
    --learning-rate=0.01
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={4,2048}
    --norm-type=LN)


add_multitarget_test(NAME BatchNormFc_Batch16_Acts256_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,256}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormFc_Batch16_Acts256_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,256}
    --num-groups=16
    --norm-type=GN)

add_multitarget_test(NAME LayerNormFc_Batch16_Acts256_SmallEps
  COMMAND norm_layer
    --eps=0.00001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={16,256}
    --norm-type=LN)

add_multitarget_test(NAME BatchNormFc_Batch8_Acts512_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormFc_Batch8_Acts512_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --num-groups=16
    --norm-type=GN)

add_multitarget_test(NAME LayerNormFc_Batch8_Acts512_LargeEps
  COMMAND norm_layer
    --eps=0.01
    --learning-rate=0.01
    --unbiased-var-est=0
    --data-type=half
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --num-groups=16
    --norm-type=LN)

add_multitarget_test(NAME BatchNormFc_Batch8_Acts512_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --norm-type=BN)

add_multitarget_test(NAME GroupNormFc_Batch8_Acts512_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --num-groups=8
    --norm-type=GN)

add_multitarget_test(NAME LayerNormFc_Batch8_Acts512_DataFloat_PartialsFloat
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={8,256}
    --norm-type=LN)

add_multitarget_test(NAME InstanceNormCOnv_Batch2_field_dims_4
  COMMAND norm_layer
    --eps=0.001
    --learning-rate=0.01
    --unbiased-var-est=1
    --data-type=float
    --partials-type=float
    --tiles-per-ipu=16
    --dims={2,4,6,7,8,9}
    --norm-type=IN)

# General matrix multiply tests.

# the ignore data option disables validation, this test basically just checks
# that the option or any of the profiling options do not cause a crash.
add_multitarget_test(
  NAME gemm_5x0x4_ignore_data_profile
  COMMAND general_matrix_multiply
          --m 5
          --k 0
          --n 4
          --tiles-per-ipu=1
          --ignore-data
          --profile
          --report-plan
          --show-execution-steps
          --show-var-storage)

add_multitarget_test(
  NAME gemm_5x0x4_groups3_ignore_data_profile
  COMMAND general_matrix_multiply
          --g 3
          --m 5
          --k 0
          --n 4
          --tiles-per-ipu=1
          --ignore-data
          --profile
          --report-plan
          --show-execution-steps
          --show-var-storage)

add_multitarget_test(
  NAME gemm_5x0x4
  COMMAND general_matrix_multiply
          --m 5
          --k 0
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=normal
          --right-matrix-op=normal)

add_multitarget_test(
  NAME gemm_0x3x4
  COMMAND general_matrix_multiply
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=normal
          --right-matrix-op=normal)

add_multitarget_test(
  NAME gemm_5x3x0
  COMMAND general_matrix_multiply
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=normal
          --right-matrix-op=normal)


add_multitarget_test(
  NAME gemm_5x0x4_trA
  COMMAND general_matrix_multiply
          --m 5
          --k 0
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=transpose
          --right-matrix-op=normal)

add_multitarget_test(
  NAME gemm_0x3x4_trA
  COMMAND general_matrix_multiply
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=transpose
          --right-matrix-op=normal)

add_multitarget_test(
  NAME gemm_5x3x0_trA
  COMMAND general_matrix_multiply
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=transpose
          --right-matrix-op=normal)

add_multitarget_test(
  NAME gemm_5x0x4_trB
  COMMAND general_matrix_multiply
          --m 5
          --k 0
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=normal
          --right-matrix-op=transpose)

add_multitarget_test(
  NAME gemm_0x3x4_trB
  COMMAND general_matrix_multiply
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=normal
          --right-matrix-op=transpose)

add_multitarget_test(
  NAME gemm_5x3x0_trB
  COMMAND general_matrix_multiply
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=normal
          --right-matrix-op=transpose)

add_multitarget_test(
  NAME gemm_5x0x4_trA_trB
  COMMAND general_matrix_multiply
          --m 5
          --k 0
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=transpose
          --right-matrix-op=transpose)

add_multitarget_test(
  NAME gemm_0x3x4_trA_trB
  COMMAND general_matrix_multiply
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=transpose
          --right-matrix-op=transpose)

add_multitarget_test(
  NAME gemm_5x3x0_trA_trB
  COMMAND general_matrix_multiply
          --m 0
          --k 3
          --n 4
          --alpha 1
          --beta 1
          --tiles-per-ipu=1
          --left-matrix-op=transpose
          --right-matrix-op=transpose)

add_multitarget_test(
  NAME gemm_1x3x1
  COMMAND general_matrix_multiply
          --m 1
          --k 3
          --n 1
          --alpha 1
          --beta 1
          --tiles-per-ipu=16
          --left-matrix-op=normal
          --right-matrix-op=normal)

add_multitarget_test(
NAME gemm_1x1000x1
COMMAND general_matrix_multiply
        --m 1
        --k 1000
        --n 1
        --alpha 2
        --beta 1
        --tiles-per-ipu=16
        --left-matrix-op=normal
        --right-matrix-op=normal)


add_multitarget_test(
 NAME gemm_40x40x40
 COMMAND general_matrix_multiply
         --m 40
         --k 40
         --n 40
         --alpha 2
         --beta 1
         --tiles-per-ipu=16
         --left-matrix-op=normal
         --right-matrix-op=normal)


add_multitarget_test(
 NAME gemm_10x20x30_groups5
 COMMAND general_matrix_multiply
         --g 5
         --m 10
         --k 20
         --n 30
         --alpha 2
         --beta 3
         --tiles-per-ipu=16
         --left-matrix-op=normal
         --right-matrix-op=normal)

add_multitarget_test(
 NAME gemm_40x39x38
 COMMAND general_matrix_multiply
         --m 40
         --k 39
         --n 38
         --alpha 2
         --beta 1
         --tiles-per-ipu=16
         --left-matrix-op=normal
         --right-matrix-op=normal)

add_multitarget_test(
 NAME gemm_40x39x38_left_transpose
 COMMAND general_matrix_multiply
         --m 40
         --k 39
         --n 38
         --alpha 2
         --beta 1
         --tiles-per-ipu=16
         --left-matrix-op=transpose
         --right-matrix-op=normal)


add_multitarget_test(
 NAME gemm_40x39x38_right_transpose
 COMMAND general_matrix_multiply
         --m 40
         --k 39
         --n 38
         --alpha 2
         --beta 1
         --tiles-per-ipu=16
         --left-matrix-op=normal
         --right-matrix-op=transpose)

add_multitarget_test(
 NAME gemm_40x40x40_halffloat
 COMMAND general_matrix_multiply
         --m 40
         --k 40
         --n 40
         --alpha 2
         --beta 1
         --tiles-per-ipu=16
         --left-matrix-op=normal
         --right-matrix-op=normal
         --input-type=half
         --output-type=float)

add_multitarget_test(
 NAME gemm_40x39x38_floathalf
 COMMAND general_matrix_multiply
         --m 40
         --k 39
         --n 38
         --alpha 2
         --beta 1
         --tiles-per-ipu=16
         --left-matrix-op=normal
         --right-matrix-op=normal
         --input-type=float
         --output-type=half)

add_multitarget_test(
 NAME gemm_40x39x38_left_transpose_halffloat
 COMMAND general_matrix_multiply
         --m 40
         --k 39
         --n 38
         --alpha 2
         --beta 1
         --tiles-per-ipu=16
         --left-matrix-op=transpose
         --right-matrix-op=normal
         --input-type=half
         --output-type=float)

add_multitarget_test(
 NAME gemm_40x39x38_right_transpose_floathalf
 COMMAND general_matrix_multiply
         --m 40
         --k 39
         --n 38
         --alpha 2
         --beta 1
         --tiles-per-ipu=16
         --left-matrix-op=normal
         --right-matrix-op=transpose
         --input-type=float
         --output-type=half)

add_multitarget_test(
 NAME gemm_100x100x100_serial_split_nx2
 COMMAND general_matrix_multiply
         --m 100
         --k 100
         --n 100
         --tiles-per-ipu=16
         --plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/pc_serial_split_ocx2.json
         )

add_multitarget_test(
 NAME gemm_200x100x50_serial_split_nx2
 COMMAND general_matrix_multiply
         --m 200
         --k 100
         --n 50
         --tiles-per-ipu=16
         --plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/pc_serial_split_ocx2.json
         )

add_multitarget_test(
 NAME gemm_40x80x140_serial_split_nx4
 COMMAND general_matrix_multiply
         --m 40
         --k 80
         --n 140
         --tiles-per-ipu=16
         --plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/pc_serial_split_ocx4.json
         )

add_multitarget_test(
 NAME gemm_20x40x10_input_channel_serialization
 COMMAND general_matrix_multiply
         --m 20
         --k 40
         --n 10
         --tiles-per-ipu=4
         --input-type=half
         --output-type=half
         --plan-constraints={\"0\":{\"partition\":{\"inChanSplit\":{\"serial\":5,\"parallel\":4}}}})

add_multitarget_test(
 NAME gemm_using_fast_reduce_vertex
 COMMAND general_matrix_multiply
         --m 20
         --k 256
         --n 100
         --tiles-per-ipu=4
         --input-type=half
         --output-type=half
         --enable-fast-reduce=true
         )

# Single conv layer tests

if (PYTHONINTERP_FOUND AND PYTHON_VERSION_MAJOR EQUAL 3)
  # Random single_conv_layer tests with fixed seed.
  set(NUM_CONV_RANDOM_TESTS 50)
  foreach(n RANGE 1 ${NUM_CONV_RANDOM_TESTS})
  add_multitarget_test(
          NAME conv_random_${n}
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_SOURCE_DIR}/tools/single_conv_layer_random.py
                  --seed ${n}
                  --binary $<TARGET_FILE:single_conv_layer>
          LABELS single_conv_layer
          ALLOW_SKIP)
  endforeach()

  foreach(n RANGE 1 500)
    add_multitarget_test(
            NAME conv_random_quarter${n}
            COMMAND ${PYTHON_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/tools/single_conv_layer_random.py
                    --seed ${n}
                    --binary $<TARGET_FILE:single_conv_layer>
                    --input-type=quarter
            VARIANTS "Sim21"
            ALLOW_SKIP)
  endforeach()

  # Random single_conv_layer tests with fixed seed on multi ipus
  set(NUM_CONV_RANDOM_TESTS 25)
  foreach(n RANGE 1 ${NUM_CONV_RANDOM_TESTS})
    add_multitarget_test(
            NAME multi_ipu_conv_random_${n}
            COMMAND ${PYTHON_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/tools/single_conv_layer_random.py
                    --seed ${n}
                    --ipus 2
                    --binary $<TARGET_FILE:single_conv_layer>
            VARIANTS "Hw;${IPUMODEL_VARIANTS}"
            LABELS single_conv_layer
            ALLOW_SKIP)
  endforeach()

  # Random multi_conv_layer tests with fixed seed.
  set(NUM_CONV_RANDOM_TESTS 25)
  foreach(n RANGE 1 ${NUM_CONV_RANDOM_TESTS})
    add_multitarget_test(
          NAME multi_conv_random_${n}
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_SOURCE_DIR}/tools/single_conv_layer_random.py
                  --seed ${n}
                  --json
                  --large
                  --num-convs 0
                  --binary $<TARGET_FILE:multi_conv_layer>
          ALLOW_SKIP)
  endforeach()
endif()

# a few multi ipu tests that don't take too long that we can run on the Sim.
add_multitarget_test(NAME multi_ipu_conv_smoke_1
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={6,3}
                 --input-channels=4
                 --in-dilation={3,1}
                 --padding-upper={1,1}
                 --padding-lower={1,1}
                 --output-channels=16
                 --kernel-size={1,2}
                 --kernel-dilation={1,3}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={0,0}
                 --stride={2,2}
                 --tiles-per-ipu=24
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS};IpuModel21")

add_multitarget_test(NAME multi_ipu_conv_smoke_2
         COMMAND single_conv_layer
                 --data-type=float
                 --conv-groups=1
                 --batch-size=2
                 --field={3,2}
                 --input-channels=12
                 --in-dilation={1,1}
                 --padding-upper={2,2}
                 --padding-lower={1,2}
                 --output-channels=19
                 --kernel-size={1,1}
                 --kernel-dilation={1,2}
                 --kernel-padding-upper={0,1}
                 --kernel-padding-lower={0,2}
                 --stride={4,2}
                 --tiles-per-ipu=24
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_3
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={12,3}
                 --input-channels=3
                 --in-dilation={1,1}
                 --padding-upper={1,1}
                 --padding-lower={2,1}
                 --output-channels=16
                 --kernel-size={5,3}
                 --kernel-dilation={1,1}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={2,2}
                 --stride={2,2}
                 --tiles-per-ipu=24
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_4
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={4,2}
                 --input-channels=16
                 --in-dilation={1,3}
                 --padding-upper={1,1}
                 --padding-lower={1,1}
                 --output-channels=3
                 --kernel-size={3,2}
                 --kernel-dilation={1,4}
                 --kernel-padding-upper={0,1}
                 --kernel-padding-lower={0,0}
                 --stride={1,1}
                 --tiles-per-ipu=16
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_5
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=3
                 --field={9,4}
                 --input-channels=4
                 --in-dilation={1,1}
                 --padding-upper={2,1}
                 --padding-lower={1,3}
                 --output-channels=1
                 --kernel-size={2,1}
                 --kernel-dilation={4,1}
                 --kernel-padding-upper={0,1}
                 --kernel-padding-lower={0,1}
                 --stride={2,1}
                 --tiles-per-ipu=24
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_6
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={9,1}
                 --input-channels=16
                 --in-dilation={2,1}
                 --padding-upper={1,2}
                 --padding-lower={1,2}
                 --output-channels=4
                 --kernel-size={1,2}
                 --kernel-dilation={1,1}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={1,0}
                 --stride={1,3}
                 --tiles-per-ipu=16
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_7
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={8,1}
                 --input-channels=46
                 --in-dilation={1,2}
                 --padding-upper={3,2}
                 --padding-lower={1,1}
                 --output-channels=4
                 --kernel-size={1,1}
                 --kernel-dilation={1,2}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={2,1}
                 --stride={1,1}
                 --tiles-per-ipu=16
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_8
         COMMAND single_conv_layer
                 --data-type=float
                 --conv-groups=1
                 --batch-size=1
                 --field={2,1}
                 --input-channels=4
                 --in-dilation={2,2}
                 --padding-upper={1,4}
                 --padding-lower={1,3}
                 --output-channels=20
                 --kernel-size={2,1}
                 --kernel-dilation={4,2}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={0,0}
                 --stride={1,2}
                 --tiles-per-ipu=1
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_9
         COMMAND single_conv_layer
                 --data-type=float
                 --conv-groups=1
                 --batch-size=1
                 --field={4,11}
                 --input-channels=4
                 --in-dilation={1,3}
                 --padding-upper={1,1}
                 --padding-lower={2,1}
                 --output-channels=4
                 --kernel-size={1,2}
                 --kernel-dilation={3,3}
                 --kernel-padding-upper={0,0}
                 --kernel-padding-lower={1,0}
                 --stride={1,2}
                 --tiles-per-ipu=16
                 --ipus=2
                 --convolution-options={\"partialsType\":\"float\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(NAME multi_ipu_conv_smoke_10
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=2
                 --field={1,3}
                 --input-channels=16
                 --in-dilation={3,1}
                 --padding-upper={3,1}
                 --padding-lower={1,2}
                 --output-channels=16
                 --kernel-size={1,2}
                 --kernel-dilation={2,1}
                 --kernel-padding-upper={0,1}
                 --kernel-padding-lower={0,0}
                 --stride={1,2}
                 --tiles-per-ipu=24
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\"}
         VARIANTS "Hw;${IPUMODEL_VARIANTS}")

# Basic convolution to test profiling option
add_multitarget_test(NAME conv_profile
         COMMAND single_conv_layer
         --input-channels=16
         --output-channels=16
         --field={1}
         --kernel-size=1
         --tiles-per-ipu=16
         --profile)

add_multitarget_test(NAME conv1x1_float_input_odd
         COMMAND single_conv_layer
         --single-phase=fwd
         --input-channels=37
         --output-channels=1
         --field={5,6,15}
         --kernel-size=1
         --data-type=float
         --tiles-per-ipu=12
         --padding=0
         --stride=1)

# If the partial type is smaller than the output type we just ignore it.
add_multitarget_test(NAME conv1x1_half_partials_float_output
        COMMAND single_conv_layer
        --input-channels=32
        --output-channels=32
        --field={7,7}
        --tiles-per-ipu=4
        --stride=1
        --kernel-size=1
        --data-type=float
        --convolution-options={\"partialsType\":\"half\"})

add_multitarget_test(NAME conv3x3_zero_batch
        COMMAND single_conv_layer
        --conv-groups=2
        --batch-size=0
        --input-channels=16
        --output-channels=16
        --field={32}
        --kernel-size=3
        --padding=1
        --data-type=float
        --tiles-per-ipu=16
        --stride=1)

add_multitarget_test(NAME conv3x3_zero_in_chans
        COMMAND single_conv_layer
        --conv-groups=2
        --batch-size=2
        --input-channels=0
        --output-channels=16
        --field={32}
        --kernel-size=3
        --padding=1
        --data-type=float
        --tiles-per-ipu=16
        --stride=1)

add_multitarget_test(NAME conv3x3_zero_out_chans
        COMMAND single_conv_layer
        --conv-groups=2
        --batch-size=2
        --input-channels=16
        --output-channels=0
        --field={32}
        --kernel-size=3
        --padding=1
        --data-type=float
        --tiles-per-ipu=16
        --stride=1)

#  ORIG Layer tests
add_multitarget_test(NAME conv3x3_stride_2_128_out
         COMMAND single_conv_layer
                 --input-channels=48
                 --output-channels=128
                 --field={20,12}
                 --kernel-size=3
                 --data-type=half
                 --tiles-per-ipu=12
                 --padding=2
                 --stride=2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=128
             --output-channels=256
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_no_bias
         COMMAND single_conv_layer
            --input-channels=64
            --output-channels=32
            --field={7,7}
            --tiles-per-ipu=8
            --stride=1
            --kernel-size=1
            --bias=0
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
  NAME conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=128
            --output-channels=257
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=129
            --output-channels=256
            --field={7,7}
            --tiles-per-ipu=12
            --stride=1
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_3
         COMMAND single_conv_layer
             --input-channels=64
             --output-channels=128
             --field={14,14}
             --tiles-per-ipu=8
             --stride=3
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_thinly_spread
         COMMAND single_conv_layer
             --field={1,56}
             --input-channels=16
             --output-channels=8
             --tiles-per-ipu=24)

add_multitarget_test(
         NAME conv3x3_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=3
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=2
             --kernel-size=3)

add_multitarget_test(
         NAME conv3x3_asymmetric_stride
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,10}
             --tiles-per-ipu=16
             --stride={2,3}
             --kernel-size=3)

add_multitarget_test(
         NAME conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --padding=1
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --padding=2
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=2
                 --stride=2
                 --kernel-size=3)

add_multitarget_test(
         NAME conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=1
                 --stride=3
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_batch_2
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={7,7}
                  --tiles-per-ipu=2
                  --stride=2
                  --kernel-size=3
                  --batch-size=2)

add_multitarget_test(
         NAME conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={28,20}
        --kernel-size=3
        --data-type=half
        --padding={1,2}
        --tiles-per-ipu=16
        --stride=1
        VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={7,7}
        --tiles-per-ipu=2
        --stride=2
        --kernel-size=3
        --padding={1,2}
        --batch-size=2
        VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_1_asym_padding
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=24
                --field={20,12}
                --kernel-size=3
                --data-type=half
                --padding-lower=1
                --padding-upper=2
                --tiles-per-ipu=12
                --stride=1
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_1_asym_padding_2
         COMMAND single_conv_layer
                --input-channels=32
                --output-channels=40
                --field={20,12}
                --kernel-size=3
                --data-type=half
                --padding-lower=3
                --padding-upper=0
                --tiles-per-ipu=16
                --stride=1
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv5x5_stride_2_asym_padding
         COMMAND single_conv_layer
                  --input-channels=32
                  --output-channels=32
                  --field={20,12}
                  --kernel-size=5
                  --data-type=half
                  --padding-lower={1,2}
                  --padding-upper={3,3}
                  --stride=2
                  --tiles-per-ipu=12
                  VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --field={5,5}
                 --tiles-per-ipu=24
                 --convolution-options={\"partialsType\":\"float\"})

add_multitarget_test(
         NAME small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --kernel-size=3
                 --field={7,7}
                 --tiles-per-ipu=24
                 --convolution-options={\"partialsType\":\"float\"})

add_multitarget_test(
         NAME conv1x1_shallow_input
         COMMAND single_conv_layer
                 --input-channels=8
                 --output-channels=512
                 --field={8,8}
                 --data-type=half
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=1
                 --data-type=half
                 --tiles-per-ipu=2)

add_multitarget_test(
         NAME conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=2
                 --data-type=half
                 --tiles-per-ipu=2)

add_multitarget_test(
         NAME conv_large_kernel
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --stride=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_large_kernel_bwd
         COMMAND single_conv_layer
                 --single-phase=bwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --stride=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                 --field={1,16}
                 --padding={1,6}
                 --kernel-size={1,7}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_flip_input_and_kernel_1
         COMMAND single_conv_layer
                 --field={6,2}
                 --padding={0,1}
                 --flip-input={0,1}
                 --kernel-size={1,3}
                 --kernel-padding={0,0}
                 --flip-kernel={0,1}
                 --input-channels=2
                 --output-channels=1
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_flip_input_and_kernel_2
         COMMAND single_conv_layer
                 --field={1,32}
                 --padding={0,16}
                 --flip-input={0,1}
                 --kernel-size={1,48}
                 --flip-kernel={0,1}
                 --input-channels=32
                 --output-channels=8
                 --tiles-per-ipu=4
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_flip_input_only
         COMMAND single_conv_layer
                 --field={20,20}
                 --padding=1
                 --flip-input=1
                 --kernel-size=3
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=8
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_flip_kernel_only
         COMMAND single_conv_layer
                 --field={20,20}
                 --padding=1
                 --kernel-size=3
                 --flip-kernel=1
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=8
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=12
                 --data-type=half
                 --batch-size=4
                 --single-phase=bwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_float
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_group
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=256
                 --output-channels=64
                 --batch-size=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

foreach(PARTIALS_TYPE half float)
foreach(IN_CHANS 2 4 8)
add_multitarget_test(
         NAME "conv_mac_${PARTIALS_TYPE}_in_chan_${IN_CHANS}"
         COMMAND single_conv_layer
                 --input-channels=${IN_CHANS}
                 --output-channels=1
                 --field={7,4}
                 --kernel-size={4,3}
                 --tiles-per-ipu=1
                 --single-phase=fwd
                 --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                 --fwd-plan-constraints={\"method\":\"HMAC\"}
                 --bias=0)
endforeach() # IN_CHANS

foreach(OUT_CHANS 1 2 4)
add_multitarget_test(
         NAME "conv_mac_${PARTIALS_TYPE}_out_chan_${OUT_CHANS}"
         COMMAND single_conv_layer
                 --input-channels=4
                 --output-channels=${OUT_CHANS}
                 --field={5,2}
                 --kernel-size={2,2}
                 --tiles-per-ipu=1
                 --single-phase=fwd
                 --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                 --fwd-plan-constraints={\"method\":\"HMAC\"}
                 --bias=0)
endforeach() # OUT_CHANS

foreach(FLIP_INPUT 0 1)
add_multitarget_test(
         NAME "conv_mac_${PARTIALS_TYPE}_stride_4_flip_input_${FLIP_INPUT}"
         COMMAND single_conv_layer
                 --input-channels=8
                 --output-channels=6
                 --field={10,7}
                 --kernel-size={4,4}
                 --tiles-per-ipu=1
                 --single-phase=fwd
                 --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                 --fwd-plan-constraints={"method":"HMAC","0":{"transform":{"swapOperands":false}}}
                 --bias=0
                 --flip-input=${FLIP_INPUT}
                 --in-dilation={1,4}
                 --stride={4,1})
endforeach() # FLIP_INPUT
endforeach() # PARTIALS_TYPE

foreach(CONV_GROUPS_PER_GROUP 4 8 16)
  foreach(CONV_GROUPS 1 3 4 5 11 16)
    foreach(BATCH_SIZE 1 5 9)
      foreach(PARTIALS_TYPE half float)
          if ((${PARTIALS_TYPE} STREQUAL float) AND (${CONV_GROUPS_PER_GROUP} GREATER 4))
            # Skip 8 and 16 codelets as they don't have assembly implementation
            continue()
          endif()
          add_multitarget_test(
            NAME "conv_vmac_cg_${CONV_GROUPS}_b_${BATCH_SIZE}_partials_${PARTIALS_TYPE}_cgpg_${CONV_GROUPS_PER_GROUP}"
              COMMAND single_conv_layer
                    --input-channels=1
                    --output-channels=1
                    --field={10,7}
                    --kernel-size={4,4}
                    --conv-groups=${CONV_GROUPS}
                    --batch-size=${BATCH_SIZE}
                    --tiles-per-ipu=1
                    --single-phase=wu
                    --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                    --wu-plan-constraints={"method":"VMAC","convGroupsPerGroup":${CONV_GROUPS_PER_GROUP},"0":{"transform":{"swapOperands":false}}}
                    --bias=0
                    --flip-input=0
                    --in-dilation={2,1}
                    --kernel-dilation={3,2}
                    --stride={3,5})
      endforeach() # PARTIALS_TYPE
    endforeach() # BATCH_SIZE
  endforeach() # CONV_GROUPS
endforeach() # CONV_GROUPS_PER_GROUP

foreach(CONV_GROUPS_PER_GROUP 4 8 16)
  foreach(FLIP_INPUT 0 1)
    foreach(CONV_GROUPS 5 11)
      foreach(BATCH_SIZE 3 7)
        foreach(PARTIALS_TYPE half float)
                if ((${PARTIALS_TYPE} STREQUAL float) AND (${CONV_GROUPS_PER_GROUP} GREATER 4))
                  # Skip 8 and 16 codelets as they don't have assembly implementation
                  continue()
                endif()
                add_multitarget_test(
                NAME "conv_vmac_cg_${CONV_GROUPS}_b_${BATCH_SIZE}_flip_input_${FLIP_INPUT}_partials_${PARTIALS_TYPE}_cgpg_${CONV_GROUPS_PER_GROUP}"
                COMMAND single_conv_layer
                        --input-channels=1
                        --output-channels=1
                        --field={13,9}
                        --kernel-size={5,4}
                        --conv-groups=${CONV_GROUPS}
                        --batch-size=${BATCH_SIZE}
                        --tiles-per-ipu=1
                        --single-phase=wu
                        --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                        --wu-plan-constraints={"method":"VMAC","convGroupsPerGroup":${CONV_GROUPS_PER_GROUP},"0":{"transform":{"swapOperands":false}}}
                        --bias=0
                        --flip-input=${FLIP_INPUT}
                        --in-dilation={2,3}
                        --kernel-dilation={3,5}
                        --stride={2,3})
        endforeach() # PARTIALS_TYPE
      endforeach() # BATCH_SIZE
    endforeach() # CONV_GROUPS
  endforeach() # FLIP_INPUT
endforeach() # CONV_GROUPS_PER_GROUP

# Tests to target non-limited HMAC codelets
add_multitarget_test(
         NAME "conv_mac_half_half_non_limited"
         COMMAND single_conv_layer
                 --input-channels=4
                 --output-channels=2
                 --field={7,4}
                 --tiles-per-ipu=1
                 --input-type=half
                 --output-type=half
                 --convolution-options={"partialsType":"half"}
                 --single-phase=fwd
                 --fwd-plan-constraints={"method":"HMAC","useLimitedVersion":false}
                 --bias=0)

add_multitarget_test(
         NAME "conv_mac_half_float_non_limited"
         COMMAND single_conv_layer
                 --input-channels=4
                 --output-channels=2
                 --field={7,4}
                 --tiles-per-ipu=1
                 --input-type=half
                 --output-type=float
                 --convolution-options={"partialsType":"float"}
                 --single-phase=fwd
                 --fwd-plan-constraints={"method":"HMAC","useLimitedVersion":false}
                 --bias=0)

add_multitarget_test(
         NAME "conv_mac_float_float_non_limited"
         COMMAND single_conv_layer
                 --input-channels=4
                 --output-channels=2
                 --field={7,4}
                 --tiles-per-ipu=1
                 --data-type=float
                 --convolution-options={"partialsType":"float"}
                 --single-phase=fwd
                 --fwd-plan-constraints={"method":"HMAC","useLimitedVersion":false}
                 --bias=0)

foreach(CHANS 8 16 32 64)
add_multitarget_test(
         NAME "conv_dual_amp_half_1x1_channels_${CHANS}"
         COMMAND single_conv_layer
                 --input-channels=${CHANS}
                 --output-channels=${CHANS}
                 --field={1,1}
                 --kernel-size=1
                 --tiles-per-ipu=1
                 --single-phase=fwd
                 --convolution-options={\"partialsType\":\"half\"}
                 --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                 VARIANTS "IpuModel2;Sim2")
endforeach()

foreach(FILED_SIZE 1 2 3 7)
add_multitarget_test(
         NAME "conv_dual_amp_half_1x1_field_size_${FILED_SIZE}"
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field=${FILED_SIZE}
                 --kernel-size=1
                 --tiles-per-ipu=1
                 --single-phase=fwd
                 --convolution-options={\"partialsType\":\"half\"}
                 --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                 VARIANTS "IpuModel2;Sim2;IpuModel21;Sim21")
endforeach()

foreach(OUT_CHANS 16 32)
foreach(IN_CHANS 4 8 16 32)
add_multitarget_test(
        NAME "conv_dual_amp_half_Nx1_inChans_${IN_CHANS}_outChans_${OUT_CHANS}"
        COMMAND single_conv_layer
                --input-channels=${IN_CHANS}
                --output-channels=${OUT_CHANS}
                --field={13,13}
                --kernel-size=5
                --tiles-per-ipu=1
                --single-phase=fwd
                --convolution-options={\"partialsType\":\"half\"}
                --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                --in-dilation=2
                --flip-input=0
                VARIANTS "IpuModel2;Sim2")
endforeach()
endforeach()

foreach(KERNEL_SIZE 2 5 9 13)
add_multitarget_test(
        NAME "conv_dual_amp_half_Nx1_kernel-size_${KERNEL_SIZE}"
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field={13,13}
                --kernel-size=${KERNEL_SIZE}
                --tiles-per-ipu=1
                --single-phase=fwd
                --convolution-options={\"partialsType\":\"half\"}
                --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                --in-dilation=2
                --flip-input=0
                VARIANTS "IpuModel2;Sim2")
endforeach()

foreach(FILED_SIZE 3 8 11 13)
add_multitarget_test(
        NAME "conv_dual_amp_half_Nx1_field_${FILED_SIZE}"
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field=${FILED_SIZE}
                --kernel-size=5
                --tiles-per-ipu=1
                --single-phase=fwd
                --convolution-options={\"partialsType\":\"half\"}
                --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                --in-dilation=2
                --flip-input=0
                VARIANTS "IpuModel2;Sim2")
endforeach()

add_multitarget_test(
        NAME "conv_dual_amp_half_Nx1_flip-input_true"
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field={13,13}
                --kernel-size=5
                --tiles-per-ipu=1
                --single-phase=fwd
                --convolution-options={\"partialsType\":\"half\"}
                --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                --in-dilation=2
                --flip-input=1
                VARIANTS "IpuModel2;Sim2")

foreach(CHANS 8 16 32 64)
add_multitarget_test(
         NAME "conv_dual_amp_float_1x1_channels_${CHANS}"
         COMMAND single_conv_layer
                 --input-channels=${CHANS}
                 --output-channels=${CHANS}
                 --field={1,1}
                 --kernel-size=1
                 --tiles-per-ipu=1
                 --single-phase=fwd
                 --data-type=float
                 --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                 VARIANTS "IpuModel2;Sim2")
endforeach()

foreach(FILED_SIZE 1 2 3 7)
add_multitarget_test(
         NAME "conv_dual_amp_float_1x1_field_size_${FILED_SIZE}"
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field=${FILED_SIZE}
                 --kernel-size=1
                 --tiles-per-ipu=1
                 --single-phase=fwd
                 --data-type=float
                 --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                 VARIANTS "IpuModel2;Sim2")
endforeach()

foreach(OUT_CHANS 16 32)
foreach(IN_CHANS 4 8 16 32)
add_multitarget_test(
        NAME "conv_dual_amp_float_Nx1_inChans_${IN_CHANS}_outChans_${OUT_CHANS}"
        COMMAND single_conv_layer
                --input-channels=${IN_CHANS}
                --output-channels=${OUT_CHANS}
                --field={13,13}
                --kernel-size=5
                --tiles-per-ipu=1
                --single-phase=fwd
                 --data-type=float
                --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                --in-dilation=2
                --flip-input=0
                VARIANTS "IpuModel2;Sim2")
endforeach()
endforeach()

foreach(KERNEL_SIZE 2 5 9 13)
add_multitarget_test(
        NAME "conv_dual_amp_float_Nx1_kernel-size_${KERNEL_SIZE}"
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field={13,13}
                --kernel-size=${KERNEL_SIZE}
                --tiles-per-ipu=4
                --single-phase=fwd
                 --data-type=float
                --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                --in-dilation=2
                --flip-input=0
                VARIANTS "IpuModel2;Sim2")
endforeach()

foreach(FILED_SIZE 3 8 11 13)
add_multitarget_test(
        NAME "conv_dual_amp_float_Nx1_field_${FILED_SIZE}"
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field=${FILED_SIZE}
                --kernel-size=5
                --tiles-per-ipu=1
                --single-phase=fwd
                 --data-type=float
                --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                --in-dilation=2
                --flip-input=0
                VARIANTS "IpuModel2;Sim2")
endforeach()

add_multitarget_test(
        NAME "conv_dual_amp_float_Nx1_flip-input_true"
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field={13,13}
                --kernel-size=5
                --tiles-per-ipu=1
                --single-phase=fwd
                 --data-type=float
                --fwd-plan-constraints={\"method\":\"AMP\",\"numAmpConvUnits\":\"16\"}
                --in-dilation=2
                --flip-input=1
                VARIANTS "IpuModel2;Sim2")

# conv_batch_float test split into the three component phases
add_multitarget_test(
         NAME conv_batch_float_fwd
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=12
                 --data-type=float
                 --batch-size=4
                 --single-phase=fwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_float_bwd
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=12
                 --data-type=float
                 --batch-size=4
                 --single-phase=bwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_batch_float_wu
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=32
                 --output-channels=16
                 --tiles-per-ipu=12
                 --data-type=float
                 --batch-size=4
                 --single-phase=wu)

add_multitarget_test(
         NAME conv_negative_input_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_kernel_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,1}
                --kernel-size={1,2}
                --kernel-truncation-upper={0,1}
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_positive_and_negative_kernel_padding
         COMMAND single_conv_layer
         --input-channels=16
         --output-channels=16
         --field={5,5}
         --kernel-size=2
         --kernel-padding-lower=1
         --kernel-truncation-upper=1
         --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_input_padding_lower_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,2}
                 --kernel-size={1,1}
                 --truncation-upper={0,1}
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME conv_negative_kernel_padding_lower_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --kernel-truncation-lower={0,1}
                 --tiles-per-ipu=1)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_multitarget_test(
         NAME conv_large_output_tensor
         COMMAND single_conv_layer
                --field={16,16}
                --input-channels=16
                --output-channels=128
                --tiles-per-ipu=12
                --data-type=half
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_large_weights
         COMMAND single_conv_layer
                --field={14,16}
                --padding={2,0}
                --flip-input={1,0}
                --input-channels=64
                --output-channels=64
                --kernel-size={16,1}
                --flip-kernel={1,0}
                --tiles-per-ipu=16
                --data-type=half
                --single-phase=fwd
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_outer_product
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=384
                 --field={1,128}
                 --tiles-per-ipu=12
                 VARIANTS ${TimesOutOnSim})

# Caused a failure due to the outer product method being selected despite it
# not supporting output padding (see T26443)
add_multitarget_test(
         NAME conv_output_padding
         COMMAND single_conv_layer
                 --data-type=float
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --output-padding-lower={1,1}
                 --tiles-per-ipu=1
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_1_half_partials
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=16
                 --stride=1
                 --kernel-size=3
                 --convolution-options={\"partialsType\":\"half\"}
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_kernel_with_padding_larger_than_field
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --field={1,128}
                 --tiles-per-ipu=4
                 --padding={1,64})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=64
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=3
                 --stride=3
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor2
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=4
                 --stride=2
                 --tiles-per-ipu=12
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=2
                 --stride=4
                 --tiles-per-ipu=8
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor4
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field={14,14}
                --kernel-size={3,3}
                --in-dilation=4
                --stride=6
                --tiles-per-ipu=8
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_input_dilation_and_stride_share_factor5
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --field={14,14}
                --kernel-size={3,3}
                --in-dilation=6
                --stride=4
                --tiles-per-ipu=8
                VARIANTS ${TimesOutOnSim})

# Example where the amount of padding that must be added to start of the input
# window for the conv nx1 vertex exceeds the amount of lower padding for the
# layer.
add_multitarget_test(
         NAME conv_nx1_vertex_lower_padding
         COMMAND single_conv_layer
                 --field={16,32}
                 --input-channels=4
                 --padding-lower={1,0}
                 --output-channels=8
                 --kernel-size={2,1}
                 --kernel-dilation={2,1}
                 --stride={3,1}
                 --tiles-per-ipu=1
                 VARIANTS ${TimesOutOnSim})

# Test where, after padding and dilation, the last input index that would be
# multiplied by the a kernel element is in the middle of the lower input
# padding.
add_multitarget_test(
         NAME conv1x1_input_ignored_with_stride
         COMMAND single_conv_layer
                 --field={2,2}
                 --in-dilation=2
                 --padding-lower=2
                 --padding-upper=1
                 --stride=7
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=1)

# Test with large amounts of output padding
add_multitarget_test(
         NAME conv1x1_output_padding
         COMMAND single_conv_layer
                 --field={1,1}
                 --input-channels=16
                 --output-channels=16
                 --output-padding=10
                 --tiles-per-ipu=1)

# Convolution with output padding and non-view-only expandDims transform
add_multitarget_test(
         NAME conv1x20_expanddims_with_output_padding
         COMMAND single_conv_layer
                 --field={1,20}
                 --input-channels=4
                 --output-channels=4
                 --kernel-size={1,10}
                 --padding-lower={1,0}
                 --inference-only
                 --tiles-per-ipu=16
                 --fwd-plan-constraints={\"0\":{\"transform\":{\"expandDims\":[1,0]}}})

#  Grouped convolution layer tests
add_multitarget_test(
         NAME grouped_conv147x147_stride_1_8_out
         COMMAND single_conv_layer
                 --input-channels=4
                 --output-channels=8
                 --field={147,147}
                 --kernel-size={49,49}
                 --data-type=float
                 --padding=0
                 --conv-groups=2
                 --stride=1
                 --kernel-dilation={3,3}
                 --inference-only
                 VARIANTS Cpu)

add_multitarget_test(
         NAME grouped_conv3x3_stride_2_32_out
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={20,12}
                 --kernel-size=3
                 --data-type=half
                 --padding=2
                 --conv-groups=2
                 --stride=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=4
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME zero_conv_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=0
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=64
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=8
            --stride=1
            --conv-groups=2
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=128
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --conv-groups=2
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=12
            --stride=1
            --conv-groups=2
            --kernel-size=1
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_3_odd_conv_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={14,14}
             --tiles-per-ipu=16
             --stride=3
             --conv-groups=3
             --kernel-size=1
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_thinly_spread
         COMMAND single_conv_layer
             --field={1,128}
             --input-channels=16
             --conv-groups=2
             --output-channels=8
             --tiles-per-ipu=24)

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_odd_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=3
             --kernel-size=3
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=2
             --conv-groups=2
             --kernel-size=3
             VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_asymmetric_stride
         COMMAND single_conv_layer
             --input-channels=16
             --output-channels=8
             --field={7,10}
             --tiles-per-ipu=16
             --stride={2,3}
             --conv-groups=2
             --kernel-size=3)

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --conv-groups=2
              --padding=1
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=16
              --output-channels=8
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --conv-groups=2
              --padding=2
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=2
                 --stride=2
                 --conv-groups=2
                 --kernel-size=3
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=4
                 --conv-groups=2
                 --stride=3)

add_multitarget_test(
         NAME grouped_conv3x3_batch_2
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={7,7}
                  --tiles-per-ipu=4
                  --stride=2
                  --kernel-size=3
                  --conv-groups=2
                  --batch-size=2
                  VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={28,20}
        --kernel-size=3
        --data-type=half
        --padding={1,2}
        --conv-groups=2
        --stride=1
        --tiles-per-ipu=12
        VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={7,7}
        --tiles-per-ipu=4
        --stride=2
        --kernel-size=3
        --padding={1,2}
        --conv-groups=2
        --batch-size=2
        VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_asym_padding
        COMMAND single_conv_layer
                --input-channels=16
                --output-channels=24
                --field={20,12}
                --kernel-size=3
                --data-type=half
                --padding-lower=1
                --padding-upper=2
                --conv-groups=2
                --tiles-per-ipu=12
                --stride=1
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_stride_1_asym_padding_2
         COMMAND single_conv_layer
                --input-channels=16
                --output-channels=24
                --field={20,12}
                --kernel-size=3
                --data-type=half
                --padding-lower=3
                --padding-upper=0
                --conv-groups=2
                --tiles-per-ipu=12
                --stride=1
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv5x5_stride_2_asym_padding
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={20,12}
                  --kernel-size=5
                  --data-type=half
                  --padding-lower={1,2}
                  --padding-upper={3,3}
                  --stride=2
                  --conv-groups=2
                  --tiles-per-ipu=12
                  VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --conv-groups=2
                 --tiles-per-ipu=12
                 --convolution-options={\"partialsType\":\"float\"}
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --kernel-size=3
                 --field={7,7}
                 --conv-groups=2
                 --tiles-per-ipu=20
                 --convolution-options={\"partialsType\":\"float\"}
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv1x1_shallow_input
        COMMAND single_conv_layer
                --input-channels=8
                --output-channels=384
                --field={8,8}
                --conv-groups=2
                --data-type=half
                --tiles-per-ipu=16
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=1
                 --conv-groups=2
                 --data-type=half
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME grouped_conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=2
                 --conv-groups=2
                 --data-type=half
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME grouped_conv_large_kernel_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=8
                 --padding=1
                 --conv-groups=2
                 --stride=2
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_large_kernel_bwd
         COMMAND single_conv_layer
                 --single-phase=bwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=8
                 --padding=1
                 --conv-groups=2
                 --stride=2
                 --tiles-per-ipu=20
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                --field={1,16}
                 --padding={6,1}
                 --kernel-size={7,1}
                 --input-channels=16
                 --output-channels=16
                 --conv-groups=2
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME grouped_conv_batch_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=half
                 --conv-groups=2
                 --batch-size=4
                 --single-phase=fwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_batch_float
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --conv-groups=2
                 --batch-size=2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_batch_group
         COMMAND single_conv_layer
                 --field={7,7}
                 --input-channels=256
                 --output-channels=64
                 --tiles-per-ipu=32
                 --conv-groups=2
                 --batch-size=2
                 VARIANTS ${TimesOutOnSim})

# conv_batch_bwd_float test split into the three component phases
add_multitarget_test(
         NAME grouped_conv_batch_float_fwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=fwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_batch_float_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=bwd
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_batch_float_wu
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=wu
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_negative_input_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --conv-groups=2
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_kernel_padding_upper_simple
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --kernel-truncation-lower={0,1}
                 --conv-groups=2
                 --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_input_padding_lower_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --truncation-upper={0,1}
                --conv-groups=2
                --tiles-per-ipu=1)

add_multitarget_test(
         NAME grouped_conv_negative_kernel_padding_lower_simple_${phase}
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field={1,1}
                 --kernel-size={1,2}
                 --conv-groups=2
                 --kernel-truncation-lower={0,1}
                 --tiles-per-ipu=1)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_multitarget_test(
         NAME grouped_conv_large_output_tensor
         COMMAND single_conv_layer
                --field={16,16}
                --input-channels=16
                --output-channels=64
                --tiles-per-ipu=12
                --conv-groups=2
                --data-type=half
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_large_weights
         COMMAND single_conv_layer
                --field={14,16}
                --padding={2,0}
                --flip-input={1,0}
                --input-channels=32
                --output-channels=16
                --kernel-size={16,1}
                --flip-kernel={1,0}
                --tiles-per-ipu=12
                --data-type=half
                --conv-groups=2
                --single-phase=fwd
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_conv_outer_product
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=256
                 --field={1,128}
                 --conv-groups=2
                 --tiles-per-ipu=20
                 VARIANTS ${TimesOutOnSim})

foreach(GROUPING "1611" "811" "411" "222" "144")
  foreach(PARTIALS_TYPE float half)
    if ((GROUPING STREQUAL "1611") OR (GROUPING STREQUAL "811"))
      set(TARGETS "Hw;IpuModel2")
      if (${PARTIALS_TYPE} STREQUAL float)
        # Skip 8 and 16 codelets as they don't exist for float partials
        continue()
      endif()
    else()
      set(TARGETS "Hw;${IPUMODEL_VARIANTS}")
    endif()

    set(SLIC_CONV_PLAN ${CMAKE_CURRENT_SOURCE_DIR}/json/slic${GROUPING}.json)
    add_multitarget_test(
              NAME conv_slic_${GROUPING}_${PARTIALS_TYPE}_partials_16groups_no_padding
              COMMAND single_conv_layer
                      --field={2,12}
                      --kernel-size={2,8}
                      --batch-size=3
                      --conv-groups=16
                      --input-channels=1
                      --output-channels=1
                      --fwd-plan-constraints-file=${SLIC_CONV_PLAN}
                      --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                      --tiles-per-ipu=1
                      VARIANTS ${TARGETS})

    add_multitarget_test(
              NAME conv_slic_${GROUPING}_${PARTIALS_TYPE}_partials_16groups_pad_kernel_implicit
              COMMAND single_conv_layer
                      --field={2,12}
                      --kernel-size={2,7}
                      --batch-size=3
                      --conv-groups=16
                      --input-channels=1
                      --output-channels=1
                      --fwd-plan-constraints-file=${SLIC_CONV_PLAN}
                      --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                      --tiles-per-ipu=1
                      VARIANTS ${TARGETS})

    add_multitarget_test(
              NAME conv_slic_${GROUPING}_${PARTIALS_TYPE}_partials_16groups_pad_input
              COMMAND single_conv_layer
                      --field={1,10}
                      --padding-lower={1,2}
                      --kernel-size={2,8}
                      --batch-size=3
                      --conv-groups=16
                      --input-channels=1
                      --output-channels=1
                      --fwd-plan-constraints-file=${SLIC_CONV_PLAN}
                      --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                      --tiles-per-ipu=1
                      VARIANTS ${TARGETS})

    add_multitarget_test(
              NAME conv_slic_${GROUPING}_${PARTIALS_TYPE}_partials_16groups_dilate_input
              COMMAND single_conv_layer
                      --field={2,6}
                      --in-dilation={1,2}
                      --kernel-size={2,8}
                      --batch-size=3
                      --conv-groups=16
                      --input-channels=1
                      --output-channels=1
                      --fwd-plan-constraints-file=${SLIC_CONV_PLAN}
                      --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                      --tiles-per-ipu=1
                      VARIANTS ${TARGETS})

    add_multitarget_test(
              NAME conv_slic_${GROUPING}_${PARTIALS_TYPE}_partials_16groups_truncate_input
              COMMAND single_conv_layer
                      --field={3,13}
                      --truncation-upper={1,2}
                      --kernel-size={2,8}
                      --batch-size=3
                      --conv-groups=16
                      --input-channels=1
                      --output-channels=1
                      --fwd-plan-constraints-file=${SLIC_CONV_PLAN}
                      --convolution-options={\"partialsType\":\"${PARTIALS_TYPE}\"}
                      --tiles-per-ipu=1
                      VARIANTS ${TARGETS})
  endforeach()
endforeach()

foreach(GROUPING 4 8 16)
  foreach(OUTPUT_STRIDE 1 2)
    add_multitarget_test(
            NAME "conv_slic_${GROUPING}_half_1x4_stride_${OUTPUT_STRIDE}"
            COMMAND single_conv_layer
                    --input-channels=1
                    --output-channels=1
                    --conv-groups=16
                    --field={8,32}
                    --kernel-size={1,4}
                    --stride={1,${OUTPUT_STRIDE}}
                    --tiles-per-ipu=4
                    --single-phase=fwd
                    --data-type=half
                    --batch-size=2
                    --convolution-options={\"partialsType\":\"half\"}
                    --fwd-plan-constraints={"method":"SLIC","convGroupsPerGroup":${GROUPING}}
                    VARIANTS "IpuModel2;Hw")
  endforeach()
endforeach()

add_multitarget_test(
         NAME conv0d_simple
         COMMAND single_conv_layer
                 --field={}
                 --batch-size=4
                 --input-channels=128
                 --output-channels=64
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_simple_remap_output
         COMMAND single_conv_layer
                 --field={45}
                 --batch-size=4
                 --input-channels=123
                 --output-channels=63
                 --remap-output-tensor=true
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv_simple_gather_output
         COMMAND single_conv_layer
                 --field={45}
                 --batch-size=4
                 --input-channels=123
                 --output-channels=63
                 --convolution-options={\"gatherConvOutput\":\"true\"}
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME conv1d_simple
         COMMAND single_conv_layer
                  --field={40}
                  --kernel-size=3
                  --padding=1
                  --input-channels=128
                  --output-channels=128
                  --tiles-per-ipu=20
                  VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3d_simple
         COMMAND single_conv_layer
                 --field={5,6,7}
                 --kernel-size={4,3,2}
                 --padding={1,1,1}
                 --input-channels=16
                 --output-channels=32
                 --tiles-per-ipu=20
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3d_1_output_chan
         COMMAND single_conv_layer
                 --field={5,6,7}
                 --kernel-size={4,3,2}
                 --padding={1,1,1}
                 --input-channels=32
                 --output-channels=1
                 --tiles-per-ipu=8
                 VARIANTS ${TimesOutOnSim})

# 3D convolution with multiple batches, convolution groups, input dilation,
# kernel dilation and striding.
add_multitarget_test(
         NAME conv3d_complex
         COMMAND single_conv_layer
                 --conv-groups=2
                 --batch-size=2
                 --field={4,3,16}
                 --in-dilation={1,2,1}
                 --padding={1,0,1}
                 --kernel-size={3,2,3}
                 --kernel-dilation={1,1,2}
                 --input-channels=16
                 --output-channels=32
                 --stride={1,1,3}
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
        NAME conv3x3_stride_3_in_dilation_2_fwd_conv_dither
        COMMAND single_conv_layer
                --single-phase=fwd
                --input-channels=16
                --output-channels=16
                --field={14,14}
                --kernel-size=3
                --tiles-per-ipu=1
                --stride=3
                --in-dilation=2
                --convolution-options={\"enableConvDithering\":\"true\"})

add_multitarget_test(
         NAME conv1x1_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --tiles-per-ipu=1
                 --in-dilation=2)

add_multitarget_test(
         NAME conv1x1_stride_3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2)

add_multitarget_test(
         NAME conv3x3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation=2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2)

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd_minMem
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2
                 --convolution-options={\"availableMemoryProportion\":\"0\"})

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd_minCycles
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2
                 --convolution-options={\"availableMemoryProportion\":\"1\"})

add_multitarget_test(
          NAME conv3x3_stride_3_in_dilation_2_fwd_128bit_load
          COMMAND single_conv_layer
                  --single-phase=fwd
                  --input-channels=16
                  --output-channels=16
                  --field={14,14}
                  --kernel-size=3
                  --tiles-per-ipu=1
                  --stride=3
                  --in-dilation=2
                  --convolution-options={\"use128BitConvUnitLoad\":\"true\"})

add_multitarget_test(
         NAME conv3x3_in_dilation_asymmetric
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation={2,3}
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_kernel_dilation_amp
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --wu-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/method_amp.json
                --field={14,28}
                --kernel-size={3,5}
                --kernel-dilation={3,2}
                --tiles-per-ipu=12
                VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_flattening
         COMMAND single_conv_layer
                --input-channels=46
                --output-channels=4
                --fwd-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/zero_conv_flattening.json
                --field={8,1}
                --in-dilation={1,2}
                --padding-upper={3,2}
                --padding-lower={1,1}
                --kernel-size={1,1}
                --kernel-dilation={1,2}
                --kernel-padding-lower={2,1}
                --stride={1,1}
                --ipus=2
                --tiles-per-ipu=20
                VARIANTS "Hw;${IPUMODEL_VARIANTS}")

add_multitarget_test(
         NAME conv3x3_kernel_padding_amp
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --wu-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/method_amp.json
                 --field={14,14}
                 --kernel-size={3,3}
                 --kernel-padding-lower={4,2}
                 --kernel-padding-upper={4,3}
                 --tiles-per-ipu=16
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_stride_3_in_dilation_2_fwd_minCycles_halffloat
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2
                 --input-type=half
                 --output-type=float
                 --convolution-options={\"availableMemoryProportion\":\"1\"})

add_multitarget_test(
         NAME conv3x3_in_dilation_asymmetric_floathalf
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation={2,3}
                 --input-type=float
                 --output-type=half
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_kernel_dilation_amp_halffloat
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --wu-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/method_amp.json
                --field={14,28}
                --kernel-size={3,5}
                --kernel-dilation={3,2}
                --tiles-per-ipu=12
                --input-type=half
                --output-type=float
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv3x3_kernel_padding_amp_floathalf
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --wu-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/method_amp.json
                 --field={14,14}
                 --kernel-size={3,3}
                 --kernel-padding-lower={4,2}
                 --kernel-padding-upper={4,3}
                 --tiles-per-ipu=16
                 --input-type=float
                 --output-type=half
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv1x1_stride_1_fwd_serial_split_ocx2
         COMMAND single_conv_layer
             --input-channels=128
             --output-channels=256
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=1
             --single-phase=fwd
             --fwd-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/pc_serial_split_ocx2.json
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
        NAME conv3x3_stride_2_128_out_fwd_serial_split_ocx4
        COMMAND single_conv_layer
            --input-channels=48
            --output-channels=128
            --field={20,12}
            --kernel-size=3
            --data-type=half
            --tiles-per-ipu=12
            --padding=2
            --stride=2
            --fwd-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/pc_serial_split_ocx4.json
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME T8298_conv
         COMMAND single_conv_layer
                 --data-type=half
                 --conv-groups=1
                 --batch-size=1
                 --field={32,7,1}
                 --input-channels=2
                 --in-dilation={1,1,1}
                 --padding-upper={2,1,2}
                 --padding-lower={1,1,1}
                 --output-channels=1
                 --kernel-size={1,1,3}
                 --kernel-dilation={2,2,1}
                 --kernel-padding-upper={0,1,1}
                 --kernel-padding-lower={1,1,0}
                 --stride={1,2,1}
                 --tiles-per-ipu=2
                 --ipus=2
                 --convolution-options={\"partialsType\":\"half\"}
                 VARIANTS "Hw;Cpu;${IPUMODEL_VARIANTS}")

# TODO T27251: Temporarily disabled as it times out with the latest TBB
# # Convolution seen in resnet 50 which exposed a bug when generating WU pass
# add_multitarget_test(
#          NAME conv7x7_stride_2_1024_in_512_out_serial_fail_case
#          COMMAND single_conv_layer
#             --input-channels=1024
#             --output-channels=512
#             --field={14,14}
#             --stride={2,2}
#             --batch-size=4
#             --single-phase=wu
#             --ipus=1
#             --wu-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/pc_conv7x7_stride_2_1024_in_512_out_serial_fail_case.json
#             VARIANTS ${TimesOutOnSim})

# convolution with kernel dilation which gave incorrect memory estimates
add_multitarget_test(
         NAME conv3x3_stride_2_64_in_64_out_field_14x14
         COMMAND single_conv_layer
            --input-channels=64
            --output-channels=64
            --field={14,14}
            --stride={2,2}
            --kernel-size={3,3}
            --padding=1
            --batch-size=20
            --ipus=1
            --tiles-per-ipu 64
            --convolution-options={\"partialsType\":\"half\",\"availableMemoryProportion\":\"0.175\"}
            VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_T10392
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1
                 --field=1
                 --kernel-size=3
                 --kernel-truncation-lower=1
                 --kernel-truncation-upper=1
                 --single-phase=fwd
                 --fwd-plan-constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/json/pc_T10392.json
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME conv_serialization_and_post_conv_dilation
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --field={3,3}
		 --kernel-size={1,1}
		 --in-dilation={2,2}
		 --padding-upper={1,1}
		 --batch-size=1
                 --single-phase=fwd
		 --tiles-per-ipu=16
		 --convolution-options={\"availableMemoryProportion\":\"0.1\"}
		 --fwd-plan-constraints={\"0\":{\"partition\":{\"outChanSplit\":{\"serial\":2}}}})

add_multitarget_test(NAME conv5x200_1_in_100_out
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=100
                 --field={15,200}
                 --kernel-size={5,200}
                 --batch-size=16
                 --data-type=half
                 --tiles-per-ipu=20
                 --fwd-plan-constraints={\"0\":{\"transform\":{\"swapOperands\":false,\"expandDims\":[1]}}})

add_multitarget_test(NAME conv5x200_1_in_100_out_swap_operands
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=100
                 --field={15,200}
                 --kernel-size={5,200}
                 --batch-size=16
                 --data-type=half
                 --tiles-per-ipu=20
                 --fwd-plan-constraints={\"0\":{\"transform\":{\"swapOperands\":true,\"expandDims\":[1]}}})

# Test Fwd/Bwd symmetric convolution with code reuse
add_multitarget_test(NAME conv8x6_8_in_8_out_reuse
         COMMAND single_conv_layer
                 --input-channels=8
                 --output-channels=8
                 --field={8,6}
                 --padding-lower={1,2}
                 --padding-upper={1,2}
                 --kernel-size={3,5}
                 --batch-size=8
                 --data-type=half
                 --tiles-per-ipu=16
                 --enable-convolution-reuse 1
                 --fwd-plan-constraints={\"0\":{\"transform\":{\"swapOperands\":false,\"expandDims\":[1]}}})

# Test the --use-create-input option
foreach(USE_CREATE_INPUT "0" "1")
    add_multitarget_test(
            NAME conv_use_create_input_${USE_CREATE_INPUT}
            COMMAND single_conv_layer
                    --input-channels=16
                    --output-channels=16
                    --field={7,7}
                    --padding=1
                    --kernel-size=3
                    --conv-groups=2
                    --data-type=half
                    --tiles-per-ipu=2
                    --use-create-input=${USE_CREATE_INPUT})
endforeach()

# Fully connected layer tests

add_multitarget_test(
         NAME fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half)

add_multitarget_test(
         NAME fully_connected_half_no_bias
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --bias 0
                 --data-type=half)

add_multitarget_test(
         NAME fully_connected_half_fwd
         COMMAND fully_connected_layer
                 --single-phase=fwd
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half)

add_multitarget_test(
        NAME fully_connected_half_bwd
        COMMAND fully_connected_layer
                --single-phase=bwd
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half)

add_multitarget_test(
        NAME fully_connected_small_field_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 4
                --output-size 4
                --tiles-per-ipu 16
                --data-type=half)

add_multitarget_test(
        NAME fully_connected_large_field_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half)

add_multitarget_test(
        NAME fully_connected_half_fwd_128_bit_load
        COMMAND fully_connected_layer
                --single-phase=fwd
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half
                --matmul-options={\"use128BitConvUnitLoad\":\"true\"})


add_multitarget_test(
        NAME fully_connected_half_fwd_gather_output
        COMMAND fully_connected_layer
                --single-phase=fwd
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half
                --matmul-options={\"gatherOutput\":\"true\"})

add_test(
         NAME fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type=half)

add_multitarget_test(
         NAME fully_connected_float
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float)

add_test(
         NAME fully_connected_float_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type=float)

add_multitarget_test(
         NAME fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim})

add_test(
         NAME fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --ipus=2
                 --batch-size=4)

add_multitarget_test(
      NAME fully_connected_half_batch_7
      COMMAND fully_connected_layer
              --input-size 96
              --output-size 96
              --tiles-per-ipu 8
              --data-type=half
              --batch-size=7
              VARIANTS ${TimesOutOnSim})


add_multitarget_test(
         NAME fully_connected_float_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim})

add_test(
         NAME fully_connected_float_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float
                 --ipus=2
                 --batch-size=4)

add_multitarget_test(
         NAME fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 220
                 --output-size 84
                 --tiles-per-ipu 1
                 --data-type=half
                 --batch-size=4
                 --inference-only)

add_multitarget_test(
         NAME fully_connected_input_size_1
         COMMAND fully_connected_layer
                 --input-size=1
                 --output-size=100
                 --tiles-per-ipu=4
                 --data-type=half
                 --batch-size=16)

add_multitarget_test(
         NAME fully_connected_output_size_1
         COMMAND fully_connected_layer
                 --input-size=100
                 --output-size=1
                 --tiles-per-ipu=4
                 --data-type=half
                 --batch-size=16)

add_multitarget_test(
         NAME grouped_fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 160
                 --output-size 100
                 --tiles-per-ipu 12
                 --num-groups 2
                 --data-type=half)


add_test(
         NAME grouped_fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --num-groups 3
                 --data-type=half)


add_multitarget_test(
         NAME grouped_fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 160
                 --output-size 100
                 --tiles-per-ipu 12
                 --data-type=half
                 --num-groups 2
                 --batch-size=4
                 VARIANTS ${TimesOutOnSim})

add_test(
         NAME grouped_fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --num-groups 3
                 --ipus=2
                 --batch-size=4)

add_multitarget_test(
      NAME grouped_fully_connected_half_batch_7
      COMMAND fully_connected_layer
              --input-size 112
              --output-size 112
              --tiles-per-ipu 8
              --data-type=half
              --num-groups 3
              --batch-size=7
              VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 108
                 --output-size 84
                 --tiles-per-ipu 1
                 --data-type=half
                 --batch-size=2
                 --num-groups 2
                 --inference-only)

add_multitarget_test(
         NAME grouped_fully_connected_half_batch_4_halffloat
         COMMAND fully_connected_layer
                 --input-size 160
                 --output-size 100
                 --tiles-per-ipu 12
                 --num-groups 2
                 --batch-size=4
                 --input-type=half
                 --output-type=float
         VARIANTS ${TimesOutOnSim})

add_test(
         NAME grouped_fully_connected_half_batch_4_two_ipus_floathalf
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --num-groups 3
                 --ipus=2
                 --batch-size=4
                 --input-type=float
                 --output-type=half)

add_multitarget_test(
      NAME grouped_fully_connected_half_batch_7_halffloat
      COMMAND fully_connected_layer
              --input-size 112
              --output-size 112
              --tiles-per-ipu 8
              --num-groups 3
              --batch-size=7
              --input-type=half
              --output-type=float
      VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME grouped_fully_connected_batch_fwd_one_tile_floathalf
         COMMAND fully_connected_layer
                 --input-size 108
                 --output-size 84
                 --tiles-per-ipu 1
                 --batch-size=2
                 --num-groups 2
                 --inference-only
                 --input-type=float
                 --output-type=half)

# Fully connected layer where the RHS has fewer elements than the LHS
add_multitarget_test(
         NAME fully_connected_small_rhs
         COMMAND fully_connected_layer
                 --batch-size=64
                 --input-size=512
                 --tiles-per-ipu=16
                 --output-size=4
                 --data-type=half
                 VARIANTS ${TimesOutOnSim})

# test that hit a bug where we would generate an incorrect vertex due to trying
# to use a joint plan when the input and output types didn't match.
add_multitarget_test(
         NAME fully_connected_T33085
         COMMAND fully_connected_layer
                 --batch-size=16
                 --input-size=384
                 --output-size=384
                 --num-groups=2
                 --input-type=half
                 --output-type=float
                 --matmul-options={\"partialsType\":\"half\"}
                 --single-phase=fwd
                 VARIANTS ${TimesOutOnSim})

# test that fit a bug where no vertex candidates were found for the backward
# pass of a joint plan due to the partial type being smaller than the output
# type
add_multitarget_test(
         NAME fully_connected_half_partials_float_output
         COMMAND fully_connected_layer
                 --input-size=2048
                 --output-size=1000
                 --batch-size=4
                 --data-type=float
                 --matmul-options={\"partialsType\":\"half\"}
                 VARIANTS ${TimesOutOnSim})

# Test the --use-create-input option
foreach(USE_CREATE_INPUT "0" "1")
    add_multitarget_test(
          NAME fully_connected_use_create_input_${USE_CREATE_INPUT}
          COMMAND fully_connected_layer
                  --batch-size=8
                  --input-size=64
                  --output-size=32
                  --num-groups=2
                  --tiles-per-ipu=2
                  --data-type=half
                  --use-create-input=${USE_CREATE_INPUT})
endforeach()

foreach(PLAN_TYPE parallel serial)
  add_multitarget_test(NAME octconv_basic_${PLAN_TYPE}
           COMMAND oct_conv_layer
                   --input-channels=2
                   --output-channels=4
                   --field={26,78}
                   --kernel-size={2,3}
                   --conv-groups=2
                   --batch-size=3
                   --padding-lower={1,2}
                   --data-type=half
                   --tiles-per-ipu=32
                   --alpha=0.5
                   --options={\"planType\":\"${PLAN_TYPE}\",\"perConvReservedTiles\":4})

    add_multitarget_test(NAME octconv_basic_${PLAN_TYPE}_no_reserved_tiles
           COMMAND oct_conv_layer
                   --input-channels=2
                   --output-channels=4
                   --field={26,78}
                   --kernel-size={2,3}
                   --conv-groups=2
                   --batch-size=3
                   --padding-lower={1,2}
                   --data-type=half
                   --tiles-per-ipu=32
                   --alpha=0.5
                   --options={\"planType\":\"${PLAN_TYPE}\"})

  add_multitarget_test(NAME octconvLL_${PLAN_TYPE}
           COMMAND oct_conv_layer
                   --input-channels=2
                   --output-channels=2
                   --field={40,100}
                   --kernel-size={2,3}
                   --conv-groups=2
                   --batch-size=3
                   --padding-lower={1,2}
                   --data-type=half
                   --tiles-per-ipu=16
                   --alpha=1.0
                   --options={\"planType\":\"${PLAN_TYPE}\",\"perConvReservedTiles\":4})
  add_multitarget_test(NAME octconvLH_${PLAN_TYPE}
           COMMAND oct_conv_layer
                   --input-channels=2
                   --output-channels=2
                   --field={40,100}
                   --kernel-size={2,3}
                   --conv-groups=2
                   --batch-size=3
                   --padding-lower={1,2}
                   --data-type=half
                   --tiles-per-ipu=16
                   --alpha-in=1.0
                   --alpha-out=0.0
                   --options={\"planType\":\"${PLAN_TYPE}\",\"perConvReservedTiles\":4})
  add_multitarget_test(NAME octconvHL_${PLAN_TYPE}
           COMMAND oct_conv_layer
                   --input-channels=2
                   --output-channels=2
                   --field={40,100}
                   --kernel-size={2,3}
                   --conv-groups=2
                   --batch-size=3
                   --padding-lower={1,2}
                   --data-type=half
                   --tiles-per-ipu=16
                   --alpha-in=0.0
                   --alpha-out=1.0
                   --options={\"planType\":\"${PLAN_TYPE}\",\"perConvReservedTiles\":4})
  add_multitarget_test(NAME octconvHH_${PLAN_TYPE}
           COMMAND oct_conv_layer
                   --input-channels=2
                   --output-channels=2
                   --field={40,100}
                   --kernel-size={2,3}
                   --conv-groups=2
                   --batch-size=3
                   --padding-lower={1,2}
                   --data-type=half
                   --tiles-per-ipu=16
                   --alpha=0.0
                   --options={\"planType\":\"${PLAN_TYPE}\",\"perConvReservedTiles\":4})
endforeach()

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/json/depthwise_conv_half.json DEPTHWISE_CONV)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/json/pointwise_conv_float.json POINTWISE_CONV)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/json/pointwise_reusable_conv_float.json POINTWISE_REUSABLE_CONV)
add_multitarget_test(
         NAME multi_conv_different_types
         COMMAND multi_conv_layer
                 --tiles-per-ipu 128
                 --conv ${DEPTHWISE_CONV}
                 --conv ${POINTWISE_CONV}
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME multi_conv_different_types_single_tile
         COMMAND multi_conv_layer
                 --tiles-per-ipu 1
                 --conv ${DEPTHWISE_CONV}
                 --conv ${DEPTHWISE_CONV}
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME multi_conv_different_types_bwd
         COMMAND multi_conv_layer
         	 --tiles-per-ipu 128
                 --conv ${DEPTHWISE_CONV}
                 --conv ${POINTWISE_REUSABLE_CONV}
         	 --bwd 1
         	 --enable-convolution-reuse 0
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME multi_conv_different_types_bwd_reuse
         COMMAND multi_conv_layer
         	 --tiles-per-ipu 128
                 --conv ${DEPTHWISE_CONV}
                 --conv ${POINTWISE_REUSABLE_CONV}
         	 --bwd 1
         	 --enable-convolution-reuse 1
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME triangular_solve_4_31_10_4_float_left_lower_unit_validation
         COMMAND matrix_solver
                 --batches 4
                 --a-rank 31
                 --b-rank 10
                 --block-size 4
                 --data-type float
                 --left-side true
                 --lower true
                 --unit-diagonal true
         VARIANTS "${IPUMODEL_VARIANTS}")

#FAILING
add_multitarget_test(
         NAME triangular_solve_4_31_10_4_float_left_upper_non_unit_validation
         COMMAND matrix_solver
                 --batches 4
                 --a-rank 31
                 --b-rank 10
                 --block-size 4
                 --data-type float
                 --left-side true
                 --lower false
                 --unit-diagonal false
         VARIANTS "${IPUMODEL_VARIANTS}")

add_multitarget_test(
         NAME triangular_solve_4_31_10_4_half_right_lower_unit_validation
         COMMAND matrix_solver
                 --batches 4
                 --a-rank 31
                 --b-rank 10
                 --block-size 4
                 --data-type half
                 --left-side false
                 --lower true
                 --unit-diagonal true
         VARIANTS "${IPUMODEL_VARIANTS}")

add_multitarget_test(
         NAME triangular_solve_4_31_10_4_half_right_upper_non_unit_validation
         COMMAND matrix_solver
                 --batches 4
                 --a-rank 31
                 --b-rank 10
                 --block-size 4
                 --data-type half
                 --left-side false
                 --lower false
                 --unit-diagonal false
         VARIANTS "${IPUMODEL_VARIANTS}")
