find_package(Boost REQUIRED unit_test_framework timer system program_options)

add_subdirectory(popsolver)

function(add_popnn_unit_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:popconv,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} popnn
                        poplar
                        poplib_test
                        Boost::unit_test_framework
                        Boost::timer
                        Boost::system
                        Boost::program_options)
  if(NOT ${Boost_USE_STATIC_LIBS})
    target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  endif()
  add_test(NAME ${name}
    COMMAND ${name})
endfunction()

# Unit tests
add_popnn_unit_test(RangeTest RangeTest.cpp)
add_popnn_unit_test(ConvPlanTest ConvPlanTest.cpp)
add_popnn_unit_test(GraphFunctionTest GraphFunctionTest.cpp)
add_popnn_unit_test(WinogradConvolution WinogradConv.cpp)
add_popnn_unit_test(NonLinearityTest NonLinearityTest.cpp)
add_popnn_unit_test(GraphProgLocationTest GraphProgLocationTest.cpp)
add_popnn_unit_test(StdOperatorTest StdOperatorsTest.cpp)

# Layer tests
add_test(NAME conv3x3_stride_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=128
                 --output-channels=256
                 --width=20
                 --height=28
                 --kernel-size=3
                 --data-type=half
                 --padding=2
                 --stride=2)

add_test(NAME conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=128
             --output-channels=256
             --width=7
             --height=7
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=1)

add_test(NAME conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=128
            --output-channels=257
            --width=7
            --height=7
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1)

add_test(NAME conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=129
            --output-channels=256
            --width=7
            --height=7
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1)

add_test(NAME conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=129
            --output-channels=257
            --width=7
            --height=7
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1)

add_test(NAME conv1x1_stride_3
         COMMAND single_conv_layer
             --input-channels=64
             --output-channels=128
             --width=14
             --height=14
             --tiles-per-ipu=8
             --stride=3
             --kernel-size=1)

add_test(NAME conv3x3_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --width=7
             --height=7
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=3)


add_test(NAME conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --width=7
             --height=7
             --tiles-per-ipu=16
             --stride=2
             --kernel-size=3)

add_test(NAME conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --width=7
              --height=7
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --padding 1)

add_test(NAME conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --width=7
              --height=7
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --padding 2)

add_test(NAME conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --width=7
                 --height=7
                 --tiles-per-ipu=2
                 --stride=2
                 --kernel-size=3)

add_test(NAME conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --width=7
                 --height=7
                 --tiles-per-ipu=1
                 --stride=3)

add_test(NAME conv3x3_batch_2
         COMMAND single_conv_layer
                  --single-phase=fwd
                  --input-channels=16
                  --output-channels=16
                  --width=7
                  --height=7
                  --tiles-per-ipu=2
                  --stride=2
                  --kernel-size=3
                  --batch-size=2)

add_test(NAME conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --single-phase=fwd
        --input-channels=16
        --output-channels=16
        --width=20
        --height=28
        --kernel-size=3
        --data-type=half
        --padding-height=1
        --padding-width=2
        --stride=1)

add_test(NAME conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --width=7
        --height=7
        --tiles-per-ipu=2
        --stride=2
        --kernel-size=3
        --padding-height=1
        --padding-width=2
        --batch-size=2)

add_test(NAME conv3x3_stride_1_fwd_asym_padding
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --width=20
                --height=28
                --kernel-size=3
                --data-type=half
                --padding-height-lower=1
                --padding-width-lower=1
                --padding-height-upper=2
                --padding-width-upper=2
                --stride=1)


add_test(NAME conv5x5_stride_2_fwd_asym_padding
         COMMAND single_conv_layer
                  --single-phase=fwd
                  --input-channels=128
                  --output-channels=256
                  --width=20
                  --height=28
                  --kernel-size=5
                  --data-type=half
                  --padding-height-lower=1
                  --padding-width-lower=2
                  --padding-height-upper=3
                  --padding-width-upper=3
                  --stride=2)

add_test(NAME small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --width=14
                 --height=14
                 --partials-type=float
                 --tiles-per-ipu=1216)

add_test(NAME small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --kernel-size 3
                 --width=14
                 --height=14
                 --partials-type=float
                 --tiles-per-ipu=1216)

add_test(NAME conv1x1_shallow_input
        COMMAND single_conv_layer
                --input-channels=8
                --output-channels=1024
                --width=16
                --height=16
                --data-type=half)

add_test(NAME conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size 1
                 --input-channels 1
                 --output-channels 1
                 --width 8
                 --height 8
                 --kernel-size 3
                 --padding 1
                 --stride 1
                 --single-phase=fwd
                 --data-type half) 

add_test(NAME conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size 1
                 --input-channels 1
                 --output-channels 1
                 --width 8
                 --height 8
                 --kernel-size 3
                 --padding 1
                 --single-phase=fwd
                 --stride 2
                 --data-type half) 

add_test(NAME conv_large_kernel
         COMMAND single_conv_layer
                 --width 14
                 --height 64
                 --kernel-height 62
                 --kernel-width 3
                 --input-channels 16
                 --output-channels 16
                 --padding 1
                 --stride 2)

add_test(NAME conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                 --width=16
                 --height=1
                 --padding-height=6
                 --kernel-height=7
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16)

add_test(NAME conv_batch_bwd
         COMMAND single_conv_layer
                 --width 14
                 --height 14
                 --input-channels 32
                 --output-channels 32
                 --tiles-per-ipu=16
                 --data-type=half
                 --batch-size=4)

add_test(NAME conv_batch_bwd_float
         COMMAND single_conv_layer
                 --width 14
                 --height 14
                 --input-channels 32
                 --output-channels 32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4)

add_test(NAME conv_batch_group
         COMMAND single_conv_layer
                 --height 28
                 --width 28
                 --input-channels 512
                 --output-channels 128
                 --single-phase=fwd
                 --batch-size 2)

# conv_batch_bwd_float test split into the three component phases
add_test(NAME conv_batch_bwd_float_fwd
         COMMAND single_conv_layer
                 --width 14
                 --height 14
                 --input-channels 32
                 --output-channels 32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=fwd)

add_test(NAME conv_batch_bwd_float_bwd
         COMMAND single_conv_layer
                 --width 14
                 --height 14
                 --input-channels 32
                 --output-channels 32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=bwd)

add_test(NAME conv_batch_bwd_float_wu
         COMMAND single_conv_layer
                 --width 14
                 --height 14
                 --input-channels 32
                 --output-channels 32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=wu)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_test(NAME conv_large_output_tensor
         COMMAND single_conv_layer
                --width 16
                --height 16
                --input-channels 16
                --output-channels 128
                --tiles-per-ipu=16
                --data-type=half)

add_test(NAME conv_large_weights
         COMMAND single_conv_layer
                --width=16
                --height=14
                --input-channels=64
                --output-channels=64
                --kernel-height=16
                --kernel-width=1
                --tiles-per-ipu=16
                --data-type=half
                --single-phase=fwd)

add_test(NAME fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type half)

add_test(NAME fully_connected_half_fwd
         COMMAND fully_connected_layer
                 --single-phase=fwd
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type half)

add_test(NAME fully_connected_half_bwd
        COMMAND fully_connected_layer
                --single-phase=bwd
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type half)

add_test(NAME fully_connected_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type half)

add_test(NAME fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type half)

add_test(NAME fully_connected_float
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type float)

add_test(NAME fully_connected_float_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type float)

add_test(NAME fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type half
                 --batch-size 4)

add_test(NAME fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type half
                 --ipus=2
                 --batch-size 4)


add_test(NAME fully_connected_float_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type float
                 --batch-size 4)

add_test(NAME fully_connected_float_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type float
                 --ipus=2
                 --batch-size 4)                 

add_test(NAME fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 1
                 --data-type half
                 --batch-size 4
                 --inference-only)

add_test(NAME max_pool_layer_half
         COMMAND max_pool_layer
                 --channels 32
                 --width 28
                 --height 28
                 --kernel-size 2
                 --stride 2
                 --data-type half)

add_test(NAME max_pool_layer_float
         COMMAND max_pool_layer
                 --channels 32
                 --width 28
                 --height 28
                 --kernel-size 2
                 --stride 2
                 --data-type float)

add_test(NAME max_pool_layer_1chan_overlapping_kernel
         COMMAND max_pool_layer
                 --channels 1
                 --width 100
                 --height 100
                 --kernel-size 3
                 --stride 2
                 --padding 1
                 --data-type float)

add_test(NAME max_pool_layer_half_batch
         COMMAND max_pool_layer
                 --batch-size 4
                 --channels 32
                 --width 28
                 --height 28
                 --kernel-size 2
                 --stride 2
                 --data-type half)

add_test(NAME max_pool_layer_float_batch
         COMMAND max_pool_layer
                 --batch-size 4
                 --channels 32
                 --width 28
                 --height 28
                 --kernel-size 2
                 --stride 2
                 --data-type float)

add_test(NAME max_pool_layer_2ipu
         COMMAND max_pool_layer
                 --channels 64
                 --bwd-chans-per-group 8
                 --width 112
                 --height 112
                 --kernel-size 3
                 --stride 2
                 --ipus 2)

add_test(NAME max_pool_layer_1x4
         COMMAND max_pool_layer
                 --channels 32
                 --width 32 
                 --height 1 
                 --kernel-height 1
                 --kernel-width 4
                 --stride-height 1
                 --stride-width 4)

add_test(NAME gemm_1x3x1
          COMMAND general_matrix_multiply
                  --m 1
                  --k 3
                  --n 1
                  --alpha 1
                  --beta 1
                  --left-matrix-op=normal
                  --right-matrix-op=normal)

add_test(NAME gemm_1x1000x1
        COMMAND general_matrix_multiply
                --m 1
                --k 1000
                --n 1
                --alpha 2
                --beta 1
                --left-matrix-op=normal
                --right-matrix-op=normal)


add_test(NAME gemm_40x40x40
         COMMAND general_matrix_multiply
                 --m 40
                 --k 40
                 --n 40 
                 --alpha 2
                 --beta 1 
                 --left-matrix-op=normal
                 --right-matrix-op=normal)


add_test(NAME gemm_40x39x38
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38 
                 --alpha 2
                 --beta 1 
                 --left-matrix-op=normal
                 --right-matrix-op=normal)

add_test(NAME gemm_40x39x38_left_transpose
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38 
                 --alpha 2
                 --beta 1 
                 --left-matrix-op=transpose
                 --right-matrix-op=normal)


add_test(NAME gemm_40x39x38_right_transpose
         COMMAND general_matrix_multiply
                 --m 40
                 --k 39
                 --n 38 
                 --alpha 2
                 --beta 1 
                 --left-matrix-op=normal
                 --right-matrix-op=transpose)

add_test(NAME rnn_3x2x38_no_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --output-size 38
                 --batch-size 2
                 --nonlinearity-type sigmoid)


add_test(NAME rnn_3x32x2x38_with_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --input-size 32
                 --output-size 38
                 --batch-size 2
                 --nonlinearity-type sigmoid
                 --apply-feedforward-weights) 
                 
add_test(NAME basic_lstm_40x4x38_seq_2_half_data
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size 4
                 --output-size 38
                 --sequence-size 2)

add_test(NAME basic_lstm_40x4x38_seq_2_float_data
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size 4
                 --output-size 38
                 --sequence-size 2
                 --data-type float)
