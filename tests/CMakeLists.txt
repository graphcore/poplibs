find_package(Boost REQUIRED unit_test_framework program_options)

function(add_popnn_unit_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      ${Boost_INCLUDE_DIRS})
  target_link_libraries(${name} popnn poplar popnn_ref ${Boost_LIBRARIES})
  target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  add_test(NAME ${name}
    COMMAND ${name})
endfunction()
# Unit tests
add_popnn_unit_test(WinogradConvolution WinogradConv.cpp)
add_popnn_unit_test(NonLinearityTest NonLinearityTest.cpp)

# Layer tests
add_test(NAME conv3x3_stride_2_fwd
         COMMAND single_conv_layer
                 --inference-only
                 --input-channels=128
                 --output-channels=256
                 --width=20
                 --height=28
                 --kernel-size=3
                 --data-type=half
                 --padding=2
                 --stride=2
                 --fwd-out-chans-per-group=16)

add_test(NAME conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=128
             --output-channels=256
             --width=7
             --height=7
             --fwd-out-chans-per-group=8
             --bwd-out-chans-per-group=8
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=1)

add_test(NAME conv1x1_stride_3
         COMMAND single_conv_layer
             --input-channels=64
             --output-channels=128
             --width=14
             --height=14
             --fwd-out-chans-per-group=8
             --bwd-out-chans-per-group=8
             --tiles-per-ipu=8
             --stride=3
             --kernel-size=1)

add_test(NAME conv3x3_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --width=7
             --height=7
             --fwd-out-chans-per-group=8
             --bwd-out-chans-per-group=8
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=3)


add_test(NAME conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --width=7
             --height=7
             --fwd-out-chans-per-group=8
             --bwd-out-chans-per-group=8
             --tiles-per-ipu=16
             --stride=2
             --kernel-size=3)

add_test(NAME conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --width=7
              --height=7
              --fwd-out-chans-per-group=8
              --bwd-out-chans-per-group=8
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
              --fwd-out-chans-per-group=8
              --bwd-out-chans-per-group=8
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
                 --fwd-out-chans-per-group=16
                 --bwd-out-chans-per-group=16
                 --tiles-per-ipu=2
                 --stride=2
                 --kernel-size=3)

add_test(NAME conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --inference-only
                 --input-channels=16
                 --output-channels=16
                 --width=7
                 --height=7
                 --fwd-out-chans-per-group=16
                 --tiles-per-ipu=1
                 --stride=3)

 add_test(NAME conv3x3_batch_2
          COMMAND single_conv_layer
                  --inference-only
                  --input-channels=16
                  --output-channels=16
                  --width=7
                  --height=7
                  --fwd-out-chans-per-group=16
                  --bwd-out-chans-per-group=16
                  --tiles-per-ipu=2
                  --stride=2
                  --kernel-size=3
                  --batch-size=2)

add_test(NAME small_conv_many_tiles
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --width=14
                 --height=14
                 --partials-type=float
                 --tiles-per-ipu=1216)

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

add_test(NAME fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type half)

add_test(NAME fully_connected_float
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type float)

add_test(NAME fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type half
                 --batch-size 4)

add_test(NAME fully_connected_float_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type float
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
