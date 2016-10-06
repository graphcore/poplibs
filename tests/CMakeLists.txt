find_package(Boost REQUIRED unit_test_framework program_options)

function(add_popnn_unit_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      ${Boost_INCLUDE_DIRS})
  target_link_libraries(${name} popnn poplar ${Boost_LIBRARIES})
  target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  add_test(NAME ${name}
    COMMAND ${name})
endfunction()
# Unit tests
add_popnn_unit_test(FullyConnectedTest FullyConnectedTest.cpp)
add_popnn_unit_test(WinogradConvolution WinogradConv.cpp)
add_popnn_unit_test(MaxPoolTest MaxPoolTest.cpp)

# Layer tests
add_test(NAME conv3x3_stride_2_fwd
         COMMAND single_conv_layer
                 --inference-only
                 --input-channels=128
                 --output-channels=256
                 --width=20
                 --height=28
                 --kernel-height=3
                 --kernel-width=3
                 --non-linearity=relu
                 --data-type=half
                 --padding-height=2
                 --padding-width=2
                 --stride-height=2
                 --stride-weight=2
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
             --stride-height=1
             --stride-width=1
             --kernel-height=1
             --kernel-width=1)

add_test(NAME conv1x1_stride_3
         COMMAND single_conv_layer
             --input-channels=64
             --output-channels=128
             --width=14
             --height=14
             --fwd-out-chans-per-group=8
             --bwd-out-chans-per-group=8
             --tiles-per-ipu=8
             --stride-height=3
             --stride-width=3
             --kernel-height=1
             --kernel-width=1)

add_test(NAME conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --width=7
                 --height=7
                 --fwd-out-chans-per-group=16
                 --bwd-out-chans-per-group=16
                 --tiles-per-ipu=2
                 --stride-height=2
                 --stride-width=2
                 --kernel-height=3
                 --kernel-width=3)

add_test(NAME conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --inference-only
                 --input-channels=16
                 --output-channels=16
                 --width=7
                 --height=7
                 --fwd-out-chans-per-group=16
                 --tiles-per-ipu=1
                 --stride-height=3
                 --stride-width=3)

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
                  --stride-height=2
                  --stride-width=2
                  --kernel-height=3
                  --kernel-width=3
                  --batch-size=2)
