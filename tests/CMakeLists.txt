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
                 --kernel-size=3
                 --non-linearity=relu
                 --data-type=half
                 --padding=2
                 --stride=2
                 --fwd-out-chans-per-group=16)

add_test(NAME conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=256
             --output-channels=512
             --width=7
             --height=7
             --fwd-out-chans-per-group=8
             --bwd-out-chans-per-group=8
             --tiles-per-ipu=64
             --stride=1
             --kernel-size=1)

add_test(NAME conv1x1_stride_3
         COMMAND single_conv_layer
             --input-channels=256
             --output-channels=512
             --width=14
             --height=14
             --fwd-out-chans-per-group=8
             --bwd-out-chans-per-group=8
             --tiles-per-ipu=64
             --stride=3
             --kernel-size=1)

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
