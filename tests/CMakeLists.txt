find_package(Boost REQUIRED unit_test_framework timer system program_options)

add_subdirectory(popsolver)

function(add_popnn_unit_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:popconv,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} poprand popnn
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
add_popnn_unit_test(RandomGenTest RandomGenTest.cpp)
add_popnn_unit_test(ConvUtilTest ConvUtilTest.cpp)
add_popnn_unit_test(RangeTest RangeTest.cpp)
add_popnn_unit_test(ConvPlanTest ConvPlanTest.cpp)
add_popnn_unit_test(StdArithmeticTests StdArithmeticTests.cpp)
add_popnn_unit_test(GraphFunctionTest GraphFunctionTest.cpp)
add_popnn_unit_test(WinogradConvolution WinogradConv.cpp)
add_popnn_unit_test(NonLinearityTest NonLinearityTest.cpp)
add_popnn_unit_test(GraphProgLocationTest GraphProgLocationTest.cpp)
add_popnn_unit_test(StdOperatorTest StdOperatorsTest.cpp)
add_popnn_unit_test(BatchNormTest BatchNormTest.cpp)
add_popnn_unit_test(ReductionTests ReductionTests.cpp)
add_popnn_unit_test(LargeSplitRegionsTest LargeSplitRegionsTest.cpp)
add_popnn_unit_test(DynamicSliceTest DynamicSliceTest.cpp)
add_popnn_unit_test(CircBufTests CircBufTests.cpp)

#  ORIG Layer tests
add_test(NAME conv3x3_stride_2_128_out
         COMMAND single_conv_layer
                 --input-channels=64
                 --output-channels=128
                 --field={28,20}
                 --kernel-size=3
                 --data-type=half
                 --padding=2
                 --stride=2)

add_test(NAME conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=128
             --output-channels=256
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=1)

add_test(NAME conv1x1_no_bias
        COMMAND single_conv_layer
            --input-channels=64
            --output-channels=32
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1
            --bias=0)

add_test(NAME conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=128
            --output-channels=257
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1)

add_test(NAME conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=129
            --output-channels=256
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1)

add_test(NAME conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=129
            --output-channels=257
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --kernel-size=1)

add_test(NAME conv1x1_stride_3
         COMMAND single_conv_layer
             --input-channels=64
             --output-channels=128
             --field={14,14}
             --tiles-per-ipu=8
             --stride=3
             --kernel-size=1)

add_test(NAME conv1x1_thinly_spread
         COMMAND single_conv_layer
             --field={1,128}
             --input-channels=16
             --output-channels=8)

add_test(NAME conv3x3_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --kernel-size=3)


add_test(NAME conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=2
             --kernel-size=3)

add_test(NAME conv3x3_asymmetric_stride
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,10}
             --tiles-per-ipu=16
             --stride={2,3}
             --kernel-size=3)

add_test(NAME conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --padding=1)

add_test(NAME conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --padding=2)

add_test(NAME conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=2
                 --stride=2
                 --kernel-size=3)

add_test(NAME conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=1
                 --stride=3)

add_test(NAME conv3x3_batch_2
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={7,7}
                  --tiles-per-ipu=2
                  --stride=2
                  --kernel-size=3
                  --batch-size=2)

add_test(NAME conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={28,20}
        --kernel-size=3
        --data-type=half
        --padding={1,2}
        --stride=1)

add_test(NAME conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={7,7}
        --tiles-per-ipu=2
        --stride=2
        --kernel-size=3
        --padding={1,2}
        --batch-size=2)

add_test(NAME conv3x3_stride_1_fwd_asym_padding
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --field={28,20}
                --kernel-size=3
                --data-type=half
                --padding-lower=1
                --padding-upper=2
                --stride=1)
add_test(NAME conv3x3_stride_1_fwd_asym_padding_2
         COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --field={28,20}
                --kernel-size=3
                --data-type=half
                --padding-lower=3
                --padding-upper=0
                --stride=1)

add_test(NAME conv5x5_stride_2_fwd_asym_padding
         COMMAND single_conv_layer
                  --input-channels=64
                  --output-channels=64
                  --field={28,20}
                  --kernel-size=5
                  --data-type=half
                  --padding-lower={1,2}
                  --padding-upper={3,3}
                  --stride=2
                  --tiles-per-ipu=16)

add_test(NAME small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --partials-type=float
                 --tiles-per-ipu=1216)

add_test(NAME small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --kernel-size=3
                 --field={14,14}
                 --partials-type=float
                 --tiles-per-ipu=1216)

add_test(NAME conv1x1_shallow_input
         COMMAND single_conv_layer
                 --input-channels=8
                 --output-channels=1024
                 --field={16,16}
                 --data-type=half)

add_test(NAME conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=1
                 --data-type=half)

add_test(NAME conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=2
                 --data-type=half)

add_test(NAME conv_large_kernel
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --stride=2)

add_test(NAME conv_large_kernel_bwd
         COMMAND single_conv_layer
                 --single-phase=bwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --stride=2)

add_test(NAME conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                 --field={1,16}
                 --padding={1,6}
                 --kernel-size={1,7}
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=16)

add_test(NAME conv_kernel_larger_than_field_1
         COMMAND single_conv_layer
                 --field={6,2}
                 --kernel-size={1,3}
                 --input-channels=2
                 --output-channels=1
                 --tiles-per-ipu=1
                 --single-phase=fwd)

add_test(NAME conv_kernel_larger_than_field_2
         COMMAND single_conv_layer
                 --field={1,32}
                 --kernel-size={1,48}
                 --input-channels=32
                 --output-channels=8
                 --tiles-per-ipu=4
                 --single-phase=fwd)

add_test(NAME conv_batch_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=half
                 --batch-size=4)

add_test(NAME conv_batch_bwd_float
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4)

add_test(NAME conv_batch_group
         COMMAND single_conv_layer
                 --field={28,28}
                 --input-channels=512
                 --output-channels=128
                 --batch-size=2)

# conv_batch_bwd_float test split into the three component phases
add_test(NAME conv_batch_bwd_float_fwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=fwd)

add_test(NAME conv_batch_bwd_float_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=bwd)

add_test(NAME conv_batch_bwd_float_wu
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --single-phase=wu)

add_test(NAME conv_negative_input_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --padding-upper={0,-1}
                --tiles-per-ipu=1)

# Note WU pass disabled due to T1737
foreach(phase fwd bwd)
  add_test(NAME conv_negative_kernel_padding_upper_simple_${phase}
          COMMAND single_conv_layer
                  --input-channels=1
                  --output-channels=1
                  --field={1,1}
                  --kernel-size={1,2}
                  --kernel-padding-lower={0,-1}
                  --tiles-per-ipu=1
                  --single-phase=${phase})

  add_test(NAME conv_positive_and_negative_kernel_padding_${phase}
           COMMAND single_conv_layer
           --input-channels=16
           --output-channels=16
           --field={5,5}
           --kernel-size=2
           --kernel-padding-lower=1
           --kernel-padding-upper=-1
           --tiles-per-ipu=1
           --single-phase=${phase})
endforeach(phase)

add_test(NAME conv_negative_input_padding_lower_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --padding-upper={0,-1}
                --tiles-per-ipu=1)

# Note WU pass disabled due to T1737
foreach(phase fwd bwd)
  add_test(NAME conv_negative_kernel_padding_lower_simple_${phase}
          COMMAND single_conv_layer
                  --input-channels=1
                  --output-channels=1
                  --field={1,1}
                  --kernel-size={1,2}
                  --kernel-padding-lower={0,-1}
                  --tiles-per-ipu=1
                  --single-phase=${phase})
endforeach(phase)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_test(NAME conv_large_output_tensor
         COMMAND single_conv_layer
                --field={16,16}
                --input-channels=16
                --output-channels=128
                --tiles-per-ipu=16
                --data-type=half)

add_test(NAME conv_large_weights
         COMMAND single_conv_layer
                --field={14,16}
                --input-channels=64
                --output-channels=64
                --kernel-size={16,1}
                --tiles-per-ipu=16
                --data-type=half
                --single-phase=fwd)

add_test(NAME conv_outer_product
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1024
                 --field={1,512}
                 --tiles-per-ipu=64)

add_test(NAME conv3x3_stride_1_half_partials
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=16
                 --stride=1
                 --kernel-size=3
                 --partials-type=half)

add_test(NAME conv1x1_kernel_with_padding_larger_than_field
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=8
                 --field={1,128}
                 --tiles-per-ipu=2
                 --padding={1,64})

add_test(NAME conv_input_dilation_and_stride_share_factor1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=64
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=3
                 --stride=3
                 --tiles-per-ipu=16)

add_test(NAME conv_input_dilation_and_stride_share_factor2
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=64
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=4
                 --stride=2
                 --tiles-per-ipu=16)

add_test(NAME conv_input_dilation_and_stride_share_factor3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=64
                 --field={14,14}
                 --kernel-size={3,3}
                 --in-dilation=2
                 --stride=4
                 --tiles-per-ipu=16)

add_test(NAME conv_input_dilation_and_stride_share_factor4
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=64
                --field={14,14}
                --kernel-size={3,3}
                --in-dilation=4
                --stride=6
                --tiles-per-ipu=16)

add_test(NAME conv_input_dilation_and_stride_share_factor5
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=64
                --field={14,14}
                --kernel-size={3,3}
                --in-dilation=6
                --stride=4
                --tiles-per-ipu=16)

# Example where the amount of padding that must be added to start of the input
# window for the conv nx1 vertex exceeds the amount of lower padding for the
# layer.
add_test(NAME conv_nx1_vertex_lower_padding
         COMMAND single_conv_layer
                 --field={16,32}
                 --input-channels=4
                 --padding-lower={1,0}
                 --output-channels=8
                 --kernel-size={2,1}
                 --kernel-dilation={2,1}
                 --stride={3,1}
                 --tiles-per-ipu=1)

# Test where, after padding and dilation, the last input index that would be
# multiplied by the a kernel element is in the middle of the lower input
# padding.
add_test(NAME conv1x1_input_ignored_with_stride
         COMMAND single_conv_layer
                 --field={2,2}
                 --in-dilation=2
                 --padding-lower=2
                 --padding-upper=1
                 --stride=7
                 --input-channels=16
                 --output-channels=16
                 --tiles-per-ipu=1)

#  Grouped convolution layer tests
add_test(NAME grouped_conv3x3_stride_2_256_out
         COMMAND single_conv_layer
                 --input-channels=64
                 --output-channels=128
                 --field={28,20}
                 --kernel-size=3
                 --data-type=half
                 --padding=2
                 --conv-groups=2
                 --stride=2)

add_test(NAME grouped_conv1x1_stride_1
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=4
             --kernel-size=1)

add_test(NAME grouped_conv1x1_stride_1_odd_outchans
        COMMAND single_conv_layer
            --input-channels=64
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --conv-groups=2
            --kernel-size=1)

add_test(NAME grouped_conv1x1_stride_1_odd_inchans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=128
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --conv-groups=2
            --kernel-size=1)

add_test(NAME grouped_conv1x1_stride_1_odd_in_and_out_chans
        COMMAND single_conv_layer
            --input-channels=65
            --output-channels=129
            --field={7,7}
            --tiles-per-ipu=16
            --stride=1
            --conv-groups=2
            --kernel-size=1)

add_test(NAME grouped_conv1x1_stride_3_odd_conv_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=64
             --field={14,14}
             --tiles-per-ipu=8
             --stride=3
             --conv-groups=3
             --kernel-size=1)

add_test(NAME grouped_conv1x1_thinly_spread
         COMMAND single_conv_layer
             --field={1,128}
             --input-channels=16
             --conv-groups=2
             --output-channels=8)

add_test(NAME grouped_conv3x3_stride_1_odd_groups
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=1
             --conv-groups=3
             --kernel-size=3)


add_test(NAME grouped_conv3x3_stride_2
         COMMAND single_conv_layer
             --input-channels=32
             --output-channels=16
             --field={7,7}
             --tiles-per-ipu=16
             --stride=2
             --conv-groups=2
             --kernel-size=3)

add_test(NAME grouped_conv3x3_asymmetric_stride
         COMMAND single_conv_layer
             --input-channels=16
             --output-channels=8
             --field={7,10}
             --tiles-per-ipu=16
             --stride={2,3}
             --conv-groups=2
             --kernel-size=3)

add_test(NAME grouped_conv3x3_stride_1_padding_1
         COMMAND single_conv_layer
              --input-channels=32
              --output-channels=16
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=3
              --conv-groups=2
              --padding=1)

add_test(NAME grouped_conv4x4_stride_1_padding_2
         COMMAND single_conv_layer
              --input-channels=16
              --output-channels=8
              --field={7,7}
              --tiles-per-ipu=16
              --stride=1
              --kernel-size=4
              --conv-groups=2
              --padding=2)

add_test(NAME grouped_conv3x3_stride_2_multi_tile
         COMMAND single_conv_layer
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=2
                 --stride=2
                 --conv-groups=2
                 --kernel-size=3)

add_test(NAME grouped_conv1x1_stride_3_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={7,7}
                 --tiles-per-ipu=1
                 --conv-groups=2
                 --stride=3)

add_test(NAME grouped_conv3x3_batch_2
         COMMAND single_conv_layer
                  --input-channels=16
                  --output-channels=16
                  --field={7,7}
                  --tiles-per-ipu=2
                  --stride=2
                  --kernel-size=3
                  --conv-groups=2
                  --batch-size=2)

add_test(NAME grouped_conv3x3_16_to_16_stride_1_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={28,20}
        --kernel-size=3
        --data-type=half
        --padding={1,2}
        --conv-groups=2
        --stride=1)

add_test(NAME grouped_conv3x3_batch_2_differing_padding
        COMMAND single_conv_layer
        --input-channels=16
        --output-channels=16
        --field={7,7}
        --tiles-per-ipu=2
        --stride=2
        --kernel-size=3
        --padding={1,2}
        --conv-groups=2
        --batch-size=2)

add_test(NAME grouped_conv3x3_stride_1_fwd_asym_padding
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --field={28,20}
                --kernel-size=3
                --data-type=half
                --padding-lower=1
                --padding-upper=2
                --conv-groups=2
                --stride=1)
add_test(NAME grouped_conv3x3_stride_1_fwd_asym_padding_2
         COMMAND single_conv_layer
                --input-channels=32
                --output-channels=48
                --field={28,20}
                --kernel-size=3
                --data-type=half
                --padding-lower=3
                --padding-upper=0
                --conv-groups=2
                --stride=1)

add_test(NAME grouped_conv5x5_stride_2_fwd_asym_padding
         COMMAND single_conv_layer
                  --input-channels=64
                  --output-channels=64
                  --field={28,20}
                  --kernel-size=5
                  --data-type=half
                  --padding-lower={1,2}
                  --padding-upper={3,3}
                  --stride=2
                  --conv-groups=2
                  --tiles-per-ipu=16)

add_test(NAME grouped_small_conv_many_tiles_1x1
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --field={14,14}
                 --partials-type=float
                 --conv-groups=2
                 --tiles-per-ipu=1216)

add_test(NAME grouped_small_conv_many_tiles_3x3
         COMMAND single_conv_layer
                 --input-channels=32
                 --output-channels=32
                 --kernel-size=3
                 --field={14,14}
                 --partials-type=float
                 --conv-groups=2
                 --tiles-per-ipu=1216)

add_test(NAME grouped_conv1x1_shallow_input
        COMMAND single_conv_layer
                --input-channels=8
                --output-channels=1024
                --field={16,16}
                --conv-groups=2
                --data-type=half)

add_test(NAME grouped_conv3x3_feature8x8_shallow
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=1
                 --conv-groups=2
                 --data-type=half)

add_test(NAME grouped_conv3x3_feature8x8_shallow_stride2
         COMMAND single_conv_layer
                 --batch-size=1
                 --input-channels=1
                 --output-channels=1
                 --field={8,8}
                 --kernel-size=3
                 --padding=1
                 --stride=2
                 --conv-groups=2
                 --data-type=half)

add_test(NAME grouped_conv_large_kernel
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --conv-groups=2
                 --stride=2)

add_test(NAME grouped_conv_large_kernel_bwd
         COMMAND single_conv_layer
                 --single-phase=bwd
                 --field={64,14}
                 --kernel-size={62,3}
                 --input-channels=16
                 --output-channels=16
                 --padding=1
                 --conv-groups=2
                 --stride=2)

add_test(NAME grouped_conv_large_kernel_with_small_field
         COMMAND single_conv_layer
                --field={1,16}
                 --padding={6,1}
                 --kernel-size={7,1}
                 --input-channels=16
                 --output-channels=16
                 --conv-groups=2
                 --tiles-per-ipu=16)

add_test(NAME grouped_conv_batch_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=half
                 --conv-groups=2
                 --batch-size=4)

add_test(NAME grouped_conv_batch_bwd_float
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --conv-groups=2
                 --batch-size=4)

add_test(NAME grouped_conv_batch_group
         COMMAND single_conv_layer
                 --field={28,28}
                 --input-channels=512
                 --output-channels=128
                 --conv-groups=2
                 --batch-size=2)

# conv_batch_bwd_float test split into the three component phases
add_test(NAME grouped_conv_batch_bwd_float_fwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=fwd)

add_test(NAME grouped_conv_batch_bwd_float_bwd
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=bwd)

add_test(NAME grouped_conv_batch_bwd_float_wu
         COMMAND single_conv_layer
                 --field={14,14}
                 --input-channels=32
                 --output-channels=32
                 --tiles-per-ipu=16
                 --data-type=float
                 --batch-size=4
                 --conv-groups=2
                 --single-phase=wu)

add_test(NAME grouped_conv_negative_input_padding_upper_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --padding-upper={0,-1}
                --conv-groups=2
                --tiles-per-ipu=1)

# Note WU pass disabled due to T1737
foreach(phase fwd bwd)
  add_test(NAME grouped_conv_negative_kernel_padding_upper_simple_${phase}
          COMMAND single_conv_layer
                  --input-channels=1
                  --output-channels=1
                  --field={1,1}
                  --kernel-size={1,2}
                  --kernel-padding-lower={0,-1}
                  --conv-groups=2
                  --tiles-per-ipu=1
                  --single-phase=${phase})
endforeach(phase)

add_test(NAME grouped_conv_negative_input_padding_lower_simple
        COMMAND single_conv_layer
                --input-channels=1
                --output-channels=1
                --field={1,2}
                --kernel-size={1,1}
                --padding-upper={0,-1}
                --conv-groups=2
                --tiles-per-ipu=1)

# Note WU pass disabled due to T1737
foreach(phase fwd bwd)
  add_test(NAME grouped_conv_negative_kernel_padding_lower_simple_${phase}
          COMMAND single_conv_layer
                  --input-channels=1
                  --output-channels=1
                  --field={1,1}
                  --kernel-size={1,2}
                  --conv-groups=2
                  --kernel-padding-lower={0,-1}
                  --tiles-per-ipu=1
                  --single-phase=${phase})
endforeach(phase)

# Test where the size of the output tensor is much bigger than the size weight
# tensor. As such it likely to be better to rearranging the weight deltas
# instead of the deltas in the weight update phase.
add_test(NAME grouped_conv_large_output_tensor
         COMMAND single_conv_layer
                --field={16,16}
                --input-channels=16
                --output-channels=64
                --tiles-per-ipu=16
                --conv-groups=2
                --data-type=half)

add_test(NAME grouped_conv_large_weights
         COMMAND single_conv_layer
                --field={14,16}
                --input-channels=32
                --output-channels=32
                --kernel-size={16,1}
                --tiles-per-ipu=16
                --data-type=half
                --conv-groups=2
                --single-phase=fwd)

add_test(NAME grouped_conv_outer_product
         COMMAND single_conv_layer
                 --input-channels=1
                 --output-channels=1024
                 --field={1,512}
                 --conv-groups=2
                 --tiles-per-ipu=64)

add_test(NAME conv3d_simple
         COMMAND single_conv_layer
                 --field={5,6,7}
                 --kernel-size={4,3,2}
                 --padding={1,1,1}
                 --input-channels=32
                 --output-channels=64
                 --tiles-per-ipu=48)

add_test(NAME conv3d_1_output_chan
         COMMAND single_conv_layer
                 --field={5,6,7}
                 --kernel-size={4,3,2}
                 --padding={1,1,1}
                 --input-channels=32
                 --output-channels=1
                 --tiles-per-ipu=48)

# 3D convolution with multiple batches, convolution groups, input dilation,
# kernel dilation and striding.
add_test(NAME conv3d_complex
         COMMAND single_conv_layer
                 --conv-groups=2
                 --batch-size=2
                 --field={4,3,32}
                 --in-dilation={1,2,1}
                 --padding={1,0,1}
                 --kernel-size={3,2,3}
                 --kernel-dilation={1,1,2}
                 --input-channels=16
                 --output-channels=64
                 --stride={1,1,3}
                 --tiles-per-ipu=16)

add_test(NAME fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half)

add_test(NAME fully_connected_half_fwd
         COMMAND fully_connected_layer
                 --single-phase=fwd
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half)

add_test(NAME fully_connected_half_bwd
        COMMAND fully_connected_layer
                --single-phase=bwd
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half)

add_test(NAME fully_connected_small_field_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 4
                --output-size 4
                --tiles-per-ipu 16
                --data-type=half)

add_test(NAME fully_connected_large_field_half_wu
        COMMAND fully_connected_layer
                --single-phase=wu
                --input-size 300
                --output-size 100
                --tiles-per-ipu 16
                --data-type=half)

add_test(NAME fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type=half)

add_test(NAME fully_connected_float
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float)

add_test(NAME fully_connected_float_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --data-type=float)

add_test(NAME fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --batch-size=4)

add_test(NAME fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --ipus=2
                 --batch-size=4)

add_test(NAME fully_connected_half_batch_7
      COMMAND fully_connected_layer
              --input-size 256
              --output-size 256
              --tiles-per-ipu 16
              --data-type=half
              --batch-size=7)

add_test(NAME fully_connected_half_batch_7_bwd_as_wu
      COMMAND fully_connected_layer
              --input-size 256
              --output-size 256
              --tiles-per-ipu 16
              --data-type=half
              --batch-size=7)

add_test(NAME fully_connected_float_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float
                 --batch-size=4)

add_test(NAME fully_connected_float_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=float
                 --ipus=2
                 --batch-size=4)

add_test(NAME fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 1
                 --data-type=half
                 --batch-size=4
                 --inference-only)

add_test(NAME grouped_fully_connected_half
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --num-groups 2
                 --data-type=half)


add_test(NAME grouped_fully_connected_half_two_ipu
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --ipus=2
                 --num-groups 3
                 --data-type=half)


add_test(NAME grouped_fully_connected_half_batch_4
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --num-groups 2
                 --batch-size=4)

add_test(NAME grouped_fully_connected_half_batch_4_two_ipus
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 16
                 --data-type=half
                 --num-groups 3
                 --ipus=2
                 --batch-size=4)

add_test(NAME grouped_fully_connected_half_batch_7
      COMMAND fully_connected_layer
              --input-size 256
              --output-size 256
              --tiles-per-ipu 16
              --data-type=half
              --num-groups 3
              --batch-size=7)

add_test(NAME grouped_fully_connected_batch_fwd_one_tile
         COMMAND fully_connected_layer
                 --input-size 300
                 --output-size 100
                 --tiles-per-ipu 1
                 --data-type=half
                 --batch-size=4
                 --num-groups 2
                 --inference-only)

# Fully connected layer where the RHS has fewer elements than the LHS
add_test(NAME fully_connected_small_rhs
         COMMAND fully_connected_layer
                 --batch-size=64
                 --input-size=512
                 --output-size=4
                 --data-type=half)

add_test(NAME max_pool_layer_half
         COMMAND pooling_layer
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME max_pool_layer_float
         COMMAND pooling_layer
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)
               

add_test(NAME max_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --width=100
                 --height=100
                 --kernel-size=3
                 --stride=2
                 --padding=1
                 --data-type=float)

add_test(NAME max_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME max_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_test(NAME max_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --width=112
                 --height=112
                 --kernel-size=3
                 --stride=2
                 --ipus 2)

add_test(NAME max_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --width=32
                 --height=1
                 --kernel-height=1
                 --kernel-width=4
                 --stride-height=1
                 --stride-width=4)

add_test(NAME max_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=1
                --padding-width-upper=2
                --padding-height-lower=3
                --padding-height-upper=4)

add_test(NAME max_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=-1
                --padding-width-upper=2
                --padding-height-lower=-3
                --padding-height-upper=4)

add_test(NAME avg_pool_layer_half
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=avg
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME avg_pool_layer_float
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=avg
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_test(NAME avg_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --pooling-type=avg
                 --width=100
                 --height=100
                 --kernel-size=3
                 --stride=2
                 --padding=1
                 --data-type=float)

add_test(NAME avg_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=avg
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME avg_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=avg
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_test(NAME avg_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --pooling-type=avg
                 --bwd-chans-per-group=8
                 --width=112
                 --height=112
                 --kernel-size=3
                 --stride=2
                 --ipus 2)

add_test(NAME avg_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=avg
                 --width=32
                 --height=1
                 --kernel-height=1
                 --kernel-width=4
                 --stride-height=1
                 --stride-width=4)

add_test(NAME avg_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --pooling-type=avg
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=1
                --padding-width-upper=2
                --padding-height-lower=3
                --padding-height-upper=4)

add_test(NAME avg_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --pooling-type=avg
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=-1
                --padding-width-upper=2
                --padding-height-lower=-3
                --padding-height-upper=4)

add_test(NAME sum_pool_layer_half
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=sum
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME sum_pool_layer_float
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=sum
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)
               

add_test(NAME sum_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --pooling-type=sum
                 --width=100
                 --height=100
                 --kernel-size=3
                 --stride=2
                 --padding=1
                 --data-type=float)

add_test(NAME sum_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=sum
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_test(NAME sum_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --pooling-type=sum
                 --channels 32
                 --width=28
                 --height=28
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_test(NAME sum_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --pooling-type=sum
                 --bwd-chans-per-group=8
                 --width=112
                 --height=112
                 --kernel-size=3
                 --stride=2
                 --ipus 2)

add_test(NAME sum_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --pooling-type=sum
                 --width=32
                 --height=1
                 --kernel-height=1
                 --kernel-width=4
                 --stride-height=1
                 --stride-width=4)

add_test(NAME sum_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --pooling-type=sum
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=1
                --padding-width-upper=2
                --padding-height-lower=3
                --padding-height-upper=4)

add_test(NAME sum_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --pooling-type=sum
                --width=32
                --height=16
                --kernel-height=5
                --kernel-width=3
                --stride-height=1
                --stride-width=4
                --padding-width-lower=-1
                --padding-width-upper=2
                --padding-height-lower=-3
                --padding-height-upper=4)

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
                 --batch-size=2
                 --nonlinearity-type sigmoid)


add_test(NAME rnn_3x32x2x38_with_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --input-size 32
                 --output-size 38
                 --batch-size=2
                 --phase all
                 --nonlinearity-type sigmoid
                 --apply-feedforward-weights) 
                 
add_test(NAME basic_lstm_40x4x38_seq_2_half_data
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --phase all
                 --sequence-size 2)

add_test(NAME basic_lstm_40x4x38_seq_2_half_data_preweight_inp
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --pre-weight-input=1
                 --phase all
                 --sequence-size 2)

add_test(NAME basic_lstm_40x4x38_seq_2_float_data
         COMMAND lstm_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --sequence-size 2
                 --phase all
                 --data-type=float)

add_test(NAME conv1x1_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --tiles-per-ipu=1
                 --in-dilation=2)

add_test(NAME conv1x1_stride_3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2)

add_test(NAME conv3x3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation=2)

add_test(NAME conv3x3_stride_3_in_dilation_2_fwd
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --stride=3
                 --in-dilation=2)

add_test(NAME conv3x3_in_dilation_asymmetric
         COMMAND single_conv_layer
                 --single-phase=fwd
                 --input-channels=16
                 --output-channels=16
                 --field={14,14}
                 --kernel-size=3
                 --tiles-per-ipu=1
                 --in-dilation={2,3})

add_test(NAME conv3x3_kernel_dilation_amp
        COMMAND single_conv_layer
                --input-channels=32
                --output-channels=32
                --weight-update-method=amp
                --field={14,28}
                --kernel-size={3,5}
                --kernel-dilation={3,2}
                --tiles-per-ipu=16)


# Note WU pass disabled due to T1737
foreach(phase fwd bwd)
  add_test(NAME conv3x3_kernel_padding_amp_${phase}
           COMMAND single_conv_layer
                   --input-channels=32
                   --output-channels=32
                   --weight-update-method=amp
                   --field={14,14}
                   --kernel-size={3,3}
                   --kernel-padding-lower={4,2}
                   --kernel-padding-upper={4,3}
                   --tiles-per-ipu=16
                   --single-phase=${phase})
endforeach(phase)
