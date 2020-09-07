add_subdirectory(codelets)

add_unit_test(BigNLVertices BigNLVertices.cpp)
add_unit_test(GraphProgLocationTest GraphProgLocationTest.cpp)
add_unit_test(LossTest LossTest.cpp
              SUITES ArgMinMax TopK SUM_SQUARED_LOSS_suite
                     CROSS_ENTROPY_LOSS_suite Accuracy)

# NonLinearity Grad Sweep tests
add_multi_target_test_executable(NonLinearityGradSweepTest NonLinearityGradSweepTest.cpp)

add_multitarget_test(NAME NonLinearityGradSweepTest
  COMMAND NonLinearityGradSweepTest
  --nl-type gelu)

# NonLinearity Sweep tests
add_multi_target_test_executable(NonLinearitySweepTest NonLinearitySweepTest.cpp)

add_multitarget_test(NAME NonLinearitySweepTest
  COMMAND NonLinearitySweepTest
  --nl-type gelu)

add_unit_test(NonLinearityTest NonLinearityTest.cpp)
add_unit_test(SpatialSoftmaxTest SpatialSoftmaxTest.cpp)


add_multitarget_test(NAME max_pool_layer_half_with_introspection
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --stride=2
                 --data-type=half
                 --use-introspection=1)

add_multitarget_test(NAME max_pool_layer_half_with_introspection_and_scale_grad
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --stride=2
                 --data-type=half
                 --use-introspection=1)

add_multitarget_test(NAME max_pool_layer_3d_half_with_introspection_and_scale_grad
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14,7}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --stride={2,2,1}
                 --data-type=half
                 --use-introspection=1)

add_multitarget_test(NAME max_pool_layer_half_without_introspection
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --stride=2
                 --data-type=half
                 --use-introspection=0)

add_multitarget_test(NAME max_pool_layer_float
         COMMAND pooling_layer
                 --channels 16
                 --field={9,14}
                 --kernel-size=2
                 --tiles-per-ipu=16
                 --stride=2
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_1chan_overlapping_kernel
         COMMAND pooling_layer
                 --channels 1
                 --field={50,50}
                 --kernel-size=3
                 --tiles-per-ipu=16
                 --stride=2
                 --padding-lower=1
                 --padding-upper=1
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_1chan_overlapping_kernel_scaled_grad
         COMMAND pooling_layer
                 --channels 1
                 --field={50,50}
                 --kernel-size=3
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --stride=2
                 --padding-lower=1
                 --padding-upper=1
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_3d_1chan_overlapping_kernel_scaled_grad
         COMMAND pooling_layer
                 --channels 1
                 --field={50,50,7}
                 --kernel-size={3,3,1}
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --stride=2
                 --padding-lower=1
                 --padding-upper=1
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_half_batch
         COMMAND pooling_layer
                 --batch-size=4
                 --channels 16
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=half)

add_multitarget_test(NAME max_pool_layer_float_batch
         COMMAND pooling_layer
                 --batch-size=2
                 --channels 16
                 --field={14,14}
                 --tiles-per-ipu=16
                 --kernel-size=2
                 --stride=2
                 --data-type=float)

add_multitarget_test(NAME max_pool_layer_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --field={56,56}
                 --kernel-size=3
                 --tiles-per-ipu 16
                 --stride=2
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_3d_2ipu
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --field={14,14,4}
                 --kernel-size=3
                 --tiles-per-ipu 16
                 --stride={2,4,1}
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_2ipu_scaled_grad
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --field={56,56}
                 --kernel-size=3
                 --tiles-per-ipu 304
                 --stride=2
                 --use-scaled-grad=1
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_3d_2ipu_scaled_grad
         COMMAND pooling_layer
                 --channels 64
                 --bwd-chans-per-group=8
                 --field={14,14,4}
                 --kernel-size=3
                 --tiles-per-ipu 16
                 --stride={2,4,1}
                 --use-scaled-grad=1
                 --ipus 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(NAME max_pool_layer_1x4
         COMMAND pooling_layer
                 --channels 32
                 --field={1,32}
                 --tiles-per-ipu=16
                 --kernel-size={1,4}
                 --stride={1,4})

add_multitarget_test(NAME max_pool_layer_1x4_scaled_grad
         COMMAND pooling_layer
                 --channels 32
                 --field={1,32}
                 --tiles-per-ipu=16
                 --use-scaled-grad=1
                 --kernel-size={1,4}
                 --stride={1,4})

add_multitarget_test(NAME max_pool_layer_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --field={16,32}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={3,1}
                --padding-upper={4,2})

add_multitarget_test(NAME max_pool_layer_mixed_padding_scaled_grad
        COMMAND pooling_layer
                --channels 32
                --field={16,32}
                --tiles-per-ipu=16
                --use-scaled-grad=1
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={3,1}
                --padding-upper={4,2})

add_multitarget_test(NAME max_pool_layer_negative_mixed_padding
        COMMAND pooling_layer
                --channels 32
                --field={16,32}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --padding-lower={-3,-1}
                --padding-upper={4,2})

add_multitarget_test(NAME max_pool_layer_negative_mixed_padding_scaled_grad
        COMMAND pooling_layer
                --channels 32
                --field={16,32}
                --tiles-per-ipu=16
                --kernel-size={5,3}
                --stride={1,4}
                --use-scaled-grad=1
                --padding-lower={-3,-1}
                --padding-upper={4,2})

add_multitarget_test(NAME max_pool_layer_folded_spatial_dims
        COMMAND pooling_layer
                 --channels 1
                 --field={3,3}
                 --kernel-size={1,1}
                 --tiles-per-ipu=16
                 --stride={2,2}
                 --data-type=half
                 --use-introspection=1)


foreach(OPTIMIZE_FOR_SPEED true false)

  # Presently implementations that optimise for speed only cover average and
  # sum pooling, so verify those variants.
  # PLUS at least one max pool to ensure that it is correctly unaffected

  add_multitarget_test(NAME max_pool_layer_same_field_and_kernel_size_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 16
                   --pooling-type=max
                   --field={7,7}
                   --tiles-per-ipu=16
                   --kernel-size=7
                   --stride=7
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=half)


  add_multitarget_test(NAME avg_pool_layer_half_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 32
                   --pooling-type=avg
                   --field={28,28}
                   --tiles-per-ipu=16
                   --kernel-size=2
                   --stride=2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=half)

  add_multitarget_test(NAME avg_pool_layer_float_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 32
                   --pooling-type=avg
                   --field={28,28}
                   --tiles-per-ipu=16
                   --kernel-size=2
                   --stride=2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=float)

  add_multitarget_test(NAME avg_pool_layer_1chan_overlapping_kernel_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 1
                   --pooling-type=avg
                   --field={100,100}
                   --tiles-per-ipu=16
                   --kernel-size=3
                   --stride=2
                   --padding-lower=1
                   --padding-upper=1
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=float)

  add_multitarget_test(NAME avg_pool_layer_same_field_and_kernel_size_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 16
                   --pooling-type=avg
                   --field={7,7}
                   --tiles-per-ipu=16
                   --kernel-size=7
                   --stride=7
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=half)

  add_multitarget_test(NAME sum_pool_layer_same_field_and_kernel_size_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 16
                   --pooling-type=sum
                   --field={7,7}
                   --tiles-per-ipu=16
                   --kernel-size=7
                   --stride=7
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=half)

  add_multitarget_test(NAME sum_pool_layer_3d_same_field_and_kernel_size_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 16
                   --pooling-type=sum
                   --field={4,4,4}
                   --tiles-per-ipu=16
                   --kernel-size=4
                   --stride=4
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=half)

  add_multitarget_test(NAME avg_pool_layer_half_batch_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --batch-size=4
                   --pooling-type=avg
                   --channels 16
                   --field={14,14}
                   --tiles-per-ipu=16
                   --kernel-size=2
                   --stride=2
                  --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                  --data-type=half)

  add_multitarget_test(NAME avg_pool_layer_float_batch_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --batch-size=4
                   --pooling-type=avg
                   --channels 32
                   --field={14,28}
                   --tiles-per-ipu=16
                   --kernel-size=2
                   --stride=2
                   --data-type=float
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   VARIANTS ${TimesOutOnSim})

    add_multitarget_test(NAME avg_pool_layer_2ipu_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 64
                   --pooling-type=avg
                   --bwd-chans-per-group=8
                   --field={56,56}
                   --kernel-size=3
                   --tiles-per-ipu=304
                   --stride=2
                   --ipus 2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   VARIANTS ${TimesOutOnSim})

  add_multitarget_test(NAME avg_pool_layer_1x4_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 16
                   --pooling-type=avg
                   --field={1,32}
                   --tiles-per-ipu=8
                   --kernel-size={1,4}
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --stride={1,4})

  add_multitarget_test(NAME avg_pool_layer_mixed_padding_${OPTIMIZE_FOR_SPEED}
          COMMAND pooling_layer
                  --channels 16
                  --pooling-type=avg
                  --field={16,24}
                  --tiles-per-ipu=16
                  --kernel={5,3}
                  --stride={1,4}
                  --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                  --padding-lower={3,1}
                  --padding-upper={4,2})

  add_multitarget_test(NAME avg_pool_layer_negative_mixed_padding_${OPTIMIZE_FOR_SPEED}
          COMMAND pooling_layer
                  --channels 16
                  --pooling-type=avg
                  --field={16,24}
                  --tiles-per-ipu=16
                  --kernel-size={5,3}
                  --stride={1,4}
                  --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                  --padding-lower={-3,-1}
                  --padding-upper={4,2})

  add_multitarget_test(NAME avg_pool_layer_3d_negative_mixed_padding_${OPTIMIZE_FOR_SPEED}
          COMMAND pooling_layer
                  --channels 16
                  --pooling-type=avg
                  --field={16,24,4}
                  --tiles-per-ipu=16
                  --kernel-size={5,3,1}
                  --stride={1,4,1}
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                  --padding-lower={-3,-1,-1}
                  --padding-upper={4,2,-1})

  add_multitarget_test(NAME sum_pool_layer_half_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 16
                   --pooling-type=sum
                   --field={14,14}
                   --tiles-per-ipu=16
                   --kernel-size=2
                   --stride=2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=half)

  add_multitarget_test(NAME sum_pool_layer_float_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 32
                   --pooling-type=sum
                   --field={14,14}
                   --tiles-per-ipu=16
                   --kernel-size=2
                   --stride=2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=float)

  add_multitarget_test(NAME sum_pool_layer_1chan_overlapping_kernel_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 1
                   --pooling-type=sum
                   --field={100,100}
                   --tiles-per-ipu=16
                   --kernel-size=3
                   --stride=2
                   --padding-lower=1
                   --padding-upper=1
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=float)

  add_multitarget_test(NAME sum_pool_layer_half_batch_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --batch-size=4
                   --pooling-type=sum
                   --channels 16
                   --field={14,14}
                   --tiles-per-ipu=16
                   --kernel-size=2
                   --stride=2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=half)

  add_multitarget_test(NAME sum_pool_layer_float_batch_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --batch-size=4
                   --pooling-type=sum
                   --channels 16
                   --field={14,14}
                   --tiles-per-ipu=16
                   --kernel-size=2
                   --stride=2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --data-type=float)

  add_multitarget_test(NAME sum_pool_layer_2ipu_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 64
                   --pooling-type=sum
                   --bwd-chans-per-group=8
                   --field={56,56}
                   --kernel-size=3
                   --tiles-per-ipu=16
                   --stride=2
                   --ipus 2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   VARIANTS ${TimesOutOnSim})

  add_multitarget_test(NAME sum_pool_layer_3d_2ipu_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 64
                   --pooling-type=sum
                   --bwd-chans-per-group=8
                   --field={14,14,4}
                   --kernel-size={3,3,1}
                   --tiles-per-ipu=16
                   --stride=2
                   --ipus 2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   VARIANTS ${TimesOutOnSim})

  add_multitarget_test(NAME sum_pool_layer_1x4_${OPTIMIZE_FOR_SPEED}
           COMMAND pooling_layer
                   --channels 32
                   --pooling-type=sum
                   --field={1,32}
                   --tiles-per-ipu=16
                   --kernel-size={1,4}
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --stride={1,4})

  add_multitarget_test(NAME sum_pool_layer_mixed_padding_${OPTIMIZE_FOR_SPEED}
          COMMAND pooling_layer
                  --channels 16
                  --pooling-type=sum
                  --field={16,24}
                  --tiles-per-ipu=16
                  --kernel-size={5,3}
                  --stride={1,4}
                  --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                  --padding-lower={3,1}
                  --padding-upper={4,2})

  add_multitarget_test(NAME sum_pool_layer_negative_mixed_padding_${OPTIMIZE_FOR_SPEED}
          COMMAND pooling_layer
                  --channels 16
                  --pooling-type=sum
                  --field={16,24}
                  --tiles-per-ipu=16
                  --kernel-size={5,3}
                  --stride={1,4}
                  --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                  --padding-lower={-3,-1}
                  --padding-upper={4,2})

  add_multitarget_test(NAME avg_pool_half_single_channel_${OPTIMIZE_FOR_SPEED}
          COMMAND pooling_layer
                  --channels=1
                  --field={10,10}
                  --kernel-size={5,5}
                  --padding-lower={0,0}
                  --padding-upper={0,0}
                  --batch-size=1
                  --pooling-type=avg
                  --data-type=half
                  --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                  VARIANTS ${IPUMODEL_VARIANTS};IpuModel)
endforeach() # OPTIMISE_FOR_SPEED

add_multitarget_test(
         NAME rnn_3x2x38_no_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --output-size 38
                 --tiles-per-ipu=16
                 --batch-size=2
                 --nonlinearity-type sigmoid)

add_multitarget_test(
         NAME rnn_3x32x2x38_with_feedfwd
         COMMAND rnn_layer
                 --sequence-size 3
                 --input-size 24
                 --output-size 32
                 --batch-size=2
                 --phase all
                 --tiles-per-ipu=16
                 --nonlinearity-type sigmoid
                 --apply-feedforward-weights)

foreach(CELL_ORDER "{forget,input,cell,output}" "{input,output,forget,cell}")
        foreach(PARTIALS_TYPE half float)
                add_multitarget_test(
                        NAME basic_lstm_40x4x38_seq_2_half_data_${PARTIALS_TYPE}_partials_${CELL_ORDER}
                        COMMAND lstm_layer
                                --input-size 40
                                --batch-size=4
                                --output-size 38
                                --tiles-per-ipu=16
                                --phase all
                                --sequence-size 2
                                --partials-type=${PARTIALS_TYPE}
                                --accumulators-type=${PARTIALS_TYPE}
                                --cell-order=${CELL_ORDER}
                                VARIANTS ${TimesOutOnSim}
                                LABELS lstm)
        endforeach()

        foreach(PARTIALS_TYPE half float)
        add_multitarget_test(
                NAME basic_lstm_40x4x38_seq_2_half_data_preweight_inp_${PARTIALS_TYPE}_partials_${CELL_ORDER}
                COMMAND lstm_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --pre-weight-input=1
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --partials-type=${PARTIALS_TYPE}
                        --accumulators-type=${PARTIALS_TYPE}
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim}
                        LABELS lstm)
        endforeach()

        add_multitarget_test(
                NAME basic_lstm_40x4x38_seq_2_float_data_${CELL_ORDER}
                COMMAND lstm_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --sequence-size 2
                        --tiles-per-ipu=16
                        --phase all
                        --data-type=float
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim}
                        LABELS lstm)

        add_multitarget_test(
                NAME basic_lstm_40x4x38_seq_2_float_data_fwd_only_${CELL_ORDER}
                COMMAND lstm_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --sequence-size 2
                        --tiles-per-ipu=16
                        --phase fwd
                        --data-type=float
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim}
                        LABELS lstm)

        foreach(PARTIALS_TYPE half float)
        add_multitarget_test(
                NAME basic_lstm_40x4x38_seq_2_half_data_celltanhrecomp_${PARTIALS_TYPE}_partials_${CELL_ORDER}
                COMMAND lstm_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --recomputation-mode=cellAndTanh
                        --partials-type=${PARTIALS_TYPE}
                        --accumulators-type=${PARTIALS_TYPE}
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim}
                        LABELS lstm)
        endforeach()

        foreach(PARTIALS_TYPE half float)
        add_multitarget_test(
                NAME basic_lstm_40x4x38_seq_2_half_data_runs_2_${PARTIALS_TYPE}_partials_${CELL_ORDER}
                COMMAND lstm_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --runs 2
                        --partials-type=${PARTIALS_TYPE}
                        --accumulators-type=${PARTIALS_TYPE}
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim}
                        LABELS lstm)
        endforeach()
endforeach()

add_multitarget_test(
         NAME basic_gru_40x4x38_seq_2_half_data
         COMMAND gru_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --tiles-per-ipu=16
                 --phase all
                 --sequence-size 2
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME basic_gru_40x4x38_seq_2_float_data
         COMMAND gru_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --sequence-size 2
                 --tiles-per-ipu=16
                 --phase all
                 --data-type=float
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME basic_gru_40x4x38_seq_2_float_data_fwd_only
         COMMAND gru_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --sequence-size 2
                 --tiles-per-ipu=16
                 --phase fwd
                 --data-type=float
                 VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME basic_gru_40x4x38_seq_2_half_data_runs_2
         COMMAND gru_layer
                 --input-size 40
                 --batch-size=4
                 --output-size 38
                 --tiles-per-ipu=16
                 --phase all
                 --sequence-size 2
                 --runs 2
                 VARIANTS ${TimesOutOnSim})
