add_subdirectory(codelets)

add_unit_test(BigNLVertices BigNLVertices.cpp)
add_unit_test(GraphProgLocationTest GraphProgLocationTest.cpp)
add_unit_test(LossTest LossTest.cpp
              SUITES ArgMinMax TopK SUM_SQUARED_LOSS_suite
                     CROSS_ENTROPY_LOSS_suite Accuracy)

# NonLinearity Grad and activations Sweep tests
add_multi_target_test_executable(NonLinearityGradSweepTest NonLinearityGradSweepTest.cpp)
add_multi_target_test_executable(NonLinearitySweepTest NonLinearitySweepTest.cpp)


foreach(NL_TYPE gelu swish)
  add_multitarget_test(NAME NonLinearity_${NL_TYPE}_SweepTest
    COMMAND NonLinearitySweepTest
    --nl-type ${NL_TYPE})

  add_multitarget_test(NAME NonLinearity_${NL_TYPE}_GradSweepTest
    COMMAND NonLinearityGradSweepTest
    --nl-type ${NL_TYPE})
endforeach()

# Distributed batch norm
add_multi_target_test_executable(DistributedBatchNorm DistributedBatchNorm.cpp)
foreach(STABLE_ALGO true false)
  foreach(UNBIASED_VAR true false)
    foreach(DATA_TYPE half float)
      foreach(NUM_REPLICAS 1 2 4)
        add_multitarget_test(NAME DistributedBatchNorm_stable_${STABLE_ALGO}_unbiased_${UNBIASED_VAR}_datatype_${DATA_TYPE}_repl_${NUM_REPLICAS}
          COMMAND DistributedBatchNorm
          --eps 0.00001
          --tiles-per-ipu 16
          --stable-algo-for-stats ${STABLE_ALGO}
          --unbiased-var-est ${UNBIASED_VAR}
          --data-type ${DATA_TYPE}
          --num-replicas ${NUM_REPLICAS}
          --partials-type float
          --dims={2,32,28,28}
          VARIANTS ${IPUMODEL_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

add_unit_test(NonLinearityTest NonLinearityTest.cpp)
add_unit_test(SpatialSoftmaxTest SpatialSoftmaxTest.cpp)
add_unit_test(LogSoftmaxTest LogSoftmaxTest.cpp)

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
  foreach(USE_FLOAT_PARTIALS true false)
  add_multitarget_test(NAME avg_pool_layer_half_${OPTIMIZE_FOR_SPEED}_${USE_FLOAT_PARTIALS}
           COMMAND pooling_layer
                   --channels 32
                   --pooling-type=avg
                   --field={32,32}
                   --tiles-per-ipu=16
                   --kernel-size=4
                   --stride=2
                   --optimize-for-speed=${OPTIMIZE_FOR_SPEED}
                   --use-float-partials=${USE_FLOAT_PARTIALS}
                   --data-type=half)
  endforeach()
endforeach()

# TODO - Reinstate this once T40690 is fixed
#add_multitarget_test(NAME avg_pool_layer_with_transform
#           COMMAND pooling_layer
#                   --channels 1
#                   --pooling-type=avg
#                   --field={1,4,4,1}
#                   --tiles-per-ipu=1
#                   --kernel-size=1
#                   --stride={1,2,2,1}
#                   --use-float-partials=true
#                  --data-type=half)

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
                  VARIANTS ${IPUMODEL_VARIANTS})
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

        add_multitarget_test(
                NAME basic_lstm_40x4x38_seq_2_float_data_bwd_only_${CELL_ORDER}
                COMMAND lstm_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --sequence-size 2
                        --tiles-per-ipu=16
                        --phase bwd
                        --data-type=float
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim}
                        LABELS lstm)

        add_multitarget_test(
                NAME basic_lstm_40x4x38_seq_2_float_data_wu_only_${CELL_ORDER}
                COMMAND lstm_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --sequence-size 2
                        --tiles-per-ipu=16
                        --phase wu
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

foreach(ACTIVATION tanh sigmoid hard_sigmoid relu softmax)
        foreach(RECURRENT_ACTIVATION tanh sigmoid hard_sigmoid softmax)
                add_multitarget_test(
                        NAME basic_lstm_act_config_40x1x38_seq_2_float_${ACTIVATION}_${RECURRENT_ACTIVATION}
                        COMMAND lstm_layer
                                --input-size 40
                                --output-size 38
                                --tiles-per-ipu=16
                                --data-type=float
                                --sequence-size 2
                                --shards=1
                                --phase all
                                --activation ${ACTIVATION}
                                --recurrent-activation ${RECURRENT_ACTIVATION}
                                VARIANTS ${TimesOutOnSim}
                                LABELS lstm)
                endforeach()
endforeach()



foreach(CELL_ORDER "{reset,update,cell}" "{update,cell,reset}")
        add_multitarget_test(
                NAME basic_gru_40x4x38_seq_2_half_data_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --cell-order=${CELL_ORDER}
                        --reset-after=false
                        VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
                NAME basic_gru_40x4x38_seq_2_float_data_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --sequence-size 2
                        --tiles-per-ipu=16
                        --phase all
                        --data-type=float
                        --cell-order=${CELL_ORDER}
                        --reset-after=false
                        VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
                NAME basic_gru_40x4x38_seq_2_float_data_fwd_only_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --sequence-size 2
                        --tiles-per-ipu=16
                        --phase fwd
                        --data-type=float
                        --cell-order=${CELL_ORDER}
                        --reset-after=false
                        VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
                NAME basic_gru_40x4x38_seq_2_half_data_runs_2_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --runs 2
                        --cell-order=${CELL_ORDER}
                        --reset-after=false
                        VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
                NAME basic_gru_40x4x38_seq_2_float_data_reset_after_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --output-size 38
                        --sequence-size 2
                        --tiles-per-ipu=16
                        --phase all
                        --data-type=float
                        --cell-order=${CELL_ORDER}
                        --reset-after=true
                        VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
                NAME augru_40x4x38_seq_2_half_data_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --with-attention true
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
                NAME augru_40x4x38_seq_2_float_data_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --with-attention true
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
                NAME augru_with_real_time_steps_40x4x38_seq_2_half_data_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --with-attention true
                        --with-real-time-steps true
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim})

        add_multitarget_test(
                NAME augru_with_real_time_steps_40x4x38_seq_2_float_data_${CELL_ORDER}
                COMMAND gru_layer
                        --input-size 40
                        --batch-size=4
                        --with-attention true
                        --with-real-time-steps true
                        --output-size 38
                        --tiles-per-ipu=16
                        --phase all
                        --sequence-size 2
                        --cell-order=${CELL_ORDER}
                        VARIANTS ${TimesOutOnSim})
endforeach()

foreach(ACTIVATION tanh sigmoid hard_sigmoid relu softmax)
        foreach(RECURRENT_ACTIVATION tanh sigmoid hard_sigmoid softmax) # relu cannot be used for recurrent activation, must be in [0,1] range.
                add_multitarget_test(
                        NAME basic_gru_act_config_40x1x38_seq_2_float_${ACTIVATION}_${RECURRENT_ACTIVATION}
                        COMMAND gru_layer
                                --input-size 40
                                --output-size 38
                                --tiles-per-ipu=16
                                --data-type=float
                                --sequence-size 2
                                --shards=1
                                --phase all
                                --activation ${ACTIVATION}
                                --recurrent-activation ${RECURRENT_ACTIVATION}
                                VARIANTS ${TimesOutOnSim})
                add_multitarget_test(
                        NAME augru_act_config_40x1x38_seq_2_float_${ACTIVATION}_${RECURRENT_ACTIVATION}
                        COMMAND gru_layer
                                --input-size 40
                                --output-size 38
                                --tiles-per-ipu=16
                                --data-type=float
                                --sequence-size 2
                                --shards=1
                                --phase all
                                --with-attention true
                                --activation ${ACTIVATION}
                                --recurrent-activation ${RECURRENT_ACTIVATION}
                                VARIANTS ${TimesOutOnSim})
        endforeach()
endforeach()

# Sharding tests
foreach(NUM_SHARDS 3 5)
        foreach(PARTIALS_TYPE half float)

                # Test LSTM sharding
                foreach(CELL_ORDER "{forget,input,cell,output}" "{input,output,forget,cell}")
                        foreach(PRE_WEIGHT_INPUT 0 1)
                                add_multitarget_test(
                                        NAME lstm_16x2x32_seq_601_shards_${NUM_SHARDS}_partials_${PARTIALS_TYPE}_order_${CELL_ORDER}_pre_weight_${PRE_WEIGHT_INPUT}
                                        COMMAND lstm_layer
                                                --input-size 16
                                                --batch-size=2
                                                --output-size 32
                                                --pre-weight-input=${PRE_WEIGHT_INPUT}
                                                --tiles-per-ipu=20
                                                --phase all
                                                --sequence-size 601
                                                --partials-type=${PARTIALS_TYPE}
                                                --accumulators-type=${PARTIALS_TYPE}
                                                --cell-order=${CELL_ORDER}
                                                --shards=${NUM_SHARDS}
                                                VARIANTS "Hw")
                        endforeach()
                endforeach()
                add_multitarget_test(
                        NAME lstm_16x2x32_seq_601_shards_${NUM_SHARDS}_partials_${PARTIALS_TYPE}_ignore_input_gradient
                        COMMAND lstm_layer
                                --input-size 16
                                --batch-size=2
                                --output-size 32
                                --tiles-per-ipu=20
                                --phase all
                                --sequence-size 601
                                --partials-type=${PARTIALS_TYPE}
                                --accumulators-type=${PARTIALS_TYPE}
                                --shards=${NUM_SHARDS}
                                --ignore-input-gradient
                                VARIANTS "Hw")

                # Test GRU sharding
                foreach(RESET_AFTER 0 1)
                        foreach(CELL_ORDER "{reset,update,cell}" "{update,cell,reset}")
                                add_multitarget_test(
                                        NAME gru_16x2x32_seq_200_shards_${NUM_SHARDS}_data_${PARTIALS_TYPE}_reset_after_${RESET_AFTER}_order_${CELL_ORDER}
                                        COMMAND gru_layer
                                                --input-size 16
                                                --batch-size=2
                                                --output-size 8
                                                --tiles-per-ipu=20
                                                --phase all
                                                --data-type=${PARTIALS_TYPE}
                                                --sequence-size 200
                                                --cell-order=${CELL_ORDER}
                                                --reset-after=${RESET_AFTER}
                                                --shards=${NUM_SHARDS}
                                                VARIANTS "Hw")
                        endforeach()
                        add_multitarget_test(
                                NAME gru_16x2x32_seq_200_shards_${NUM_SHARDS}_data_${PARTIALS_TYPE}_reset_after_${RESET_AFTER}_ignore_input_gradient
                                COMMAND gru_layer
                                        --input-size 16
                                        --batch-size=2
                                        --output-size 8
                                        --tiles-per-ipu=20
                                        --phase all
                                        --data-type=${PARTIALS_TYPE}
                                        --sequence-size 200
                                        --reset-after=${RESET_AFTER}
                                        --shards=${NUM_SHARDS}
                                        --ignore-input-gradient
                                        VARIANTS "Hw")
                        foreach(REAL_TIME_STEPS false true)
                                foreach(ATTENTION false true)
                                        foreach(CELL_ORDER "{reset,update,cell}" "{update,cell,reset}")
                                                add_multitarget_test(
                                                        NAME augru_16x2x32_seq_102_shards_${NUM_SHARDS}_reset_after_${RESET_AFTER}_attention_${ATTENTION}_real_time_steps_${REAL_TIME_STEPS}_data_${PARTIALS_TYPE}_order_${CELL_ORDER}
                                                        COMMAND gru_layer
                                                                --input-size 16
                                                                --batch-size=2
                                                                --with-attention ${ATTENTION}
                                                                --with-real-time-steps ${REAL_TIME_STEPS}
                                                                --output-size 8
                                                                --tiles-per-ipu=20
                                                                --phase all
                                                                --data-type=${PARTIALS_TYPE}
                                                                --sequence-size 102
                                                                --cell-order=${CELL_ORDER}
                                                                --reset-after=${RESET_AFTER}
                                                                --shards=${NUM_SHARDS}
                                                                VARIANTS "Hw")
                                        endforeach()
                                        add_multitarget_test(
                                                NAME augru_16x2x32_seq_102_shards_${NUM_SHARDS}_reset_after_${RESET_AFTER}_attention_${ATTENTION}_real_time_steps_${REAL_TIME_STEPS}_data_${PARTIALS_TYPE}_ignore_input_gradient
                                                COMMAND gru_layer
                                                        --input-size 16
                                                        --batch-size=2
                                                        --with-attention ${ATTENTION}
                                                        --with-real-time-steps ${REAL_TIME_STEPS}
                                                        --output-size 8
                                                        --tiles-per-ipu=20
                                                        --phase all
                                                        --data-type=${PARTIALS_TYPE}
                                                        --sequence-size 102
                                                        --reset-after=${RESET_AFTER}
                                                        --shards=${NUM_SHARDS}
                                                        --ignore-input-gradient
                                                        VARIANTS "Hw")
                                endforeach()
                        endforeach()
                endforeach()
        endforeach()
endforeach()

# Lstm/Gru shard auto calculation with and without code reuse.
foreach(REUSE false true)
        add_multitarget_test(
                NAME augru_16x1x16_seq_1000_reuse_${REUSE}
                COMMAND gru_layer
                        --input-size 16
                        --batch-size=1
                        --with-attention true
                        --with-real-time-steps true
                        --output-size 16
                        --tiles-per-ipu=20
                        --phase all
                        --data-type=float
                        --sequence-size 1000
                        --code-reuse ${REUSE}
                        VARIANTS ${IPUMODEL_VARIANTS})
        add_multitarget_test(
                NAME lstm_16x1x16_seq_1000_reuse_${REUSE}
                COMMAND lstm_layer
                        --input-size 16
                        --batch-size=1
                        --output-size 16
                        --tiles-per-ipu=20
                        --phase all
                        --data-type=float
                        --sequence-size 1000
                        --code-reuse ${REUSE}
                        VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

# Test variable time-steps for LSTM/GRU
foreach(ALL_OUTPUT  0 1)
        add_multitarget_test(
                NAME lstm_16x2x32_seq_601_shards_3_allOutput_${ALL_OUTPUT}
                COMMAND lstm_layer
                        --input-size 16
                        --batch-size=2
                        --output-size 32
                        --tiles-per-ipu=20
                        --phase all
                        --sequence-size 601
                        --shards=3
                        --output-all-sequence=${ALL_OUTPUT}
                        VARIANTS "Hw")
        add_multitarget_test(
                NAME lstm_16x2x32_seq_601_shards_3_allOutput_${ALL_OUTPUT}_ignore_input_gradient
                COMMAND lstm_layer
                        --input-size 16
                        --batch-size=2
                        --output-size 32
                        --tiles-per-ipu=20
                        --phase all
                        --sequence-size 601
                        --shards=3
                        --output-all-sequence=${ALL_OUTPUT}
                        --ignore-input-gradient
                        VARIANTS "Hw")

        foreach(ATTENTION false true)
                # real-time-steps=true is only supported with --output-all-sequence=true
                add_multitarget_test(
                        NAME augru_16x2x32_seq_102_shards_3_attention_${ATTENTION}_real_time_steps_0_allOutput_${ALL_OUTPUT}
                        COMMAND gru_layer
                                --input-size 16
                                --batch-size=2
                                --with-attention ${ATTENTION}
                                --with-real-time-steps 0
                                --output-size 8
                                --tiles-per-ipu=20
                                --phase all
                                --sequence-size 102
                                --shards=3
                                --output-all-sequence=${ALL_OUTPUT}
                                VARIANTS "Hw")
                add_multitarget_test(
                        NAME augru_16x2x32_seq_102_shards_3_attention_${ATTENTION}_real_time_steps_0_allOutput_${ALL_OUTPUT}_ignore_input_gradient
                        COMMAND gru_layer
                                --input-size 16
                                --batch-size=2
                                --with-attention ${ATTENTION}
                                --with-real-time-steps 0
                                --output-size 8
                                --tiles-per-ipu=20
                                --phase all
                                --sequence-size 102
                                --shards=3
                                --output-all-sequence=${ALL_OUTPUT}
                                --ignore-input-gradient
                                VARIANTS "Hw")
        endforeach()
endforeach()

# Test variable time-steps for LSTM/GRU with variable weight update cadence
foreach(ALL_OUTPUT  0 1)
        foreach(PARTIALS_TYPE half float)
                foreach(STEPS_PER_WU 21 217)
                        foreach(IGNORE_FINAL_CELL_STATE 0 1)
                                add_multitarget_test(
                                        NAME lstm_16x2x32_seq_123_shards_3_allOutput_${ALL_OUTPUT}_partials_${PARTIALS_TYPE}_varTS_57_stepsPerWU_${STEPS_PER_WU}_ignoreFinalCellState_${IGNORE_FINAL_CELL_STATE}
                                        COMMAND lstm_layer
                                                --data-type=half
                                                --input-size 16
                                                --batch-size=2
                                                --output-size 32
                                                --tiles-per-ipu=20
                                                --phase all
                                                --sequence-size 123
                                                --shards=3
                                                --partials-type=${PARTIALS_TYPE}
                                                --accumulators-type=${PARTIALS_TYPE}
                                                --output-all-sequence=${ALL_OUTPUT}
                                                --variable-time-steps={57}
                                                --steps-per-wu=${STEPS_PER_WU}
                                                --ignore-final-cell-state=${IGNORE_FINAL_CELL_STATE}
                                                VARIANTS "Hw;${IPUMODEL_VARIANTS}")
                                add_multitarget_test(
                                        NAME lstm_16x2x32_seq_123_shards_3_allOutput_${ALL_OUTPUT}_partials_${PARTIALS_TYPE}_varTS_57_stepsPerWU_${STEPS_PER_WU}_ignoreFinalCellState_${IGNORE_FINAL_CELL_STATE}_ignore_input_gradient
                                        COMMAND lstm_layer
                                                --data-type=half
                                                --input-size 16
                                                --batch-size=2
                                                --output-size 32
                                                --tiles-per-ipu=20
                                                --phase all
                                                --sequence-size 123
                                                --shards=3
                                                --partials-type=${PARTIALS_TYPE}
                                                --accumulators-type=${PARTIALS_TYPE}
                                                --output-all-sequence=${ALL_OUTPUT}
                                                --variable-time-steps={57}
                                                --steps-per-wu=${STEPS_PER_WU}
                                                --ignore-final-cell-state=${IGNORE_FINAL_CELL_STATE}
                                                --ignore-input-gradient
                                                VARIANTS "Hw;${IPUMODEL_VARIANTS}")
                                add_multitarget_test(
                                        NAME lstm_16x3x32_seq_123_shards_3_allOutput_${ALL_OUTPUT}_partials_${PARTIALS_TYPE}_varTS_23_41_80_stepsPerWU_${STEPS_PER_WU}_ignoreFinalCellState_${IGNORE_FINAL_CELL_STATE}
                                        COMMAND lstm_layer
                                                --data-type=half
                                                --input-size 16
                                                --batch-size=3
                                                --output-size 32
                                                --tiles-per-ipu=20
                                                --phase all
                                                --sequence-size 123
                                                --shards=3
                                                --partials-type=${PARTIALS_TYPE}
                                                --accumulators-type=${PARTIALS_TYPE}
                                                --output-all-sequence=${ALL_OUTPUT}
                                                --variable-time-steps={23,41,80}
                                                --steps-per-wu=${STEPS_PER_WU}
                                                --ignore-final-cell-state=${IGNORE_FINAL_CELL_STATE}
                                                VARIANTS "Hw;${IPUMODEL_VARIANTS}")
                                add_multitarget_test(
                                        NAME lstm_16x3x32_seq_123_shards_3_allOutput_${ALL_OUTPUT}_partials_${PARTIALS_TYPE}_varTS_23_41_80_stepsPerWU_${STEPS_PER_WU}_ignoreFinalCellState_${IGNORE_FINAL_CELL_STATE}_ignore_input_gradient
                                        COMMAND lstm_layer
                                                --data-type=half
                                                --input-size 16
                                                --batch-size=3
                                                --output-size 32
                                                --tiles-per-ipu=20
                                                --phase all
                                                --sequence-size 123
                                                --shards=3
                                                --partials-type=${PARTIALS_TYPE}
                                                --accumulators-type=${PARTIALS_TYPE}
                                                --output-all-sequence=${ALL_OUTPUT}
                                                --variable-time-steps={23,41,80}
                                                --steps-per-wu=${STEPS_PER_WU}
                                                --ignore-final-cell-state=${IGNORE_FINAL_CELL_STATE}
                                                --ignore-input-gradient
                                                VARIANTS "Hw;${IPUMODEL_VARIANTS}")
                        endforeach()
                endforeach()
        endforeach()
endforeach()

# Test LSTM/GRU variable time-steps for larger sequence sizes
foreach(ALL_OUTPUT  0 1)
        add_multitarget_test(
                NAME lstm_16x2x32_seq_601_shards_3_allOutput_${ALL_OUTPUT}_varTS_253
                COMMAND lstm_layer
                        --data-type=float
                        --input-size 16
                        --batch-size=2
                        --output-size 32
                        --tiles-per-ipu=20
                        --phase all
                        --sequence-size 601
                        --shards=3
                        --output-all-sequence=${ALL_OUTPUT}
                        --variable-time-steps={253}
                        VARIANTS "Hw;${IPUMODEL_VARIANTS}")
        add_multitarget_test(
                NAME lstm_16x2x32_seq_601_shards_3_allOutput_${ALL_OUTPUT}_varTS_253_ignore_input_gradient
                COMMAND lstm_layer
                        --data-type=float
                        --input-size 16
                        --batch-size=2
                        --output-size 32
                        --tiles-per-ipu=20
                        --phase all
                        --sequence-size 601
                        --shards=3
                        --output-all-sequence=${ALL_OUTPUT}
                        --variable-time-steps={253}
                        --ignore-input-gradient
                        VARIANTS "Hw;${IPUMODEL_VARIANTS}")
        add_multitarget_test(
                NAME lstm_16x3x32_seq_601_shards_3_allOutput_${ALL_OUTPUT}_varTS_100_200_300
                COMMAND lstm_layer
                        --input-size 16
                        --batch-size=3
                        --output-size 32
                        --tiles-per-ipu=20
                        --phase all
                        --sequence-size 601
                        --shards=3
                        --output-all-sequence=${ALL_OUTPUT}
                        --variable-time-steps={100,200,300}
                        VARIANTS "Hw;${IPUMODEL_VARIANTS}")
        add_multitarget_test(
                NAME lstm_16x3x32_seq_601_shards_3_allOutput_${ALL_OUTPUT}_varTS_100_200_300_ignore_input_gradient
                COMMAND lstm_layer
                        --input-size 16
                        --batch-size=3
                        --output-size 32
                        --tiles-per-ipu=20
                        --phase all
                        --sequence-size 601
                        --shards=3
                        --output-all-sequence=${ALL_OUTPUT}
                        --variable-time-steps={100,200,300}
                        --ignore-input-gradient
                        VARIANTS "Hw;${IPUMODEL_VARIANTS}")

        foreach(ATTENTION false true)
                add_multitarget_test(
                        NAME augru_16x2x32_seq_99_shards_3_attention_${ATTENTION}_real_time_steps_0_allOutput_${ALL_OUTPUT}_varTS_31
                        COMMAND gru_layer
                                --input-size 16
                                --batch-size=2
                                --with-attention ${ATTENTION}
                                --with-real-time-steps 0
                                --output-size 8
                                --tiles-per-ipu=20
                                --phase all
                                --sequence-size 99
                                --variable-time-steps={31}
                                --shards=3
                                --output-all-sequence=${ALL_OUTPUT}
                                VARIANTS "Hw;${IPUMODEL_VARIANTS}")
                add_multitarget_test(
                        NAME augru_16x2x32_seq_99_shards_3_attention_${ATTENTION}_real_time_steps_0_allOutput_${ALL_OUTPUT}_varTS_31_ignore_input_gradient
                        COMMAND gru_layer
                                --input-size 16
                                --batch-size=2
                                --with-attention ${ATTENTION}
                                --with-real-time-steps 0
                                --output-size 8
                                --tiles-per-ipu=20
                                --phase all
                                --sequence-size 99
                                --variable-time-steps={31}
                                --shards=3
                                --output-all-sequence=${ALL_OUTPUT}
                                --ignore-input-gradient
                                VARIANTS "Hw;${IPUMODEL_VARIANTS}")
                add_multitarget_test(
                        NAME augru_16x3x32_seq_99_shards_3_attention_${ATTENTION}_real_time_steps_0_allOutput_${ALL_OUTPUT}_varTS_97_67_87
                        COMMAND gru_layer
                                --input-size 16
                                --batch-size=3
                                --with-attention ${ATTENTION}
                                --with-real-time-steps 0
                                --output-size 8
                                --tiles-per-ipu=20
                                --phase all
                                --sequence-size 99
                                --variable-time-steps={97,67,87}
                                --shards=3
                                --output-all-sequence=${ALL_OUTPUT}
                                VARIANTS "Hw;${IPUMODEL_VARIANTS}")
                add_multitarget_test(
                        NAME augru_16x3x32_seq_99_shards_3_attention_${ATTENTION}_real_time_steps_0_allOutput_${ALL_OUTPUT}_varTS_97_67_87_ignore_input_gradient
                        COMMAND gru_layer
                                --input-size 16
                                --batch-size=3
                                --with-attention ${ATTENTION}
                                --with-real-time-steps 0
                                --output-size 8
                                --tiles-per-ipu=20
                                --phase all
                                --sequence-size 99
                                --variable-time-steps={97,67,87}
                                --shards=3
                                --output-all-sequence=${ALL_OUTPUT}
                                --ignore-input-gradient
                                VARIANTS "Hw;${IPUMODEL_VARIANTS}")
        endforeach()
endforeach()

add_unit_test(NormStatisticsTest
              NormStatisticsTest.cpp
              VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS}
              LABELS T28462)

add_unit_test(CTCLossPlanTest CTCLossPlanTest.cpp VARIANTS ${IPUMODEL_VARIANTS})

# Sanity check that LogProbs API works as expected
add_multitarget_test(
      NAME CTC_Loss_LogProbs_santity
      COMMAND ctc_loss
              --logit-inputs=false
      VARIANTS ${IPUMODEL_VARIANTS})

# Verify cases where the individual entry is <= the maximum timesteps or
# maximum label length defined.
foreach(last_blank_separate 0 1)
  foreach(time 25 28 32)
    foreach(labels 2 3 6 8 12)
      add_multitarget_test(
        NAME CTC_Loss_1_Time_${time}_Labels_${labels}_${last_blank_separate}
        COMMAND ctc_loss
                --batch=1
                --max-time=32
                --max-label-length=12
                --time=${time}
                --label-length=${labels}
                --plan-constraints={\"parallel\":{\"time\":4,\"label\":4,\"lastBlankOnSeparateTile\":${last_blank_separate}}}
        VARIANTS ${IPUMODEL_VARIANTS})
    endforeach()
  endforeach()
endforeach()

foreach(last_blank_separate 0 1)
  foreach(time 12 15 16)
    foreach(labels 3 5)
      add_multitarget_test(
        NAME CTC_Loss_2_Time_${time}_Labels_${labels}_${last_blank_separate}
        COMMAND ctc_loss
                --batch=1
                --max-time=32
                --max-label-length=12
                --time=${time}
                --label-length=${labels}
                --plan-constraints={\"parallel\":{\"time\":4,\"label\":4,\"lastBlankOnSeparateTile\":${last_blank_separate}}}
        VARIANTS ${IPUMODEL_VARIANTS})
    endforeach()
  endforeach()
endforeach()

# Exersise batch > 1 and other type combinations to those above (which are
# float in, float out)
foreach(out_type half float)
    add_multitarget_test(
      NAME CTC_Loss_2_Batch_And_Type_Time_${out_type}
      COMMAND ctc_loss
              --batch=4
              --max-time=32
              --min-time=3
              --max-label-length=12
              --min-label-length=1
              --in-type=half
              --out-type=${out_type}
      VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

foreach(partials_type half float)
    add_multitarget_test(
      NAME CTC_Loss_in_half_partials_${partials_type}_out_half
      COMMAND ctc_loss
              --in-type=half
              --partials-type=${partials_type}
              --out-type=half
      VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

foreach(blank_class 0 4 10)
  add_multitarget_test(
    NAME CTC_Loss_Blank_${blank_class}
    COMMAND ctc_loss
            --num-classes=11
            --batch=1
            --max-time=32
            --min-time=32
            --max-label-length=12
            --min-label-length=12
            --blank-class=${blank_class}
    VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

# Specific test asking the grad-given-alpha codelet to do more work
add_multitarget_test(
  NAME CTC_Loss_One_Time_Partition_Codelet
  COMMAND ctc_loss
          --batch=1
          --max-time=16
          --time=16
          --max-label-length=4
          --label-length=4
          --plan-constraints={\"parallel\":{\"time\":1}}
  VARIANTS ${IPUMODEL_VARIANTS})

# Edge cases caused by specific partitioning in time or label dimension
foreach(time_partitions 1 2 3)
  foreach(label_partitions 1 2 3)
    add_multitarget_test(
      NAME CTC_Loss_Partitions_t${time_partitions}_l${label_partitions}
      COMMAND ctc_loss
              --batch=1
              --max-time=32
              --max-label-length=12
              --time=20
              --label-length=10
              --plan-constraints={\"parallel\":{\"time\":${time_partitions},\"label\":${label_partitions}}}
      VARIANTS ${IPUMODEL_VARIANTS})
  endforeach()
endforeach()

add_multitarget_test(
  NAME CTC_Loss_Codelet_Result_only
  COMMAND ctc_loss
          --test-reduced-codelet-result=1
  VARIANTS ${IPUMODEL_VARIANTS})

# CTC inference beam search.  Batch size provides multiple random inputs
foreach(beam_width 1 2 3 10 15)
  foreach(num_classes 3 5 10)
    foreach(blank_class 0 2)
      set(top_paths 2)
      if(beam_width EQUAL 1)
        set(top_paths 1)
      endif()
      add_multitarget_test(
        NAME ctc_beam_search_batch_beam${beam_width}_classes${num_classes}_blank${blank_class}
          COMMAND ctc_beam_search
            --batch=16
            --max-label-length=16
            --min-label-length=5
            --max-time=100
            --top-paths=${top_paths}
            --beamwidth=${beam_width}
            --num-classes=${num_classes}
            --blank-class=${blank_class}
          VARIANTS ${IPUMODEL_VARIANTS})
    endforeach()
  endforeach()
endforeach()

foreach(beam_width 1 2 3 5)
  foreach(seed RANGE 1 10)
    foreach(num_classes 3 5 10)
      set(top_paths 2)
      if(beam_width EQUAL 1)
        set(top_paths 1)
      endif()
      add_multitarget_test(
        NAME ctc_beam_search_seed${seed}_beam${beam_width}_classes${num_classes}
          COMMAND ctc_beam_search
            --batch=2
            --seed=${seed}
            --max-label-length=20
            --min-label-length=15
            --max-time=100
            --min-time=40
            --top-paths=${top_paths}
            --beamwidth=${beam_width}
            --num-classes=${num_classes}
          VARIANTS ${IPUMODEL_VARIANTS})
    endforeach()
  endforeach()
endforeach()

# Check case where empty beams exist for a few timesteps
add_multitarget_test(
  NAME ctc_beam_search_empty_beam_propagates
    COMMAND ctc_beam_search
      --batch=4
      --beamwidth=15
      --num-classes=3
      --max-time=6
      --disable-always-satisfiable-error
      --max-label-length=4
      --blank-class=2
    VARIANTS ${IPUMODEL_VARIANTS})

# Verify ctc beam search half operation with both float and half partials
foreach(partials_type half float)
  add_multitarget_test(
    NAME ctc_beam_search_inout_half_partials_${partials_type}
      COMMAND ctc_beam_search
        --partials-type=${partials_type}
        --in-type=half
        --out-type=half
      VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

# Test with non-zero blank class
foreach(seed RANGE 1 10)
  add_multitarget_test(
    NAME ctc_beam_search_non_zero_blank${seed}
      COMMAND ctc_beam_search
        --batch=8
        --seed=${seed}
        --beamwidth=5
        --num-classes=11
        --max-time=100
        --min-time=40
        --max-label-length=40
        --min-label-length=20
        --top-paths=5
        --blank-class=${seed}
      VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

# Verify ctc beam search with tile constraints
# Ensure that the number of classes results in tests where
# tiles < batch_size
foreach(batch_size 1 2 4 8)
  foreach(num_tiles 1 2 3 )
    add_multitarget_test(
      NAME ctc_beam_search_tile_constraint_batch${batch_size}_tiles${num_tiles}
        COMMAND ctc_beam_search
          --batch=${batch_size}
          --beamwidth=2
          --num-classes=16
          --tiles-per-ipu=${num_tiles}
        VARIANTS ${IPUMODEL_VARIANTS})
  endforeach()
endforeach()

# Test empty beams (Edge case for short inputs)
foreach(seed RANGE 1 5)
  add_multitarget_test(
    NAME ctc_beam_search_empty_beam${seed}
      COMMAND ctc_beam_search
        --batch=1
        --seed=${seed}
        --beamwidth=5
        --num-classes=3
        --max-time=1
        --max-label-length=1
        --top-paths=5
      VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

# Some coverage in simulation and hardware
# Batch > 1 and a range of min to max time tests early completion
foreach(seed RANGE 10 20)
  foreach(num_classes 3 5 10)
    foreach(beamwidth 4 5 50)
      add_multitarget_test(
        NAME ctc_beam_search_sim_${beamwidth}_${seed}_${num_classes}
          COMMAND ctc_beam_search
            --batch=2
            --seed=${seed}
            --num-classes=${num_classes}
            --beamwidth=${beamwidth}
            --max-time=8
            --min-time=4
            --max-label-length=2
            --top-paths=4
            --tiles-per-ipu=16
          VARIANTS "Hw;Sim2")
    endforeach()
  endforeach()
endforeach()

# Always a large beamwidth, and short time which exposes cases where
# invalid initial beam state and its treatment is important.
# Picking specific seeds where the test produced a problem
foreach(seed 12 15 36 60 64 65 77)
  foreach(beamwidth 20 50)
    foreach(classes 2 3 4 25)
      foreach(time 4 5 6)
        add_multitarget_test(
          NAME ctc_beam_search_large_beamwidth_${beamwidth}_${classes}_${time}_${seed}
            COMMAND ctc_beam_search
              --batch=1
              --seed=${seed}
              --beamwidth=${beamwidth}
              --num-classes=${classes}
              --max-time=${time}
              --time=${time}
              --max-label-length=4
              --disable-always-satisfiable-error
              --top-paths=2
            VARIANTS ${IPUMODEL_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Force 3 stage sort
add_multitarget_test(
  NAME ctc_beam_search_3_stage_sort
    COMMAND ctc_beam_search
      --num-classes=400
        --options={\"sortStageGroups\":\"16,4,1\"}
    VARIANTS ${IPUMODEL_VARIANTS})


# Sweep of number of tiles to check constrained sort cases
foreach(tiles RANGE 1 20)
  foreach(batch 1 2 5)
    foreach(classes 4 37 80)
      add_multitarget_test(
        NAME ctc_beam_search_constrained_sort${tiles}_${batch}_${classes}
          COMMAND ctc_beam_search
            --batch=${batch}
            --beamwidth=5
            --num-classes=${classes}
            --tiles-per-ipu=${tiles}
          VARIANTS ${IPUMODEL_VARIANTS})
    endforeach()
  endforeach()
endforeach()
