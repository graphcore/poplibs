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
