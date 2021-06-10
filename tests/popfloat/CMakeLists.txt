add_unit_test(DebugInfoTest DebugInfoTest.cpp)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_10_RZ
  COMMAND cast_to_gfloat
          --input-size=40001
          --tiles-per-ipu=16
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_10_bias30_no_inf_RA
  COMMAND cast_to_gfloat
          --input-size=40001
          --round-mode=RA
          --tiles-per-ipu=32
          --bias=30
          --enable-infs-and-nans=0
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_4_11_bias13_no_denorm_RN_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=4
          --man=11
          --bias=13
          --enable-denorms=0
          --round-mode=RN
          --tiles-per-ipu=16
          --storage-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_4_11_bias13_no_denorm_RN_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=4
          --man=11
          --bias=13
          --enable-denorms=0
          --round-mode=RN
          --tiles-per-ipu=32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_3_4_bias3_no_inf_RD_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=3
          --man=4
          --bias=3
          --enable-infs-and-nans=0
          --round-mode=RD
          --tiles-per-ipu=16
          --storage-type=FP16
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_3_4_bias3_no_inf_RD_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=3
          --man=4
          --bias=3
          --enable-infs-and-nans=0
          --round-mode=RD
          --tiles-per-ipu=16
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_3_4_bias3_no_inf_RD_quantise_fp32_calc_type
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=3
          --man=4
          --bias=3
          --enable-infs-and-nans=0
          --round-mode=RD
          --tiles-per-ipu=16
          --storage-type=FP16
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_3_4_bias3_no_inf_RD_gfloat_as_int_fp32_calc_type
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=3
          --man=4
          --bias=3
          --enable-infs-and-nans=0
          --round-mode=RD
          --tiles-per-ipu=32
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_6_9_bias62_no_inf_no_denorm_RA_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=6
          --man=9
          --bias=62
          --enable-infs-and-nans=0
          --enable-denorms=0
          --round-mode=RA
          --tiles-per-ipu=16
          --storage-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_6_9_bias62_no_inf_no_denorm_RA_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=6
          --man=9
          --bias=62
          --enable-infs-and-nans=0
          --enable-denorms=0
          --round-mode=RA
          --tiles-per-ipu=32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_2_bias29_no_inf_RU_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=5
          --man=2
          --bias=29
          --enable-infs-and-nans=0
          --round-mode=RD
          --storage-type=FP16
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_2_bias29_no_inf_RU_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=5
          --man=2
          --bias=29
          --enable-infs-and-nans=0
          --round-mode=RD
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_2_bias29_no_inf_RU_quantise_fp32_calc_type
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=5
          --man=2
          --bias=29
          --enable-infs-and-nans=0
          --round-mode=RD
          --storage-type=FP16
          --calc-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_2_bias29_no_inf_RU_gfloat_as_int_fp32_calc_type
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=5
          --man=2
          --bias=29
          --enable-infs-and-nans=0
          --round-mode=RD
          --calc-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_2_bias29_nanoo_RD_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=5
          --man=2
          --bias=29
          --enable-nanoo-mode=1
          --round-mode=RD
          --tiles-per-ipu=100
          --storage-type=FP16
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_2_bias29_nanoo_RD_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=5
          --man=2
          --bias=29
          --enable-nanoo-mode=1
          --round-mode=RD
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_2_bias29_nanoo_RD_quantise_fp32_calc_type
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=5
          --man=2
          --bias=29
          --enable-nanoo-mode=1
          --round-mode=RD
          --tiles-per-ipu=100
          --storage-type=FP16
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_5_2_bias29_nanoo_RD_gfloat_as_int_fp32_calc_type
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=5
          --man=2
          --bias=29
          --enable-nanoo-mode=1
          --round-mode=RD
          --calc-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_6_9_bias62_no_inf_RZ_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=6
          --man=9
          --bias=62
          --enable-infs-and-nans=0
          --round-mode=RZ
          --storage-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_6_9_bias62_no_inf_RZ_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=6
          --man=9
          --bias=62
          --enable-infs-and-nans=0
          --round-mode=RZ
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_1_14_bias0_noInf_RU_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=1
          --man=14
          --bias=0
          --enable-infs-and-nans=0
          --round-mode=RU
          --storage-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_1_14_bias0_noInf_RU_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=1
          --man=14
          --bias=0
          --enable-infs-and-nans=0
          --round-mode=RU
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_1_6_bias0_noInf_RA_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=1
          --man=6
          --bias=0
          --enable-infs-and-nans=0
          --round-mode=RA
          --storage-type=FP16
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_1_6_bias0_noInf_RA_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=1
          --man=6
          --bias=0
          --enable-infs-and-nans=0
          --round-mode=RA
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_1_6_bias0_noInf_RA_quantise_fp32_calc_type
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=1
          --man=6
          --bias=0
          --enable-infs-and-nans=0
          --round-mode=RA
          --storage-type=FP16
          --calc-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_1_6_bias0_noInf_RA_gfloat_as_int_fp32_calc_type
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=1
          --man=6
          --bias=0
          --enable-infs-and-nans=0
          --round-mode=RA
          --calc-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_0_15_bias-1_RA_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=0
          --man=15
          --bias=-1
          --enable-infs-and-nans=0
          --round-mode=RA
          --storage-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_0_15_bias-1_RA_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=0
          --man=15
          --bias=-1
          --enable-infs-and-nans=0
          --round-mode=RA
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_6_1_bias31_RU_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=6
          --man=1
          --bias=31
          --round-mode=RU
          --storage-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_6_1_bias31_RU_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=6
          --man=1
          --bias=31
          --round-mode=RU
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_7_0_bias63_RN_quantise
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=7
          --man=0
          --bias=63
          --round-mode=RN
          --storage-type=FP32
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_1_7_0_bias63_RN_gfloat_as_int
  COMMAND cast_to_gfloat
          --input-size=40001
          --exp=7
          --man=0
          --bias=63
          --round-mode=RN
          --tiles-per-ipu=100
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_Uniform
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=6
          --man=9
          --bias=31
          --sr-noise-density=Uniform
          --sr-noise-min=0.65
          --sr-noise-max=0.85
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Uniform
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=4
          --man=3
          --bias=7
          --sr-noise-density=Uniform
          --sr-noise-min=0.05
          --sr-noise-max=0.34
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Uniform_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=4
          --man=3
          --bias=7
          --sr-noise-density=Uniform
          --sr-noise-min=0.05
          --sr-noise-max=0.34
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_Normal
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=6
          --man=2
          --bias=31
          --sr-noise-density=Normal
          --sr-noise-min=0.0
          --sr-noise-max=0.42
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Normal
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=3
          --man=6
          --bias=1
          --sr-noise-density=Normal
          --sr-noise-min=0.0 --sr-noise-max=0.22
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Normal_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=3
          --man=6
          --bias=1
          --sr-noise-density=Normal
          --sr-noise-min=0.0
          --sr-noise-max=0.22
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_TruncatedNormal
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=5
          --man=10
          --bias=30
          --sr-noise-density=TruncatedNormal
          --sr-noise-min=-0.15
          --sr-noise-max=0.25
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_TruncatedNormal
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=5
          --man=2
          --bias=15
          --sr-noise-density=TruncatedNormal
          --sr-noise-min=-0.1
          --sr-noise-max=0.5
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_TruncatedNormal_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=5
          --man=2
          --bias=15
          --sr-noise-density=TruncatedNormal
          --sr-noise-min=-0.1
          --sr-noise-max=0.5
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_TruncatedLaplace
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=7
          --man=1
          --bias=15
          --sr-noise-density=TruncatedLaplace
          --sr-noise-min=-0.2
          --sr-noise-max=0.5
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_TruncatedLaplace
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=5
          --man=2
          --bias=15
          --sr-noise-density=TruncatedLaplace
          --sr-noise-min=-0.16
          --sr-noise-max=0.43
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_TruncatedLaplace_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=5
          --man=2
          --bias=15
          --sr-noise-density=TruncatedLaplace
          --sr-noise-min=-0.16
          --sr-noise-max=0.43
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_Laplace
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=4
          --man=13
          --bias=10
          --sr-noise-density=Laplace
          --sr-noise-min=0.0
          --sr-noise-max=0.5
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Laplace
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=2
          --man=4
          --bias=6
          --sr-noise-density=Laplace
          --sr-noise-min=0.0
          --sr-noise-max=0.3
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Laplace_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=2
          --man=4
          --bias=6
          --sr-noise-density=Laplace
          --sr-noise-min=-0.41
          --sr-noise-max=0.3
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_TruncatedLogistic
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=4
          --man=11
          --bias=7
          --sr-noise-density=TruncatedLogistic
          --sr-noise-min=-0.35
          --sr-noise-max=-0.15
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_Logistic
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=6
          --man=7
          --bias=17
          --sr-noise-density=Logistic
          --sr-noise-min=-0.35
          --sr-noise-max=0.15
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Logistic
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=0
          --man=7
          --bias=4
          --sr-noise-density=Logistic
          --sr-noise-min=-0.5
          --sr-noise-max=0.15
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Logistic_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=2
          --man=9
          --bias=4
          --sr-noise-density=Logistic
          --sr-noise-min=-0.5
          --sr-noise-max=0.15
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_TruncatedLogitNormal
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=3
          --man=16
          --bias=7
          --sr-noise-density=TruncatedLogitNormal
          --sr-noise-min=0.15
          --sr-noise-max=0.85
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_TruncatedLogitNormal
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=4
          --man=7
          --bias=3
          --sr-noise-density=TruncatedLogitNormal
          --sr-noise-min=0.0
          --sr-noise-max=0.9
          --input-value=2047
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_TruncatedLogitNormal_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=4
          --man=7
          --bias=4
          --sr-noise-density=TruncatedLogitNormal
          --sr-noise-min=0.0
          --sr-noise-max=0.9
          --input-value=2047
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_LogitNormal
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=2
          --man=5
          --bias=4
          --sr-noise-density=LogitNormal
          --sr-noise-min=0.0
          --sr-noise-max=0.5
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_LogitNormal
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=6
          --man=7
          --bias=31
          --sr-noise-density=LogitNormal
          --sr-noise-min=0.05
          --sr-noise-max=0.45
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_LogitNormal_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=2
          --man=5
          --bias=4
          --sr-noise-density=LogitNormal
          --sr-noise-min=0.0
          --sr-noise-max=0.5
          --calc-type=FP32
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_float_Bernoulli
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=8
          --man=7
          --bias=127
          --sr-noise-density=Bernoulli
          --sr-prob-truncate=0.45
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Bernoulli
  COMMAND cast_to_gfloat_sr
         --input-size=40001
         --tiles-per-ipu=32
         --exp=0
         --man=3
         --bias=4
         --sr-noise-density=Bernoulli
         --sr-prob-truncate=0.26
          VARIANTS Hw)

add_multitarget_test(
  NAME cast_to_gfloat_sr_half_Bernoulli_fp32_calc
  COMMAND cast_to_gfloat_sr
          --input-size=40001
          --tiles-per-ipu=32
          --exp=4
          --man=6
          --bias=4
          --sr-noise-density=Bernoulli
          --sr-prob-truncate=0.76
          --calc-type=FP32
          VARIANTS Hw)
