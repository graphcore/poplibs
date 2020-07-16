add_multitarget_test(
         NAME random_gen_uniform_half
         COMMAND random_generator
                 --rand-test=Uniform
                 --min-val=-2.0
                 --max-val=0.0
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --half-data-type=true
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_int
         COMMAND random_generator
                 --rand-test=UniformInt
                 --min-val=-20.0
                 --max-val=2.0
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=2
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_int_max_range
         COMMAND random_generator
                 --rand-test=UniformInt
                 --min-val=-2147483648
                 --max-val=2147483647
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --in-size=40001
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_float
         COMMAND random_generator
                 --rand-test=Uniform
                 --min-val=-1.0
                 --max-val=1.0
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_float_1_0
         COMMAND random_generator
                 --rand-test=Uniform
                 --min-val=1e-9
                 --max-val=1.0
                 --percent-error=5.0
                 --seed=4538342
                 --seed-modifier=1243547
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_half
         COMMAND random_generator
                 --rand-test=Bernoulli
                 --half-data-type=true
                 --prob=0.75
                 --percent-error=5.0
                 --seed=6513234
                 --seed-modifier=24543
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_float
         COMMAND random_generator
                 --rand-test=Bernoulli
                 --prob=0.25
                 --percent-error=5.0
                 --seed=986427
                 --seed-modifier=1234
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_int
         COMMAND random_generator
                 --rand-test=BernoulliInt
                 --prob=0.5
                 --percent-error=5.0
                 --seed=9885343
                 --seed-modifier=8643232
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_float_prob0
         COMMAND random_generator
                 --rand-test=Bernoulli
                 --prob=0.0
                 --percent-error=5.0
                 --seed=9786553
                 --seed-modifier=6765354
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_bernoulli_int_prob1
         COMMAND random_generator
                 --rand-test=BernoulliInt
                 --prob=1.0
                 --percent-error=5.0
                 --seed=876854
                 --seed-modifier=89766465
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_normal_half
         COMMAND random_generator
                 --rand-test=Normal
                 --half-data-type=true
                 --mean=2.5
                 --std-dev=2.5
                 --percent-error=5.0
                 --seed=827437
                 --seed-modifier=7665654
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_normal_float
         COMMAND random_generator
                 --rand-test=Normal
                 --mean=-0.5
                 --std-dev=2.5
                 --percent-error=5.0
                 --seed=1452764
                 --seed-modifier=7861245
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_truncated_normal_half
         COMMAND random_generator
                 --rand-test=TruncatedNormal
                 --half-data-type=true
                 --mean=1.0
                 --std-dev=1.0
                 --alpha=2.0
                 --percent-error=5.0
                 --seed=1387532
                 --seed-modifier=985436
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_truncated_normal_float
         COMMAND random_generator
                 --rand-test=TruncatedNormal
                 --mean=-1.0
                 --std-dev=1.0
                 --alpha=2.0
                 --percent-error=5.0
                 --seed=8956342
                 --seed-modifier=249065
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_dropout_float
         COMMAND random_generator
                 --rand-test=Dropout
                 --prob=0.3
                 --percent-error=2.0
                 --seed=9887532
                 --seed-modifier=575329
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=2
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_dropout_half
         COMMAND random_generator
                 --rand-test=Dropout
                 --half-data-type=true
                 --prob=0.25
                 --percent-error=2.0
                 --seed=9077511
                 --seed-modifier=709815
                 --in-size=20001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)
add_multitarget_test(
         NAME random_gen_dropout_half_rem2
         COMMAND random_generator
                 --rand-test=Dropout
                 --half-data-type=true
                 --prob=0.25
                 --percent-error=2.0
                 --seed=9077511
                 --seed-modifier=709815
                 --in-size=20002
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)
add_multitarget_test(
         NAME random_gen_dropout_half_rem3
         COMMAND random_generator
                 --rand-test=Dropout
                 --half-data-type=true
                 --prob=0.25
                 --percent-error=2.0
                 --seed=9077511
                 --seed-modifier=709815
                 --in-size=20003
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_gen_uniform_float_no_seed
         COMMAND random_generator
                 --rand-test=Uniform
                 --min-val=-1.0
                 --max-val=1.0
                 --percent-error=5.0
                 --in-size=40001
                 --fp-checking=true
                 --tiles-per-ipu=1
                 VARIANTS ${IPUMODEL_VARIANTS};${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_set_seeds
         COMMAND random_generator
                 --rand-test=SetSeeds
                 --tiles-per-ipu=16
                 VARIANTS ${SIM_VARIANTS};Hw)

add_multitarget_test(
         NAME random_set_hw_seeds
         COMMAND random_generator
                 --rand-test=SetHwSeeds
                 --tiles-per-ipu=16
                 VARIANTS ${SIM_VARIANTS};Hw)
