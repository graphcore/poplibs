add_subdirectory(codelets)

add_unit_test(AllTrueTest AllTrueTest.cpp)

# Binary, Unary Operator tests
add_multi_target_test_executable(BinaryOpTest BinaryOpTest.cpp)
add_multi_target_test_executable(UnaryOpTest UnaryOpTest.cpp)

# Verify large input sizes for Binary/Unary Ops, but only where data actually fits
  add_multitarget_test(NAME UnaryOpLargeNonLinearity_float
    COMMAND UnaryOpTest
    --ope=relu
    --data=float
    --shape={40000}
    --tiles=1
    --in-place)

foreach(DATA_SIZE 30000 65536)
  add_multitarget_test(NAME UnaryOpLargeNonLinearity_half_${DATA_SIZE}
    COMMAND UnaryOpTest
    --ope=relu
    --data=half
    --shape={${DATA_SIZE}}
    --tiles=1
    --in-place)
endforeach()

foreach(DATA_TYPE short ushort ulonglong longlong)
  add_multitarget_test(NAME UnaryOpLargeBitwise_${DATA_TYPE}
    COMMAND UnaryOpTest
    --ope=bitwise_not
    --data=${DATA_TYPE}
    --shape={100000}
    --tiles=1
    --in-place)
endforeach()

foreach(DATA_TYPE short ushort)
  add_multitarget_test(NAME BinaryOpLargeBitwise_${DATA_TYPE}
    COMMAND BinaryOpTest
    --ope=bitwise_xor
    --data=${DATA_TYPE}
    --shape1={100000}
    --shape2={1}
    --tiles=1
    --in-place)
endforeach()


foreach(DATA_TYPE longlong ulonglong)
  add_multitarget_test(NAME BinaryOpMediumMul_${DATA_TYPE}
    COMMAND BinaryOpTest
    --ope=mul
    --data=${DATA_TYPE}
    --shape1={10000}
    --shape2={1}
    --tiles=1
    --in-place)
endforeach()


# Would overflow unless the correct divide by 6 method is used to divide work
foreach(DATA_TYPE half float)
  add_multitarget_test(NAME BinaryOpLargeArith_${DATA_TYPE}
    COMMAND BinaryOpTest
    --ope=add
    --data=${DATA_TYPE}
    --shape1={100000}
    --shape2={1}
    --tiles=1
    --in-place
    VARIANTS Sim2)
endforeach()

# This won't fit in memory in a sim, but OK on IpuModel.
# The size will force the selection of 2D vertices
foreach(DATA_TYPE half float)
  add_multitarget_test(NAME BinaryOpVeryLargeArith_${DATA_TYPE}
    COMMAND BinaryOpTest
    --ope=add
    --data=${DATA_TYPE}
    --shape1={1000000}
    --shape2={1}
    --tiles=1
    --in-place
    VARIANTS ${IPUMODEL_VARIANTS})
endforeach()

add_multitarget_test(NAME BinaryOpConstInput
  COMMAND BinaryOpTest
  --ope=add
  --data=float
  --shape1={1}
  --shape2={2}
  --map1=c:0
  --tiles=1)

# Broadcast Optimise tests
add_multi_target_test_executable(BroadcastOptimiseTest BroadcastOptimiseTest.cpp)

add_multitarget_test(NAME BroadcastOptimiseTest_multiply_inplace
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,16,16}
  --operation MULTIPLY
  --dim=2
  --dim-shuffle={0,1,2}
  --tiles 16)

add_multitarget_test(NAME BroadcastOptimiseTest_multiply_inplace_2
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,16,16}
  --operation MULTIPLY
  --dim=0
  --dim-shuffle={0,1,2}
  --tiles 16)

add_multitarget_test(NAME BroadcastOptimiseTest_multiply_inplace_3
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,2,64}
  --operation MULTIPLY
  --dim=2
  --dim-shuffle={0,1,2}
  --tiles 4)

add_multitarget_test(NAME BroadcastOptimiseTest_subtract_inplace
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,2,16,16}
  --operation SUBTRACT
  --dim=3
  --dim-shuffle={3,2,1,0}
  --tiles 5)

add_multitarget_test(NAME BroadcastOptimiseTest_add_inplace
  COMMAND BroadcastOptimiseTest
  --data-type float
  --dims={2,16,16}
  --operation ADD
  --dim=1
  --dim-shuffle={0,1,2}
  --tiles 1)

add_multitarget_test(NAME BroadcastOptimiseTest_add_inplace_2
  COMMAND BroadcastOptimiseTest
  --data-type float
  --dims={2,8,64}
  --operation ADD
  --dim=1
  --dim-shuffle={0,1,2}
  --tiles 5)

add_multitarget_test(NAME BroadcastOptimiseTest_add_inplace_3
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={4,16,128}
  --operation ADD
  --dim=2
  --dim-shuffle={0,1,2}
  --tiles 3)

add_multitarget_test(NAME BroadcastOptimiseTest_multiply
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,8,4,16}
  --operation MULTIPLY
  --dim=0
  --in-place 0
  --dim-shuffle={0,1,2,3}
  --tiles 16)

add_multitarget_test(NAME BroadcastOptimiseTest_subtract
  COMMAND BroadcastOptimiseTest
  --data-type float
  --dims={2,16,16}
  --operation SUBTRACT
  --dim=2
  --in-place 0
  --dim-shuffle={2,1,0}
  --tiles 16)

# Test float32 alias for float
add_multitarget_test(NAME BroadcastOptimiseTest_subtract_1
  COMMAND BroadcastOptimiseTest
  --data-type float32
  --dims={2,16,16}
  --operation SUBTRACT
  --dim=2
  --in-place 0
  --dim-shuffle={2,1,0}
  --tiles 16)

add_multitarget_test(NAME BroadcastOptimiseTest_subtract_2
  COMMAND BroadcastOptimiseTest
  --data-type half
  --dims={2,16,20}
  --operation SUBTRACT
  --dim=1
  --dim-shuffle={0,1,2}
  --tiles 1)

# Test float16 alias for half
add_multitarget_test(NAME BroadcastOptimiseTest_subtract_3
  COMMAND BroadcastOptimiseTest
  --data-type float16
  --dims={2,16,20}
  --operation SUBTRACT
  --dim=1
  --dim-shuffle={0,1,2}
  --tiles 1)

add_multitarget_test(NAME BroadcastOptimiseTest_add
  COMMAND BroadcastOptimiseTest
  --data-type float
  --dims={2,16,16}
  --operation ADD
  --dim=2
  --in-place 0
  --dim-shuffle={2,0,1}
  --tiles 16)

# Broadcast Patterns Generator
add_multi_target_test_executable(BroadcastGeneratePatterns BroadcastGeneratePatterns.cpp)

add_multitarget_test(NAME BroadcastGeneratePatterns_multiply_inplace
  COMMAND BroadcastGeneratePatterns
  --data-type half
  --pattern={3,2,5}
  --tiles 5
  --regions-per-tile 3
  --operation MULTIPLY
  --in-place 1)

add_unit_test(BertSlicing BertSlicing.cpp VARIANTS Cpu;Hw;${IPUMODEL_VARIANTS})
add_unit_test(CircBufTests CircBufTests.cpp)
add_unit_test(RegroupTest RegroupTest.cpp)
add_unit_test(DynamicSliceCreation.cpp DynamicSliceCreation.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(DynamicSliceTest DynamicSliceTest.cpp
              SUITES SingleDim MultiDim LargeBuffer Misc MultiSlice CheckIndexValidation
              VARIANTS Hw;${IPUMODEL_VARIANTS};${SIM_VARIANTS})

add_unit_test(MultiUpdateTest DynamicSliceTest.cpp
              SUITES Update MultiUpdate  MultiUpdateSingles MultiUpdateMultiples)

add_unit_test(MultiUpdatePlanTest DynamicSliceTest.cpp
              SUITES MultiUpdatePlanLookupSplit MultiUpdatePlan VARIANTS IpuModel2)

add_unit_test(MultiUpdateTestMax DynamicSliceTest.cpp
              SUITES MultiUpdateMaxHalf MultiUpdateMaxFloat VARIANTS Hw;Sim2;IpuModel2)

add_unit_test(MultiUpdateAddHalfScaleTest DynamicSliceTest.cpp
              SUITES MultiUpdateAddHalfScale VARIANTS Hw;Sim2;IpuModel2)

add_unit_test(SeqSliceTest SeqSliceTest.cpp)
add_unit_test(DynamicSliceTestCpu DynamicSliceTest.cpp SUITES CpuChecks VARIANTS Cpu)
add_unit_test(ElementWiseUtilTest ElementWiseUtilTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(EncodingTest EncodingTest.cpp)
add_unit_test(ExprName ExprName.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(GatherSimpleTest GatherSimpleTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(GatherTest GatherTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(HostSliceTensorTest HostSliceTensorTest.cpp VARIANTS ${SIM_VARIANTS})
add_unit_test(MapExprOptimisations MapExprOptimisations.cpp)



# MapFusionTests
macro(add_map_fusion_test test)
  add_multitarget_test(NAME MapFusion_${test} COMMAND MapFusionTest --test ${test})
endmacro()

add_multi_target_test_executable(MapFusionTest MapFusionTest.cpp)

add_map_fusion_test(Abs)
add_map_fusion_test(Add)
add_map_fusion_test(And)
add_map_fusion_test(Atan2)
add_map_fusion_test(BitwiseAnd)
add_map_fusion_test(BitwiseNot)
add_map_fusion_test(BitwiseOr)
add_map_fusion_test(BitwiseXnor)
add_map_fusion_test(BitwiseXor)
add_map_fusion_test(Cbrt)
add_map_fusion_test(Ceil)
add_map_fusion_test(Clamp)
add_map_fusion_test(Cos)
add_map_fusion_test(Divide)
add_map_fusion_test(Equal)
add_map_fusion_test(Erf)
add_map_fusion_test(GeluErf)
add_map_fusion_test(Exp)
add_map_fusion_test(Expm1)
add_map_fusion_test(Floor)
add_map_fusion_test(Gt)
add_map_fusion_test(Gte)
add_map_fusion_test(Inv)
add_map_fusion_test(IsFinite)
add_map_fusion_test(Log)
add_map_fusion_test(Log1p)
add_map_fusion_test(Lt)
add_map_fusion_test(Lte)
add_map_fusion_test(Max)
add_map_fusion_test(Min)
add_map_fusion_test(Mul)
add_map_fusion_test(Neg)
add_map_fusion_test(Not)
add_map_fusion_test(NotEqual)
add_map_fusion_test(Or)
add_map_fusion_test(Pow)
add_map_fusion_test(Rem)
add_map_fusion_test(Round)
add_map_fusion_test(Rsqrt)
add_map_fusion_test(Select)
add_map_fusion_test(Shl)
add_map_fusion_test(Shr)
add_map_fusion_test(ShrSE)
add_map_fusion_test(Sigmoid)
add_map_fusion_test(Signum)
add_map_fusion_test(Sin)
add_map_fusion_test(Sqrt)
add_map_fusion_test(Square)
add_map_fusion_test(Sub)
add_map_fusion_test(Tanh)

add_map_fusion_test(Fusion)
add_map_fusion_test(MissingPlaceholder)

# MapScalarTests
add_unit_test(MapExprScalar MapExprScalar.cpp)

add_unit_test(PaddingTest PaddingTest.cpp)
add_unit_test(ReduceEdgeCases ReduceEdgeCases.cpp)

# Check reduction patterns.
add_test_executable (ReductionPatternsTest ReductionPatternsTest.cpp)
add_test(NAME ReductionPatternsTest COMMAND ReductionPatternsTest)


add_multi_target_test_executable(MapExprRemAndDivOptimTest MapExprRemAndDivideOptimisations.cpp)

foreach(dtype unsigned int)
  foreach(inPlace true false)
    foreach(op REMAINDER DIVIDE)
      foreach(power2 true false)
        foreach(rhsSigned true false)
          add_multitarget_test(
          NAME MapExprRemAndDivOptimTest_${dtype}_inplace_${inPlace}_op_${op}_po2_${power2}_signed_${rhsSigned}
            COMMAND MapExprRemAndDivOptimTest
              --rhs-is-signed=${rhsSigned}
              --data-type=${dtype}
              --test=${op}
              --power-of-two=${power2}
              --in-place=${inPlace})
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

add_multi_target_test_executable(MapExprMultiVertexTest MapExprMultiVertex.cpp)
foreach(dtype unsigned half float)
  foreach(length 1 2 3 4 6 12 24 25 26 27)
    foreach(slices 1 2)
      add_multitarget_test(
            NAME MapExprMultiVertexTest_${dtype}_${length}_${slices}
            COMMAND MapExprMultiVertexTest
            --data-type ${dtype}
            --length ${length}
            --slices ${slices}
            --in-place true
            VARIANTS ${IPUMODEL_VARIANTS};Sim2)
    endforeach()
  endforeach()
endforeach()

foreach(dtype unsigned half float)
  foreach(inPlace true false)
    foreach(slices 1 2)
      add_multitarget_test(
            NAME MapExprMultiVertexTest_${dtype}_${inPlace}_${slices}
            COMMAND MapExprMultiVertexTest
            --data-type ${dtype}
            --length 25
            --slices ${slices}
            --in-place ${inPlace}
            VARIANTS ${IPUMODEL_VARIANTS};Sim2)
    endforeach()
  endforeach()
endforeach()

foreach(inPlace true false)
  foreach(dtype unsigned half float)
    add_multitarget_test(
          NAME MapExprMultiVertexTest_non_vectorizable_${dtype}_${inPlace}
          COMMAND MapExprMultiVertexTest
          --data-type ${dtype}
          --length 16
          --slices 1
          --vectorizable-expression=false
          --in-place ${inPlace}
          VARIANTS ${IPUMODEL_VARIANTS};Sim2)
  endforeach()
endforeach()

add_multi_target_test_executable(ReductionTests ReductionTests.cpp)
add_multitarget_test(NAME Reduce_100x100_float_float_noupdate
  COMMAND ReductionTests
    --dims={100,100}
    --partials-type=float
    --out-type=float
    --k=1.0
    --update=false
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_10x200_half_half
  COMMAND ReductionTests
    --dims={10,200}
    --partials-type=half
    --out-type=half
    --k=2.0
    --update=false
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_scale_half_half
  COMMAND ReductionTests
    --dims={31,201}
    --partials-type=half
    --out-type=half
    --k=3.0
    --update=false
    --scale=true
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_scale_float_half
  COMMAND ReductionTests
    --dims={31,201}
    --partials-type=float
    --out-type=half
    --k=-1.5
    --update=false
    --scale=true
    --test=Add)

add_multitarget_test(NAME Reduce_1x201_scale_float_half
  COMMAND ReductionTests
    --dims={1,201}
    --partials-type=float
    --out-type=half
    --k=-1.5
    --update=false
    --scale=true
    --test=Add)

add_multitarget_test(NAME Reduce_1x201_scale_half_half
  COMMAND ReductionTests
    --dims={1,201}
    --partials-type=half
    --out-type=half
    --k=-1.5
    --update=false
    --scale=true
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_update_float_float
  COMMAND ReductionTests
    --dims={31,101}
    --partials-type=float
    --out-type=float
    --k=-1.5
    --update=true
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_update_half_half
  COMMAND ReductionTests
    --dims={31,101}
    --partials-type=half
    --out-type=half
    --k=2.0
    --update=true
    --scale=false
    --test=Add)

add_multitarget_test(NAME Reduce_31x201_update_float_half
  COMMAND ReductionTests
    --dims={31,101}
    --partials-type=float
    --out-type=half
    --k=-1.5
    --update=true
    --scale=false
    --test=Add)

add_multitarget_test(NAME ReduceMany_31x201_noupdate_float_half
    COMMAND ReductionTests
      --many 3
      --dims={31,101}
      --partials-type=float
      --out-type=half
      --k=-1.5
      --update=false
      --scale=false
      --test=Add)

add_multitarget_test(NAME ReduceMany_31x201_update_float_half
    COMMAND ReductionTests
      --many 3
      --dims={31,101}
      --partials-type=float
      --out-type=half
      --k=-1.5
      --update=true
      --scale=false
      --test=Add)

add_multitarget_test(NAME Reduce_Add_float
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=float
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_Add_half
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=half
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_Add_int
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=int
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_SquareAdd_float
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=float
    --operation=SQUARE_ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_SquareAdd_half
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=half
    --operation=SQUARE_ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_SquareAdd_int
  COMMAND ReductionTests
    --dims={10,20,30}
    --red-vect={0}
    --out-type=int
    --operation=SQUARE_ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_Mul_float
  COMMAND ReductionTests
    --dims={33,22,11}
    --red-vect={0}
    --out-type=float
    --operation=MUL
    --test=Ops)

add_multitarget_test(NAME Reduce_Mul_half
  COMMAND ReductionTests
    --dims={33,22,11}
    --red-vect={0}
    --out-type=half
    --operation=MUL
    --test=Ops)

add_multitarget_test(NAME Reduce_Mul_int
  COMMAND ReductionTests
    --dims={33,22,11}
    --red-vect={0}
    --out-type=int
    --operation=MUL
    --test=Ops)

add_multitarget_test(NAME Reduce_Max_float
  COMMAND ReductionTests
    --dims={20,30,40}
    --red-vect={0,1}
    --out-type=float
    --operation=MAX
    --test=Ops)

add_multitarget_test(NAME Reduce_Max_half
  COMMAND ReductionTests
    --dims={20,30,40}
    --red-vect={0,1}
    --out-type=half
    --operation=MAX
    --test=Ops)

add_multitarget_test(NAME Reduce_Max_int
  COMMAND ReductionTests
    --dims={20,30,40}
    --red-vect={0,1}
    --out-type=int
    --operation=MAX
    --test=Ops)

add_multitarget_test(NAME Reduce_Min_float
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=float
    --operation=MIN
    --test=Ops)

add_multitarget_test(NAME Reduce_Min_half
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=half
    --operation=MIN
    --test=Ops)

add_multitarget_test(NAME Reduce_Min_int
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=int
    --operation=MIN
    --test=Ops)

add_multitarget_test(NAME Reduce_And_bool
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=bool
    --operation=LOGICAL_AND
    --test=Ops)

add_multitarget_test(NAME Reduce_Or_bool
  COMMAND ReductionTests
    --dims={20,30,10}
    --red-vect={0,1}
    --out-type=bool
    --operation=LOGICAL_OR
    --test=Ops)

add_multitarget_test(NAME Reduce_All_ADD_float
  COMMAND ReductionTests
    --dims={20,30,11}
    --red-vect={1,0,2}
    --out-type=float
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_None_ADD_float
  COMMAND ReductionTests
    --dims={20,30,11}
    --red-vect={}
    --out-type=float
    --operation=ADD
    --test=Ops)

add_multitarget_test(NAME Reduce_Skip_ADD_float
  COMMAND ReductionTests
    --dims={1,1,11}
    --red-vect={0,1}
    --out-type=float
    --operation=ADD
    --test=Ops)

# Reduction tests
foreach(type half float)
  foreach(update true false)
    foreach(scale 0.0 1.2)
      foreach(result_width 1 16)
        add_multitarget_test(
          NAME reduce_op_reduceLogAdd_${result_width}_${type}_${update}_${scale}
          COMMAND $<TARGET_FILE:reduce_op>
            "--shape=32,4,${result_width}"
            "--dims=0,1"
            --type=${type}
            --scale ${scale}
            --update ${update}
            --operation LOG_ADD
            --tiles-per-ipu=1)
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(type half float)
  foreach(update true false)
    foreach(scale 1.0 1.2)
      foreach(op ADD SQUARE_ADD)
        add_multitarget_test(
          NAME reduce_op_no_reduce_${op}_${type}_${update}_${scale}
          COMMAND $<TARGET_FILE:reduce_op>
            "--shape=32,4,1"
            "--dims=2"
            --type=${type}
            --scale ${scale}
            --update ${update}
            --operation ${op}
            --tiles-per-ipu=1)
      endforeach()
    endforeach()
  endforeach()
endforeach()


foreach(type half float)
  foreach(update true false)
    foreach(scale 1.0 1.2)
      add_multitarget_test(
        NAME reduce_op_${partialsType}_${type}_${update}_${scale}
        COMMAND $<TARGET_FILE:reduce_op>
          "--shape=32,4,4"
          "--dims=0"
          --type=${type}
          --scale ${scale}
          --update ${update}
          --operation ADD
          --tiles-per-ipu=1)
    endforeach()
  endforeach()
endforeach()

foreach(operation ADD SQUARE_ADD)
  add_multitarget_test(
    NAME reduce_op_empty_dims_${operation}
    COMMAND $<TARGET_FILE:reduce_op>
      "--shape=32,4,4"
      --type=half
      --scale=1.2
      --update false
      --operation ${operation}
      --tiles-per-ipu=1)
endforeach()

foreach(type half float)
  foreach(operation ADD SQUARE_ADD MUL MIN MAX)
    foreach(update true false)
      foreach(computesetapi true false)
        add_multitarget_test(
          NAME reduce_op_empty_reduce_${operation}_${type}_${update}_${computesetapi}
          COMMAND $<TARGET_FILE:reduce_op>
            "--shape=0,64"
            "--dims=0"
            --type=${type}
            --update=${update}
            --computesetapi=${computesetapi}
            --operation=${operation}
            --tiles-per-ipu=1)
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(operation LOGICAL_OR LOGICAL_AND)
  foreach(update true false)
    foreach(computesetapi true false)
      add_multitarget_test(
        NAME reduce_op_empty_reduce_${operation}_${update}_${computesetapi}
        COMMAND $<TARGET_FILE:reduce_op>
          "--shape=0,64"
          "--dims=0"
          --type=bool
          --update=${update}
          --computesetapi=${computesetapi}
          --operation=${operation}
          --tiles-per-ipu=1)
      endforeach()
    endforeach()
endforeach()

foreach(type half float)
  foreach(update true false)
    foreach(scale 1.0 1.2)
      foreach(tiles 1 2)
        add_multitarget_test(
          NAME reduce_op_shuffle_${partialsType}_${type}_${update}_${scale}_${tiles}
          COMMAND $<TARGET_FILE:reduce_op>
            "--shape=128,8,2"
            "--initial-shape=256,4,2"
            "--shuffle=0,2,1"
            "--dims=0"
            --type=${type}
            --scale ${scale}
            --update ${update}
            --operation ADD
            --tiles-per-ipu=${tiles})
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Shuffle to produce memory layouts to make many strided reduce vertices
# with an outer stride
foreach(type half float)
  foreach(update true false)
    foreach(scale 1.0 1.2)
      foreach(tiles 1 2 3 4 8)
        add_multitarget_test(
          NAME reduce_op_shuffle_outer_${partialsType}_${type}_${update}_${scale}_${tiles}
          COMMAND $<TARGET_FILE:reduce_op>
            "--shape=4,32,32,16"
            "--initial-shape=16,32,4,32"
            "--shuffle=0,1,3,2"
            "--dims=0,1,2"
            --type=${type}
            --scale ${scale}
            --update ${update}
            --operation ADD
            --tiles-per-ipu=${tiles})
      endforeach()
    endforeach()
  endforeach()
endforeach()

# Create a large tensor per tile
foreach(tiles 1 2 3 4 8)
  add_multitarget_test(
    NAME reduce_op_shuffle_large_size_per_tile_${tiles}
    COMMAND $<TARGET_FILE:reduce_op>
      "--shape=4,320,320,16"
      "--initial-shape=16,320,4,320"
      "--shuffle=1,0,3,2"
      "--dims=0,2,3"
      --type=half
      --scale 2.0
      --update true
      --operation ADD
      --tiles-per-ipu=${tiles})
endforeach()


# Check that various cases of reduction with shuffled columns are correct.
# The tests should produce the named sequence of stages
foreach(withOutput true false)
    add_multitarget_test(
      NAME reduce_op_shuffle_inter_output_${withOutput}
      COMMAND $<TARGET_FILE:reduce_op>
      "--type=half"
      "--operation=ADD"
      "--withoutput=${withOutput}"
      "--shape=1024,4,32"
      "--dims=0"
      "--initial-shape=1024,4,32"
      "--shuffle=0,2,1"
       VARIANTS "${IPUMODEL_VARIANTS}")

    add_multitarget_test(
      NAME reduce_op_shuffle_in_inter_output_${withOutput}
      COMMAND $<TARGET_FILE:reduce_op>
      "--type=half"
      "--operation=ADD"
      "--withoutput=${withOutput}"
      "--shape=200,2,2"
      "--dims=0"
      "--initial-shape=200,2,2"
      "--shuffle=0,2,1"
       VARIANTS "${IPUMODEL_VARIANTS}")

   add_multitarget_test(
      NAME reduce_op_shuffle_output_${withOutput}
      COMMAND $<TARGET_FILE:reduce_op>
      "--type=half"
      "--operation=ADD"
      "--withoutput=${withOutput}"
      "--shape=2,4,32"
      "--dims=0"
      "--initial-shape=2,4,32"
      "--shuffle=0,2,1"
       VARIANTS "${IPUMODEL_VARIANTS}")
endforeach()

# Random reduce_op tests with fixed seed.
set(NUM_REDUCE_RANDOM_TESTS 150)
foreach(n RANGE 1 ${NUM_REDUCE_RANDOM_TESTS})
  add_multitarget_test(NAME reduce_random_${n}
           COMMAND $<TARGET_FILE:reduce_op> --seed ${n})
endforeach()

# Random reduce_op tests with fixed seed.
set(NUM_REDUCE_RANDOM_TESTS 15)
foreach(n RANGE 1 ${NUM_REDUCE_RANDOM_TESTS})
  add_multitarget_test(NAME multi_ipu_reduce_random_${n}
           COMMAND $<TARGET_FILE:reduce_op> --seed ${n} --ipus=2
           VARIANTS "Hw;${IPUMODEL_VARIANTS}")
endforeach()

add_unit_test(ScaledAddTest ScaledAddTest.cpp)
add_unit_test(ScatterTest ScatterTest.cpp)
add_unit_test(ScatterUpdateTest ScatterUpdateTest.cpp)
add_unit_test(SelectScalarFromRows SelectScalarFromRowsTest.cpp)
add_unit_test(SortTest SortTest.cpp)
add_unit_test(StdArithmeticTests StdArithmeticTests.cpp
              VARIANTS  "${IPUMODEL_VARIANTS};Sim21")

# StdOperatorsTests
macro(add_std_operators_test test)
  add_multitarget_test(
    NAME StdOperators_${test}
    COMMAND StdOperatorsTest --test ${test})
endmacro()

add_multi_target_test_executable(StdOperatorsTest StdOperatorsTest.cpp)
add_std_operators_test(AbsFloat)
add_std_operators_test(AbsInt)
add_std_operators_test(AddFloat)
add_std_operators_test(Asin)
add_std_operators_test(Atan2Float)
add_std_operators_test(AddInt)
add_std_operators_test(BitwiseAndInt)
add_std_operators_test(BitwiseOrInt)
add_std_operators_test(BitwiseNotInt)
add_std_operators_test(BitwiseXorInt)
add_std_operators_test(BitwiseXnorInt)
add_std_operators_test(CubeRoot)
add_std_operators_test(Ceil)
add_std_operators_test(Cos)
add_std_operators_test(CountLeadingZeros)
add_std_operators_test(DivideInt)
add_std_operators_test(DivideHalf)
add_std_operators_test(DivideFloat)
add_std_operators_test(EqualFloat)
add_std_operators_test(GreaterThanBool)
add_std_operators_test(GreaterThanEqualBool)
add_std_operators_test(LessThanBool)
add_std_operators_test(LessThanEqualBool)
add_std_operators_test(Erf)
add_std_operators_test(GeluErf)
add_std_operators_test(Exponent)
add_std_operators_test(ExponentMinus1)
add_std_operators_test(Floor)
add_std_operators_test(GreaterThanFloat)
add_std_operators_test(GreaterThanInt)
add_std_operators_test(GreaterThanEqualFloat)
add_std_operators_test(LessThanFloat)
add_std_operators_test(LessThanEqualFloat)
add_std_operators_test(Logarithm)
add_std_operators_test(Logarithm1Plus)
add_std_operators_test(LogicalAnd)
add_std_operators_test(LogicalNot)
add_std_operators_test(LogicalOr)
add_std_operators_test(MaxFloat)
add_std_operators_test(MaxInt)
add_std_operators_test(MinFloat)
add_std_operators_test(MinInt)
add_std_operators_test(Multiply)
add_std_operators_test(NotEqualFloat)
add_std_operators_test(NotEqualBool)
add_std_operators_test(NegateFloat)
add_std_operators_test(NegateInt)
add_std_operators_test(Popcount)
add_std_operators_test(Power)
add_std_operators_test(RemainderFloat)
add_std_operators_test(RemainderInt)
add_std_operators_test(ShiftLeftInt)
add_std_operators_test(ShiftRightInt)
add_std_operators_test(ShiftRightSignExtendInt)
add_std_operators_test(SignumFloat)
add_std_operators_test(SignumInt)
add_std_operators_test(Sin)
add_std_operators_test(Tan)
add_std_operators_test(Tanh)
add_std_operators_test(Square)
add_std_operators_test(SquareRoot)
add_std_operators_test(SubtractFloat)
add_std_operators_test(SubtractHalf)
add_std_operators_test(SubtractInt)
add_std_operators_test(RoundFloat)
add_std_operators_test(SelectFloat)
add_std_operators_test(SelectFloatLHSConst)
add_std_operators_test(SelectFloatRHSConst)
add_std_operators_test(SelectFloatLHSAndRHSConst)
add_std_operators_test(SelectHalfLHSAndRHSConst)
add_std_operators_test(SelectInt)
add_std_operators_test(SelectUInt)
add_std_operators_test(SelectInPlaceInt)
add_std_operators_test(SelectInPlaceUInt)
add_std_operators_test(BroadcastSelectorSelectInt)
add_std_operators_test(BroadcastSelectorSelectUInt)
add_std_operators_test(BroadcastSelectorSelectFloat)
add_std_operators_test(BroadcastSelectorSelectInPlaceInt)
add_std_operators_test(BroadcastSelectorSelectInPlaceUInt)
add_std_operators_test(BroadcastSelectorSelectInPlaceFloat)
add_std_operators_test(ClampFloat)
add_std_operators_test(ClampFloatMinConst)
add_std_operators_test(ClampFloatMaxConst)
add_std_operators_test(ClampInt)
add_std_operators_test(ClampInPlaceFloat)
add_std_operators_test(BroadcastClampInt)
add_std_operators_test(BroadcastClampInPlaceInt)
add_std_operators_test(BroadcastClampFloat)
add_std_operators_test(BroadcastClampInPlaceFloat)
add_std_operators_test(BroadcastClampSingleElementSrcFloat)
add_std_operators_test(BinaryOutputMapChoice)
add_std_operators_test(TrinaryOutputMapChoice)
add_std_operators_test(AllTrueBad)
add_std_operators_test(AllTrue)
add_std_operators_test(IsFinite)
add_std_operators_test(Map)
add_std_operators_test(MapCast)
add_std_operators_test(MapCastInPlace)
add_std_operators_test(MapCastIntToFloat)
add_std_operators_test(MapMultiTensor)
add_std_operators_test(MapInPlace)
add_std_operators_test(MapInPlaceBroadcast)
add_std_operators_test(MapInferType)
add_std_operators_test(MapInferTypeCast)
add_std_operators_test(MapInferTypeEqual)
add_std_operators_test(MapInferTypeNot)
add_std_operators_test(AddInPlace)
add_std_operators_test(MapAllScalar)
add_std_operators_test(MapSomeScalar)
add_std_operators_test(BinaryConcat)
add_std_operators_test(UnaryConcat)
add_std_operators_test(MultiplyFloatInPlaceConstScalarTest)
add_std_operators_test(AddHalfConstScalarTest)

add_unit_test(UpdateScalarInRows UpdateScalarInRowsTest.cpp)

# Embedding layer tests
add_multitarget_test(
   NAME embedding_layer_half_pad_grain1
   COMMAND embedding_layer
           --shape {5,4}
           --num-indices 3
           --grain-size 1
           --use-embedding-plan=0
           --data-type half
           --tiles-per-ipu 4)


add_multitarget_test(
   NAME embedding_layer_half_pad_grain3
   COMMAND embedding_layer
           --shape {5,4}
           --num-indices 3
           --grain-size 3
           --use-embedding-plan=0
           --data-type half
           --tiles-per-ipu 4)


add_multitarget_test(
   NAME embedding_layer_half_sorted_indices
   COMMAND embedding_layer
     --shape {128,8}
     --num-indices 32
     --use-embedding-plan=1
     --data-type half
     --indices-are-sorted=1
     --tiles-per-ipu 4)

add_multitarget_test(
   NAME embedding_layer_half_sorted_indices_float_partial
   COMMAND embedding_layer
     --shape {128,8}
     --num-indices 32
     --use-embedding-plan=1
     --data-type half
     --partial-type float
     --indices-are-sorted=1
     --tiles-per-ipu 4)

add_multitarget_test(
   NAME embedding_layer_half_unsorted_indices_float_partial_large_lookup
   COMMAND embedding_layer
     --shape {128,2}
     --num-indices 128
     --use-embedding-plan=1
     --data-type half
     --partial-type float
     --indices-are-sorted=0
     --tiles-per-ipu 4)

add_multitarget_test(
   NAME embedding_layer_float_sorted_indices
   COMMAND embedding_layer
     --shape {32,8}
     --num-indices 32
     --use-embedding-plan=1
     --data-type float
     --indices-are-sorted=1
     --tiles-per-ipu 4)

add_multitarget_test(
   NAME embedding_layer_half_single_index
   COMMAND embedding_layer
     --shape {32,32}
     --num-indices 1
     --use-embedding-plan=1
     --data-type half
     --tiles-per-ipu 4)


add_multitarget_test(
         NAME embedding_layer_float_10x20
         COMMAND embedding_layer
                 --data-type=float
                 --shape={10,20}
                 --num-indices=50
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME embedding_layer_half_10x20
         COMMAND embedding_layer
                 --data-type=half
                 --shape={10,20}
                 --num-indices=50
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME embedding_layer_float_10x20_2_index_sets
         COMMAND embedding_layer
                 --data-type=float
                 --shape={10,20}
                 --num-indices={100,10}
                 --tiles-per-ipu=16)

add_multitarget_test(
         NAME embedding_layer_half_10x20_2_index_sets
         COMMAND embedding_layer
                 --data-type=half
                 --shape={10,20}
                 --num-indices={5,20}
                 --tiles-per-ipu=16)

# These tests are quite slow to execute on the simulator
add_multitarget_test(
         NAME embedding_layer_big_embedding
         COMMAND embedding_layer
                 --data-type=half
                 --shape={100000,200}
                 --num-indices=1440
                 --use-embedding-plan=1
                 --ignore-data
         VARIANTS ${IPUMODEL_VARIANTS};Hw)

add_multitarget_test(
         NAME embedding_layer_many_lookups
         COMMAND embedding_layer
                 --data-type=half
                 --shape={1000,200}
                 --num-indices=18000
                 --use-embedding-plan=1
                 --ignore-data
         VARIANTS ${IPUMODEL_VARIANTS};Hw)

# This test is too slow and fails on MACS
add_multitarget_test(
         NAME embedding_layer_vmany_lookups
         COMMAND embedding_layer
                 --data-type=half
                 --shape={1000,200}
                 --num-indices=40001
                 --use-embedding-plan=1
                 --ignore-data
         VARIANTS ${IPUMODEL_VARIANTS};Hw)

# This will also split the unsliced dim
add_multitarget_test(
         NAME embedding_layer_quick_split_sliced_dim
         COMMAND embedding_layer
                 --data-type=half
                 --shape={5000,100}
                 --num-indices=1440
                 --use-embedding-plan=1
                 --ignore-data
                 --tiles-per-ipu=30
                 --plan-constraints={\"slicedDimSplit\":\"6\"})

# This will also split the unsliced dim
add_multitarget_test(
         NAME embedding_layer_quick_split_lookup
         COMMAND embedding_layer
                 --data-type=half
                 --shape={100,200}
                 --num-indices=450
                 --use-embedding-plan=1
                 --ignore-data
                 --tiles-per-ipu=30
                 --plan-constraints={\"lookupSplit\":\"3\"})

add_multitarget_test(
         NAME embedding_layer_quick_unsliced_split_lookup
         COMMAND embedding_layer
                 --data-type=half
                 --shape={100,200}
                 --num-indices=450
                 --use-embedding-plan=1
                 --ignore-data
                 --tiles-per-ipu=30
                 --plan-constraints={\"unslicedDimSplit\":\"10\"})

add_multitarget_test(
         NAME embedding_layer_T24704
         COMMAND embedding_layer
                 --shape {61,6}
                 --num-indices 2010
                 --data-type float
                 --plan-constraints={\"unslicedDimSplit\":\"3\",\"slicedDimSplit\":\"8\",\"lookupSplit\":\"21\"}
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME embedding_layer_T25957
         COMMAND embedding_layer
                 --shape {12,12}
                 --num-indices 1361
                 --data-type float
                 --plan-constraints={\"unslicedDimSplit\":\"6\",\"slicedDimSplit\":\"6\",\"lookupSplit\":\"19\"}
         VARIANTS ${TimesOutOnSim})

add_multitarget_test(
         NAME embedding_layer_T30781
         COMMAND embedding_layer
                 --num-indices {36}
                 --shape {180,1}
                 --data-type=half
                 --tiles-per-ipu 30)

# Histogram
add_multi_target_test_executable(HistogramTest HistogramTest.cpp)

add_multitarget_test(NAME HistogramTest_largeOnTile
  COMMAND HistogramTest
  --type float
  --inner-dim-size=50000
  --limits-size=8
  --tiles-per-ipu=1)

foreach(DATA_TYPE half float)
  add_multitarget_test(NAME Histogram_by_data_2dinput_${DATA_TYPE}
                       COMMAND HistogramTest
                       --inner-dim-size=6400
                       --limits-size=4
                       --type=${DATA_TYPE}
                       --two-d=true
                       --absolute=false
                       --tiles-per-ipu=4)
endforeach()

foreach(useFloat true false)
  foreach(update true false)
    foreach(withOutput true false)
      add_multitarget_test(NAME HistogramTest_${withOutput}_${update}_${useFloat}
        COMMAND HistogramTest
        --type float
        --inner-dim-size=200
        --limits-size=8
        --with-output=${withOutput}
        --update=${update}
        --use-float-arithmetic=${useFloat})
    endforeach()
  endforeach()
endforeach()

# Note that useFloat works here despite counting 20,000,000 elements -
# with 7 limits = 8 histogram entries and random data there are few enough
# values counted in each to be represented in float as an exact integer
foreach(useFloat true false)
  foreach(update true false)
    foreach(withOutput true false)
      add_multitarget_test(NAME HistogramTest_large_${withOutput}_${update}_${useFloat}
        COMMAND HistogramTest
        --type half
        --inner-dim-size=20000000
        --limits-size=7
        --with-output=${withOutput}
        --update=${update}
        --use-float-arithmetic=${useFloat}
        --tiles-per-ipu=1024
        VARIANTS ${IPUMODEL_VARIANTS})
    endforeach()
  endforeach()
endforeach()


add_multi_target_test_executable(NaNAndInfTest NaNTest.cpp)


# FULL tensor tests
foreach(dType half float)
  foreach(hasInf true false)
    foreach(hasNaN true false)
      foreach(checkNaNOnly true false)
        foreach(use2D true false)
          foreach(tiles 1 4)
            add_multitarget_test(NAME HasNaNTest_full_${dType}_inf${hasInf}_nan${hasNaN}_2D${use2D}_checkNaN${checkNaNOnly}_${tiles}
              COMMAND NaNAndInfTest
              --add-inf ${hasInf}
              --add-nan ${hasNaN}
              --data-type ${dType}
              --full-size true
              --two-d ${use2D}
              --check-nan-only ${checkNaNOnly}
              --num-tiles ${tiles})
          endforeach()
        endforeach()

        foreach (len 1 2 3 4 5 6 7)
            add_multitarget_test(NAME HasNaNTest_rem_${dType}_inf${hasInf}_nan${hasNaN}_checkNaN${checkNaNOnly}_len${len}
              COMMAND NaNAndInfTest
              --add-inf ${hasInf}
              --add-nan ${hasNaN}
              --data-type ${dType}
              --full-size false
              --size ${len}
              --two-d false
              --check-nan-only ${checkNaNOnly}
              --num-tiles 1)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

# NormaliseImage tests
add_multi_target_test_executable(NormaliseImageTest NormaliseImageTest.cpp)
foreach(dType uchar half float)
  foreach(b 1 2 4)
    foreach(fsize 4 224)
      add_multitarget_test(NAME NormaliseImageTest_${dType}_b${b}_size${fsize}
        COMMAND NormaliseImageTest
          --data-type ${dType}
          --batch-size ${b}
          --field-size ${fsize})
    endforeach()
  endforeach()
  # Test with an odd number of elements
  add_multitarget_test(NAME NormaliseImageTest_${dType}_b1_size3
                       COMMAND NormaliseImageTest --data-type ${dType} --batch-size 1 --field-size 3)
endforeach()

foreach(DATA_TYPE half float unsigned int)
  add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_b0_n1000_k100_edge_case
    COMMAND topk
      --n 1000
      --k 100
      --b 0
      --data-type ${DATA_TYPE}
      --index-type uint
      --api=popops
      --tiles-per-ipu=4
    VARIANTS ${IPUMODEL_VARIANTS})
  foreach(BATCH_SIZE 1 3)
    add_multitarget_test(NAME topk_popnn_${DATA_TYPE}_unsigned_b${BATCH_SIZE}_n1000_k1000_sorted
      COMMAND topk
        --n 1000
        --k 1000
        --batch-size ${BATCH_SIZE}
        --data-type ${DATA_TYPE}
        --index-type uint
        --sort-order=descending
        --api=popnn
        --tiles-per-ipu=4)

    add_multitarget_test(NAME topk_popnn_${DATA_TYPE}_unsigned_b${BATCH_SIZE}_n1000_k100_sorted
      COMMAND topk
        --n 1000
        --k 100
        --batch-size ${BATCH_SIZE}
        --data-type ${DATA_TYPE}
        --index-type uint
        --sort-order=descending
        --api=popnn
        --tiles-per-ipu=4)

    add_multitarget_test(NAME topk_popnn_${DATA_TYPE}_unsigned_b${BATCH_SIZE}_n1000_k1
      COMMAND topk
        --n 1000
        --k 100
        --batch-size ${BATCH_SIZE}
        --data-type ${DATA_TYPE}
        --index-type uint
        --api=popnn
        --tiles-per-ipu=4
      VARIANTS ${IPUMODEL_VARIANTS})

    add_multitarget_test(NAME topk_popnn_${DATA_TYPE}_unsigned_b${BATCH_SIZE}_n1000_k1000_unsorted
      COMMAND topk
        --n 1000
        --k 1000
        --batch-size ${BATCH_SIZE}
        --data-type ${DATA_TYPE}
        --index-type uint
        --sort-order=none
        --api=popnn
        --tiles-per-ipu=4
      VARIANTS ${IPUMODEL_VARIANTS})

    add_multitarget_test(NAME topk_popnn_${DATA_TYPE}_unsigned_b${BATCH_SIZE}_n1000_k100_unsorted
      COMMAND topk
        --n 1000
        --k 100
        --batch-size ${BATCH_SIZE}
        --data-type ${DATA_TYPE}
        --index-type uint
        --sort-order=none
        --api=popnn
        --tiles-per-ipu=4
      VARIANTS ${IPUMODEL_VARIANTS})
  endforeach()

  foreach(INDICES 0 1)
    foreach(BATCH_SIZE 1 3)
      foreach(SORT_ORDER ascending descending)

        add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1_k1_${SORT_ORDER}_edge_case
          COMMAND topk
            --n 1
            --k 1
            --batch-size ${BATCH_SIZE}
            --data-type ${DATA_TYPE}
            --index-type uint
            --return-indices ${INDICES}
            --api=popops
            --sort-order=${SORT_ORDER}
            --tiles-per-ipu=4
          VARIANTS Cpu;${IPUMODEL_VARIANTS})

        add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1000_k1000_${SORT_ORDER}
          COMMAND topk
            --n 1000
            --k 1000
            --batch-size ${BATCH_SIZE}
            --data-type ${DATA_TYPE}
            --index-type uint
            --return-indices ${INDICES}
            --sort-order=${SORT_ORDER}
            --api=popops
            --tiles-per-ipu=4
          VARIANTS Cpu;${IPUMODEL_VARIANTS})

        add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1234_k1234_${SORT_ORDER}
          COMMAND topk
            --n 1234
            --batch-size ${BATCH_SIZE}
            --data-type ${DATA_TYPE}
            --index-type uint
            --return-indices ${INDICES}
            --api=popops
            --sort-order=${SORT_ORDER}
            --tiles-per-ipu=4)

        foreach(LARGEST 0 1)
          add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1000_k100_largest${LARGEST}_${SORT_ORDER}
            COMMAND topk
              --n 1000
              --k 100
              --batch-size ${BATCH_SIZE}
              --data-type ${DATA_TYPE}
              --index-type uint
              --return-indices ${INDICES}
              --largest ${LARGEST}
              --sort-order=${SORT_ORDER}
              --api=popops
              --tiles-per-ipu=4
            VARIANTS Cpu;${IPUMODEL_VARIANTS})

          add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1234_k123_largest${LARGEST}_${SORT_ORDER}
            COMMAND topk
              --n 1234
              --k 123
              --batch-size ${BATCH_SIZE}
              --data-type ${DATA_TYPE}
              --index-type uint
              --return-indices ${INDICES}
              --api=popops
              --largest ${LARGEST}
              --sort-order=${SORT_ORDER}
              --tiles-per-ipu=4)
        endforeach()
      endforeach()

      add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1000_k0_edge_case
        COMMAND topk
          --n 1000
          --k 0
          --batch-size ${BATCH_SIZE}
          --data-type ${DATA_TYPE}
          --index-type uint
          --return-indices ${INDICES}
          --api=popops
          --tiles-per-ipu=4
        VARIANTS ${IPUMODEL_VARIANTS})

      add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1000_k1000_unsorted
        COMMAND topk
          --n 1000
          --k 1000
          --batch-size ${BATCH_SIZE}
          --data-type ${DATA_TYPE}
          --index-type uint
          --return-indices ${INDICES}
          --sort-order=none
          --api=popops
          --tiles-per-ipu=4
        VARIANTS ${IPUMODEL_VARIANTS})

      add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1234_k1234_unsorted
        COMMAND topk
          --n 1234
          --batch-size ${BATCH_SIZE}
          --data-type ${DATA_TYPE}
          --index-type uint
          --return-indices ${INDICES}
          --api=popops
          --sort-order=none
          --tiles-per-ipu=4
        VARIANTS ${IPUMODEL_VARIANTS})

      foreach(LARGEST 0 1)
        add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1000_k100_largest${LARGEST}_unsorted
          COMMAND topk
            --n 1000
            --k 100
            --batch-size ${BATCH_SIZE}
            --data-type ${DATA_TYPE}
            --index-type uint
            --return-indices ${INDICES}
            --largest ${LARGEST}
            --sort-order=none
            --api=popops
            --tiles-per-ipu=4
          VARIANTS ${IPUMODEL_VARIANTS})

        add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1000_k1_largest${LARGEST}
          COMMAND topk
            --n 1000
            --k 100
            --batch-size ${BATCH_SIZE}
            --data-type ${DATA_TYPE}
            --index-type uint
            --return-indices ${INDICES}
            --largest ${LARGEST}
            --api=popops
            --tiles-per-ipu=4
          VARIANTS ${IPUMODEL_VARIANTS})

        add_multitarget_test(NAME topk_popops_${DATA_TYPE}_unsigned_indices${INDICES}_b${BATCH_SIZE}_n1234_k123_largest${LARGEST}_unsorted
          COMMAND topk
            --n 1234
            --k 123
            --batch-size ${BATCH_SIZE}
            --data-type ${DATA_TYPE}
            --index-type uint
            --return-indices ${INDICES}
            --api=popops
            --largest ${LARGEST}
            --sort-order=none
            --tiles-per-ipu=4
          VARIANTS ${IPUMODEL_VARIANTS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

add_unit_test(LoopTest LoopTest.cpp)
