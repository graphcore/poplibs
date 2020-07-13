add_unit_test(AllTrueTest AllTrueTest.cpp)

# Binary Operator tests - currently not used.
# add_multi_target_test_executable(BinaryOpTest BinaryOpTest.cpp)

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


add_multitarget_test(NAME BroadcastOptimiseTest_subtract_2
  COMMAND BroadcastOptimiseTest
  --data-type half
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

add_unit_test(CircBufTests CircBufTests.cpp)
add_unit_test(collective-control-code
              collective-control-code.cpp
              VARIANTS Hw
              LABELS multicard;CE_reproducers;Collectives)
add_unit_test(DynamicSliceCreation.cpp DynamicSliceCreation.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(DynamicSlicePlanningTest DynamicSlicePlanningTest.cpp VARIANTS Hw;Sim;IpuModel)
add_unit_test(DynamicSliceTest DynamicSliceTest.cpp
              SUITES SingleDim MultiDim LargeBuffer Update Misc MultiSlice
                     MultiUpdate MultiUpdateSingles MultiUpdateMultiples)
add_unit_test(DynamicSliceTestCpu DynamicSliceTest.cpp SUITES CpuChecks VARIANTS Cpu)
add_unit_test(ElementWiseUtilTest ElementWiseUtilTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(EncodingTest EncodingTest.cpp)
add_unit_test(ExprName ExprName.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(GatherSimpleTest GatherSimpleTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(GatherTest GatherTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(HostSliceTensorTest HostSliceTensorTest.cpp VARIANTS ${SIM_VARIANTS})

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
add_map_fusion_test(Ceil)
add_map_fusion_test(Clamp)
add_map_fusion_test(Cos)
add_map_fusion_test(Divide)
add_map_fusion_test(Equal)
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

add_unit_test(NaNTest NaNTest.cpp)
add_unit_test(PaddingTest PaddingTest.cpp)
add_unit_test(ReduceEdgeCases ReduceEdgeCases.cpp)

# Check reduction patterns.
add_test_executable (ReductionPatternsTest ReductionPatternsTest.cpp)
add_test(NAME ReductionPatternsTest COMMAND ReductionPatternsTest)

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


add_unit_test(ReplicatedAllToAll
  ReplicatedAllToAll.cpp
  VARIANTS Hw
  LABELS multicard;Collectives)

add_unit_test(ReplicatedReduceScatter
  ReplicatedReduceScatter.cpp
  VARIANTS Hw
  LABELS multicard;Collectives)

add_unit_test(ScaledAddEquivalenceTest ScaledAddEquivalenceTest.cpp)
add_unit_test(ScatterTest ScatterTest.cpp)
add_unit_test(ScatterUpdateTest ScatterUpdateTest.cpp)
add_unit_test(SelectScalarFromRows SelectScalarFromRowsTest.cpp)
add_unit_test(SortTest SortTest.cpp)
add_unit_test(StdArithmeticTests StdArithmeticTests.cpp)

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
add_std_operators_test(BroadcastSelectorSelectInt)
add_std_operators_test(BroadcastSelectorSelectFloat)
add_std_operators_test(BroadcastSelectorSelectInPlaceInt)
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
