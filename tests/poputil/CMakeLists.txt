add_unit_test(BroadcastToMatchTest BroadcastToMatchTest.cpp)
add_unit_test(CopyToIpu CopyToIpu.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(DuplicateTensor DuplicateTensor.cpp VARIANTS ${IPUMODEL_VARIANTS})
# GraphFunctionTest is variant-independent
add_unit_test(GraphFunctionTest GraphFunctionTest.cpp
              VARIANTS ${IPUMODEL_VARIANTS} ${SIM_VARIANTS})
add_unit_test(GraphReplication GraphReplication.cpp)
add_unit_test(LargeSplitRegionsTest LargeSplitRegionsTest.cpp)
add_unit_test(LoopTest_poputil_deprecated LoopTest.cpp)
add_unit_test(TileMappingTest TileMappingTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(VarStructureTest VarStructureTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
