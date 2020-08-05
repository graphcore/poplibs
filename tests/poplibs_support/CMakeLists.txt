add_unit_test(AlgorithmTest AlgorithmTest.cpp VARIANTS ${IPUMODEL_VARIANTS})
add_unit_test(MultiArrayTest MultiArrayTest.cpp VARIANTS NoTarget)
add_unit_test(PlanConstraintsTest PlanConstraintsTest.cpp VARIANTS NoTarget)

add_unit_test(LoggingTest
              LoggingTest.cpp VARIANTS NoTarget)
if(TARGET LoggingTest)
  set(logging_test NoTarget_default_LoggingTest)
  if (TEST ${logging_test})
    set(LoggingPrintoutExample "")
    list(APPEND LoggingPrintoutExample "PL:POPFLOAT  \\[I\\] Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPLIN    \\[I\\] Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPNN     \\[I\\] Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPOPS    \\[I\\] Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPRAND   \\[I\\] Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPSOLVER \\[I\\] Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPSPARSE \\[I\\] Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPUTIL   \\[I\\] Hello world")
    set(SelectiveLoggingPass "I'm printed")
    set(v "${LoggingPrintoutExample};${SelectiveLoggingPass}")
    list(JOIN v ".*" LoggingPassRegex)

    set(SelectiveLoggingFail "I'm not printed")

    set_tests_properties(${logging_test} PROPERTIES
      PASS_REGULAR_EXPRESSION "${LoggingPassRegex}")
    set_tests_properties(${logging_test} PROPERTIES
      FAIL_REGULAR_EXPRESSION "${SelectiveLoggingFail}")
  else()
    message(WARNING "Could not find logging test")
  endif()
endif()
