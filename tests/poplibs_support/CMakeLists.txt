add_unit_test(MultiArrayTest MultiArrayTest.cpp VARIANTS NoTarget)
add_unit_test(PlanConstraintsTest PlanConstraintsTest.cpp VARIANTS NoTarget)
add_unit_test(StridedRegionsTest StridedRegionsTest.cpp VARIANTS NoTarget)

add_unit_test(LoggingTest
              LoggingTest.cpp VARIANTS NoTarget)
if(TARGET LoggingTest)
  set(logging_test NoTarget_default_LoggingTest)
  if (TEST ${logging_test})
    set(LoggingPrintoutExample "")
    list(APPEND LoggingPrintoutExample "PL:POPFLOAT  [0-9]+\\.[0-9]+ I: Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPLIN    [0-9]+\\.[0-9]+ I: Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPNN     [0-9]+\\.[0-9]+ I: Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPOPS    [0-9]+\\.[0-9]+ I: Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPRAND   [0-9]+\\.[0-9]+ I: Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPSOLVER [0-9]+\\.[0-9]+ I: Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPSPARSE [0-9]+\\.[0-9]+ I: Hello world")
    list(APPEND LoggingPrintoutExample "PL:POPUTIL   [0-9]+\\.[0-9]+ I: Hello world")
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
