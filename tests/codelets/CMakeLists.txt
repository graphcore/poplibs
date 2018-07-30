# NonLinearitySupervisor<dataType, nlType>
foreach(DATA_TYPE half float)
  foreach(NL_TYPE sigmoid relu tanh)
    foreach(NL_VERTEX Supervisor 2D)
      set(VARIANT_INPUT "NonLinearity${NL_VERTEX}.cpp.in")
      set(VARIANT_NAME "NonLinearity${NL_VERTEX}_${DATA_TYPE}_${NL_TYPE}")
      set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
      string(TOUPPER "${DATA_TYPE}" DATA_TYPE_UPPER)
      string(TOUPPER "${NL_TYPE}" NL_TYPE_UPPER)
      configure_file(${VARIANT_INPUT} ${VARIANT_SRC})
      add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC})
    endforeach()
  endforeach()
endforeach()

# Loss(LT_TYPE)Transform<dataType>
foreach(DATA_TYPE half float)
  foreach(LT_TYPE SumSquared Softmax)
    set(VARIANT_NAME "Loss${LT_TYPE}Transform_${DATA_TYPE}")
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
    string(TOUPPER "${DATA_TYPE}" DATA_TYPE_UPPER)
    string(TOUPPER "${LT_TYPE}_LOSS" LT_TYPE_UPPER)
    configure_file(LossTransform.cpp.in ${VARIANT_SRC})
    add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC})
  endforeach()
endforeach()

#Clamp<InType>
foreach(DATA_TYPE half float int)
   set(VARIANT_NAME "ClampTest_${DATA_TYPE}")
   set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
   string(TOUPPER "${DATA_TYPE}" DATA_TYPE_UPPER)
   configure_file(ClampTest.cpp.in ${VARIANT_SRC})
   add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC} VARIANTS "IpuModel;Sim:cpp;Sim")
endforeach()

foreach(OUTER_DIM RANGE 1 15)
  foreach(INNER_DIM RANGE 1 3)
    foreach(PARTIALS_TYPE HALF FLOAT)
      foreach(OUT_TYPE HALF FLOAT)
        set(VARIANT_NAME
            "ReduceAdd_${OUTER_DIM}_${INNER_DIM}_${PARTIALS_TYPE}_${OUT_TYPE}")
        set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
        configure_file(ReduceAdd.cpp ${VARIANT_SRC})
        add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC}
	                           VARIANTS "IpuModel;Sim:cpp;Sim")
        endforeach()
      endforeach()
  endforeach()
endforeach()

#Transpose2d<InType>
   add_popnn_unit_test(Transpose2dTest Transpose2dTest.cpp VARIANTS "IpuModel;Sim:cpp;Sim")

# MaxPooling<dataType> and ScaledSumPooling<dataType, poolType>
foreach(DATA_TYPE half float)
  foreach(POOL_TYPE avg sum max)
    set(VARIANT_NAME "Pooling_${DATA_TYPE}_${POOL_TYPE}")
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")

    # set up the variables used in the test.
    if("${POOL_TYPE}" STREQUAL "avg")
      set(VERTEX
        "popnn::ScaledSumPooling<${DATA_TYPE},popnn::PoolingType::AVG>")
      set(EXPECTED "pooling/AvgPoolingExpected.hpp")
    elseif("${POOL_TYPE}" STREQUAL "sum")
      set(VERTEX
        "popnn::ScaledSumPooling<${DATA_TYPE},popnn::PoolingType::SUM>")
      set(EXPECTED "pooling/SumPoolingExpected.hpp")
    elseif("${POOL_TYPE}" STREQUAL "max")
      set(VERTEX "popnn::MaxPooling<${DATA_TYPE}>")
      set(EXPECTED "pooling/MaxPoolingExpected.hpp")
    endif()

    string(TOUPPER "${DATA_TYPE}" TYPE)
    if("${DATA_TYPE}" STREQUAL "half")
      set(ATOL "1e-7")
    else()
      set(ATOL "1e-20")
    endif()

    # create the test.
    configure_file(Pooling.cpp.in ${VARIANT_SRC})
    configure_file(${EXPECTED} ${CMAKE_CURRENT_BINARY_DIR}/${EXPECTED} COPYONLY)
    add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC})
  endforeach()
endforeach()

add_popnn_unit_test(AddToChannel
                    AddToChannel.cpp
                    VARIANTS ${TimesOutOnSim})
add_popnn_unit_test(AddToChannel2D
                    AddToChannel2D.cpp
                    VARIANTS ${TimesOutOnSim})

# Select
foreach(DATA_TYPE int float half bool)
  set(VARIANT_NAME "Select_${DATA_TYPE}")
  set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
  string(TOUPPER "${DATA_TYPE}" TYPE)
  set(DATA_HEADER "select/${DATA_TYPE}.hpp")
  configure_file(Select.cpp.in ${VARIANT_SRC})
  configure_file(${DATA_HEADER} ${CMAKE_CURRENT_BINARY_DIR}/${DATA_HEADER} COPYONLY)
  add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC})
endforeach()
