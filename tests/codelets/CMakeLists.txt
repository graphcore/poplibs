# Function to add a multi-target test executable for codelets
function(add_codelet_test_executable name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/tests
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poplin,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name}
    poprand popnn poputil poplar poplibs_test
    Boost::program_options ${CMAKE_THREAD_LIBS_INIT})
endfunction()

# NonLinearity{Supervisor,2D}<dataType, nlType>
foreach(NL_VERTEX Supervisor 2D)
  set(TEST_EXECUTABLE "NonLinearity${NL_VERTEX}")
  set(TEST_SRC "${TEST_EXECUTABLE}.cpp")
  add_codelet_test_executable(${TEST_EXECUTABLE} ${TEST_SRC})
  foreach(DATA_TYPE half float)
    foreach(NL_TYPE sigmoid relu tanh)
      set(VARIANT_NAME "NonLinearity${NL_VERTEX}_${DATA_TYPE}_${NL_TYPE}")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ${TEST_EXECUTABLE}
                             --data-type=${DATA_TYPE}
                             --nl-type=${NL_TYPE})
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
   add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC} )
endforeach()

add_codelet_test_executable(ReduceAdd ReduceAdd.cpp)
foreach(OUTER_DIM RANGE 1 15)
  foreach(INNER_DIM RANGE 1 3)
    foreach(PARTIALS_TYPE half float)
      foreach(OUT_TYPE half float)
        set(VARIANT_NAME
            "ReduceAdd_${OUTER_DIM}_${INNER_DIM}_${PARTIALS_TYPE}_${OUT_TYPE}")
        add_multitarget_test(NAME ${VARIANT_NAME}
                             COMMAND ReduceAdd
                               --partials-type=${PARTIALS_TYPE}
                               --out-type=${OUT_TYPE}
                               --outer-dim=${OUTER_DIM}
                               --inner-dim=${INNER_DIM})
        endforeach()
      endforeach()
  endforeach()
endforeach()

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


add_codelet_test_executable(Reduce Reduce.cpp)
foreach(OUTER_DIM RANGE 1 16)
  foreach(IN_TYPE half float)
    foreach(OUT_TYPE half float)
      set(VARIANT_NAME "Reduce_${IN_TYPE}_${OUT_TYPE}_${OUTER_DIM}")
      set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
      add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND Reduce
                            --in-type=${IN_TYPE}
                            --out-type=${OUT_TYPE}
                            --outer-dim=${OUTER_DIM})
    endforeach()
  endforeach()
endforeach()


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

add_codelet_test_executable(ReduceMaxClassGather ReduceMaxClassGather.cpp)
foreach(ACTIVATION_TYPE half float)
  foreach(SIZE RANGE 1 24)
    set(VARIANT_NAME "ReduceMaxClassGather_${ACTIVATION_TYPE}_${SIZE}/4")
    add_multitarget_test(NAME ${VARIANT_NAME}
                           COMMAND ReduceMaxClassGather
                           --activation-type=${ACTIVATION_TYPE}
                           --divisor=4
                           --size=${SIZE})
  endforeach()
endforeach()

add_codelet_test_executable(ReduceMaxClassSparse ReduceMaxClassSparse.cpp)
foreach(SIZE RANGE 1 4)
  set(VARIANT_NAME "ReduceMaxClassSparse_${SIZE}")
  add_multitarget_test(NAME ${VARIANT_NAME}
                         COMMAND ReduceMaxClassSparse
                         --size=${SIZE})
endforeach()

add_popnn_unit_test(AddToChannel
                    AddToChannel.cpp)
add_popnn_unit_test(AddToChannel2D
                    AddToChannel2D.cpp)
add_popnn_unit_test(CastTest
                    CastTest.cpp)
add_popnn_unit_test(ChannelMul
                    ChannelMul.cpp)
add_popnn_unit_test(ChannelMul2D
                    ChannelMul2D.cpp)
add_popnn_unit_test(OuterProductTest
                    OuterProductTest.cpp)
add_popnn_unit_test(ScaledAdd2D_fp
                    ScaledAdd2D_fp.cpp)
add_popnn_unit_test(ScaledAdd2D_integral
                    ScaledAdd2D_integral.cpp)
add_popnn_unit_test(ScaledAddSupervisor_fp
                    ScaledAddSupervisor_fp.cpp)
add_popnn_unit_test(ScaledAddSupervisor_integral
                    ScaledAddSupervisor_integral.cpp)
add_popnn_unit_test(Transpose2dTest
                    Transpose2dTest.cpp)
