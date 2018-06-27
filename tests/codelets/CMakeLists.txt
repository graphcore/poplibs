# NonLinearitySupervisor<dataType, nlType>
foreach(DATA_TYPE half float)
  foreach(NL_TYPE sigmoid relu tanh)
    set(VARIANT_NAME "NonLinearitySupervisor_${DATA_TYPE}_${NL_TYPE}")
    set(VARIANT_SRC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT_NAME}.cpp")
    string(TOUPPER "${DATA_TYPE}" DATA_TYPE_UPPER)
    string(TOUPPER "NON_LINEARITY_${NL_TYPE}" NL_TYPE_UPPER)
    configure_file(NonLinearitySupervisor.cpp.in ${VARIANT_SRC})
    # TODO: Re-enable 'Sim:cpp' variants when T3574 is fixed
    add_popnn_unit_test(${VARIANT_NAME} ${VARIANT_SRC}
                        VARIANTS "IpuModel;Sim")
  endforeach()
endforeach()
