find_package(Boost REQUIRED unit_test_framework timer system program_options)

function(add_popsolver_unit_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name}
                        popsolver
                        Boost::unit_test_framework
                        Boost::timer
                        Boost::system)
  if(NOT ${Boost_USE_STATIC_LIBS})
    target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  endif()
  add_test(NAME ${name}
    COMMAND ${name})
endfunction()

add_popsolver_unit_test(Simple Simple.cpp)
