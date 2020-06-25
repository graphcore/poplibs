
function(add_popsolver_unit_test name)
  set(test_name "popsolver_${name}")
  add_executable(${test_name} ${ARGN})
  target_link_libraries(${test_name}
                        popsolver
                        Boost::unit_test_framework
                        Boost::timer
                        Boost::system)
  target_include_directories(${test_name}
    PRIVATE $<TARGET_PROPERTY:popsolver,INCLUDE_DIRECTORIES>)
  if(NOT Boost_UNIT_TEST_FRAMEWORK_LIBRARY MATCHES "\\.a$")
    target_compile_definitions(${test_name} PRIVATE -DBOOST_TEST_DYN_LINK)
  endif()
  add_test(NAME ${test_name}
    COMMAND ${test_name})
endfunction()

add_popsolver_unit_test(Div Div.cpp)
add_popsolver_unit_test(GenericAssignment GenericAssignment.cpp)
add_popsolver_unit_test(Less Less.cpp)
add_popsolver_unit_test(LessOrEqual LessOrEqual.cpp)
add_popsolver_unit_test(Max Max.cpp)
add_popsolver_unit_test(Min Min.cpp)
add_popsolver_unit_test(Mod Mod.cpp)
add_popsolver_unit_test(Product Product.cpp)
add_popsolver_unit_test(Simple Simple.cpp)
add_popsolver_unit_test(Sum Sum.cpp)
