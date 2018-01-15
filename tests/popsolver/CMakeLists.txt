find_package(Boost REQUIRED unit_test_framework timer system program_options)

function(add_popsolver_unit_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name}
                        popsolver
                        Boost::unit_test_framework
                        Boost::timer
                        Boost::system)
  target_include_directories(${name}
    PRIVATE $<TARGET_PROPERTY:popsolver,INCLUDE_DIRECTORIES>)
  if(NOT ${Boost_USE_STATIC_LIBS})
    target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  endif()
  add_test(NAME ${name}
    COMMAND ${name})
endfunction()

add_popsolver_unit_test(Div Div.cpp)
add_popsolver_unit_test(Less Less.cpp)
add_popsolver_unit_test(LessOrEqual LessOrEqual.cpp)
add_popsolver_unit_test(Product Product.cpp)
add_popsolver_unit_test(Simple Simple.cpp)
add_popsolver_unit_test(Sum Sum.cpp)
