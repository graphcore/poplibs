find_package(Boost REQUIRED unit_test_framework program_options)

function(add_popnn_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      ${Boost_INCLUDE_DIRS})
  target_link_libraries(${name} popnn poplar ${Boost_LIBRARIES})
  target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  add_test(NAME ${name}
    COMMAND ${name})
endfunction()

add_popnn_test(FullyConnectedTest FullyConnectedTest.cpp)
