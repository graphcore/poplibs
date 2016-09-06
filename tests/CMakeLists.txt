find_package(Boost REQUIRED unit_test_framework program_options)

function(add_popnn_unit_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      $<TARGET_PROPERTY:popnn,INCLUDE_DIRECTORIES>
      ${Boost_INCLUDE_DIRS})
  target_link_libraries(${name} popnn poplar ${Boost_LIBRARIES})
  target_compile_definitions(${name} PRIVATE -DBOOST_TEST_DYN_LINK)
  add_test(NAME ${name}
    COMMAND ${name})
endfunction()

# Unit tests
add_popnn_unit_test(DimShuffleTest DimShuffleTest.cpp)
add_popnn_unit_test(FullyConnectedTest FullyConnectedTest.cpp)

# Layer tests
add_test(NAME conv3x3_stride_2_fwd
         COMMAND single_conv_layer
                 --input-channels=128
                 --output-channels=256
                 --width=20
                 --height=28
                 --kernel-size=3
                 --non-linearity=relu
                 --data-type=half
                 --padding=2
                 --stride=2
                 --out-chans-per-group=16)
