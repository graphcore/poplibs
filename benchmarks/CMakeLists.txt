include(GNUInstallDirs)

find_package(Boost REQUIRED program_options)

function(add_benchmark name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} popnn)
endfunction()

add_benchmark(alexnet alexnet.cpp)

add_benchmark(resnet34 resnet34.cpp)

add_benchmark(resnet50 resnet50.cpp)

add_benchmark(single_conv_layer single_conv_layer.cpp)
target_include_directories(single_conv_layer PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(single_conv_layer
                      popnn_ref
                      ${Boost_PROGRAM_OPTIONS_LIBRARY})

add_custom_command(
  OUTPUT
    run_benchmarks.py
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/run_benchmarks.py run_benchmarks.py
  DEPENDS
    run_benchmarks.py
)

add_custom_command(
  OUTPUT
    benchmark_report.py
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_report.py benchmark_report.py
  DEPENDS
    benchmark_report.py
)
