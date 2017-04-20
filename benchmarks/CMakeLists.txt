include(GNUInstallDirs)

find_package(Boost REQUIRED program_options)

function(add_benchmark name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} enigma)
endfunction()

add_benchmark(alexnet alexnet.cpp)

add_benchmark(resnet18 resnet18.cpp)

add_benchmark(resnet34 resnet34.cpp)

add_benchmark(resnet50 resnet50.cpp)

add_benchmark(vgg16 vgg16.cpp)

add_benchmark(vgg19 vgg19.cpp)

add_benchmark(arxiv1701.00008 arxiv1701.00008.cpp)

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

add_custom_command(
  OUTPUT
    deepbench_rnn.py
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/deepbench_rnn.py deepbench_rnn.py
  DEPENDS
    deepbench_rnn.py
)

add_custom_target(python_scripts DEPENDS benchmark_report.py run_benchmarks.py deepbench_rnn.py)
add_dependencies(alexnet python_scripts)
