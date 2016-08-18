cmake_minimum_required(VERSION 2.8.11)
project(popnn)
find_package(Boost REQUIRED)
enable_testing()

set(Boost_USE_STATIC_LIBS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(benchmarks)
add_subdirectory(tests)

install(FILES)
