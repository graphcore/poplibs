cmake_minimum_required(VERSION 2.8.11)
project(popnn)

find_package(poplar REQUIRED)

include(CheckCXXCompilerFlag)

enable_testing()

set(Boost_USE_STATIC_LIBS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

check_cxx_compiler_flag(-Wno-maybe-uninitialized COMPILER_SUPPORTS_W_NMI)
if(COMPILER_SUPPORTS_W_NMI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

check_cxx_compiler_flag(-Werror=vla COMPILER_SUPPORTS_W_VLA)
if(COMPILER_SUPPORTS_W_VLA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=vla")
endif()

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(benchmarks)
add_subdirectory(tests)
add_subdirectory(tools)

install(FILES)
