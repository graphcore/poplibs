cmake_minimum_required(VERSION 3.5.0)
project(poplibs)

include(CheckCXXCompilerFlag)

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" )

enable_testing()

if(${BUILD_SHARED_LIBS})
  set(Boost_USE_STATIC_LIBS OFF)
else()
  set(Boost_USE_STATIC_LIBS ON)
endif()

find_package(poplar REQUIRED)
find_package(Boost REQUIRED regex)
find_package(Sparsehash REQUIRED)

# Require C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add some warning flags. check_cxx_compiler_flag isn't very good at detecting
# unsupported flags that don't cause an error, so it is simpler to switch based
# on the compiler ID.

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror=vla -Wno-maybe-uninitialized")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror=vla")
endif()

add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(tools)

install(FILES)
