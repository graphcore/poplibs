cmake_minimum_required(VERSION 3.6.0)
project(poplibs)

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" )

include(AddGPLibrary)
include(FindRuntimeAndArchPaths)

enable_testing()

set(DEFAULT_TEST_VARIANTS "IpuModel;Sim;Hw" CACHE STRING
    "The device variants to run tests on when no variants are set explicitly")

if (DEFINED ENV{POPLAR_SDK_ENABLED})
  set(ENABLED_TEST_VARIANTS "NoTarget;Cpu;IpuModel;IpuModel0" CACHE STRING
      "Generate tests for these devices and variants")
else()
  set(ENABLED_TEST_VARIANTS "NoTarget;Cpu;IpuModel;IpuModel0;Sim;Sim0;Sim:cpp" CACHE STRING
      "Generate tests for these devices and variants")
endif()

# Enable colour output if the CLICOLOR_FORCE environment variable is set.
if($ENV{CLICOLOR_FORCE})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "Enabling GNU colour output")
    add_compile_options("-fdiagnostics-color=always")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message(STATUS "Enabling Clang colour output")
    add_compile_options("-fcolor-diagnostics")
  endif()
endif()

if(${BUILD_SHARED_LIBS})
  set(Boost_USE_STATIC_LIBS OFF)
else()
  set(Boost_USE_STATIC_LIBS ON)
endif()

find_package(poplar REQUIRED)
find_package(Boost 1.65.1 REQUIRED regex)
find_package(TBB REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add some warning flags. check_cxx_compiler_flag isn't very good at detecting
# unsupported flags that don't cause an error, so it is simpler to switch based
# on the compiler ID.

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror -Wvla -Wno-maybe-uninitialized")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror -Wvla")
endif()

if(DEFINED POPLAR_INCLUDE_PATH)
  string(APPEND CMAKE_CXX_FLAGS " -I ${POPLAR_INCLUDE_PATH}")
endif()

# Add an explicit exported symbols file (a linker flag)

if(APPLE)
  string(APPEND CMAKE_SHARED_LINKER_FLAGS
         " -Wl,-exported_symbols_list,"
         "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_osx.lds")
else()
  string(APPEND CMAKE_SHARED_LINKER_FLAGS
         " -Wl,--version-script,"
         "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_linux.lds")
endif()

# Set up popc flags.
set(POPC_FLAGS -I ${CMAKE_SOURCE_DIR}/include)
if(DEFINED ARCH_MAN_PATH)
  list(APPEND POPC_FLAGS -I ${ARCH_MAN_PATH}/include)
endif()
if(DEFINED POPLAR_INCLUDE_PATH)
  list(APPEND POPC_FLAGS -I ${POPLAR_INCLUDE_PATH})
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  list(APPEND POPC_FLAGS -g)
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
  list(APPEND POPC_FLAGS -O3 -g)
else()
  list(APPEND POPC_FLAGS -O3)
endif()

# If requested, optimise the debug builds too.
if (OPTIMISE_IPU_DEBUG AND CMAKE_BUILD_TYPE STREQUAL Debug)
  list(APPEND POPC_FLAGS -O3)
endif()

# For poplibs, we don't want surprises in code-generation.
# -Wdouble-promotion will catch any kind of implicit floating point promotion.
list(APPEND POPC_FLAGS -Wdouble-promotion -Werror=double-promotion)

add_definitions("-DBOOST_ICL_USE_STATIC_BOUNDED_INTERVALS")

add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(tools)

install(FILES)

if (DEFINED ENV{POPLAR_SDK_ENABLED})
  file(WRITE ${CMAKE_BINARY_DIR}/enable.sh [=[
#!/bin/bash
/bin/cat <<MessageEnd
ERROR: This Poplibs build must be installed (e.g. ninja install) before it can
be enabled. Once installed, source enable.sh from the install folder.
NOTE: You only need to enable if you are building another project that links
to this build.
MessageEnd
]=])
  install(FILES ${CMAKE_SOURCE_DIR}/packaging_files/enable.sh.in
          RENAME enable.sh
          DESTINATION .)
endif()
