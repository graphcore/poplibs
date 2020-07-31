cmake_minimum_required(VERSION 3.10.2) # Default version in Ubuntu 18.04.
project(poplibs)

include(GNUInstallDirs)

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" )

include(AddGPLibrary)

enable_testing()

find_package(poplar REQUIRED)

if(NOT DEFINED POPLIBS_ENABLED_IPU_ARCH_NAMES)
  # by default build for all architectures that the Poplar we are using supports.
  get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

  execute_process(COMMAND ${POPC_EXECUTABLE} --list-targets
                  OUTPUT_VARIABLE POPC_LIST_TARGETS_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "," ";" TMP "${POPC_LIST_TARGETS_OUTPUT}")
  set(POPLIBS_ENABLED_IPU_ARCH_NAMES "${TMP}" CACHE STRING
      "The architectures to build the poplibs library objects for")
endif()
message("Enabled architectures: ${POPLIBS_ENABLED_IPU_ARCH_NAMES}")

# allow users to specify a subset of test variants to test on.
set(POPLIBS_ENABLED_IPU_ARCH_TEST_VARIANTS "${POPLIBS_ENABLED_IPU_ARCH_NAMES}" CACHE STRING
    "The architectures to run tests for, defaults to all enabled architectures")
message("Enabled test architectures: ${POPLIBS_ENABLED_IPU_ARCH_TEST_VARIANTS}")

# allow users easily to opt out from testing :cpp variants, this is off by
# default because this code is not packaged.
set(POPLIBS_ENABLE_CPP_TEST_VARIANTS OFF CACHE STRING
    "Whether to include cpp tests for codelets that otherwise have an assembly implentation")
message("cpp tests enabled: ${POPLIBS_ENABLE_CPP_TEST_VARIANTS}")

# ipu1 is our default arch, so convert ipu1->ipu
string(REPLACE "ipu1" "ipu" ABBREV_TARGETS "${POPLIBS_ENABLED_IPU_ARCH_TEST_VARIANTS}")

set(CPU_VARIANTS "")
set(IPUMODEL_VARIANTS "")
set(SIM_VARIANTS "")
set(SIMCPP_VARIANTS "")
foreach(IPU_ARCH_TEST_VARIANT ${ABBREV_TARGETS})
  if (IPU_ARCH_TEST_VARIANT STREQUAL "cpu")
    list(APPEND CPU_VARIANTS "Cpu")
  elseif(IPU_ARCH_TEST_VARIANT MATCHES "^ipu")
    string(REPLACE "ipu" "IpuModel" IPUMODEL_VARIANT "${IPU_ARCH_TEST_VARIANT}")
    list(APPEND IPUMODEL_VARIANTS "${IPUMODEL_VARIANT}")

    string(REPLACE "ipu" "Sim" SIM_VARIANT "${IPU_ARCH_TEST_VARIANT}")
    list(APPEND SIM_VARIANTS "${SIM_VARIANT}")
    if (POPLIBS_ENABLE_CPP_TEST_VARIANTS)
      list(APPEND SIMCPP_VARIANTS "${SIM_VARIANT}:cpp")
    endif()
  else()
    message(FATAL_ERROR "Unrecognised architecture: ${IPU_ARCH_TEST_VARIANT}")
  endif()
endforeach()

set(ALL_POSSIBLE_TEST_VARIANTS "NoTarget;Cpu;IpuModel;IpuModel2;Sim;Sim2;Sim:cpp;Sim2:cpp;Hw")

set(DEFAULT_TEST_VARIANTS "${IPUMODEL_VARIANTS};${SIM_VARIANTS};${SIMCPP_VARIANTS};Hw" CACHE STRING
    "The device variants to run tests on when no variants are set explicitly")

if (DEFINED ENV{POPLAR_SDK_ENABLED})
set(ENABLED_TEST_VARIANTS "NoTarget;${CPU_VARIANTS};${IPUMODEL_VARIANTS}" CACHE STRING
      "Generate tests for these devices and variants")
else()
set(ENABLED_TEST_VARIANTS "NoTarget;${CPU_VARIANTS};${IPUMODEL_VARIANTS};${SIM_VARIANTS};${SIMCPP_VARIANTS}" CACHE STRING
      "Generate tests for these devices and variants")
endif()

message("Default test variants: ${DEFAULT_TEST_VARIANTS}")
message("Enabled test variants: ${ENABLED_TEST_VARIANTS}")

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

# Enable LLVM sanitisers if requested (only supported for Clang currently).
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if ("asan" IN_LIST SANITIZERS)
    message(STATUS "Enabling LLVM address sanitizer")
    string(APPEND CMAKE_CXX_FLAGS " -fno-omit-frame-pointer -fsanitize=address")
    string(APPEND CMAKE_LINKER_FLAGS " -fno-omit-frame-pointer -fsanitize=address")

    # Find llvm-symbolizer or addr2line and set ASAN_SYMBOLIZER_PATH to point
    # to it. We need symbolised backtraces so the suppressions file works.
    find_program(symbolizer_path NAMES "llvm-symbolizer" "addr2line")
    if (NOT symbolizer_path)
      message(FATAL_ERROR "llvm-symbolizer or addr2line not found")
    endif()

    list(APPEND TEST_ENVIRONMENT
      "LSAN_OPTIONS=suppressions=${PROJECT_SOURCE_DIR}/lsan.supp"
      "ASAN_SYMBOLIZER_PATH=${symbolizer_path}"
      "CXXFLAGS=-fsanitize=address"
    )

    list(APPEND TEST_CXXFLAGS "-fsanitize=address")

    list(APPEND POPC_ENVIRONMENT
      "ASAN_OPTIONS=detect_leaks=0,halt_on_error=0"
      "MSAN_OPTIONS=halt_on_error=0"
    )
  endif()
endif()


if(${BUILD_SHARED_LIBS})
  set(Boost_USE_STATIC_LIBS OFF)
else()
  set(Boost_USE_STATIC_LIBS ON)
endif()

find_package(Boost 1.70.0 REQUIRED regex program_options filesystem unit_test_framework timer system)
find_package(TBB REQUIRED)
find_package(zoltan QUIET)
find_package(Threads REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer -g3")

# Add some warning flags. check_cxx_compiler_flag isn't very good at detecting
# unsupported flags that don't cause an error, so it is simpler to switch based
# on the compiler ID.

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror -pedantic -pedantic-errors -Wvla -Wno-maybe-uninitialized")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror -pedantic -pedantic-errors -Wvla")
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
list(APPEND POPC_FLAGS -Werror -Wdouble-promotion)

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
MessageEnd
]=])
  install(FILES ${CMAKE_SOURCE_DIR}/packaging_files/enable.sh.in
          RENAME enable.sh
          DESTINATION .)
endif()
