// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cstdio>

#define BOOST_TEST_MODULE ConsistentExecutableTest
#include <boost/test/unit_test.hpp>

static int checkExit(int rc) {
  // Return 0 if the process exited normally otherwise the exit status code.
  return WIFEXITED(rc) ? WEXITSTATUS(rc) : rc;
}

BOOST_AUTO_TEST_CASE(ConsistentExecutable) {
  // Create two executables and check they're identical.
#ifndef EXECUTABLE
#error "EXECUTABLE must be defined"
#endif
  BOOST_TEST(checkExit(system(EXECUTABLE " one.exe")) == EXIT_SUCCESS);
  BOOST_TEST(checkExit(system(EXECUTABLE " two.exe")) == EXIT_SUCCESS);
  BOOST_TEST(checkExit(system("cmp one.exe two.exe")) == EXIT_SUCCESS);
  std::remove("one.exe");
  std::remove("two.exe");
}
