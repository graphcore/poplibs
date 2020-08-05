// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LoggingTest
#include <boost/test/unit_test.hpp>

#include <poplibs_support/logging.hpp>

using namespace poplibs_support::logging;

// The output of these tests are checked by CMake regular expressions and are
// not checked here explicitly.

BOOST_AUTO_TEST_CASE(LoggingPrintoutExample) {
  setLogLevel(Module::popfloat, Level::Trace);
  setLogLevel(Module::poplin, Level::Trace);
  setLogLevel(Module::popnn, Level::Trace);
  setLogLevel(Module::popops, Level::Trace);
  setLogLevel(Module::poprand, Level::Trace);
  setLogLevel(Module::popsolver, Level::Trace);
  setLogLevel(Module::popsparse, Level::Trace);
  setLogLevel(Module::poputil, Level::Trace);
  popfloat::info("Hello world");
  poplin::info("Hello world");
  popnn::info("Hello world");
  popops::info("Hello world");
  poprand::info("Hello world");
  popsolver::info("Hello world");
  popsparse::info("Hello world");
  poputil::info("Hello world");

  // Note that poplibs isn't a specific logging module. So the following line
  // will not compile
  // poplibs::err("Hello world");
}

BOOST_AUTO_TEST_CASE(SelectiveLogging) {
  // POPLIBS_POPFLOAT_LOG_LEVEL=ERR
  setLogLevel(Module::popfloat, Level::Err);
  // POPLIBS_POPLIN_LOG_LEVEL=TRACE
  setLogLevel(Module::poplin, Level::Trace);

  popfloat::info("I'm not printed");
  poplin::info("I'm printed");
}
