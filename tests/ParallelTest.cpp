// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE ParallelTest
#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>
#include <popops/codelets.hpp>

#include <atomic>
#include <iostream>
#include <thread>

using namespace poplar;
using namespace popops;

BOOST_AUTO_TEST_CASE(ManyParallelGraphLoads) {
  std::atomic<bool> success{true};

  const size_t nthreads = std::thread::hardware_concurrency();

  std::vector<std::thread> threads;

  for (unsigned t = 0; t < nthreads; t++) {
    threads.push_back(std::thread([&]() {
      // Exceptions can't be thrown across threads so just
      // catch everything and print a message if it failed.
      try {
        auto device = createTestDevice(TEST_TARGET);

        Graph graph(device.getTarget());
        popops::addCodelets(graph);
      } catch (const std::exception &e) {
        std::cout << ((std::string("Exception: ") + e.what()) + "\n");
        success = false;
      }
    }));
  }

  for (unsigned t = 0; t < nthreads; t++) {
    threads[t].join();
  }

  BOOST_CHECK(success);
}
