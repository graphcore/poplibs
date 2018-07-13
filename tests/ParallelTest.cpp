#define BOOST_TEST_MODULE ParallelTest
#include <boost/test/unit_test.hpp>
#include <popops/codelets.hpp>
#include "TestDevice.hpp"

#include <thread>

using namespace poplar;
using namespace popops;

BOOST_AUTO_TEST_CASE(ManyParallelGraphLoads){
    const size_t nthreads = std::thread::hardware_concurrency();

    if (TEST_TARGET == DeviceType::Hw) {
      BOOST_FAIL("We need to make a version of this "
                 "test appropriate for Hw (T3632)");
    } else {
      std::vector<std::thread> threads;

      for (unsigned t = 0; t<nthreads; t++)
      {
        threads.push_back(std::thread([]() {
          auto device = createTestDevice(TEST_TARGET);

          Graph graph(device);
          popops::addCodelets(graph);
        }));
      }

      for (unsigned t = 0; t<nthreads; t++) {
        threads[t].join();
      }
    }
}
