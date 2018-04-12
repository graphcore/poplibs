#define BOOST_TEST_MODULE ParallelTest
#include <boost/test/unit_test.hpp>
#include <popops/codelets.hpp>
#include <poplar/IPUModel.hpp>

#include <thread>

using namespace poplar;
using namespace popops;

BOOST_AUTO_TEST_CASE(ManyParallelGraphLoads){
    const size_t nthreads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;

    for (unsigned t = 0; t<nthreads; t++)
    {
        threads.push_back(std::thread([]() {
          IPUModel ipuModel;
          auto device = ipuModel.createDevice();

          Graph graph(device);
          popops::addCodelets(graph);
        }));
    }

    for (unsigned t = 0; t<nthreads; t++) {
        threads[t].join();
    }
}
