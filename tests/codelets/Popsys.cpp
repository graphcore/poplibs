#include <poplar/Engine.hpp>
#define BOOST_TEST_MODULE Popsys
#include <boost/test/unit_test.hpp>
#include "TestDevice.hpp"

#include "popsys/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_test/Util.hpp"
#include "poplar/Target.hpp"
#include "popsys/CycleCount.hpp"
#include <cstdint>

using namespace poplar;
using namespace poplar::program;

using namespace poputil;
using namespace poplibs_test::util;

const unsigned maxProfilingOverhead = 100;

BOOST_AUTO_TEST_CASE(Popsys) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popsys::addCodelets(graph);
  graph.addCodelets("Delay1000.gp");

  std::string vertexClass = "popsys::TimeItStart";

  auto cs = graph.addComputeSet("cs");
  auto v = graph.addVertex(cs, "Delay1000");
  graph.setTileMapping(v, 0);
  Sequence prog;
  prog.add(Execute(cs));
  auto counts = popsys::cycleCount(graph, prog, 0);
  graph.createHostRead("counts", counts);

  Engine e(graph, prog,
           OptionFlags{{"target.textSectionSizeInBytes", "0x9000"}});
  e.load(device);

  uint64_t cycles;
  e.run();
  e.readTensor("counts", &cycles);

  BOOST_CHECK(cycles >= 1000 && cycles < 1000 + maxProfilingOverhead);
}
