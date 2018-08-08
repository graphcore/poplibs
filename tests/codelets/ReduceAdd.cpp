#include <poplar/Engine.hpp>
#define BOOST_TEST_MODULE ReduceAdd
#include <boost/test/unit_test.hpp>
#include "TestDevice.hpp"
// codelets
#include "popconv/codelets.hpp"
#include "popconv/Convolution.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_test/Util.hpp"
#include "poplibs_test/Util.hpp"
#include "poplar/Target.hpp"
#include <string.h>


#define OUTER_DIM @OUTER_DIM@
#define INNER_DIM @INNER_DIM@
#define PARTIALS_TYPE @PARTIALS_TYPE@
#define OUT_TYPE @OUT_TYPE@

using namespace poplar;
using namespace poplar::program;
using namespace popconv;
using namespace poputil;
using namespace poplibs_test::util;

BOOST_AUTO_TEST_CASE(ReduceAdd) {
  auto device = createTestDevice(TEST_TARGET);
  auto &target = device.getTarget();
  Graph graph(device);
  popconv::addCodelets(graph);

  // Claim enough space for floats
  unsigned char data[INNER_DIM * OUTER_DIM * 4];
  float nums[INNER_DIM * OUTER_DIM];
  for (unsigned i = 0; i < INNER_DIM * OUTER_DIM; ++i) {
    nums[i] = 1.0 * (i % OUTER_DIM);
  }
  copy(target, nums, INNER_DIM*OUTER_DIM, PARTIALS_TYPE, data);
  float answers[OUTER_DIM + 1];
  char ans_data[(OUTER_DIM + 1) * 4];
  for (unsigned i = 0; i < OUTER_DIM + 1; ++i) {
    answers[i] = 0.0;
  }
  memcpy(ans_data, answers, (OUTER_DIM + 1) * 4);

  Sequence prog;

  auto cs = graph.addComputeSet("cs");

  Tensor partials;
  partials = graph.addVariable(PARTIALS_TYPE, {INNER_DIM, OUTER_DIM});
  Tensor out;
  out = graph.addVariable(OUT_TYPE, {1, OUTER_DIM+1});

  const auto vertexClass = templateVertex("popconv::ReduceAdd",
                                          OUT_TYPE, PARTIALS_TYPE);
  auto v1 = graph.addVertex(cs,
                            vertexClass);

  for (int i = 0; i < INNER_DIM; ++i) {
    Tensor Row = partials.slice(i, i+1, 0);
    graph.connect(v1["partials"][i], Row.reshape({OUTER_DIM}));
  }
  graph.setFieldSize(v1["partials"], INNER_DIM);
  graph.connect(v1["out"], out.slice(0, OUTER_DIM, 1));
  graph.setInitialValue(v1["numPartials"], INNER_DIM);

  graph.setTileMapping(v1, 0);
  graph.setTileMapping(partials, 0);
  graph.setTileMapping(out, 0);

  graph.createHostWrite("partials", partials);
  graph.createHostWrite("outw", out);
  graph.createHostRead("out", out);

  prog.add(Execute(cs));

  Engine e(graph, prog,
           OptionFlags{{"target.textSectionSizeInBytes", "0x9000"}});

  e.load(device);
  e.writeTensor("partials", data);
  e.writeTensor("outw", ans_data);
  e.readTensor("out", ans_data);

  e.run();

  e.readTensor("out", ans_data);

  copy(target, OUT_TYPE, ans_data, answers, OUTER_DIM+1);

  for(int i =0; i < OUTER_DIM; ++i){
    BOOST_CHECK_EQUAL(INNER_DIM * 1.0 * i, answers[i]);
    answers[i] = 0; // zero for next iteration
  }
  BOOST_CHECK_EQUAL(answers[OUTER_DIM], 0.0);
}
