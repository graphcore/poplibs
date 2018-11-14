#include <poplar/Engine.hpp>
#define BOOST_TEST_MODULE Popsys
#include <boost/test/unit_test.hpp>
#include "TestDevice.hpp"

#include "popsys/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_test/Util.hpp"
#include "poplar/Target.hpp"
#include "popsys/CycleCount.hpp"
#include "popsys/CSRFunctions.hpp"
#include <cstdint>

#define __IPU_ARCH_VERSION__ 0
#include <tilearch.h>

using namespace poplar;
using namespace poplar::program;

using namespace popsys;

using namespace poputil;
using namespace poplibs_test::util;

const unsigned maxProfilingOverhead = 100;

//******************************************************************************
// Helper functions to access individual registers
//******************************************************************************

Tensor getWorkerCSR(Graph &graph, Sequence &prog, unsigned tile,
                            unsigned csrReg, const std::string &debugPrefix) {

  Tensor result = graph.addVariable(UNSIGNED_INT, {1});
  auto cs = graph.addComputeSet(debugPrefix + "/getWorkerCSR");
  auto v = graph.addVertex(cs, templateVertex("popsys::GetWorkerCSR", csrReg));

  graph.connect(v["out"],result);
  graph.setTileMapping(v,tile);

  prog.add(Execute(cs));

  return result;
 }

void putWorkerCSR(Graph &graph, Sequence &prog, unsigned tile, unsigned csrReg,
                        unsigned writeVal, const std::string &debugPrefix) {

  auto cs = graph.addComputeSet(debugPrefix + "/setWorkerCSR");
  auto v = graph.addVertex(cs, templateVertex("popsys::PutWorkerCSR", csrReg));

  graph.setInitialValue(v["setVal"], writeVal);
  graph.setTileMapping(v, tile);

  prog.add(Execute(cs));
}

void modifyWorkerCSR(Graph &graph,
                      Sequence &prog,
                      unsigned tile,
                      unsigned csrReg,
                      unsigned clearVal,
                      unsigned setVal,
                      const std::string &debugPrefix) {

  auto cs = graph.addComputeSet(debugPrefix + "/modifyWorkerCSR");
  auto v = graph.addVertex(cs,
              templateVertex("popsys::ModifyWorkerCSR", csrReg));

  graph.setInitialValue(v["clearVal"], clearVal);
  graph.setInitialValue(v["setVal"], setVal);
  graph.setTileMapping(v, tile);

  prog.add(Execute(cs));
}

Tensor getSupervisorCSR( Graph &graph, Sequence &prog, unsigned tile,
                            unsigned csrReg, const std::string &debugPrefix) {

  Tensor result = graph.addVariable(UNSIGNED_INT, {1});
  auto cs = graph.addComputeSet(debugPrefix + "/getSupervisorCSR");
  auto v = graph.addVertex(cs,
                        templateVertex("popsys::GetSupervisorCSR", csrReg));

  graph.connect(v["out"],result);
  graph.setTileMapping(v,tile);

  prog.add(Execute(cs));
  return result;
 }

void putSupervisorCSR(Graph &graph, Sequence &prog, unsigned tile,
          unsigned csrReg, unsigned writeVal, const std::string &debugPrefix) {

  auto cs = graph.addComputeSet(debugPrefix + "/setSupervisorCSR");
  auto v = graph.addVertex(cs,
                templateVertex("popsys::PutSupervisorCSR", csrReg));

  graph.setInitialValue(v["setVal"], writeVal);
  graph.setTileMapping(v, tile);

  prog.add(Execute(cs));
}


void modifySupervisorCSR(Graph &graph,
                          Sequence &prog,
                          unsigned tile,
                          unsigned csrReg,
                          unsigned clearVal,
                          unsigned setVal,
                          const std::string &debugPrefix) {

  auto cs = graph.addComputeSet(debugPrefix + "/modifySupervisorCSR");
  auto v = graph.addVertex(cs,
      templateVertex("popsys::ModifySupervisorCSR", csrReg));

  graph.setInitialValue(v["clearVal"], clearVal);
  graph.setInitialValue(v["setVal"], setVal);
  graph.setTileMapping(v, tile);

  prog.add(Execute(cs));
}

//******************************************************************************
// Tests : time
//******************************************************************************

BOOST_AUTO_TEST_CASE(PopsysTimeIt) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popsys::addCodelets(graph);
  graph.addCodelets("Delay1000.gp");

  auto cs = graph.addComputeSet("cs");
  auto v = graph.addVertex(cs, "Delay1000");
  graph.setTileMapping(v, 0);
  Sequence prog;
  prog.add(Execute(cs));
  auto counts = popsys::cycleCount(graph, prog, 0);
  graph.createHostRead("counts", counts);

  Engine e(graph, prog);
  e.load(device);

  uint64_t cycles;
  e.run();
  e.readTensor("counts", &cycles);

 BOOST_CHECK(cycles >= 1000 && cycles < 1000 + maxProfilingOverhead);
}
//******************************************************************************
// Tests : general register access
//******************************************************************************

BOOST_AUTO_TEST_CASE(PopsysSupervisorCSR) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popsys::addCodelets(graph);

  // Register 32 is FP_ICTL.  Only bits 0,1,2,19,20 are read/write
  auto csrReg = CSR_S_FP_ICTL__INDEX;

  Sequence prog;

  Tensor result1 = getSupervisorCSR(graph, prog, 0, csrReg, "Before");
  graph.setTileMapping(result1, 0);
  graph.createHostRead("result1", result1);

  putSupervisorCSR(graph, prog, 0, csrReg, 7, "Write register");
  modifySupervisorCSR(graph, prog, 0, csrReg, 0xf, 0x100, "Modify");

  Tensor result2 = getSupervisorCSR(graph, prog, 0, csrReg, "Second");
  graph.setTileMapping(result2, 0);
  graph.createHostRead("result2", result2);

  modifySupervisorCSR(graph, prog, 0, csrReg, 0xfffffffe, 0x0, "Modify");
  modifySupervisorCSR(graph, prog, 0, csrReg, 0xffffffff, 0x100000, "Modify");

  Tensor result3 = getSupervisorCSR(graph, prog, 0, csrReg, "Final Result");
  graph.setTileMapping(result3, 0);
  graph.createHostRead("result3", result3);

  Engine e(graph, prog);
  e.load(device);

  std::vector<unsigned> csrResult(3);
  e.run();

  e.readTensor("result1", &csrResult[0]);
  e.readTensor("result2", &csrResult[1]);
  e.readTensor("result3", &csrResult[2]);

  std::vector<unsigned> expectedResult = {0x0, 0x7, 0x100006};
  bool check = checkEqual("PopsysGetPutSupervisor", csrResult.data(), {3},
                  expectedResult.data(),expectedResult.size());
  BOOST_CHECK(check);
}


BOOST_AUTO_TEST_CASE(PopsysWorkerCSR) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popsys::addCodelets(graph);

  // Register 112 is DBG_DATA, which has all bits valid, is read/write
  // and is visible/ writable from all contexts.  This is useful for test,
  // as we can write it in a worker context (we have no control over which
  // worker), and read it back again from a worker context (any worker, again no
  // control) reliably.
  auto csrReg = CSR_C_DBG_DATA__INDEX;

  Sequence prog;

  Tensor result1 = getWorkerCSR(graph, prog, 0, csrReg, "First Result");
  graph.setTileMapping(result1, 0);

  putWorkerCSR(graph, prog, 0, csrReg, 7, "Write register");
  modifyWorkerCSR(graph, prog, 0, csrReg, 0xf, 0x100, "Modify");

  Tensor result2 = getWorkerCSR(graph, prog, 0, csrReg, "Final Result");
  graph.setTileMapping(result2, 0);

  graph.createHostRead("result1", result1);
  graph.createHostRead("result2", result2);

  Engine e(graph, prog);
  e.load(device);

  std::vector<unsigned> csrResult(2);
  e.run();

  e.readTensor("result1", &csrResult[0]);
  e.readTensor("result2", &csrResult[1]);

  std::vector<unsigned> expectedResult = {0xbaddf000, 0x107};
  bool check = checkEqual("PopsysGetPutSupervisor", csrResult.data(), {2},
                  expectedResult.data(),expectedResult.size());
  BOOST_CHECK(check);
}

BOOST_AUTO_TEST_CASE(PopsyssetFloatingPointBehaviour) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popsys::addCodelets(graph);

  // Register 32 is FP_ICTL.  Only bits 0,1,2,19,20 are read/write
  auto csrReg = CSR_S_FP_ICTL__INDEX;

  Sequence prog;

  Tensor result1 = getSupervisorCSR(graph, prog, 0, csrReg, "Result 1");
  graph.setTileMapping(result1, 0);

  floatingPointBehaviour behaviour;
  behaviour.div0 = false;
  behaviour.oflo = false;
  setFloatingPointBehaviour(graph, prog, behaviour, 0, "Set");

  Tensor result2 = getSupervisorCSR(graph, prog, 0, csrReg, "Result 2");
  graph.setTileMapping(result2, 0);

  behaviour.div0 = true;
  behaviour.esr = false;
  setFloatingPointBehaviour(graph, prog, behaviour, 0, "SetClr");

  Tensor result3 = getSupervisorCSR(graph, prog, 0, csrReg, "Result 3");
  graph.setTileMapping(result3, 0);

  setStochasticRounding(graph, prog, true, 0, "Rounding");

  Tensor result4 = getSupervisorCSR(graph, prog, 0, csrReg, "Result 4");
  graph.setTileMapping(result4, 0);

  graph.createHostRead("result1", result1);
  graph.createHostRead("result2", result2);
  graph.createHostRead("result3", result3);
  graph.createHostRead("result4", result4);

  Engine e(graph, prog);
  e.load(device);

  std::vector<unsigned> csrResult(4);
  e.run();

  e.readTensor("result1", &csrResult[0]);
  e.readTensor("result2", &csrResult[1]);
  e.readTensor("result3", &csrResult[2]);
  e.readTensor("result4", &csrResult[3]);

  std::vector<unsigned> expectedResult = {0x0, 0x180001, 0x100003, 0x180003};
  bool check = checkEqual("PopsysGetPutSupervisor", csrResult.data(), {4},
                  expectedResult.data(),expectedResult.size());
  BOOST_CHECK(check);
}
