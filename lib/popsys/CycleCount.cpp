#include "popsys/CycleCount.hpp"
//#include "popops/ElementWise.hpp"

using namespace poplar;
using namespace poplar::program;
//using namespace poputil;

static unsigned id = 0;

namespace popsys {
Tensor cycleCount(Graph &graph, Sequence &prog, unsigned tile,
                  const std::string &debugPrefix) {
  // Would be better if could force a sync here as time could vary
  // depending on tile
  Sequence timerSequence;
  // longs not supported on IPU backend so vector of 2 uints
  Tensor beforeProgram = graph.addVariable(UNSIGNED_INT, {2});
  Tensor afterProgram = graph.addVariable(UNSIGNED_INT, {2});

  auto beforeCS = graph.addComputeSet(debugPrefix + "/timeCS_"
                                      + std::to_string(++id));
  auto afterCS = graph.addComputeSet(debugPrefix + "/timeCS_"
                                     + std::to_string(++id));

  auto beforeVertex = graph.addVertex(beforeCS, "popsys::TimeItStart");
  auto afterVertex = graph.addVertex(afterCS, "popsys::TimeItEnd");

  //connect stuff
  graph.connect(beforeVertex["out"], beforeProgram);
  graph.connect(afterVertex["startCount"], beforeProgram);
  graph.connect(afterVertex["out"], afterProgram);

  graph.setTileMapping(beforeVertex, tile);
  graph.setTileMapping(afterVertex, tile);
  graph.setTileMapping(beforeProgram, tile);
  graph.setTileMapping(afterProgram, tile);

  timerSequence.add(Execute(beforeCS));
  timerSequence.add(prog);
  timerSequence.add(Execute(afterCS));

  prog = timerSequence;
  // Alternative to replacing original could add prepend method to Sequence



  return afterProgram;

}
} // end namespace popsys
