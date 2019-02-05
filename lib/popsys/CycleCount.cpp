#include "popsys/CycleCount.hpp"
#include "poputil/exceptions.hpp"


using namespace poplar;
using namespace poplar::program;

namespace popsys {

Tensor cycleCount(Graph &graph, Sequence &prog, unsigned tile,
                  const std::string &debugPrefix) {
  if (graph.getTarget().getTargetType() != poplar::TargetType::IPU) {
    throw poputil::poplibs_error(
        "cycleCount is only available for ipu targets");
  }

  // Would be better if could force a sync here as time could vary
  // depending on tile
  Sequence timerSequence;
  // longs not supported on IPU backend so vector of 2 uints
  Tensor beforeProgram = graph.addVariable(UNSIGNED_INT, {2});
  Tensor afterProgram = graph.addVariable(UNSIGNED_INT, {2});

  static unsigned id = 0;
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

  // Sync, record starting cycle count on chosen tile
  // execute sequence, sync, and finally record end cycle count
  // and calculate total.
  // TODO: These should probably be external syncs, if they were
  // supported
  timerSequence.add(Sync(SyncType::INTERNAL));
  timerSequence.add(Execute(beforeCS));
  timerSequence.add(prog);
  timerSequence.add(Sync(SyncType::INTERNAL));
  timerSequence.add(Execute(afterCS));

  prog = timerSequence;
  // Alternative to replacing original could add prepend method to Sequence

  return afterProgram;
}

} // end namespace popsys
