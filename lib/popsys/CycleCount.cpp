#include "popsys/CycleCount.hpp"
#include "poputil/exceptions.hpp"


using namespace poplar;
using namespace poplar::program;

namespace popsys {

Tensor cycleCount(Graph &graph, Sequence &prog, unsigned tile,
                  const std::string &debugPrefix) {
  const auto &target = graph.getTarget();
  if (target.getTargetType() != poplar::TargetType::IPU) {
    throw poputil::poplibs_error(
        "cycleCount is only available for ipu targets");
  }

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

  const auto syncType =
    target.getNumIPUs() > 1 ? SyncType::EXTERNAL : SyncType::INTERNAL;

  // Sync, record starting cycle count on chosen tile
  // execute sequence, sync, and finally record end cycle count
  // and calculate total.
  timerSequence.add(Sync(syncType));
  timerSequence.add(Execute(beforeCS));
  timerSequence.add(prog);
  timerSequence.add(Sync(syncType));
  timerSequence.add(Execute(afterCS));

  prog = timerSequence;
  // Alternative to replacing original could add prepend method to Sequence

  return afterProgram;
}

} // end namespace popsys
