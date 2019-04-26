#include "popsys/CycleStamp.hpp"
#include "poputil/exceptions.hpp"


using namespace poplar;
using namespace poplar::program;

namespace popsys {

// TODO:: could add choice of internal/external sync
Tensor cycleStamp(Graph &graph, Sequence &prog,
                  unsigned tile,
                  const std::string &debugPrefix) {
  if (graph.getTarget().getTargetType() != poplar::TargetType::IPU) {
    throw poputil::poplibs_error(
        "cycleStamp is only available for ipu targets");
  }

  // longs not supported on IPU backend so vector of 2 uints
  Tensor stamp = graph.addVariable(UNSIGNED_INT, {2});

  static unsigned id = 0;
  auto cs =
      graph.addComputeSet(debugPrefix + "/timeStampCS_" + std::to_string(++id));
  auto v = graph.addVertex(cs, "popsys::TimeItStart");

  graph.connect(v["out"], stamp);
  graph.setTileMapping(v, tile);
  graph.setTileMapping(stamp, tile);

  // Sync and record cycle count on chosen tile
  prog.add(Sync(SyncType::INTERNAL));
  prog.add(Execute(cs));

  return stamp;
}

} // end namespace popsys
