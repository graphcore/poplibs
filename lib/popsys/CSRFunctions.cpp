#include "popsys/CSRFunctions.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/VertexTemplates.hpp"

#include "poplibs_support/TileConstants.hpp"

#include <poplar/Target.hpp>

using namespace poplar;
using namespace poplar::program;

namespace popsys {

void setStochasticRounding(Graph &graph,
                           Sequence &prog,
                           bool enable,
                           const std::string &debugPrefix) {
  if (graph.getTarget().getTargetType() != TargetType::IPU)
    return;
  auto cs = graph.addComputeSet(debugPrefix + "/setStochasticRounding");
  uint32_t fpIctlWithEsrSet = graph.getTarget().makeFpIctlValue(false,
                                                                false,
                                                                false,
                                                                true, //ESR
                                                                false);
  auto numTiles = graph.getTarget().getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    auto v = graph.addVertex(
               cs,
               poputil::templateVertex("popsys::ModifySupervisorCSR",
                                       graph.getTarget().getFpIctlRegIndex()));

    graph.setInitialValue(v["clearVal"], ~fpIctlWithEsrSet);
    graph.setInitialValue(v["setVal"], enable ? fpIctlWithEsrSet : 0);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));

}

void setFloatingPointBehaviour( Graph &graph,
                                Sequence &prog,
                                const FloatingPointBehaviour &behaviour,
                                const std::string &debugPrefix) {
  if (graph.getTarget().getTargetType() != TargetType::IPU)
    return;
  uint32_t set = graph.getTarget().makeFpIctlValue(behaviour.inv,
                                                    behaviour.div0,
                                                    behaviour.oflo,
                                                    behaviour.esr,
                                                    behaviour.nanoo);

  auto cs = graph.addComputeSet(debugPrefix + "/setFloatingPointBehaviour");
  auto numTiles = graph.getTarget().getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    auto v = graph.addVertex(cs,
              poputil::templateVertex("popsys::PutSupervisorCSR",
                                      graph.getTarget().getFpIctlRegIndex()));
    graph.setInitialValue(v["setVal"], set);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));

}

} // end namespace popsys
