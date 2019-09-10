#include "popops/Zero.hpp"

#include <poplar/Graph.hpp>
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

void zero(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix) {
  const auto zero =
    graph.addConstant(t.elementType(), t.shape(), 0, debugPrefix + "/zero");
  graph.setTileMapping(zero, graph.getTileMapping(t));
  prog.add(Copy(zero, t));
}

} // end namespace popops
