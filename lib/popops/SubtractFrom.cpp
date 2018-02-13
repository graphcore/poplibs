#include "popops/SubtractFrom.hpp"

#include "popops/Add.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

namespace popops {

  void subtractFrom(Graph &graph, Tensor A, Tensor B, float k,
             Sequence &prog, const std::string &debugPrefix) {

    addTo(graph, A, B, -k, prog, debugPrefix);
  }

  void subtractFrom(Graph &graph, Tensor A, Tensor B,
               Sequence &prog, const std::string &debugPrefix) {

    subtractFrom(graph, A, B, 1.0, prog, debugPrefix);
  }

}
