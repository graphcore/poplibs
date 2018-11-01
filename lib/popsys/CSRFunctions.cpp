#include "popsys/CSRFunctions.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/VertexTemplates.hpp"

#define __IPU_ARCH_VERSION__ 0
#include <tilearch.h>


using namespace poplar;
using namespace poplar::program;

namespace popsys {

void setStochasticRounding( Graph &graph,
                                Sequence &prog,
                                bool behaviour,
                                unsigned tile,
                                const std::string &debugPrefix) {

  auto cs = graph.addComputeSet(debugPrefix + "/setStochasticRounding");
  auto v = graph.addVertex(cs,
                        poputil::templateVertex("popsys::ModifySupervisorCSR",
                        CSR_S_FP_ICTL__INDEX));

  graph.setInitialValue(v["clearVal"], ~(1 << CSR_S_FP_ICTL__ESR__SHIFT));
  graph.setInitialValue(v["setVal"],
                static_cast<unsigned>(behaviour) << CSR_S_FP_ICTL__ESR__SHIFT);

  graph.setTileMapping(v, tile);

  prog.add(Execute(cs));
}

void setFloatingPointBehaviour( Graph &graph,
                                Sequence &prog,
                                floatingPointBehaviour &behaviour,
                                unsigned tile,
                                const std::string &debugPrefix) {

  auto cs = graph.addComputeSet(debugPrefix + "/setFloatingPointBehaviour");
  auto v = graph.addVertex(cs,
              poputil::templateVertex("popsys::PutSupervisorCSR",
                                                    CSR_S_FP_ICTL__INDEX));

  unsigned set =
      (static_cast<unsigned>(behaviour.inv) << CSR_S_FP_ICTL__INV__SHIFT) |
      (static_cast<unsigned>(behaviour.div0) << CSR_S_FP_ICTL__DIV0__SHIFT) |
      (static_cast<unsigned>(behaviour.oflo) << CSR_S_FP_ICTL__OFLO__SHIFT)|
      (static_cast<unsigned>(behaviour.esr) << CSR_S_FP_ICTL__ESR__SHIFT)  |
      (static_cast<unsigned>(behaviour.nanoo) << CSR_S_FP_ICTL__NANOO__SHIFT);

  graph.setInitialValue(v["setVal"], set);

  graph.setTileMapping(v, tile);

  prog.add(Execute(cs));
}

} // end namespace popsys
