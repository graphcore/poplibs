#ifndef _scaledadd_internal_h_
#define _scaledadd_internal_h_

using namespace poplar;
using namespace poplar::program;

namespace popops {

void scaledArithmeticConstImpl(Graph &graph,
      Tensor A,
      float scaleA,
      Tensor B,
      float scaleB,
      Sequence &prog,
      const std::string &debugPrefix);

void scaledArithmeticTensorImpl(Graph &graph,
      Tensor A,
      Tensor scaleA,
      Tensor B,
      Tensor scaleB,
      const bool doSubtract,
      const bool doaXPlusbY,
      Sequence &prog,
      const std::string &debugPrefix);
}

#endif
