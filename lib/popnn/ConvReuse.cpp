#include "popnn/internal/ConvReuse.hpp"
#include <tuple>
#include <cassert>

using namespace poplar;
using namespace poplar::program;

bool ConvImplSpec::operator<(const ConvImplSpec &other) const {
  auto t1 = std::tie(tensorDims, kernelSizeY, kernelSizeX,
                     strideY, strideX, paddingY, paddingX,
                     nonLinearityType,
                     resMethod);
  auto t2 = std::tie(other.tensorDims, other.kernelSizeY, other.kernelSizeX,
                     other.strideY, other.strideX, other.paddingY,
                     other.paddingX, other.nonLinearityType, other.resMethod);
  return t1 < t2;
}

Program ReusableLayer::apply(const std::vector<Tensor> &in,
                             const std::vector<Tensor> &out) {
  auto prog = Sequence();
  assert(inputs.size() == in.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    prog.add(Copy(inputs[i], in[i]));
  }
  prog.add(this->prog);
  assert(outputs.size() == out.size());
  for (unsigned i = 0; i < out.size(); ++i) {
    prog.add(Copy(out[i], outputs[i]));
  }
  return prog;
}
