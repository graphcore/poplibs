// Copyright (c) Graphcore Ltd, All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace poplin {

template <class T>
class [[poplar::constraint("elem(*weights) != elem(**out)")]] OuterProduct
    : public Vertex {
public:
  OuterProduct();

  Input<Vector<T>> in;
  Input<Vector<T, ONE_PTR, 8>> weights;
  Vector<Output<Vector<T, ONE_PTR, 8>>> out;
  const unsigned chansPerGroup;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const auto width = in.size();
    const auto numChanGroups = out.size();

    for (unsigned g = 0; g != numChanGroups; ++g) {
      for (unsigned chanInGroup = 0; chanInGroup != chansPerGroup;
           ++chanInGroup) {
        const auto c = chanInGroup + g * chansPerGroup;
        for (unsigned x = 0; x != width; ++x) {
          out[g][chanInGroup + x * chansPerGroup] = in[x] * weights[c];
        }
      }
    }
    return true;
  }
};

template class OuterProduct<float>;
template class OuterProduct<half>;

} // end namespace poplin
