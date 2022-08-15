// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

namespace poplin {

template <typename OutType, typename PartialsType, bool singleInput,
          bool partialsMemConstraints>
class ReduceAdd : public MultiVertex {
private:
  constexpr static unsigned partialsAlign = partialsMemConstraints ? 16 : 8;

  static const bool needsAlignWorkers = false;

public:
  ReduceAdd();

  Vector<Input<Vector<PartialsType, ONE_PTR, partialsAlign,
                      partialsMemConstraints>>,
         COMPACT_PTR, 4>
      partials;
  Output<Vector<OutType, COMPACT_PTR, 8>> out;
  const unsigned short numPartials;
  const unsigned short numElems;

  IS_EXTERNAL_CODELET(true);

  void compute(unsigned wid) {
    if (wid == 0) {
      for (unsigned i = 0; i < numElems; ++i) {
        float sum = 0;
        for (unsigned j = 0; j < numPartials; ++j) {
          sum += float(partials[j][i]);
        }
        out[i] = sum;
      }
    }
  }
};

template <typename OutType, typename PartialsType, bool partialsMemConstraints>
class ReduceAdd<OutType, PartialsType, true, partialsMemConstraints>
    : public MultiVertex {
private:
  constexpr static unsigned partialsAlign = partialsMemConstraints ? 16 : 8;

  static const bool needsAlignWorkers = false;

public:
  ReduceAdd();
  // Intention is that initialPartials are those found on tile, partials
  // are those that are exchanged, although nothing makes this have to be the
  // case.  Exchange should gather all partials together, but separate from
  // initialPartial in another edge.
  // initialPartial must be of size numElems,
  // partials must be of size numPartials * numElems
  Input<
      Vector<PartialsType, COMPACT_PTR, partialsAlign, partialsMemConstraints>>
      partials;
  Output<Vector<OutType, COMPACT_PTR, 8>> out;
  const unsigned short numPartials;
  const unsigned short numElems;
  Input<Vector<PartialsType, COMPACT_PTR, 8, false>> initialPartial;

  IS_EXTERNAL_CODELET(true);

  void compute(unsigned wid) {
    if (wid == 0) {
      for (unsigned i = 0; i < numElems; ++i) {
        float sum = float(initialPartial[i]);
        for (unsigned j = 0; j != numPartials; ++j) {
          sum += float(partials[j * numElems + i]);
        }
        out[i] = sum;
      }
    }
  }
};

template class ReduceAdd<float, float, false, false>;
template class ReduceAdd<half, float, false, false>;
template class ReduceAdd<float, half, false, false>;
template class ReduceAdd<half, half, false, false>;

template class ReduceAdd<float, float, true, false>;
template class ReduceAdd<half, float, true, false>;
template class ReduceAdd<float, half, true, false>;
template class ReduceAdd<half, half, true, false>;

template class ReduceAdd<float, float, true, true>;
template class ReduceAdd<half, float, true, true>;
template class ReduceAdd<float, half, true, true>;
template class ReduceAdd<half, half, true, true>;

} // end namespace poplin
