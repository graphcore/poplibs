// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "popops/EncodingConstants.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popnn {

template <typename LabelType> class CalcAccuracy : public Vertex {
public:
  CalcAccuracy();

  Input<Vector<LabelType>> maxPerBatch;
  Input<Vector<LabelType, ONE_PTR>> expected;
  InOut<unsigned> numCorrect;

  bool compute() {
    auto count = *numCorrect;
    for (std::size_t i = 0; i < maxPerBatch.size(); ++i) {
      if (expected[i] != MASKED_LABEL_CODE) {
        count += (maxPerBatch[i] == expected[i]);
      }
    }
    *numCorrect = count;
    return true;
  }
};

template class CalcAccuracy<unsigned int>;
template class CalcAccuracy<int>;

} // namespace popnn
