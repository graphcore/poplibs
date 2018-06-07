#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

using namespace poplar;

namespace popops {

#define INSTANTIATE_TEMPLATES(name) \
  template class name<unsigned, float>; \
  template class name<unsigned, half>; \
  template class name<unsigned, unsigned>; \
  template class name<unsigned, int>; \
  template class name<int, float>; \
  template class name<int, half>; \
  template class name<int, unsigned>; \
  template class name<int, int>


template <typename IndexType, typename OutType>
class EncodeOneHot : public Vertex {
public:
  Input<IndexType> index;
  Output<Vector<OutType>> out;
  bool compute() {
    if (index < out.size()) {
      out[index] = 1;
    }
    return true;
  }
};

INSTANTIATE_TEMPLATES(EncodeOneHot);

template <typename IndexType, typename OutType>
class EncodeOneHot2D : public Vertex {
public:
  Input<Vector<IndexType>> indices;
  Output<VectorList<OutType, VectorListLayout::DELTAN>> out;
  bool compute() {
    for (std::size_t i = 0; i < indices.size(); i++) {
      const auto index = indices[i];
      if (index < out[i].size()) {
        out[i][index] = 1;
      }
    }
    return true;
  }
};

INSTANTIATE_TEMPLATES(EncodeOneHot2D);

}
