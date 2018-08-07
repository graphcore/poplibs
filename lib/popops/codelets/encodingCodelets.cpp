#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include "poplibs_support/ExternalCodelet.hpp"

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
    for (std::size_t i = 0; i < out.size(); i++) {
      out[i] = OutType(i == index);
    }
    return true;
  }
};

INSTANTIATE_TEMPLATES(EncodeOneHot);

template <typename IndexType, typename OutType>
class EncodeOneHot2D : public Vertex {
public:
  Input<Vector<IndexType, VectorLayout::ONE_PTR>> indices;
  Vector<Output<Vector<OutType>>> out;
  bool compute() {
    for (std::size_t i = 0; i < out.size(); i++) {
      const auto index = indices[i];
      for (std::size_t j = 0; j < out[i].size(); j++) {
        out[i][j] = OutType(j == index);
      }
    }
    return true;
  }
};

INSTANTIATE_TEMPLATES(EncodeOneHot2D);

}
