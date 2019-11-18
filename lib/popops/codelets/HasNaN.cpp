#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cmath>
#include <cstdlib>

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;

namespace popops {

template <typename FPType> class HasNaN : public Vertex {
public:
  HasNaN();

  Vector<Input<Vector<FPType>>> in;

  bool compute() {
    for (unsigned i = 0; i < in.size(); ++i) {
      for (unsigned j = 0; j < in[i].size(); ++j) {
        if (std::isnan(float(in[i][j]))) {
          return false;
        }
      }
    }

    return true;
  }
};

template class HasNaN<float>;
template class HasNaN<half>;

} // namespace popops
