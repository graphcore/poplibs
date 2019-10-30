#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace poplin {

template <class FPType> class WgdConvComplete : public Vertex {

public:
  /* Each input vector is a of length "vecLen"
   */
  Vector<Input<Vector<FPType>>> dIn;

  /* The output activation once non-linearity is applied
   */
  Vector<Output<Vector<FPType, ONE_PTR>>, ONE_PTR> act;

  bool compute() {
    const unsigned nGroups = dIn.size();
    const unsigned vecLen = dIn[0].size();

    for (unsigned gr = 0; gr < nGroups; ++gr) {
      for (unsigned el = 0; el < vecLen; ++el) {
        act[gr][el] = dIn[gr][el];
      }
    }
    return true;
  }
};

template class WgdConvComplete<float>;
template class WgdConvComplete<half>;

} // namespace poplin
