#include "popfloatCodelets.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <experimental/popfloat/GfloatExpr.hpp>
#include <ipudef.h>
#include <poplar/Vertex.hpp>
#include <print.h>

static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

using namespace poplar;

namespace experimental {
namespace popfloat {

class CastGf8ToFloat : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<char, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<float, SPAN, 8>>, SPAN> out;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() { return true; }
};

} // end namespace popfloat
} // end namespace experimental
