#include "poplibs_support/ExternalCodelet.hpp"
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
namespace popops {

class CircBufIncrIndex : public Vertex {
public:
  CircBufIncrIndex();

  InOut<unsigned> index;
  const unsigned hSize;
  bool compute() {
    *index = (*index + 1) % hSize;
    return true;
  }
};

class CircOffset : public Vertex {
public:
  CircOffset();

  Input<unsigned> indexIn;
  Output<unsigned> indexOut;
  const unsigned hSize;
  const unsigned offset;
  bool compute() {
    auto updated = *indexIn + offset;
    if (updated >= hSize) {
      updated -= hSize;
    }
    *indexOut = updated;
    return true;
  }
};

} // namespace popops
