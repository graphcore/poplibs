#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath>
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;
namespace popops {

class CircBufIncrIndex : public Vertex {
public:
  InOut<unsigned> index;
  unsigned hSize;
  bool compute() {
    *index = (*index + 1) % hSize;
    return true;
  }
};

class CircOffset : public Vertex {
public:
  Input<unsigned> indexIn;
  Output<unsigned> indexOut;
  unsigned hSize;
  unsigned offset;
  bool compute() {
    auto updated = *indexIn + offset;
    if (updated >= hSize) {
      updated -= hSize;
    }
    *indexOut = updated;
    return true;
  }
};

}
