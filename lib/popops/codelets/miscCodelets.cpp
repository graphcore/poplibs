#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath>

using namespace poplar;
namespace popops {

class AllTrue : public Vertex {
public:
  Vector<Input<Vector<bool>>> in;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    bool v = true;
    for (unsigned i = 0; i != in.size(); ++i) {
      for (unsigned j = 0; j != in[i].size(); ++j) {
        v = v && in[i][j];
      }
    }
    return v;
  }
};

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
