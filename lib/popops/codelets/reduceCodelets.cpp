#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <limits>

#include "util.hpp"

using namespace poplar;
namespace popops {
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto TWO_PTR = poplar::VectorLayout::TWO_PTR;

template <typename OutType, typename PartialsType>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
ReduceAdd : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType, ONE_PTR, 1, true>>> partials;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        float sum = 0;
        for (unsigned j = 0; j < numPartials; ++j) {
          sum += partials[r * numPartials + j][i];
        }
        out[r][i] = sum;
      }
    }
    return true;
  }
};

template class ReduceAdd<float, float>;
template class ReduceAdd<half, float>;
template class ReduceAdd<half, half>;
template class ReduceAdd<int, int>;

template <typename OutType, typename PartialsType>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
ReduceAddUpdate : public Vertex {
public:
  Vector<InOut<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType, ONE_PTR, 1, true>>> partials;
  float k;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        float sum = 0;
        for (unsigned j = 0; j < numPartials; ++j) {
          sum += partials[r * numPartials + j][i];
        }
        out[r][i] += k * sum;
      }
    }
    return true;
  }
};

template class ReduceAddUpdate<float, float>;
template class ReduceAddUpdate<half, float>;
template class ReduceAddUpdate<half, half>;
template class ReduceAddUpdate<int, int>;

template <typename OutType, typename PartialsType>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
ReduceAddScale : public Vertex {
public:
  Vector<InOut<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType, ONE_PTR, 1, true>>> partials;
  float k;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        float sum = 0;
        for (unsigned j = 0; j < numPartials; ++j) {
          sum += partials[r * numPartials + j][i];
        }
        out[r][i] = k * sum;
      }
    }
    return true;
  }
};

template class ReduceAddScale<float, float>;
template class ReduceAddScale<half, float>;
template class ReduceAddScale<half, half>;
template class ReduceAddScale<int, int>;


template <typename OutType, typename PartialsType>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
ReduceMul : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType, ONE_PTR, 1, true>>> partials;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        OutType prod = 1;
        for (unsigned j = 0; j < numPartials; ++j) {
          prod *= partials[r * numPartials + j][i];
        }
        out[r][i] = prod;
      }
    }
    return true;
  }
};

template class ReduceMul<float, float>;
template class ReduceMul<half, half>;
template class ReduceMul<half, float>;
template class ReduceMul<int, int>;

template <typename OutType, typename PartialsType>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
ReduceMax : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType, ONE_PTR, 1, true>>> partials;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        OutType maxVal = std::numeric_limits<PartialsType>::lowest();
        for (unsigned j = 0; j < numPartials; ++j) {
          maxVal = max(maxVal, partials[r * numPartials + j][i]);
        }
        out[r][i] = maxVal;
      }
    }
    return true;
  }
};

template class ReduceMax<float, float>;
template class ReduceMax<half, half>;
template class ReduceMax<int, int>;


template <typename OutType, typename PartialsType>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
ReduceMin : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType, TWO_PTR, 1, true>>> partials;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        OutType minVal = std::numeric_limits<PartialsType>::max();
        for (unsigned j = 0; j < numPartials; ++j) {
          minVal = min(minVal, partials[r * numPartials + j][i]);
        }
        out[r][i] = minVal;
      }
    }
    return true;
  }
};

template class ReduceMin<float, float>;
template class ReduceMin<half, half>;
template class ReduceMin<int, int>;


template <typename OutType, typename PartialsType>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
ReduceAnd : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType, TWO_PTR, 1, true>>> partials;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        OutType res = true;
        for (unsigned j = 0; j < numPartials; ++j) {
          res = res && partials[r * numPartials + j][i];
        }
        out[r][i] = res;
      }
    }
    return true;
  }
};

template class ReduceAnd<bool, bool>;


template <typename OutType, typename PartialsType>
class
[[poplar::constraint("elem(**out) != elem(**partials)")]]
ReduceOr : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType, TWO_PTR, 1, true>>> partials;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    unsigned numReductions = out.size();
    unsigned numPartials = partials.size() / numReductions;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = out[r].size();
      for (unsigned i = 0; i < numElem; ++i) {
        OutType res = false;
        for (unsigned j = 0; j < numPartials; ++j) {
          res = res || partials[r * numPartials + j][i];
        }
        out[r][i] = res;
      }
    }
    return true;
  }
};

template class ReduceOr<bool, bool>;

}
