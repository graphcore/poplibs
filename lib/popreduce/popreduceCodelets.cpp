#include "PerformanceEstimation.hpp"
#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <vector>

using namespace poplar;

namespace popreduce {

template <typename OutType, typename PartialsType>
class Reduce : public Vertex {
public:
  Vector<Output<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType>>> partials;

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

  uint64_t getCycleEstimate() const {
    std::vector<unsigned> outSizes;
    for (const auto &o : out)
      outSizes.push_back(o.size());
    return reduceCycleEstimate<OutType, PartialsType>(outSizes,
                                                      partials.size(),
                                                      dataPathWidth,
                                                      false, false);
  }
};

template class Reduce<float, float>;
template class Reduce<half, float>;
template class Reduce<half, half>;

template <typename OutType, typename PartialsType>
class ReduceUpdate : public Vertex {
public:
  Vector<InOut<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType>>> partials;
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

  uint64_t getCycleEstimate() const {
    std::vector<unsigned> outSizes;
    for (const auto &o : out)
      outSizes.push_back(o.size());
    return reduceCycleEstimate<OutType, PartialsType>(outSizes,
                                                      partials.size(),
                                                      dataPathWidth,
                                                      true, false);
  }
};

template class ReduceUpdate<float, float>;
template class ReduceUpdate<half, float>;
template class ReduceUpdate<half, half>;

template <typename OutType, typename PartialsType>
class ReduceScale : public Vertex {
public:
  Vector<InOut<Vector<OutType>>> out;
  Vector<Input<Vector<PartialsType>>> partials;
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

  uint64_t getCycleEstimate() const {
    std::vector<unsigned> outSizes;
    for (const auto &o : out)
      outSizes.push_back(o.size());
    return reduceCycleEstimate<OutType, PartialsType>(outSizes,
                                                      partials.size(),
                                                      dataPathWidth,
                                                      false, true);
  }
};

template class ReduceScale<float, float>;
template class ReduceScale<half, float>;
template class ReduceScale<half, half>;


} // end namespace popreduce
