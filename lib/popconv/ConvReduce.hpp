#ifndef ConvReduce_hpp
#define ConvReduce_hpp

#include <string>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>

namespace popconv {

poplar::Tensor
multiStageGroupedReduce(poplar::Graph &graph,
                        poplar::Tensor partials,
                        const poplar::Type &resultType,
                        std::vector<poplar::ComputeSet> &computeSets,
                        const std::string &debugPrefix);

}

#endif // ConvReduce_hpp
