#ifndef __popstd_ActivationMapping_hpp__
#define __popstd_ActivationMapping_hpp__
#include <vector>
#include "poplar/Graph.hpp"

namespace popstd {

void applyTensorMapping(poplar::Graph &graph, poplar::Tensor t,
                        const std::vector<unsigned> &mapping);

std::vector<unsigned>
computeActivationsMapping(const poplar::Graph &graph,
                          const std::string &actType,
                          const std::vector<std::size_t> &shape,
                          unsigned batchNum, unsigned batchSize);

std::vector<unsigned> computeActivationsMapping(const poplar::Graph &graph,
                                                poplar::Tensor t,
                                                unsigned batchNum,
                                                unsigned batchSize);

void mapActivations(poplar::Graph &graph, poplar::Tensor t);

std::vector<unsigned> computeTensorMapping(const poplar::Graph &graph,
                                           poplar::Tensor t,
                                           unsigned grainSize = 1);

void mapTensor(poplar::Graph &graph, poplar::Tensor t);

}  // end namespace popstd

#endif // __popstd_ActivationMapping_hpp__
