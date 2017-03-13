#ifndef __ActivationMapping_hpp__
#define __ActivationMapping_hpp__
#include <vector>
#include "poplar/Graph.hpp"

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

#endif // __ActivationMapping_hpp__
