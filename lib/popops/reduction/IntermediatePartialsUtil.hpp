#ifndef IntermediatePartialsUtil_hpp
#define IntermediatePartialsUtil_hpp

#include "IntermediatePartials.hpp"

namespace popops {

/// Given a 2D tensor with `wrapSize` columns and the given tile mapping, this
/// determines if any tile has more than one value from the same column mapped
/// to it.
bool mappingHasMultipleValuesFromOneColumnOnTheSameTile(
    const poplar::Graph::TileToTensorMapping &mapping, std::size_t wrapSize);

struct ReductionDebug;

/// Given a 2D tensor where only one value from each column is present on a
/// single tile, this converts it into the IntermediatePartials format.
///
/// If those conditions are not true an exception is thrown.
IntermediatePartials
tensorToIntermediatePartials(const poplar::Tensor &A,
                             const poplar::Graph::TileToTensorMapping &mapping,
                             ReductionDebug *debug);

}

#endif // IntermediatePartialsUtil_hpp
