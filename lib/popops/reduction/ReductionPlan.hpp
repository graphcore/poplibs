#ifndef ReductionPlan_hpp
#define ReductionPlan_hpp

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <boost/icl/split_interval_map.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>

#include "popops/Reduce.hpp"

#include "IntermediatePartials.hpp"

namespace popops {

/// Get the maximum number of tiles a 2D reduction tensor is spread over.
///
/// \param mapping     The tile mapping for the tensor.
/// \param outputSize  The number of columns in the tensor (i.e. the size of the
///                    reduced output).
std::size_t getMaxTileSpread(const poplar::Graph::TileToTensorMapping &mapping,
                             std::size_t outputSize);

/// Get the best tile mapping for the output tensor based on where the data is
/// mapped for the final stage of the reduction. This is used when the user
/// hasn't set a tile mapping for the output (or when we create the output
/// tensor). It may also be used if we decide it is fast to do the reduction
/// spread over the IPU and then exchange to its final destination.
///
/// \param target      The target
/// \param ipIn        The intermediate reduction result for the penultimate
///                    reduction stage.
/// \param outMapping  The tile mapping for the output tensor.
/// \param reducedType The output type of the reduction
/// \param numReducedElements   The number of elements in the output tensor.
///
/// \returns True if the reduction should be done at the destination, false
///          if it should be distributed over the IPU and exchanged afterwards.
bool shouldReduceAtDestination(
    const poplar::Target &target,
    const IntermediatePartials &ipIn,
    const poplar::Graph::TileToTensorMapping &outMapping,
    poplar::Type reducedType,
    std::size_t numReducedElements);

/// Split an intermediate reduction up into chunks that each tile can process.
/// It returns an interval map - each element of the map is a bit of the
/// final output tensor, and the map value is how many pieces to split it up
/// vertically. For example suppose we had the following intermediate partials,
/// and the number of pieces we are aiming for (numPieces) is 10.
///
/// .------------------------------------.
/// |       |                            |
/// |       |                            |
/// |       |                            |
/// |       |                            |
/// |       |----------------------------`
/// |       |
/// |       |
/// `-------`
///
/// We first split it horizontally as much as possible, to try to get 10
/// pieces. This tries to respect the grainSize and keep the piece width at
/// least minPieceCols.
///
/// .------------------------------------.
/// |       |        |        |          |
/// |       |        |        |          |
/// |       |        |        |          |
/// |       |        |        |          |
/// |       |----------------------------`
/// |       |
/// |       |
/// `-------`
///
/// In some cases it may not be possible to do enough horizontal splits to get
/// 10 pieces. In this case we split them vertically too.
///
/// .------------------------------------.
/// |       |        |        |          |
/// |~~~~~~~|        |        |          |
/// |       |~~~~~~~~|~~~~~~~~|~~~~~~~~~~|
/// |~~~~~~~|        |        |          |
/// |       |----------------------------`
/// |~~~~~~~|
/// |       |
/// `-------`
///
/// The result is an interval map like this:
///
/// |  4    |   2    |    2    |   2     |
///
///
/// \param grainSize   Columns will be split on grain boundaries, e.g. if this
///                    is 4, you might get pieces with 4, 8, 12, etc. columns
///                    but it will avoid (not guaranteed) 1, 2, 3, 5, 6, 7...
///
/// \param minPieceCols  Minimum number of columns in a piece (not guaranteed).
/// \param minPieceRows  Minimum number of rows in a piece (not guaranteed).
/// \param minPieceSize  Minimum number of element sin a piece (not guaranteed).
/// \param numPieces     The number of pieces to aim for - it may produce less
///                      but not more.
///
/// \returns A split map from output intervals to the number of pieces to split
///          each of those columns into.
///
boost::icl::split_interval_map<std::size_t, std::size_t>
calculateSplit(const IntermediatePartials &ir,
               std::size_t grainSize,
               std::size_t minPieceCols,
               std::size_t minPieceRows,
               std::size_t minPieceSize,
               unsigned numPieces);

enum NextStep {
  INTERMEDIATE_TO_INTERMEDIATE,
  INTERMEDIATE_TO_OUTPUT,
};

/// Decide whether or not to do another reduction stage, or to reduce everything
/// to the final output tensor.
NextStep calculateNextStep(const IntermediatePartials &ir);

}

#endif // ReductionPlan_hpp
