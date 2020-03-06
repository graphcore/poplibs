// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#ifndef ReductionConnection_hpp
#define ReductionConnection_hpp

#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include "ComputeSetList.hpp"
#include "Reduction.hpp"
#include "ReductionDebug.hpp"
#include <boost/range.hpp>
#include <boost/variant.hpp>

namespace popops {

enum class ReductionSpecialisation {
  // TODO: T12965 Swap 2&3 so that higher specialisations are cheaper.

  DEFAULT,
  SCALAR_OUTPUT_REGIONS,
  SCALAR_OUTPUT_SINGLE_INPUT,
  SINGLE_OUTPUT_REGION,
  ALL_REGIONS_CONTINUOUS,
  PARTIALS_EQUAL_SIZE
};
constexpr unsigned numReductionSpecialisations = 6;

/// This structure represents the reduction of a set of 1D input regions
/// to a single 1D output region. One reduction vertex can reduce a set
/// of these.
///
/// The regions are represented by as poplar::Tensors. The size of the partial
/// regions must be equal to a multiple of the size of the output region.
///
/// The shape of the partial and output tensors is ignored - they are treated
/// as if they are 1D. partials that are longer than the output tensor are
/// wrapped.
///
struct RegionReduction {
  // The output region.
  poplar::Tensor output;
  // The input regions.
  std::vector<poplar::Tensor> partials;
  // innerFactor indicates that each partial contains innerFactor elements
  // to be reduced into the 1st output element, followed by innerFactor elements
  // to be reduced into the second etc...  A two stage approach is used to
  // implement this.
  unsigned innerFactor = 1;
  unsigned outerFactor = 1;

  // Debug information about the partials and output.
  ReductionDebug::Output outputDebugInfo;
  std::vector<ReductionDebug::Partial> partialsDebugInfo;
};

/// Add vertices to the graph to perform the given reductions on the specified
/// tile and connect the vertex inputs and outputs.
///
/// If every partial region is exactly the same size as its output a more
/// optimal code path is automatically used.
///
/// The case where the output region is small an on-tile two-stage reduction
/// may be performed. This is why a vector of compute sets is passed
/// instead of a single one. The vector will be enlarged to the number
/// of compute sets required if necessary, which will always be 1 or 2.
///
/// If two compute sets are used, there will never be any exchange between
/// them so a local tile sync will be performed between them rather than a
/// full IPU sync.
///
/// The reductions are *distributed* between vertices, but not split (except
/// as described above). Before calling this function you should ensure that
/// the reductions are split appropriately so there are enough to distribute.
/// It makes a basic attempt to keep the split roughly even based on an estimate
/// of the number of cycles each reduction will take.
///
/// \param graph  The compute graph.
/// \param css    The compute sets to add the vertices to. This may use one
///               or two compute sets.
/// \param params The reduce operation to perform. Note that in multi-stage
///               operations you only want to do the scale or update in the
///               last stage.
/// \param inputType     The type of the input to this stage of the reduction.
/// \param partialType   The type of any partials created at this stage of the
///                      reduction.
/// \param outputType    The type of the outputs from this stage of the
///                      reduction.
/// \param tile          The tile to map the vertices to.
/// \param reductions    The set of reductions to distribute between vertices.
/// \param debugPrefix   Prefix for the compute sets that are added.
/// \param tileDebug     Will be filled with debug information.
///
void connectReductions(poplar::Graph &graph, ComputeSetList &css,
                       ReduceParams params, poplar::Type inputType,
                       poplar::Type partialType, poplar::Type outputType,
                       unsigned tile,
                       const std::vector<RegionReduction> &reductions,
                       bool reductionUsesInput, const std::string &debugPrefix,
                       ReductionDebug::TileReduction *tileDebug);

/// Find the appropriate vertex specialisation to use
/// \param graph   The compute graph
/// \param params  The reduce operation to solve
/// \param regions The set of reductions to perform
/// \param reductionUsesInput Flag - reduction is input stage or intermediate
using RegionReductionRange =
    boost::iterator_range<std::vector<RegionReduction>::const_iterator>;

ReductionSpecialisation getReductionVertexSpecialisation(
    const poplar::Graph &graph, const ReduceParams &params,
    const RegionReductionRange regions, poplar::Type partialType,
    bool reductionUsesInput);

bool inline reductionSupportsScaling(ReductionSpecialisation specialisation) {
  return specialisation == ReductionSpecialisation::DEFAULT ||
         specialisation == ReductionSpecialisation::SCALAR_OUTPUT_REGIONS ||
         specialisation == ReductionSpecialisation::SINGLE_OUTPUT_REGION ||
         specialisation == ReductionSpecialisation::ALL_REGIONS_CONTINUOUS ||
         specialisation == ReductionSpecialisation::PARTIALS_EQUAL_SIZE;
}
} // namespace popops
#endif // ReductionConnection_hpp
